from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import uuid
import json
import time
import mediapipe as mp
from scipy.signal import savgol_filter
from tqdm import tqdm
import cv2
import numpy as np
from scipy.interpolate import interp1d
from filterpy.kalman import KalmanFilter
import traceback
import tensorflow as tf  # For EfficientDet Lite 0

from pefuml.measure import (
    calculate_joint_angles,
    calculate_lunge_distance,
    calculate_flexion_extension_angles,
    calculate_torso_parameters,
    calculate_lunge_angles,
    calculate_azimuth_elevation_angles,
    calculate_distances,
    calculate_heights,
    calculate_velocities,
    calculate_center_of_gravity,
    get_default_angles
)

app = Flask(__name__)
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Models directory
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Initialize EfficientDet model
EFFICIENTDET_MODEL_PATH = os.path.join(MODELS_DIR, 'efficientdet_lite0_fp32.tflite')
person_detector_available = False
interpreter = None

# Try to load or download the model
try:
    # Check if model exists, if not try to download
    if not os.path.exists(EFFICIENTDET_MODEL_PATH):
        try:
            print(f"Downloading EfficientDet Lite 0 model...")
            import urllib.request

            # This is the TF Hub model URL
            model_url = "https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1.tflite"
            urllib.request.urlretrieve(model_url, EFFICIENTDET_MODEL_PATH)
            print(f"Model downloaded successfully to {EFFICIENTDET_MODEL_PATH}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Will use MediaPipe's built-in detection instead.")

    # Load the model
    interpreter = tf.lite.Interpreter(model_path=EFFICIENTDET_MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get model input shape
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]

    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    print(f"EfficientDet Lite 0 model loaded successfully. Input shape: {input_height}x{input_width}")
    person_detector_available = True
except Exception as e:
    print(f"Warning: EfficientDet Lite 0 model could not be loaded: {e}")
    print("Will rely on MediaPipe only.")
    person_detector_available = False

# Physical constraints for human motion
JOINT_VELOCITY_LIMITS = {
    'head': 10.0,  # rad/s - increased for more flexibility
    'shoulder': 8.0,
    'elbow': 14.0,
    'wrist': 20.0,
    'hip': 7.0,
    'knee': 12.0,
    'ankle': 16.0,
    'spine': 6.0
}

# Joint acceleration limits (rad/s²)
JOINT_ACCELERATION_LIMITS = {
    'head': 40.0,
    'shoulder': 35.0,
    'elbow': 50.0,
    'wrist': 70.0,
    'hip': 30.0,
    'knee': 45.0,
    'ankle': 55.0,
    'spine': 25.0
}

# Mapping from MediaPipe joints to joint types
JOINT_TYPE_MAPPING = {
    'NOSE': 'head',
    'LEFT_SHOULDER': 'shoulder',
    'RIGHT_SHOULDER': 'shoulder',
    'LEFT_ELBOW': 'elbow',
    'RIGHT_ELBOW': 'elbow',
    'LEFT_WRIST': 'wrist',
    'RIGHT_WRIST': 'wrist',
    'LEFT_HIP': 'hip',
    'RIGHT_HIP': 'hip',
    'LEFT_KNEE': 'knee',
    'RIGHT_KNEE': 'knee',
    'LEFT_ANKLE': 'ankle',
    'RIGHT_ANKLE': 'ankle'
}


def detect_people_efficientdet(image, confidence_threshold=0.1):
    """
    Detect people in the image using EfficientDet Lite 0.
    Returns bounding boxes of detected people.
    """
    if not person_detector_available or interpreter is None:
        return []

    try:
        # Get input details
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        input_height, input_width = input_shape[1], input_shape[2]

        # Get output details - inspect what's actually available
        output_details = interpreter.get_output_details()

        # Print output details for debugging
        print(f"Number of output tensors: {len(output_details)}")
        for i, output in enumerate(output_details):
            print(f"Output {i}: {output['name']} - shape: {output['shape']}")

        # Resize and preprocess the image
        resized_image = cv2.resize(image, (input_width, input_height))

        # Convert to RGB if needed
        if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and add batch dimension
        input_data = np.expand_dims(resized_image.astype(np.float32) / 255.0, axis=0)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensors - adjusted for EfficientDet Lite 0's actual outputs
        # EfficientDet Lite 0 typically has 4 outputs:
        # - Location boxes (normalized coordinates)
        # - Classes
        # - Scores
        # - Number of detections

        # Safely get outputs
        # The indices might vary based on the model - we'll check each output's name or shape

        # For safety, let's determine which output is which based on shape or name
        boxes_idx, classes_idx, scores_idx, count_idx = None, None, None, None

        # typical shapes from EfficientDet: locations [1,25,4], classes [1,25], scores [1,25], num_detections [1]
        for i, details in enumerate(output_details):
            shape = details['shape']
            if len(shape) == 3 and shape[2] == 4:  # Location boxes [batch, detections, 4 coords]
                boxes_idx = i
            elif len(shape) == 2 and shape[1] > 1:  # Classes or scores [batch, detections]
                if classes_idx is None:
                    classes_idx = i
                elif scores_idx is None:
                    scores_idx = i
            elif len(shape) == 1 or (len(shape) == 2 and shape[1] == 1):  # Number of detections [batch] or [batch, 1]
                count_idx = i

        # If we couldn't identify all outputs, use fixed indices as fallback
        if boxes_idx is None: boxes_idx = 0
        if classes_idx is None: classes_idx = 1
        if scores_idx is None: scores_idx = 2
        if count_idx is None: count_idx = 3

        # Now safely get the outputs
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]

        # Only attempt to get other outputs if the indices are within range
        if classes_idx < len(output_details):
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        else:
            classes = np.ones(len(boxes))  # Default all to class 1

        if scores_idx < len(output_details):
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
        else:
            scores = np.ones(len(boxes)) * 0.5  # Default confidence of 0.5

        if count_idx < len(output_details):
            num_detections = int(interpreter.get_tensor(output_details[count_idx]['index'])[0])
        else:
            num_detections = len(boxes)  # Use all boxes

        # Filter for person class (class 0 in COCO)
        person_boxes = []
        for i in range(min(num_detections, len(boxes))):
            if i < len(scores) and scores[i] > confidence_threshold:
                if i < len(classes) and (int(classes[i]) == 0 or int(classes[i]) == 1):  # Allow class 0 or 1 for person
                    # EfficientDet outputs [ymin, xmin, ymax, xmax] normalized
                    h, w = image.shape[0], image.shape[1]

                    # Handle different box formats
                    if len(boxes[i]) == 4:
                        ymin, xmin, ymax, xmax = boxes[i]
                    else:
                        # If unexpected format, use first 4 values and hope they're coordinates
                        ymin, xmin, ymax, xmax = boxes[i][:4]

                    # Convert normalized coordinates to pixel values
                    xmin_px = max(0, int(xmin * w))
                    ymin_px = max(0, int(ymin * h))
                    xmax_px = min(w, int(xmax * w))
                    ymax_px = min(h, int(ymax * h))

                    # Ensure box is valid (non-zero area)
                    if xmin_px < xmax_px and ymin_px < ymax_px:
                        person_boxes.append([xmin_px, ymin_px, xmax_px, ymax_px])

        return person_boxes

    except Exception as e:
        print(f"Error in EfficientDet person detection: {e}")
        traceback.print_exc()
        return []


def preprocess_frame(image):
    """Enhanced preprocessing for better keypoint detection."""
    try:
        # Check if image is empty or invalid
        if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            print("Warning: Empty or invalid image received for preprocessing")
            return None

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply adaptive histogram equalization for better contrast in dark areas
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(enhanced_rgb, 9, 75, 75)

        return filtered
    except Exception as e:
        print(f"Error in preprocessing frame: {e}")
        traceback.print_exc()
        return image  # Return original image if preprocessing fails


def enhance_detection(image):
    """
    Enhance detection capabilities with basic image processing techniques.
    This helps in dark areas or where subject blends with background.
    """
    try:
        # Check if image is empty or invalid
        if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            print("Warning: Empty or invalid image received for enhancement")
            return image, None

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply background subtraction techniques
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Simple edge detection
        edges = cv2.Canny(blur, 50, 150)

        # Create a mask using edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Create enhanced image for detection
        enhanced = image.copy()
        enhanced_hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)

        # Increase saturation and value to make subject stand out
        enhanced_hsv[:, :, 1] = np.clip(enhanced_hsv[:, :, 1] * 1.3, 0, 255).astype(np.uint8)  # increase saturation
        enhanced_hsv[:, :, 2] = np.clip(enhanced_hsv[:, :, 2] * 1.2, 0, 255).astype(np.uint8)  # increase value

        enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)

        return enhanced, dilated
    except Exception as e:
        print(f"Error enhancing detection: {e}")
        traceback.print_exc()
        return image, None  # Return original image if enhancement fails


def initialize_kalman_filter(dim_x=99, dim_z=99):
    """Initialize Kalman filter for keypoint smoothing using a constant velocity model."""
    try:
        kf = KalmanFilter(dim_x=dim_x * 2, dim_z=dim_x)  # State space is 2x because we track position and velocity

        # Initialize state transition matrix for constant velocity model
        kf.F = np.eye(dim_x * 2)
        # Add velocity component to position
        dt = 1 / 30.0  # Assuming 30fps
        for i in range(dim_x):
            kf.F[i, i + dim_x] = dt

        # Measurement function (we only measure position, not velocity)
        kf.H = np.zeros((dim_x, dim_x * 2))
        for i in range(dim_x):
            kf.H[i, i] = 1.0

        # Measurement noise
        kf.R = np.eye(dim_x) * 0.01

        # Process noise (related to acceleration)
        kf.Q = np.eye(dim_x * 2)
        # Position elements
        for i in range(dim_x):
            kf.Q[i, i] = 0.01
        # Velocity elements (higher process noise)
        for i in range(dim_x, dim_x * 2):
            kf.Q[i, i] = 1.0

        # Initial state uncertainty
        kf.P = np.eye(dim_x * 2) * 100

        return kf
    except Exception as e:
        print(f"Error initializing Kalman filter: {e}")
        traceback.print_exc()
        return None


def apply_biomechanical_constraints(keypoints, prev_keypoints=None, dt=1 / 30.0):
    """
    Apply biomechanical constraints to make motion more realistic.
    This includes joint angle limits, bone length preservation, and natural motion patterns.
    """
    if not keypoints:
        return keypoints

    try:
        # Clone keypoints to avoid modifying the original
        constrained_keypoints = [{
            'x': kp['x'],
            'y': kp['y'],
            'z': kp['z'],
            'visibility': kp['visibility'],
            'index': kp['index'],
            'name': kp['name']
        } for kp in keypoints]

        # If no previous keypoints, just return the current ones
        if prev_keypoints is None:
            return constrained_keypoints

        # Define bone connections for length preservation
        bones = [
            # Torso and head
            (0, 1),  # nose to left_eye
            (0, 4),  # nose to right_eye
            (1, 2),  # left_eye to left_ear
            (4, 5),  # right_eye to right_ear
            (0, 11),  # nose to left_shoulder
            (0, 12),  # nose to right_shoulder
            (11, 12),  # left_shoulder to right_shoulder
            (11, 23),  # left_shoulder to left_hip
            (12, 24),  # right_shoulder to right_hip
            (23, 24),  # left_hip to right_hip

            # Arms
            (11, 13),  # left_shoulder to left_elbow
            (13, 15),  # left_elbow to left_wrist
            (12, 14),  # right_shoulder to right_elbow
            (14, 16),  # right_elbow to right_wrist

            # Legs
            (23, 25),  # left_hip to left_knee
            (25, 27),  # left_knee to left_ankle
            (24, 26),  # right_hip to right_knee
            (26, 28),  # right_knee to right_ankle
        ]

        # Calculate original bone lengths from the previous frame
        original_lengths = {}
        for i, j in bones:
            if i < len(prev_keypoints) and j < len(prev_keypoints):
                p1 = np.array([prev_keypoints[i]['x'], prev_keypoints[i]['y'], prev_keypoints[i]['z']])
                p2 = np.array([prev_keypoints[j]['x'], prev_keypoints[j]['y'], prev_keypoints[j]['z']])
                length = np.linalg.norm(p2 - p1)
                original_lengths[(i, j)] = length

        # Apply velocity constraints
        for i in range(len(constrained_keypoints)):
            if i < len(prev_keypoints):
                p_current = np.array(
                    [constrained_keypoints[i]['x'], constrained_keypoints[i]['y'], constrained_keypoints[i]['z']])
                p_prev = np.array([prev_keypoints[i]['x'], prev_keypoints[i]['y'], prev_keypoints[i]['z']])

                # Calculate velocity
                velocity = (p_current - p_prev) / dt

                # Get the joint type
                joint_name = constrained_keypoints[i]['name'] if 'name' in constrained_keypoints[i] else ""
                joint_type = JOINT_TYPE_MAPPING.get(joint_name, 'spine')  # Default to spine constraints

                # Get velocity limit for this joint type
                vel_limit = JOINT_VELOCITY_LIMITS.get(joint_type, 5.0)

                # Check if velocity exceeds limit
                vel_magnitude = np.linalg.norm(velocity)
                if vel_magnitude > vel_limit:
                    # Scale back velocity
                    scaling_factor = vel_limit / vel_magnitude
                    constrained_velocity = velocity * scaling_factor

                    # Update position based on constrained velocity
                    p_constrained = p_prev + constrained_velocity * dt

                    # Update keypoint
                    constrained_keypoints[i]['x'] = float(p_constrained[0])
                    constrained_keypoints[i]['y'] = float(p_constrained[1])
                    constrained_keypoints[i]['z'] = float(p_constrained[2])

        # Enforce bone length preservation
        for iteration in range(2):  # Two iterations of constraint solving
            for i, j in bones:
                if (i, j) in original_lengths and i < len(constrained_keypoints) and j < len(constrained_keypoints):
                    # Get current positions
                    p1 = np.array(
                        [constrained_keypoints[i]['x'], constrained_keypoints[i]['y'], constrained_keypoints[i]['z']])
                    p2 = np.array(
                        [constrained_keypoints[j]['x'], constrained_keypoints[j]['y'], constrained_keypoints[j]['z']])

                    # Current length
                    current_length = np.linalg.norm(p2 - p1)

                    # Skip if length is already close enough
                    if abs(current_length - original_lengths[(i, j)]) < 0.01:
                        continue

                    # Direction vector
                    direction = (p2 - p1) / current_length if current_length > 1e-6 else np.array([0, 0, 0])

                    # Target length
                    target_length = original_lengths[(i, j)]

                    # Adjust positions to maintain bone length
                    # Weight adjustments based on joint hierarchy
                    if i < 11:  # Core body parts move less
                        w1, w2 = 0.1, 0.9
                    elif j < 11:
                        w1, w2 = 0.9, 0.1
                    else:
                        w1, w2 = 0.5, 0.5

                    # Only apply if both points have good visibility
                    if constrained_keypoints[i]['visibility'] > 0.5 and constrained_keypoints[j]['visibility'] > 0.5:
                        # Adjustment vector
                        adjustment = direction * (target_length - current_length)

                        # Apply adjustments
                        p1_new = p1 - adjustment * w1
                        p2_new = p2 + adjustment * w2

                        # Update keypoints
                        constrained_keypoints[i]['x'] = float(p1_new[0])
                        constrained_keypoints[i]['y'] = float(p1_new[1])
                        constrained_keypoints[i]['z'] = float(p1_new[2])

                        constrained_keypoints[j]['x'] = float(p2_new[0])
                        constrained_keypoints[j]['y'] = float(p2_new[1])
                        constrained_keypoints[j]['z'] = float(p2_new[2])

        return constrained_keypoints
    except Exception as e:
        print(f"Error applying biomechanical constraints: {e}")
        traceback.print_exc()
        return keypoints  # Return original keypoints if constraints fail


def validate_human_pose(keypoints):
    """Validate that the detected pose is likely a human using anatomical constraints."""
    try:
        # Check if key points exist
        if not keypoints or len(keypoints) < 33:
            return False

        # Extract key body points for validation
        nose = next((kp for kp in keypoints if kp['name'] == 'NOSE'), None)
        left_shoulder = next((kp for kp in keypoints if kp['name'] == 'LEFT_SHOULDER'), None)
        right_shoulder = next((kp for kp in keypoints if kp['name'] == 'RIGHT_SHOULDER'), None)
        left_hip = next((kp for kp in keypoints if kp['name'] == 'LEFT_HIP'), None)
        right_hip = next((kp for kp in keypoints if kp['name'] == 'RIGHT_HIP'), None)

        # More relaxed validation: if at least important torso points are present, accept the pose
        if not (left_shoulder and right_shoulder and left_hip and right_hip):
            # Allow the pose if high confidence for detected parts, even if some are missing
            high_confidence_parts = sum(1 for kp in keypoints if kp['visibility'] > 0.7)
            if high_confidence_parts >= 15:  # If at least 15 high confidence keypoints
                return True
            return False

        # Create numpy arrays for calculations
        left_shoulder_pos = np.array([left_shoulder['x'], left_shoulder['y'], left_shoulder['z']])
        right_shoulder_pos = np.array([right_shoulder['x'], right_shoulder['y'], right_shoulder['z']])
        left_hip_pos = np.array([left_hip['x'], left_hip['y'], left_hip['z']])
        right_hip_pos = np.array([right_hip['x'], right_hip['y'], right_hip['z']])

        # Calculate midpoints
        mid_shoulder = (left_shoulder_pos + right_shoulder_pos) / 2
        mid_hip = (left_hip_pos + right_hip_pos) / 2

        # More relaxed proportions check
        shoulder_width = np.linalg.norm(left_shoulder_pos - right_shoulder_pos)
        hip_width = np.linalg.norm(left_hip_pos - right_hip_pos)
        torso_height = np.linalg.norm(mid_shoulder - mid_hip)

        # Extremely relaxed proportions check to accommodate different camera angles
        # Shoulder width should be somewhat proportional to hip width
        if shoulder_width < 0.2 * hip_width or shoulder_width > 5.0 * hip_width:
            # If proportion is off but confidence is high, still accept
            if left_shoulder['visibility'] > 0.8 and right_shoulder['visibility'] > 0.8 and \
                    left_hip['visibility'] > 0.8 and right_hip['visibility'] > 0.8:
                return True
            return False

        # Very loose torso height constraint - basically just ensuring it's not negative or tiny
        if torso_height < 0.2 * max(shoulder_width, hip_width):
            return False

        return True

    except Exception as e:
        print(f"Error validating pose: {e}")
        traceback.print_exc()
        return True  # Default to accepting the pose if validation fails


def interpolate_missing_keypoints(all_landmarks, valid_frames_idx, total_frames):
    """Interpolate missing keypoints between valid detections."""
    if len(valid_frames_idx) < 2:
        print("Not enough valid frames for interpolation")
        return

    try:
        print(f"Interpolating between {len(valid_frames_idx)} valid frames")
        min_frame = min(valid_frames_idx)
        max_frame = max(valid_frames_idx)
        print(f"Frame range: {min_frame} to {max_frame}")

        # Create interpolation functions for each keypoint and dimension
        for kp_idx in range(33):  # MediaPipe has 33 keypoints
            # Collect valid x, y, z coordinates for this keypoint
            frames_x = []
            frames_y = []
            frames_z = []
            frames_v = []
            values_x = []
            values_y = []
            values_z = []
            values_v = []

            for frame_idx in valid_frames_idx:
                if frame_idx < len(all_landmarks) and all_landmarks[frame_idx] is not None:
                    if kp_idx < len(all_landmarks[frame_idx]):
                        keypoint = all_landmarks[frame_idx][kp_idx]
                        frames_x.append(frame_idx)
                        frames_y.append(frame_idx)
                        frames_z.append(frame_idx)
                        frames_v.append(frame_idx)
                        values_x.append(keypoint['x'])
                        values_y.append(keypoint['y'])
                        values_z.append(keypoint['z'])
                        values_v.append(keypoint['visibility'])

            if len(frames_x) >= 2:  # Need at least 2 points for interpolation
                # Create interpolation functions (use 'linear' for more stability than 'cubic')
                try:
                    interp_x = interp1d(frames_x, values_x, kind='linear', fill_value='extrapolate')
                    interp_y = interp1d(frames_y, values_y, kind='linear', fill_value='extrapolate')
                    interp_z = interp1d(frames_z, values_z, kind='linear', fill_value='extrapolate')
                    interp_v = interp1d(frames_v, values_v, kind='linear', fill_value='extrapolate')

                    # Find gaps to interpolate (only interpolate within the valid range)
                    for i in range(min_frame, max_frame + 1):
                        if i < len(all_landmarks) and (all_landmarks[i] is None or i not in valid_frames_idx):
                            # Create keypoints list if it doesn't exist
                            if all_landmarks[i] is None:
                                all_landmarks[i] = [{
                                    'x': 0.0, 'y': 0.0, 'z': 0.0, 'visibility': 0.0,
                                    'index': j, 'name': f"PoseLandmark_{j}"
                                } for j in range(33)]

                            # Ensure the keypoint exists in the list
                            if kp_idx >= len(all_landmarks[i]):
                                for j in range(len(all_landmarks[i]), kp_idx + 1):
                                    all_landmarks[i].append({
                                        'x': 0.0, 'y': 0.0, 'z': 0.0, 'visibility': 0.0,
                                        'index': j, 'name': f"PoseLandmark_{j}"
                                    })

                            # Interpolate values
                            try:
                                all_landmarks[i][kp_idx]['x'] = float(interp_x(i))
                                all_landmarks[i][kp_idx]['y'] = float(interp_y(i))
                                all_landmarks[i][kp_idx]['z'] = float(interp_z(i))
                                all_landmarks[i][kp_idx]['visibility'] = min(0.5, float(interp_v(i)))  # Cap visibility
                            except Exception as inner_e:
                                print(f"Error interpolating keypoint {kp_idx} at frame {i}: {inner_e}")
                except Exception as e:
                    print(f"Error creating interpolation function for keypoint {kp_idx}: {e}")
    except Exception as e:
        print(f"Error in interpolation: {e}")
        traceback.print_exc()


def apply_temporal_smoothing(keypoints_timeline, all_landmarks, window_size):
    """Apply Savitzky-Golay smoothing to keypoint trajectories."""
    # Find valid frames with keypoints
    valid_frames = [i for i, landmarks in enumerate(all_landmarks) if landmarks is not None]

    if len(valid_frames) <= window_size:
        print(f"Not enough valid frames ({len(valid_frames)}) for smoothing with window size {window_size}")
        return

    try:
        print(f"Applying temporal smoothing with window size {window_size} to {len(valid_frames)} valid frames")

        # Convert keypoints to numpy arrays for easier processing
        keypoint_arrays = np.zeros((len(all_landmarks), 33, 4))  # 33 landmarks, 4 values (x,y,z,visibility)
        keypoint_masks = np.zeros((len(all_landmarks), 33), dtype=bool)

        # Extract keypoints to arrays
        for i, landmarks in enumerate(all_landmarks):
            if landmarks is not None:
                for j, landmark in enumerate(landmarks):
                    if j < 33:  # Ensure we don't exceed array bounds
                        keypoint_arrays[i, j] = [landmark['x'], landmark['y'], landmark['z'], landmark['visibility']]
                        keypoint_masks[i, j] = True

        # Apply Savitzky-Golay filter to each keypoint dimension
        for j in range(33):  # Each landmark
            for k in range(3):  # Each position dimension (x,y,z)
                # Get data for this keypoint dimension
                data = keypoint_arrays[:, j, k]
                mask = keypoint_masks[:, j]

                if np.sum(mask) > window_size:  # If enough valid points
                    valid_indices = np.where(mask)[0]
                    valid_data = data[valid_indices]

                    try:
                        # Ensure window_size is odd
                        if window_size % 2 == 0:
                            window_size -= 1

                        if window_size < 3:
                            window_size = 3

                        # Apply Savitzky-Golay filter
                        smoothed = savgol_filter(valid_data, window_size, 3)

                        # Update arrays with smoothed data
                        data[valid_indices] = smoothed
                    except Exception as e:
                        print(f"Smoothing failed for landmark {j}, dimension {k}: {e}")

        # Update keypoints timeline with smoothed data
        for i, frame_data in enumerate(keypoints_timeline):
            if i < len(all_landmarks) and all_landmarks[i] is not None and keypoint_masks[i].any():
                if frame_data['keypoints'] is None:
                    frame_data['keypoints'] = [{
                        'x': 0.0, 'y': 0.0, 'z': 0.0, 'visibility': 0.0,
                        'index': j, 'name': f"PoseLandmark_{j}"
                    } for j in range(33)]

                for j in range(33):
                    if keypoint_masks[i, j] and j < len(frame_data['keypoints']):
                        frame_data['keypoints'][j]['x'] = float(keypoint_arrays[i, j, 0])
                        frame_data['keypoints'][j]['y'] = float(keypoint_arrays[i, j, 1])
                        frame_data['keypoints'][j]['z'] = float(keypoint_arrays[i, j, 2])
                        frame_data['keypoints'][j]['visibility'] = float(keypoint_arrays[i, j, 3])

    except Exception as e:
        print(f"Error in temporal smoothing: {e}")
        traceback.print_exc()


def apply_biomechanical_constraints_timeline(keypoints_timeline, fps):
    """Apply biomechanical constraints to the entire timeline to ensure realistic motion."""
    if not keypoints_timeline:
        return

    try:
        dt = 1.0 / fps

        # First pass: apply velocity constraints
        for i in range(1, len(keypoints_timeline)):
            prev_keypoints = keypoints_timeline[i - 1].get('keypoints')
            curr_keypoints = keypoints_timeline[i].get('keypoints')

            if prev_keypoints and curr_keypoints:
                constrained_keypoints = apply_biomechanical_constraints(curr_keypoints, prev_keypoints, dt)
                keypoints_timeline[i]['keypoints'] = constrained_keypoints

        # Second pass: apply acceleration constraints
        for i in range(2, len(keypoints_timeline)):
            keypoints_t0 = keypoints_timeline[i - 2].get('keypoints')
            keypoints_t1 = keypoints_timeline[i - 1].get('keypoints')
            keypoints_t2 = keypoints_timeline[i].get('keypoints')

            if keypoints_t0 and keypoints_t1 and keypoints_t2:
                # For each keypoint, check acceleration
                for j in range(len(keypoints_t2)):
                    if j < len(keypoints_t0) and j < len(keypoints_t1):
                        # Get positions
                        pos_t0 = np.array([keypoints_t0[j]['x'], keypoints_t0[j]['y'], keypoints_t0[j]['z']])
                        pos_t1 = np.array([keypoints_t1[j]['x'], keypoints_t1[j]['y'], keypoints_t1[j]['z']])
                        pos_t2 = np.array([keypoints_t2[j]['x'], keypoints_t2[j]['y'], keypoints_t2[j]['z']])

                        # Calculate velocities
                        vel_t0_t1 = (pos_t1 - pos_t0) / dt
                        vel_t1_t2 = (pos_t2 - pos_t1) / dt

                        # Calculate acceleration
                        accel = (vel_t1_t2 - vel_t0_t1) / dt
                        accel_magnitude = np.linalg.norm(accel)

                        # Get joint type for acceleration limits
                        joint_name = keypoints_t2[j]['name'] if 'name' in keypoints_t2[j] else ""
                        joint_type = JOINT_TYPE_MAPPING.get(joint_name, 'spine')
                        accel_limit = JOINT_ACCELERATION_LIMITS.get(joint_type, 20.0)

                        # If acceleration exceeds limit, adjust position
                        if accel_magnitude > accel_limit:
                            # Scale back acceleration
                            scaling_factor = accel_limit / accel_magnitude

                            # Compute adjusted velocity
                            adjusted_vel = vel_t0_t1 + (vel_t1_t2 - vel_t0_t1) * scaling_factor

                            # Compute adjusted position
                            adjusted_pos = pos_t1 + adjusted_vel * dt

                            # Update keypoint
                            keypoints_t2[j]['x'] = float(adjusted_pos[0])
                            keypoints_t2[j]['y'] = float(adjusted_pos[1])
                            keypoints_t2[j]['z'] = float(adjusted_pos[2])

    except Exception as e:
        print(f"Error applying biomechanical constraints to timeline: {e}")
        traceback.print_exc()


def extract_pose_keypoints(video_path, confidence_threshold=0.3, skip_frames=0,
                           smoothing_window=7, visualize=False, output_dir='visualize',
                           interpolate_missing=False, use_physics=False, min_track_length=10):
    """
    Enhanced function to extract pose keypoints from video with improved robustness.
    Now with physics-based constraints and EfficientDet Lite 0 for person detection.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Starting keypoint extraction for video: {video_path}")

    # Set up MediaPipe with optimal settings for improved detection
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,  # For video processing
        model_complexity=2,  # Use the most complex model
        smooth_landmarks=True,  # Enable temporal smoothing
        enable_segmentation=False,  # Enable person segmentation
        min_detection_confidence=0.2,  # Lower threshold for initial detection
        min_tracking_confidence=0.5  # More relaxed tracking threshold
    )

    # Initialize variables
    keypoints_timeline = []
    frame_count = 0
    processed_count = 0
    valid_frames = 0

    # For person tracking across frames
    person_tracks = []
    current_person_id = None
    last_valid_keypoints = None
    last_valid_frame = -1

    # For physics-based filtering
    kalman_filter = None if not use_physics else initialize_kalman_filter()
    kalman_initialized = False

    # For visualization
    if visualize:
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        os.makedirs(output_dir, exist_ok=True)
        print(f"Visualization frames will be saved to: {output_dir}")

    # Open video capture
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Try to estimate total frames if not provided by the video
            print("Warning: Could not determine total frames, trying alternative approach...")
            # Method 1: Count frames manually
            count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1
            total_frames = count
            # Reset position
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if total_frames <= 0:
                print("Warning: Still could not determine total frames, using default estimate")
                total_frames = 1000  # Reasonable default

        print(f"Video properties: {total_frames} frames at {fps} FPS")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video dimensions: {frame_width}x{frame_height}")

        # Lists for storing landmarks for later processing
        all_landmarks = [None] * total_frames
        all_landmark_confidences = [0.0] * total_frames
        valid_frames_idx = []

        # For biomechanical constraints
        previous_keypoints = None

        # Process frames
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                success, image = cap.read()
                if not success:
                    break

                # Skip frames if requested
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    pbar.update(1)
                    keypoints_timeline.append({
                        'frame': frame_count - 1,  # Adjust for 0-indexing
                        'timestamp': (frame_count - 1) / fps,
                        'keypoints': None,
                        'confidence': 0.0
                    })
                    continue

                # Initialize frame data structure
                frame_data = {
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'keypoints': None,
                    'confidence': 0.0
                }

                try:
                    # Enhanced preprocessing for better detection
                    processed_image = preprocess_frame(image)
                    if processed_image is None:
                        print(f"Warning: Failed to preprocess frame {frame_count}")
                        processed_image = image

                    # First attempt with MediaPipe directly
                    results = pose.process(processed_image)

                    # If no pose detected or low confidence, try EfficientDet and enhanced detection
                    if not results.pose_world_landmarks or \
                            (results.pose_landmarks and
                             np.mean([lm.visibility for lm in results.pose_landmarks.landmark]) < 0.3):

                        # Try EfficientDet for person detection if available
                        person_boxes = []
                        if person_detector_available:
                            person_boxes = detect_people_efficientdet(processed_image, confidence_threshold=0.1)
                            if person_boxes:
                                print(f"Frame {frame_count}: Person detected")

                        # If persons detected with EfficientDet, crop to the person region for better pose estimation
                        if person_boxes:
                            # Find the largest bounding box (likely the main subject)
                            largest_box = max(person_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
                            xmin, ymin, xmax, ymax = largest_box

                            # Add some margin to the box (10% on each side)
                            w, h = xmax - xmin, ymax - ymin
                            margin_x, margin_y = int(0.1 * w), int(0.1 * h)

                            # Ensure box stays within image bounds
                            xmin = max(0, xmin - margin_x)
                            ymin = max(0, ymin - margin_y)
                            xmax = min(image.shape[1], xmax + margin_x)
                            ymax = min(image.shape[0], ymax + margin_y)

                            # Verify box dimensions are valid
                            if xmin < xmax and ymin < ymax and xmax - xmin > 10 and ymax - ymin > 10:
                                # Crop the image to the person
                                cropped_image = processed_image[ymin:ymax, xmin:xmax]

                                # Process the cropped image with MediaPipe
                                if cropped_image.size > 0:  # Ensure the image is not empty
                                    cropped_results = pose.process(cropped_image)

                                    # If we get better results, use them
                                    if cropped_results.pose_world_landmarks:
                                        if not results.pose_landmarks or \
                                                (cropped_results.pose_landmarks and
                                                 np.mean([lm.visibility for lm in
                                                          cropped_results.pose_landmarks.landmark]) >
                                                 np.mean([lm.visibility for lm in results.pose_landmarks.landmark])):
                                            print(f"Frame {frame_count}: Using cropped detection")
                                            results = cropped_results

                                        # Visualize the bounding box if requested
                                        if visualize:
                                            box_image = image.copy()
                                            cv2.rectangle(box_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                            box_path = os.path.join(output_dir, f"box_{frame_count:06d}.jpg")
                                            cv2.imwrite(box_path, box_image)

                        # If still no good results, try enhanced detection techniques
                        if not results.pose_world_landmarks or \
                                (results.pose_landmarks and
                                 np.mean([lm.visibility for lm in results.pose_landmarks.landmark]) < 0.3):

                            # Try enhanced detection techniques
                            enhanced_image, edge_mask = enhance_detection(processed_image)
                            if enhanced_image is not None:
                                enhanced_results = pose.process(enhanced_image)

                                # If enhanced detection works, use it
                                if enhanced_results.pose_world_landmarks:
                                    if not results.pose_landmarks or \
                                            (enhanced_results.pose_landmarks and
                                             np.mean(
                                                 [lm.visibility for lm in enhanced_results.pose_landmarks.landmark]) >
                                             np.mean([lm.visibility for lm in results.pose_landmarks.landmark])):
                                        print(f"Frame {frame_count}: Using enhanced detection")
                                        results = enhanced_results

                        # If we still don't have good detection but have previous keypoints
                        if (not results.pose_world_landmarks or
                            (results.pose_landmarks and
                             np.mean([lm.visibility for lm in results.pose_landmarks.landmark]) < 0.3)) and \
                                current_person_id is not None and last_valid_keypoints is not None:

                            # Use interpolation for this frame
                            if interpolate_missing and frame_count - last_valid_frame <= 15:  # Max 0.5s gap at 30fps
                                print(
                                    f"Frame {frame_count}: Using interpolated keypoints from frame {last_valid_frame}")
                                frame_data['keypoints'] = [{
                                    'x': kp['x'],
                                    'y': kp['y'],
                                    'z': kp['z'],
                                    'visibility': kp['visibility'] * 0.8,  # Reduce visibility for interpolated points
                                    'index': kp['index'],
                                    'name': kp['name']
                                } for kp in last_valid_keypoints]

                                frame_data['confidence'] = all_landmark_confidences[
                                                               last_valid_frame] * 0.8  # Reduced confidence
                                frame_data['interpolated'] = True  # Mark as interpolated

                                all_landmarks[frame_count] = frame_data['keypoints']
                                all_landmark_confidences[frame_count] = frame_data['confidence']
                                valid_frames += 1
                                valid_frames_idx.append(frame_count)

                    # If we get valid pose landmarks
                    if results.pose_world_landmarks:
                        keypoints = []
                        avg_confidence = 0
                        valid_keypoints = 0

                        # Extract 3D landmarks
                        for i, landmark in enumerate(results.pose_world_landmarks.landmark):
                            keypoints.append({
                                'x': float(landmark.x),
                                'y': float(landmark.y),
                                'z': float(landmark.z),
                                'visibility': float(landmark.visibility),
                                'index': i,
                                'name': mp_pose.PoseLandmark(i).name if i < len(
                                    mp_pose.PoseLandmark) else f"PoseLandmark_{i}"
                            })

                            avg_confidence += landmark.visibility
                            valid_keypoints += 1

                        # Calculate average confidence for the frame
                        frame_confidence = avg_confidence / valid_keypoints if valid_keypoints > 0 else 0

                        # Verify this is a valid human pose using anatomical constraints
                        is_valid_pose = validate_human_pose(keypoints)

                        if frame_confidence >= confidence_threshold or is_valid_pose:
                            print(f"Frame {frame_count}: Valid pose detected (confidence: {frame_confidence:.2f})")

                            # Apply physics-based constraints for more realistic motion
                            if use_physics and previous_keypoints is not None:
                                keypoints = apply_biomechanical_constraints(
                                    keypoints,
                                    previous_keypoints,
                                    dt=1.0 / fps
                                )

                            # Assign person ID for tracking consistency
                            if current_person_id is None:
                                # First valid detection
                                current_person_id = len(person_tracks)
                                person_tracks.append({
                                    'id': current_person_id,
                                    'frames': [frame_count],
                                    'keypoints': [keypoints]
                                })
                            else:
                                # Add to current track
                                for track in person_tracks:
                                    if track['id'] == current_person_id:
                                        track['frames'].append(frame_count)
                                        track['keypoints'].append(keypoints)
                                        break

                            frame_data['keypoints'] = keypoints
                            frame_data['confidence'] = frame_confidence

                            all_landmarks[frame_count] = keypoints
                            all_landmark_confidences[frame_count] = frame_confidence
                            last_valid_keypoints = keypoints
                            last_valid_frame = frame_count
                            valid_frames += 1
                            valid_frames_idx.append(frame_count)

                            # Update previous keypoints for next frame
                            previous_keypoints = keypoints

                            # Apply Kalman filtering for smoother keypoints
                            if use_physics and kalman_filter is not None:
                                # Flatten keypoints for Kalman filter
                                flattened = np.array([[kp['x'], kp['y'], kp['z']] for kp in keypoints]).flatten()

                                if not kalman_initialized:
                                    # Initialize with first valid detection
                                    initial_state = np.zeros(kalman_filter.dim_x)
                                    initial_state[:len(flattened)] = flattened
                                    kalman_filter.x = initial_state
                                    kalman_initialized = True
                                else:
                                    # Update with new measurement
                                    kalman_filter.update(flattened)

                                    # Predict next state
                                    kalman_filter.predict()

                                    # Extract filtered positions
                                    filtered_positions = kalman_filter.x[:len(flattened)]

                                    # Update keypoints with filtered values
                                    for i in range(len(keypoints)):
                                        idx = i * 3
                                        if idx + 2 < len(filtered_positions):
                                            keypoints[i]['x'] = float(filtered_positions[idx])
                                            keypoints[i]['y'] = float(filtered_positions[idx + 1])
                                            keypoints[i]['z'] = float(filtered_positions[idx + 2])

                                    frame_data['keypoints'] = keypoints
                                    all_landmarks[frame_count] = keypoints

                        # Save visualization if requested
                        if visualize:
                            annotated_image = image.copy()

                            if results.pose_landmarks:
                                mp_drawing.draw_landmarks(
                                    annotated_image,
                                    results.pose_landmarks,
                                    mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                                )

                            # Draw confidence score
                            color = (0, 255, 0) if frame_data['keypoints'] is not None else (0, 0, 255)
                            cv2.putText(
                                annotated_image,
                                f"Conf: {frame_data['confidence']:.2f}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                color,
                                2
                            )

                            # Save the visualization
                            out_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                            cv2.imwrite(out_path, annotated_image)

                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    traceback.print_exc()

                keypoints_timeline.append(frame_data)
                processed_count += 1
                frame_count += 1
                pbar.update(1)

                # Safety check to avoid infinite loops
                if frame_count > total_frames * 1.5:
                    print("Warning: Exceeded expected frame count, stopping processing")
                    break

        # Find most consistent person track
        if len(person_tracks) > 1:
            # Get the longest track
            longest_track = max(person_tracks, key=lambda t: len(t['frames']))
            print(f"Multiple person tracks found. Using longest track with {len(longest_track['frames'])} frames.")

        # Apply temporal interpolation for missing frames
        if interpolate_missing and valid_frames > min_track_length:
            print(f"Interpolating missing keypoints (valid frames: {valid_frames})")
            interpolate_missing_keypoints(all_landmarks, valid_frames_idx, frame_count)

            # Update keypoints timeline with interpolated data
            for i, landmarks in enumerate(all_landmarks):
                if i < len(keypoints_timeline) and landmarks is not None and keypoints_timeline[i]['keypoints'] is None:
                    keypoints_timeline[i]['keypoints'] = landmarks
                    keypoints_timeline[i]['confidence'] = 0.5  # Default confidence for interpolated frames

        # Apply Savitzky-Golay smoothing if requested and there are enough frames
        if smoothing_window > 0 and valid_frames > smoothing_window:
            print(f"Applying temporal smoothing with window size {smoothing_window}")
            apply_temporal_smoothing(keypoints_timeline, all_landmarks, smoothing_window)

        # Apply final pass of physics-based constraints to entire timeline
        if use_physics:
            print("Applying physics-based constraints to entire timeline")
            apply_biomechanical_constraints_timeline(keypoints_timeline, fps)

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        traceback.print_exc()
    finally:
        cap.release()

    # Calculate percentage of valid frames
    valid_frames_ratio = valid_frames / processed_count if processed_count > 0 else 0
    print(f"Processing complete: {valid_frames}/{processed_count} valid frames ({valid_frames_ratio:.2%})")

    # Ensure we have some valid keypoints
    if valid_frames == 0 and processed_count > 0:
        print("WARNING: No valid frames detected. Using default poses with reduced confidence.")
        # Create default pose for at least some frames
        default_keypoints = [{
            'x': 0.5 + 0.1 * (i % 3),  # Add slight variation to avoid exact zero positions
            'y': 0.5 + 0.1 * (i % 2),
            'z': 0.0 + 0.05 * (i % 4),
            'visibility': 0.3,  # Low confidence
            'index': i,
            'name': mp_pose.PoseLandmark(i).name if i < len(mp_pose.PoseLandmark) else f"PoseLandmark_{i}"
        } for i in range(33)]

        # Add to first 10 frames to ensure we have something
        for i in range(min(10, len(keypoints_timeline))):
            keypoints_timeline[i]['keypoints'] = default_keypoints
            keypoints_timeline[i]['confidence'] = 0.3  # Low confidence

    return keypoints_timeline, fps, total_frames

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video and process for pose detection."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400

    try:
        # Save uploaded file with a unique name
        file_id = str(uuid.uuid4())
        video_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{video_file.filename}")
        video_file.save(video_path)

        print(f"Processing video: {video_path}")

        # Process video to extract pose keypoints with enhanced pipeline
        keypoints_timeline, fps, total_frames = extract_pose_keypoints(
            video_path,
            confidence_threshold=0.3,  # Lower threshold for more permissive detection
            smoothing_window=10,  # Balanced smoothing window size
            interpolate_missing=True,  # Enable interpolation
            use_physics=False,  # Enable physics-based constraints
            visualize=True
        )

        # Calculate joint angles and lunge distances for each frame
        analytics_data = []
        joints = {}
        for frame_data in keypoints_timeline:
            keypoints = frame_data['keypoints']

            # Check if we have valid keypoints
            if keypoints and len(keypoints) >= 33:
                # Extract joint coordinates
                joints_old = joints
                joints = {}
                for i, kp in enumerate(keypoints):
                    joints[i] = np.array([kp['x'], kp['y'], kp['z']])

                # Calculate each category of measurements individually
                flexion_extension = calculate_flexion_extension_angles(joints)
                torso_params = calculate_torso_parameters(joints)
                lunge_angles_data = calculate_lunge_angles(joints)
                azimuth_elevation = calculate_azimuth_elevation_angles(joints)
                distances_data = calculate_distances(joints)
                heights_data = calculate_heights(joints)
                velocities_data = calculate_velocities(joints, joints_old)
                cog_data = calculate_center_of_gravity(joints)
                lunge_distance = calculate_lunge_distance(keypoints)
            else:
                # Get default values if keypoints are not valid
                defaults = get_default_angles()

                # Use the same structure as the main function would return
                flexion_extension = {k: defaults[k] for k in [
                    'right_knee', 'left_knee', 'right_elbow', 'left_elbow',
                    'right_shoulder_elevation', 'left_shoulder_elevation',
                    'right_hip_flexion', 'left_hip_flexion', 'neck_rotation',
                    'neck_elevation', 'right_ankle_direction', 'left_ankle_direction',
                    'right_ankle_dorsiflexion', 'left_ankle_dorsiflexion'
                ]}
                # Extract other categories similarly
                torso_params = {k: defaults[k] for k in ['torso_rotation', 'torso_tilt', 'torso_lateral_tilt']}
                lunge_angles_data = {k: defaults[k] for k in ['lunge_angle_left_to_right', 'lunge_angle_right_to_left']}
                azimuth_elevation = {k: defaults[k] for k in [
                    'right_shoulder_azimuth', 'right_shoulder_elevation',
                    'left_shoulder_azimuth', 'left_shoulder_elevation',
                    'right_elbow_azimuth', 'right_elbow_elevation',
                    'left_elbow_azimuth', 'left_elbow_elevation',
                    'right_hip_azimuth', 'right_hip_elevation',
                    'left_hip_azimuth', 'left_hip_elevation'
                ]}
                distances_data = {k: defaults[k] for k in [
                    'lunge_distance_lateral', 'lunge_angle_projection', 'elbow_to_elbow_distance'
                ]}
                heights_data = {k: defaults[k] for k in [
                    'right_hip_height', 'left_hip_height', 'right_wrist_height',
                    'left_wrist_height', 'right_shoulder_height', 'left_shoulder_height'
                ]}
                velocities_data = {k: defaults[k] for k in ['right_wrist_velocity', 'left_wrist_velocity']}
                cog_data = {k: defaults[k] for k in ['body_cog_x', 'body_cog_y', 'body_cog_z']}
                lunge_distance = 0.0

            # Append to analytics data with hierarchical structure
            analytics_data.append({
                'frame': frame_data['frame'],
                'timestamp': frame_data['timestamp'],
                'measurements': {
                    'flexion_extension_angles': flexion_extension,
                    'torso_parameters': torso_params,
                    'lunge_angles': lunge_angles_data,
                    'azimuth_elevation_angles': azimuth_elevation,
                    'distances': distances_data,
                    'heights': heights_data,
                    'velocities': velocities_data,
                    'center_of_gravity': cog_data,
                    'lunge_distance': lunge_distance
                }
            })

        response = {
            'id': file_id,
            'filename': video_file.filename,
            'fps': fps,
            'total_frames': total_frames,
            'keypoints_timeline': keypoints_timeline,
            'analytics': analytics_data
        }

        # Save response to a file for later retrieval
        with open(os.path.join(UPLOAD_FOLDER, f"{file_id}_data.json"), 'w') as f:
            json.dump(response, f)

        print(f"Processing complete for {video_file.filename}. Saved as {file_id}_data.json")
        return jsonify(response), 200

    except Exception as e:
        print(f"Error in upload_video: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/videos/<video_id>', methods=['GET'])
def get_video_data(video_id):
    """Retrieve processed video data."""
    try:
        data_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_data.json")

        if not os.path.exists(data_path):
            return jsonify({'error': 'Video data not found'}), 404

        with open(data_path, 'r') as f:
            data = json.load(f)

        return jsonify(data), 200
    except Exception as e:
        print(f"Error in get_video_data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/uploads/<path:filename>', methods=['GET'])
def serve_upload(filename):
    """Directly serve a file from the uploads folder."""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except Exception as e:
        print(f"Error in serve_upload: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export', methods=['POST'])
def export_marked_frames():
    """Export marked frames with pose data and metadata."""
    try:
        data = request.json

        if not data or 'video_id' not in data or 'marked_frames' not in data:
            return jsonify({'error': 'Invalid export data'}), 400

        video_id = data['video_id']
        marked_frames = data['marked_frames']
        metadata = data.get('metadata', {})

        # Get the full keypoints timeline
        data_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_data.json")

        if not os.path.exists(data_path):
            return jsonify({'error': 'Video data not found'}), 404

        with open(data_path, 'r') as f:
            video_data = json.load(f)

        keypoints_timeline = video_data['keypoints_timeline']

        # Extract keypoints for marked frames
        export_data = {
            'video_id': video_id,
            'filename': video_data['filename'],
            'fps': video_data['fps'],
            'metadata': metadata,
            'marked_frames': []
        }

        for mark in marked_frames:
            frame_number = mark['frame']

            # Find the corresponding frame data
            frame_data = next((f for f in keypoints_timeline if f['frame'] == frame_number), None)

            if frame_data:
                export_data['marked_frames'].append({
                    'name': mark['name'],
                    'frame': frame_number,
                    'timestamp': frame_data['timestamp'],
                    'keypoints': frame_data['keypoints'],
                    'metadata': mark.get('metadata', {})
                })

        # Generate export filename
        export_filename = f"export_{video_id}_{int(time.time())}.json"
        export_path = os.path.join(UPLOAD_FOLDER, export_filename)

        with open(export_path, 'w') as f:
            json.dump(export_data, f)

        return jsonify({
            'success': True,
            'export_data': export_data,
            'filename': export_filename
        }), 200

    except Exception as e:
        print(f"Error in export_marked_frames: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)