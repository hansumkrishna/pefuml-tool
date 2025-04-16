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

app = Flask(__name__)
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_pose_keypoints(video_path, confidence_threshold=0.7, skip_frames=0, 
                          smoothing_window=5, visualize=False, output_dir=None):
   
    # Input validation
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Set up MediaPipe with optimal settings for accuracy
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Use the most complex model
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=confidence_threshold,
        min_tracking_confidence=confidence_threshold
    )
    
    keypoints_timeline = []
    frame_count = 0
    processed_count = 0
    valid_frames = 0
    
    # Create output directory for visualizations if needed
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
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # List for storing landmarks for later smoothing
        all_landmarks = [None] * total_frames
        
        # Process frames
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                
                # Skip frames if requested
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    pbar.update(1)
                    continue
                
                # Initialize frame data structure
                frame_data = {
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'keypoints': None,
                    'confidence': 0.0
                }
                
                try:
                    # Pre-process image for better detection
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Optional: Apply contrast enhancement
                    image_rgb = cv2.convertScaleAbs(image_rgb, alpha=1.2, beta=10)
                    
                    # Process the image for pose detection
                    results = pose.process(image_rgb)
                    
                    if results.pose_world_landmarks:
                        keypoints = []
                        avg_confidence = 0
                        valid_keypoints = 0
                        
                        # Extract 3D landmarks
                        for i, landmark in enumerate(results.pose_world_landmarks.landmark):
                            keypoints.append({
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z,
                                'visibility': landmark.visibility,
                                'index': i,
                                'name': mp_pose.PoseLandmark(i).name
                            })
                            
                            avg_confidence += landmark.visibility
                            valid_keypoints += 1
                        
                        # Calculate average confidence for the frame
                        frame_confidence = avg_confidence / valid_keypoints if valid_keypoints > 0 else 0
                        frame_data['confidence'] = frame_confidence
                        
                        # Store keypoints if confidence is high enough
                        if frame_confidence >= confidence_threshold:
                            frame_data['keypoints'] = keypoints
                            all_landmarks[frame_count] = results.pose_world_landmarks.landmark
                            valid_frames += 1
                        
                        # Save visualization if requested
                        if visualize:
                            annotated_image = image.copy()
                            mp_drawing.draw_landmarks(
                                annotated_image,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                            )
                            
                            # Add confidence text
                            cv2.putText(
                                annotated_image, 
                                f"Conf: {frame_confidence:.2f}", 
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (0, 255, 0) if frame_confidence >= confidence_threshold else (0, 0, 255), 
                                2
                            )
                            
                            out_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                            cv2.imwrite(out_path, annotated_image)
                
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                
                keypoints_timeline.append(frame_data)
                processed_count += 1
                frame_count += 1
                pbar.update(1)
        
        # Apply temporal smoothing if window size > 0
        if smoothing_window > 0 and valid_frames > smoothing_window:
            print(f"Applying temporal smoothing with window size {smoothing_window}...")
            
            # Convert landmarks to numpy arrays for easier processing
            landmark_arrays = np.zeros((total_frames, 33, 4))  # 33 landmarks, 4 values each (x,y,z,visibility)
            landmark_masks = np.zeros((total_frames, 33), dtype=bool)
            
            # Extract landmarks to arrays
            for i, landmarks in enumerate(all_landmarks):
                if landmarks is not None:
                    for j, landmark in enumerate(landmarks):
                        landmark_arrays[i, j] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
                        landmark_masks[i, j] = True
            
            # Apply Savitzky-Golay filter for smoothing
            for i in range(33):  # For each landmark
                for j in range(4):  # For each dimension (x,y,z,visibility)
                    data = landmark_arrays[:, i, j]
                    mask = landmark_masks[:, i]
                    
                    if np.sum(mask) > smoothing_window:  # If enough valid points
                        valid_indices = np.where(mask)[0]
                        valid_data = data[valid_indices]
                        
                        try:
                            smoothed = savgol_filter(valid_data, smoothing_window, 3)
                            data[valid_indices] = smoothed
                        except Exception as e:
                            print(f"Smoothing failed for landmark {i}, dimension {j}: {e}")
            
            # Update keypoints_timeline with smoothed data
            for i, frame_data in enumerate(keypoints_timeline):
                if all_landmarks[i] is not None and frame_data['keypoints'] is not None:
                    for j, kp in enumerate(frame_data['keypoints']):
                        kp['x'] = float(landmark_arrays[i, j, 0])
                        kp['y'] = float(landmark_arrays[i, j, 1])
                        kp['z'] = float(landmark_arrays[i, j, 2])
                        kp['visibility'] = float(landmark_arrays[i, j, 3])
    
    except Exception as e:
        print(f"Error processing video: {str(e)}")
    finally:
        cap.release()
    
    # Calculate percentage of valid frames
    valid_frames_ratio = valid_frames / processed_count if processed_count > 0 else 0
    print(f"Processing complete: {valid_frames}/{processed_count} valid frames ({valid_frames_ratio:.2%})")
    
    return keypoints_timeline, fps, total_frames

def calculate_joint_angles(keypoints):
    """Calculate joint angles from pose keypoints."""
    angles = {}
    
    # Example: Calculate knee angle
    if keypoints and len(keypoints) >= 33:  # MediaPipe has 33 keypoints
        # Right knee angle
        hip = np.array([keypoints[24]['x'], keypoints[24]['y'], keypoints[24]['z']])
        knee = np.array([keypoints[26]['x'], keypoints[26]['y'], keypoints[26]['z']])
        ankle = np.array([keypoints[28]['x'], keypoints[28]['y'], keypoints[28]['z']])
        
        v1 = hip - knee
        v2 = ankle - knee
        
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles['right_knee'] = np.degrees(angle)
        
        # Left knee angle
        hip = np.array([keypoints[23]['x'], keypoints[23]['y'], keypoints[23]['z']])
        knee = np.array([keypoints[25]['x'], keypoints[25]['y'], keypoints[25]['z']])
        ankle = np.array([keypoints[27]['x'], keypoints[27]['y'], keypoints[27]['z']])
        
        v1 = hip - knee
        v2 = ankle - knee
        
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles['left_knee'] = np.degrees(angle)
        
    return angles

def calculate_lunge_distance(keypoints):
    """Calculate lunge distance from pose keypoints."""
    if keypoints and len(keypoints) >= 33:
        # Get feet positions
        left_foot = np.array([keypoints[31]['x'], keypoints[31]['y'], keypoints[31]['z']])
        right_foot = np.array([keypoints[32]['x'], keypoints[32]['y'], keypoints[32]['z']])
        
        # Calculate distance between feet
        distance = np.linalg.norm(left_foot - right_foot)
        return distance
    
    return 0

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video and process for pose detection."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    # Save uploaded file with a unique name
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{video_file.filename}")
    video_file.save(video_path)
    
    try:
        # Process video to extract pose keypoints
        keypoints_timeline, fps, total_frames = extract_pose_keypoints(video_path)
        
        # Calculate joint angles and lunge distances for each frame
        analytics_data = []
        for frame_data in keypoints_timeline:
            if frame_data['keypoints']:
                angles = calculate_joint_angles(frame_data['keypoints'])
                lunge_distance = calculate_lunge_distance(frame_data['keypoints'])
                
                analytics_data.append({
                    'frame': frame_data['frame'],
                    'timestamp': frame_data['timestamp'],
                    'angles': angles,
                    'lunge_distance': lunge_distance
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
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/videos/<video_id>', methods=['GET'])
def get_video_data(video_id):
    """Retrieve processed video data."""
    data_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_data.json")
    
    if not os.path.exists(data_path):
        return jsonify({'error': 'Video data not found'}), 404
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return jsonify(data), 200

@app.route('/api/uploads/<path:filename>', methods=['GET'])
def serve_upload(filename):
    """Directly serve a file from the uploads folder."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/export', methods=['POST'])
def export_marked_frames():
    """Export marked frames with pose data and metadata."""
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)