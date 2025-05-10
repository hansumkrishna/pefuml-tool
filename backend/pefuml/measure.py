import numpy as np
import traceback


def calculate_joint_angles(keypoints, previous_keypoints=None, time_delta=1 / 30):
    """Main function to calculate all joint angles from pose keypoints."""
    # If no keypoints are provided, return all default values
    if not keypoints or len(keypoints) < 33:  # MediaPipe has 33 keypoints
        return get_default_angles()

    try:
        # Extract joint coordinates for use in calculations
        joints = {}
        for i, kp in enumerate(keypoints):
            joints[i] = np.array([kp['x'], kp['y'], kp['z']])

        # Previous joints for velocity calculations
        prev_joints = None
        if previous_keypoints and len(previous_keypoints) >= 33:
            prev_joints = {}
            for i, kp in enumerate(previous_keypoints):
                prev_joints[i] = np.array([kp['x'], kp['y'], kp['z']])

        # Calculate each category of measurements
        angles = {}
        angles.update(calculate_flexion_extension_angles(joints))
        angles.update(calculate_torso_parameters(joints))
        angles.update(calculate_lunge_angles(joints))
        angles.update(calculate_azimuth_elevation_angles(joints))
        angles.update(calculate_distances(joints))
        angles.update(calculate_heights(joints))
        angles.update(calculate_velocities(joints, prev_joints, time_delta))
        angles.update(calculate_center_of_gravity(joints))
        angles.update(calculate_orientation(joints))

        return angles

    except Exception as e:
        print(f"Error calculating joint angles: {e}")
        traceback.print_exc()
        return get_default_angles()


def get_default_angles():
    """Return default values for all angles."""
    return {
        # Joint Angles (Flexion/Extension)
        'right_elbow': 0.0,
        'left_elbow': 0.0,
        'right_shoulder_elevation': 0.0,
        'left_shoulder_elevation': 0.0,
        'right_knee': 0.0,
        'left_knee': 0.0,
        'right_hip_flexion': 0.0,
        'left_hip_flexion': 0.0,
        'neck_rotation': 90.0,
        'neck_elevation': 90.0,
        'right_ankle_direction': 90.0,
        'left_ankle_direction': 90.0,
        'right_ankle_dorsiflexion': 90.0,
        'left_ankle_dorsiflexion': 90.0,

        # Torso Parameters
        'torso_rotation': 90.0,
        'torso_tilt': 180.0,
        'torso_lateral_tilt': 90.0,

        # Lunge Angles
        'lunge_angle_left_to_right': 0.0,
        'lunge_angle_right_to_left': 0.0,

        # Joint Azimuthal angles
        'right_shoulder_azimuth': 0.0,
        'right_shoulder_elevation': 0.0,
        'left_shoulder_azimuth': 0.0,
        'left_shoulder_elevation': 0.0,
        'right_elbow_azimuth': 90.0,
        'right_elbow_elevation': 0.0,
        'left_elbow_azimuth': 90.0,
        'left_elbow_elevation': 0.0,
        'right_hip_azimuth': 0.0,
        'right_hip_elevation': 90.0,
        'left_hip_azimuth': 0.0,
        'left_hip_elevation': 90.0,

        # Distances
        'lunge_distance_lateral': 0.0,
        'elbow_to_elbow_distance': 0.0,

        # Heights
        'right_hip_height': 0.0,
        'left_hip_height': 0.0,
        'right_wrist_height': 0.0,
        'left_wrist_height': 0.0,
        'right_shoulder_height': 0.0,
        'left_shoulder_height': 0.0,

        # Velocities
        'right_wrist_velocity': 0.0,
        'left_wrist_velocity': 0.0,
        'body_center_velocity': 0.0,

        # Center of Gravity
        'body_cog_x': 0.0,
        'body_cog_y': 0.0,
        'body_cog_z': 0.0,

        # Orientation
        'body_orientation': 90.0
    }


def calculate_angle(point1, point2, point3):
    """Helper function to calculate angle between three points in degrees."""
    v1 = point1 - point2
    v2 = point3 - point2

    # Use dot product formula: cos(θ) = (v1·v2) / (|v1|·|v2|)
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    # Handle potential numerical errors
    if norm_product < 1e-10:
        return 0.0

    cosine = np.clip(dot_product / norm_product, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return float(angle)


def calculate_angle_vectors(v1, v2):
    """Helper function to calculate angle between two vectors in degrees."""
    # Use dot product formula: cos(θ) = (v1·v2) / (|v1|·|v2|)
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    # Handle potential numerical errors
    if norm_product < 1e-10:
        return 0.0

    cosine = np.clip(dot_product / norm_product, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return float(angle)


def calculate_flexion_extension_angles(joints):
    """
    Calculate joint flexion and extension angles according to the provided definitions.

    Keypoint indices:
    0: Nose
    11: Left shoulder, 12: Right shoulder
    13: Left elbow, 14: Right elbow
    15: Left wrist, 16: Right wrist
    23: Left hip, 24: Right hip
    25: Left knee, 26: Right knee
    27: Left ankle, 28: Right ankle
    29: Left heel, 30: Right heel
    31: Left foot_index, 32: Right foot_index
    """
    angles = {}

    # Calculate reference points and vectors
    vertical_axis = np.array([0, 1, 0])
    neck = (joints[11] + joints[12]) / 2  # midpoint between shoulders
    mid_hip = (joints[23] + joints[24]) / 2  # midpoint between hips
    torso_vector = neck - mid_hip  # Vector from mid-hip to neck

    # --- 1. Elbow Angle (left/right) ---
    # 0° = arms folded (wrist touching shoulder), 180° = arms completely extended

    # Right elbow
    right_elbow_angle = calculate_angle(joints[12], joints[14], joints[16])
    angles['right_elbow'] = right_elbow_angle

    # Left elbow
    left_elbow_angle = calculate_angle(joints[11], joints[13], joints[15])
    angles['left_elbow'] = left_elbow_angle

    # --- 2. Shoulder Elevation (left/right) ---
    # 0° = parallel to torso, 90° = perpendicular, 180° = raised all the way up

    # Right shoulder
    right_shoulder_vector = joints[14] - joints[12]  # Vector from shoulder to elbow
    if np.linalg.norm(right_shoulder_vector) > 1e-10 and np.linalg.norm(torso_vector) > 1e-10:
        angle = calculate_angle_vectors(right_shoulder_vector, torso_vector)

        # Determine if arm is raised up or down relative to torso
        cross = np.cross(torso_vector, right_shoulder_vector)

        # If cross product points toward the front, arm is raised up
        if cross[2] < 0:  # Z component negative means raised up
            angles['right_shoulder_elevation'] = angle
        else:  # Lowered down relative to torso
            angles['right_shoulder_elevation'] = 180 - angle
    else:
        angles['right_shoulder_elevation'] = 0.0

    # Left shoulder
    left_shoulder_vector = joints[13] - joints[11]  # Vector from shoulder to elbow
    if np.linalg.norm(left_shoulder_vector) > 1e-10 and np.linalg.norm(torso_vector) > 1e-10:
        angle = calculate_angle_vectors(left_shoulder_vector, torso_vector)

        # Determine if arm is raised up or down relative to torso
        cross = np.cross(torso_vector, left_shoulder_vector)

        # If cross product points toward the front, arm is raised up
        if cross[2] > 0:  # Z component positive means raised up
            angles['left_shoulder_elevation'] = angle
        else:  # Lowered down relative to torso
            angles['left_shoulder_elevation'] = 180 - angle
    else:
        angles['left_shoulder_elevation'] = 0.0

    # --- 3. Knee Angle (left/right) ---
    # 0° = ankle touching hip, 180° = straight line

    # Right knee
    right_knee_angle = calculate_angle(joints[24], joints[26], joints[28])
    angles['right_knee'] = right_knee_angle

    # Left knee
    left_knee_angle = calculate_angle(joints[23], joints[25], joints[27])
    angles['left_knee'] = left_knee_angle

    # --- 4. Hip Flexion (left/right) ---
    # 0° = knee touching shoulder, 180° = straight line

    # Right hip
    right_hip_angle = calculate_angle(joints[12], joints[24], joints[26])
    angles['right_hip_flexion'] = right_hip_angle

    # Left hip
    left_hip_angle = calculate_angle(joints[11], joints[23], joints[25])
    angles['left_hip_flexion'] = left_hip_angle

    # --- 5. Neck Rotation ---
    # 180° = looking left, 0° = looking right

    # Get the head direction vector (from neck to nose)
    head_direction = joints[0] - neck

    # Project head direction onto horizontal plane (XZ)
    head_direction_xz = np.array([head_direction[0], 0, head_direction[2]])

    # Get shoulder line vector (left to right)
    shoulder_line = joints[12] - joints[11]  # Right to left

    # Calculate neck rotation
    if np.linalg.norm(head_direction_xz) > 1e-10 and np.linalg.norm(shoulder_line) > 1e-10:
        angle = calculate_angle_vectors(head_direction_xz, shoulder_line)

        # Determine if looking left or right using cross product
        cross = np.cross(shoulder_line, head_direction_xz)

        if cross[1] < 0:  # Y component negative means looking right
            angles['neck_rotation'] = 180 - angle
        else:  # Looking left
            angles['neck_rotation'] = angle
    else:
        angles['neck_rotation'] = 90.0  # Default to middle position

    # --- 6. Neck Elevation ---
    # 0° = looking down, 180° = looking up

    # Project head direction onto vertical plane (YZ)
    head_direction_yz = np.array([0, head_direction[1], head_direction[2]])

    # Calculate angle with vertical axis
    if np.linalg.norm(head_direction_yz) > 1e-10:
        angle = calculate_angle_vectors(head_direction_yz, vertical_axis)

        # Determine if looking up or down
        if head_direction_yz[2] < 0:  # Negative Z means looking down
            angles['neck_elevation'] = angle
        else:  # Looking up
            angles['neck_elevation'] = 180 - angle
    else:
        angles['neck_elevation'] = 90.0  # Default to middle position

    # --- 7. Ankle Direction (left/right) ---
    # 180° = aligned to hip line bent to the left, 0° = aligned to hip line bent to the right

    # Hip line (from right to left hip)
    hip_line = joints[23] - joints[24]

    # Right ankle
    right_foot_vector = joints[32] - joints[28]  # Vector from ankle to foot_index
    right_foot_vector_xz = np.array([right_foot_vector[0], 0, right_foot_vector[2]])  # Project onto XZ plane

    if np.linalg.norm(right_foot_vector_xz) > 1e-10 and np.linalg.norm(hip_line) > 1e-10:
        angle = calculate_angle_vectors(right_foot_vector_xz, hip_line)

        # Determine left/right direction using cross product
        cross = np.cross(hip_line, right_foot_vector_xz)

        if cross[1] < 0:  # Y component negative means bent to the right
            angles['right_ankle_direction'] = angle
        else:  # Bent to the left
            angles['right_ankle_direction'] = 180 - angle
    else:
        angles['right_ankle_direction'] = 90.0  # Default to middle position

    # Left ankle
    left_foot_vector = joints[31] - joints[27]  # Vector from ankle to foot_index
    left_foot_vector_xz = np.array([left_foot_vector[0], 0, left_foot_vector[2]])  # Project onto XZ plane

    if np.linalg.norm(left_foot_vector_xz) > 1e-10 and np.linalg.norm(hip_line) > 1e-10:
        angle = calculate_angle_vectors(left_foot_vector_xz, hip_line)

        # Determine left/right direction using cross product
        cross = np.cross(hip_line, left_foot_vector_xz)

        if cross[1] < 0:  # Y component negative means bent to the right
            angles['left_ankle_direction'] = angle
        else:  # Bent to the left
            angles['left_ankle_direction'] = 180 - angle
    else:
        angles['left_ankle_direction'] = 90.0  # Default to middle position

    # --- 8. Ankle Dorsiflexion (left/right) ---
    # 0° = ankle bent up toes touching the respective lower leg edge, 90° = perpendicular to the leg

    # Right ankle
    right_lower_leg = joints[26] - joints[28]  # Vector from ankle to knee
    right_foot = joints[32] - joints[28]  # Vector from ankle to foot_index

    if np.linalg.norm(right_lower_leg) > 1e-10 and np.linalg.norm(right_foot) > 1e-10:
        angle = calculate_angle_vectors(right_lower_leg, right_foot)
        angles['right_ankle_dorsiflexion'] = angle
    else:
        angles['right_ankle_dorsiflexion'] = 90.0  # Default to perpendicular

    # Left ankle
    left_lower_leg = joints[25] - joints[27]  # Vector from ankle to knee
    left_foot = joints[31] - joints[27]  # Vector from ankle to foot_index

    if np.linalg.norm(left_lower_leg) > 1e-10 and np.linalg.norm(left_foot) > 1e-10:
        angle = calculate_angle_vectors(left_lower_leg, left_foot)
        angles['left_ankle_dorsiflexion'] = angle
    else:
        angles['left_ankle_dorsiflexion'] = 90.0  # Default to perpendicular

    return angles


def calculate_torso_parameters(joints):
    """
    Calculate torso orientation parameters.

    Torso Rotation: 180° = looking left, 0° = looking right
    Torso Tilt: 0° = shoulder touching knee, 180° = standing straight
    Torso Lateral Tilt: 180° = tilted left, 90° = straight, 0° = tilted right
    """
    angles = {}

    # Reference vectors
    vertical_axis = np.array([0, 1, 0])

    # Calculate key points
    left_hip = joints[23]
    right_hip = joints[24]
    left_shoulder = joints[11]
    right_shoulder = joints[12]

    # Calculate reference lines
    neck = (left_shoulder + right_shoulder) / 2
    mid_hip = (left_hip + right_hip) / 2

    # Torso vector (from mid_hip to neck)
    torso_vector = neck - mid_hip

    # Hip line (from right to left)
    hip_line = left_hip - right_hip

    # Shoulder line (from right to left)
    shoulder_line = left_shoulder - right_shoulder

    # --- 1. Torso Rotation ---
    # 180° = looking left, 0° = looking right

    # Project vectors onto horizontal plane (XZ)
    hip_line_xz = np.array([hip_line[0], 0, hip_line[2]])
    shoulder_line_xz = np.array([shoulder_line[0], 0, shoulder_line[2]])

    if np.linalg.norm(hip_line_xz) > 1e-10 and np.linalg.norm(shoulder_line_xz) > 1e-10:
        angle = calculate_angle_vectors(hip_line_xz, shoulder_line_xz)

        # Determine rotation direction (left or right)
        cross = np.cross(hip_line_xz, shoulder_line_xz)

        if cross[1] < 0:  # Y component negative means looking right
            angles['torso_rotation'] = angle
        else:  # Looking left
            angles['torso_rotation'] = 180 - angle
    else:
        angles['torso_rotation'] = 90.0  # Default to middle position

    # --- 2. Torso Tilt ---
    # 0° = shoulder touching knee, 180° = standing straight

    if np.linalg.norm(torso_vector) > 1e-10:
        angle = calculate_angle_vectors(torso_vector, vertical_axis)

        # Convert to the specified scale
        angles['torso_tilt'] = 180 - angle
    else:
        angles['torso_tilt'] = 180.0  # Default to standing straight

    # --- 3. Torso Lateral Tilt ---
    # 180° = tilted left, 90° = straight, 0° = tilted right

    # Project torso onto frontal plane (XY)
    torso_vector_xy = np.array([torso_vector[0], torso_vector[1], 0])

    if np.linalg.norm(torso_vector_xy) > 1e-10:
        angle = calculate_angle_vectors(torso_vector_xy, vertical_axis)

        # Determine tilt direction
        if torso_vector_xy[0] < 0:  # Negative X means tilted left
            angles['torso_lateral_tilt'] = 180 - angle
        else:  # Tilted right
            angles['torso_lateral_tilt'] = angle
    else:
        angles['torso_lateral_tilt'] = 90.0  # Default to straight

    return angles


def calculate_lunge_angles(joints):
    """
    Calculate lunge angles.

    Left to Right: 0° = standing with both ankles in alignment, 90° = right ankle above left ankle
    Right to Left: 0° = standing with both ankles in alignment, 90° = left ankle above right ankle
    """
    angles = {}

    # Reference vectors
    vertical_axis = np.array([0, 1, 0])

    # Left to Right Lunge angle
    # Vector from left ankle to right ankle
    ankle_vector_left_to_right = joints[28] - joints[27]

    # Project onto XZ plane (ground plane)
    ankle_vector_left_to_right_xz = np.array([ankle_vector_left_to_right[0], 0, ankle_vector_left_to_right[2]])

    if np.linalg.norm(ankle_vector_left_to_right_xz) > 1e-10:
        # Create a plane normal at left ankle
        left_ankle_plane_normal = np.cross(vertical_axis, ankle_vector_left_to_right_xz)

        # Project right ankle height onto this plane
        right_ankle_height_vector = np.array([0, joints[28][1] - joints[27][1], 0])

        if np.linalg.norm(right_ankle_height_vector) > 1e-10 and np.linalg.norm(left_ankle_plane_normal) > 1e-10:
            angle = calculate_angle_vectors(right_ankle_height_vector, left_ankle_plane_normal)

            # Adjust to match definition
            angles['lunge_angle_left_to_right'] = 90 - angle
        else:
            angles['lunge_angle_left_to_right'] = 0.0
    else:
        angles['lunge_angle_left_to_right'] = 0.0

    # Right to Left Lunge angle
    # Vector from right ankle to left ankle
    ankle_vector_right_to_left = joints[27] - joints[28]

    # Project onto XZ plane (ground plane)
    ankle_vector_right_to_left_xz = np.array([ankle_vector_right_to_left[0], 0, ankle_vector_right_to_left[2]])

    if np.linalg.norm(ankle_vector_right_to_left_xz) > 1e-10:
        # Create a plane normal at right ankle
        right_ankle_plane_normal = np.cross(vertical_axis, ankle_vector_right_to_left_xz)

        # Project left ankle height onto this plane
        left_ankle_height_vector = np.array([0, joints[27][1] - joints[28][1], 0])

        if np.linalg.norm(left_ankle_height_vector) > 1e-10 and np.linalg.norm(right_ankle_plane_normal) > 1e-10:
            angle = calculate_angle_vectors(left_ankle_height_vector, right_ankle_plane_normal)

            # Adjust to match definition
            angles['lunge_angle_right_to_left'] = 90 - angle
        else:
            angles['lunge_angle_right_to_left'] = 0.0
    else:
        angles['lunge_angle_right_to_left'] = 0.0

    return angles


def calculate_azimuth_elevation_angles(joints):
    """
    Calculate joint azimuth and elevation angles.

    Shoulder (left/right):
        Azimuth: -90° = elbow stretched back, 0° = aligned with torso, 90° = forward
        Elevation: 0° = aligned with torso, 90° = perpendicular, 180° = upward

    Elbow (left/right):
        Azimuth: 0° = wrist spun down, 90° = forward, 180° = up
        Elevation: -90° = wrist touching shoulder, 0° = perpendicular, 90° = extended

    Hip (left/right):
        Azimuth: 0° = standing straight, 90° = opened sideways, 180° = aligned with torso
        Elevation: 0° = aligned with torso, 90° = perpendicular, 180° = standing straight
    """
    angles = {}

    # Calculate reference vectors
    vertical_axis = np.array([0, 1, 0])

    # Calculate torso vector
    neck = (joints[11] + joints[12]) / 2
    mid_hip = (joints[23] + joints[24]) / 2
    torso_vector = neck - mid_hip

    # --- 1. Shoulder Azimuth and Elevation ---

    # Right shoulder
    right_upper_arm = joints[14] - joints[12]  # Elbow to shoulder

    # Project onto plane perpendicular to torso for azimuth
    # Create a plane with normal = torso_vector
    if np.linalg.norm(torso_vector) > 1e-10 and np.linalg.norm(right_upper_arm) > 1e-10:
        # Create a reference vector pointing right (perpendicular to torso)
        right_reference = np.cross(torso_vector, np.array([0, 0, 1]))
        if np.linalg.norm(right_reference) < 1e-10:
            right_reference = np.array([1, 0, 0])  # Fallback if cross product is zero

        # Normalize vectors
        right_reference = right_reference / np.linalg.norm(right_reference)
        torso_norm = torso_vector / np.linalg.norm(torso_vector)

        # Project upper arm onto the plane perpendicular to torso
        proj_right_upper_arm = right_upper_arm - np.dot(right_upper_arm, torso_norm) * torso_norm

        if np.linalg.norm(proj_right_upper_arm) > 1e-10:
            # Calculate azimuth angle
            azimuth_angle = calculate_angle_vectors(proj_right_upper_arm, right_reference)

            # Determine if arm is forward or backward
            forward_ref = np.cross(right_reference, torso_norm)
            if np.dot(proj_right_upper_arm, forward_ref) < 0:
                azimuth_angle = -azimuth_angle  # Backward

            angles['right_shoulder_azimuth'] = azimuth_angle

            # Calculate elevation angle
            elevation_angle = calculate_angle_vectors(right_upper_arm, torso_vector)
            angles['right_shoulder_elevation'] = elevation_angle
        else:
            angles['right_shoulder_azimuth'] = 0.0
            angles['right_shoulder_elevation'] = 0.0
    else:
        angles['right_shoulder_azimuth'] = 0.0
        angles['right_shoulder_elevation'] = 0.0

    # Left shoulder
    left_upper_arm = joints[13] - joints[11]  # Elbow to shoulder

    # Project onto plane perpendicular to torso for azimuth
    if np.linalg.norm(torso_vector) > 1e-10 and np.linalg.norm(left_upper_arm) > 1e-10:
        # Create a reference vector pointing left (perpendicular to torso)
        left_reference = np.cross(np.array([0, 0, 1]), torso_vector)
        if np.linalg.norm(left_reference) < 1e-10:
            left_reference = np.array([-1, 0, 0])  # Fallback if cross product is zero

        # Normalize vectors
        left_reference = left_reference / np.linalg.norm(left_reference)
        torso_norm = torso_vector / np.linalg.norm(torso_vector)

        # Project upper arm onto the plane perpendicular to torso
        proj_left_upper_arm = left_upper_arm - np.dot(left_upper_arm, torso_norm) * torso_norm

        if np.linalg.norm(proj_left_upper_arm) > 1e-10:
            # Calculate azimuth angle
            azimuth_angle = calculate_angle_vectors(proj_left_upper_arm, left_reference)

            # Determine if arm is forward or backward
            forward_ref = np.cross(torso_norm, left_reference)
            if np.dot(proj_left_upper_arm, forward_ref) < 0:
                azimuth_angle = -azimuth_angle  # Backward

            angles['left_shoulder_azimuth'] = azimuth_angle

            # Calculate elevation angle
            elevation_angle = calculate_angle_vectors(left_upper_arm, torso_vector)
            angles['left_shoulder_elevation'] = elevation_angle
        else:
            angles['left_shoulder_azimuth'] = 0.0
            angles['left_shoulder_elevation'] = 0.0
    else:
        angles['left_shoulder_azimuth'] = 0.0
        angles['left_shoulder_elevation'] = 0.0

    # --- 2. Elbow Azimuth and Elevation ---

    # Right elbow
    right_upper_arm = joints[14] - joints[12]  # Elbow to shoulder
    right_forearm = joints[16] - joints[14]  # Wrist to elbow

    if np.linalg.norm(right_upper_arm) > 1e-10 and np.linalg.norm(right_forearm) > 1e-10:
        # Create a plane with normal = right_upper_arm
        right_arm_norm = right_upper_arm / np.linalg.norm(right_upper_arm)

        # Reference vector pointing down (perpendicular to upper arm)
        down_reference = np.cross(right_arm_norm, np.array([0, 0, 1]))
        if np.linalg.norm(down_reference) < 1e-10:
            down_reference = np.array([0, -1, 0])  # Fallback if cross product is zero

        # Project forearm onto the plane perpendicular to upper arm
        proj_right_forearm = right_forearm - np.dot(right_forearm, right_arm_norm) * right_arm_norm

        if np.linalg.norm(proj_right_forearm) > 1e-10:
            # Calculate azimuth angle
            azimuth_angle = calculate_angle_vectors(proj_right_forearm, down_reference)

            # Determine orientation (up/down/forward)
            forward_ref = np.cross(down_reference, right_arm_norm)
            if np.dot(proj_right_forearm, forward_ref) > 0:
                # Forward hemisphere
                angles['right_elbow_azimuth'] = azimuth_angle
            else:
                # Backward hemisphere
                angles['right_elbow_azimuth'] = 180 - azimuth_angle

            # Calculate elevation angle
            elevation_angle = calculate_angle_vectors(right_forearm, right_upper_arm)

            # Convert to scale: -90° (touching) to 0° (perpendicular) to 90° (extended)
            angles['right_elbow_elevation'] = elevation_angle - 90
        else:
            angles['right_elbow_azimuth'] = 90.0
            angles['right_elbow_elevation'] = 0.0
    else:
        angles['right_elbow_azimuth'] = 90.0
        angles['right_elbow_elevation'] = 0.0

    # Left elbow
    left_upper_arm = joints[13] - joints[11]  # Elbow to shoulder
    left_forearm = joints[15] - joints[13]  # Wrist to elbow

    if np.linalg.norm(left_upper_arm) > 1e-10 and np.linalg.norm(left_forearm) > 1e-10:
        # Create a plane with normal = left_upper_arm
        left_arm_norm = left_upper_arm / np.linalg.norm(left_upper_arm)

        # Reference vector pointing down (perpendicular to upper arm)
        down_reference = np.cross(np.array([0, 0, 1]), left_arm_norm)
        if np.linalg.norm(down_reference) < 1e-10:
            down_reference = np.array([0, -1, 0])  # Fallback if cross product is zero

        # Project forearm onto the plane perpendicular to upper arm
        proj_left_forearm = left_forearm - np.dot(left_forearm, left_arm_norm) * left_arm_norm

        if np.linalg.norm(proj_left_forearm) > 1e-10:
            # Calculate azimuth angle
            azimuth_angle = calculate_angle_vectors(proj_left_forearm, down_reference)

            # Determine orientation (up/down/forward)
            forward_ref = np.cross(left_arm_norm, down_reference)
            if np.dot(proj_left_forearm, forward_ref) > 0:
                # Forward hemisphere
                angles['left_elbow_azimuth'] = azimuth_angle
            else:
                # Backward hemisphere
                angles['left_elbow_azimuth'] = 180 - azimuth_angle

            # Calculate elevation angle
            elevation_angle = calculate_angle_vectors(left_forearm, left_upper_arm)

            # Convert to scale: -90° (touching) to 0° (perpendicular) to 90° (extended)
            angles['left_elbow_elevation'] = elevation_angle - 90
        else:
            angles['left_elbow_azimuth'] = 90.0
            angles['left_elbow_elevation'] = 0.0
    else:
        angles['left_elbow_azimuth'] = 90.0
        angles['left_elbow_elevation'] = 0.0

    # --- 3. Hip Azimuth and Elevation ---

    # Right hip
    right_thigh = joints[26] - joints[24]  # Knee to hip

    if np.linalg.norm(right_thigh) > 1e-10 and np.linalg.norm(torso_vector) > 1e-10:
        # Reference vector pointing down (aligned with standing posture)
        down_reference = -vertical_axis

        # Project thigh onto the plane perpendicular to torso
        torso_norm = torso_vector / np.linalg.norm(torso_vector)
        proj_right_thigh = right_thigh - np.dot(right_thigh, torso_norm) * torso_norm

        if np.linalg.norm(proj_right_thigh) > 1e-10:
            # Calculate azimuth angle (0° = standing, 90° = sideways, 180° = aligned with torso)
            azimuth_angle = calculate_angle_vectors(proj_right_thigh, down_reference)

            # Determine if thigh is opened sideways
            side_ref = np.cross(torso_norm, down_reference)
            if np.dot(proj_right_thigh, side_ref) > 0:
                # Opened sideways
                angles['right_hip_azimuth'] = azimuth_angle
            else:
                # Opened inward
                angles['right_hip_azimuth'] = 180 - azimuth_angle

            # Calculate elevation angle (0° = aligned with torso, 90° = perpendicular, 180° = standing)
            elevation_angle = calculate_angle_vectors(right_thigh, torso_vector)

            # Adjust to match definition
            if np.dot(right_thigh, down_reference) > 0:
                # Thigh points downward (standing)
                angles['right_hip_elevation'] = 180 - elevation_angle
            else:
                # Thigh points upward
                angles['right_hip_elevation'] = elevation_angle
        else:
            angles['right_hip_azimuth'] = 0.0
            angles['right_hip_elevation'] = 90.0
    else:
        angles['right_hip_azimuth'] = 0.0
        angles['right_hip_elevation'] = 90.0

    # Left hip
    left_thigh = joints[25] - joints[23]  # Knee to hip

    if np.linalg.norm(left_thigh) > 1e-10 and np.linalg.norm(torso_vector) > 1e-10:
        # Reference vector pointing down (aligned with standing posture)
        down_reference = -vertical_axis

        # Project thigh onto the plane perpendicular to torso
        torso_norm = torso_vector / np.linalg.norm(torso_vector)
        proj_left_thigh = left_thigh - np.dot(left_thigh, torso_norm) * torso_norm

        if np.linalg.norm(proj_left_thigh) > 1e-10:
            # Calculate azimuth angle (0° = standing, 90° = sideways, 180° = aligned with torso)
            azimuth_angle = calculate_angle_vectors(proj_left_thigh, down_reference)

            # Determine if thigh is opened sideways
            side_ref = np.cross(down_reference, torso_norm)
            if np.dot(proj_left_thigh, side_ref) > 0:
                # Opened sideways
                angles['left_hip_azimuth'] = azimuth_angle
            else:
                # Opened inward
                angles['left_hip_azimuth'] = 180 - azimuth_angle

            # Calculate elevation angle (0° = aligned with torso, 90° = perpendicular, 180° = standing)
            elevation_angle = calculate_angle_vectors(left_thigh, torso_vector)

            # Adjust to match definition
            if np.dot(left_thigh, down_reference) > 0:
                # Thigh points downward (standing)
                angles['left_hip_elevation'] = 180 - elevation_angle
            else:
                # Thigh points upward
                angles['left_hip_elevation'] = elevation_angle
        else:
            angles['left_hip_azimuth'] = 0.0
            angles['left_hip_elevation'] = 90.0
    else:
        angles['left_hip_azimuth'] = 0.0
        angles['left_hip_elevation'] = 90.0

    return angles


def calculate_distances(joints):
    """Calculate distances between key points."""
    angles = {}

    # Lunge distance lateral (distance between ankles in XZ plane)
    left_ankle = joints[27]
    right_ankle = joints[28]

    # Project onto XZ plane
    left_ankle_xz = np.array([left_ankle[0], 0, left_ankle[2]])
    right_ankle_xz = np.array([right_ankle[0], 0, right_ankle[2]])

    # Calculate lateral distance
    lunge_distance = np.linalg.norm(right_ankle_xz - left_ankle_xz)
    angles['lunge_distance_lateral'] = float(lunge_distance)

    # Elbow to elbow distance
    left_elbow = joints[13]
    right_elbow = joints[14]
    elbow_distance = np.linalg.norm(right_elbow - left_elbow)
    angles['elbow_to_elbow_distance'] = float(elbow_distance)

    return angles


def calculate_heights(joints):
    """
    Calculate heights relative to the ground or lowest point.
    Assuming Y-up coordinate system.
    """
    angles = {}

    # Find the lowest point (ground reference)
    min_y = min(joints[i][1] for i in range(33))

    # Heights relative to ground
    angles['right_hip_height'] = float(joints[24][1] - min_y)
    angles['left_hip_height'] = float(joints[23][1] - min_y)
    angles['right_wrist_height'] = float(joints[16][1] - min_y)
    angles['left_wrist_height'] = float(joints[15][1] - min_y)
    angles['right_shoulder_height'] = float(joints[12][1] - min_y)
    angles['left_shoulder_height'] = float(joints[11][1] - min_y)

    return angles


def calculate_velocities(joints, prev_joints=None, time_delta=1 / 30):
    """Calculate velocities of key points."""
    velocities = {
        'right_wrist_velocity': 0.0,
        'left_wrist_velocity': 0.0,
        'body_center_velocity': 0.0
    }

    if prev_joints is None:
        return velocities

    # Calculate wrist velocities
    if 16 in joints and 16 in prev_joints:
        displacement = np.linalg.norm(joints[16] - prev_joints[16])
        velocities['right_wrist_velocity'] = float(displacement / time_delta)

    if 15 in joints and 15 in prev_joints:
        displacement = np.linalg.norm(joints[15] - prev_joints[15])
        velocities['left_wrist_velocity'] = float(displacement / time_delta)

    # Calculate body center velocity (using hip midpoint)
    if 23 in joints and 24 in joints and 23 in prev_joints and 24 in prev_joints:
        mid_hip_current = (joints[23] + joints[24]) / 2
        mid_hip_prev = (prev_joints[23] + prev_joints[24]) / 2
        displacement = np.linalg.norm(mid_hip_current - mid_hip_prev)
        velocities['body_center_velocity'] = float(displacement / time_delta)

    return velocities


def calculate_center_of_gravity(joints):
    """Calculate center of gravity based on body segment weights."""
    cog = {}

    # Calculate neck and mid_hip positions
    neck = (joints[11] + joints[12]) / 2  # midpoint between shoulders
    mid_hip = (joints[23] + joints[24]) / 2  # midpoint between hips

    # Approximate weights for different body segments
    weights = {
        'head': 0.08,  # 8% of body weight
        'torso': 0.55,  # 55% of body weight
        'right_arm': 0.05,  # 5% of body weight
        'left_arm': 0.05,  # 5% of body weight
        'right_leg': 0.135,  # 13.5% of body weight
        'left_leg': 0.135  # 13.5% of body weight
    }

    # Calculate weighted center of gravity
    center = np.zeros(3)

    # Head (based on nose)
    center += weights['head'] * joints[0]

    # Torso (based on midpoint of neck and mid_hip)
    center += weights['torso'] * ((neck + mid_hip) / 2)

    # Right arm (based on right shoulder, elbow, and wrist)
    right_arm_center = (joints[12] + joints[14] + joints[16]) / 3
    center += weights['right_arm'] * right_arm_center

    # Left arm (based on left shoulder, elbow, and wrist)
    left_arm_center = (joints[11] + joints[13] + joints[15]) / 3
    center += weights['left_arm'] * left_arm_center

    # Right leg (based on right hip, knee, and ankle)
    right_leg_center = (joints[24] + joints[26] + joints[28]) / 3
    center += weights['right_leg'] * right_leg_center

    # Left leg (based on left hip, knee, and ankle)
    left_leg_center = (joints[23] + joints[25] + joints[27]) / 3
    center += weights['left_leg'] * left_leg_center

    # Store the center of gravity coordinates
    cog['body_cog_x'] = float(center[0])
    cog['body_cog_y'] = float(center[1])
    cog['body_cog_z'] = float(center[2])

    return cog


def calculate_orientation(joints):
    """
    Calculate body orientation.

    Body Orientation:
    0° = looking left and hip line perpendicular to camera
    90° = looking straight
    180° = looking right and hip line perpendicular to camera
    """
    orientation = {}

    # Calculate hip line (from right to left)
    left_hip = joints[23]
    right_hip = joints[24]
    hip_line = left_hip - right_hip

    # Project onto XZ plane (assumed to be the ground plane)
    hip_line_xz = np.array([hip_line[0], 0, hip_line[2]])

    # Create a reference vector pointing into the camera (assuming Z axis is into/out of camera)
    camera_vector = np.array([0, 0, 1])

    if np.linalg.norm(hip_line_xz) > 1e-10:
        angle = calculate_angle_vectors(hip_line_xz, camera_vector)

        # Determine orientation (left or right facing)
        # If hip line X component is positive, facing right; if negative, facing left
        if hip_line_xz[0] > 0:
            orientation['body_orientation'] = 180 - angle
        else:
            orientation['body_orientation'] = angle
    else:
        orientation['body_orientation'] = 90.0  # Default to looking straight

    return orientation


def calculate_lunge_distance(keypoints):
    """Calculate lunge distance from pose keypoints."""
    try:
        if keypoints and len(keypoints) >= 33:
            # Get ankle positions
            left_ankle = np.array([keypoints[27]['x'], keypoints[27]['y'], keypoints[27]['z']])
            right_ankle = np.array([keypoints[28]['x'], keypoints[28]['y'], keypoints[28]['z']])

            # Project onto XZ plane (ground plane)
            left_ankle_xz = np.array([left_ankle[0], 0, left_ankle[2]])
            right_ankle_xz = np.array([right_ankle[0], 0, right_ankle[2]])

            # Calculate distance between ankles on ground plane
            lunge_distance = np.linalg.norm(right_ankle_xz - left_ankle_xz)

            return float(lunge_distance)
    except Exception as e:
        print(f"Error calculating lunge distance: {e}")

    return 0.0