import numpy as np
import traceback

def calculate_joint_angles(keypoints):
    """Main function to calculate all joint angles from pose keypoints."""
    # If no keypoints are provided, return all default values
    if not keypoints or len(keypoints) < 33:  # MediaPipe has 33 keypoints
        return get_default_angles()

    try:
        # Create dictionary to store all angle measurements
        angles = {}

        # Extract joint coordinates for use in calculations
        joints = {}
        for i, kp in enumerate(keypoints):
            joints[i] = np.array([kp['x'], kp['y'], kp['z']])

        # Calculate each category of measurements
        angles.update(calculate_flexion_extension_angles(joints))
        angles.update(calculate_torso_parameters(joints))
        angles.update(calculate_lunge_angles(joints))
        angles.update(calculate_azimuth_elevation_angles(joints))
        angles.update(calculate_distances(joints))
        angles.update(calculate_heights(joints))
        angles.update(calculate_velocities(joints))
        angles.update(calculate_center_of_gravity(joints))

        return angles

    except Exception as e:
        print(f"Error calculating joint angles: {e}")
        traceback.print_exc()
        return get_default_angles()


def get_default_angles():
    """Return default values for all angles."""
    return {
        # Joint Angles (Flexion/Extension)
        'right_knee': 0.0,
        'left_knee': 0.0,
        'right_elbow': 0.0,
        'left_elbow': 0.0,
        'right_shoulder_elevation': 0.0,
        'left_shoulder_elevation': 0.0,
        'right_hip_flexion': 0.0,
        'left_hip_flexion': 0.0,
        'neck_rotation': 0.0,
        'neck_elevation': 0.0,
        'right_ankle_direction': 0.0,
        'left_ankle_direction': 0.0,
        'right_ankle_dorsiflexion': 0.0,
        'left_ankle_dorsiflexion': 0.0,

        # Torso Parameters
        'torso_rotation': 0.0,
        'torso_tilt': 0.0,
        'torso_lateral_tilt': 0.0,

        # Lunge Angles
        'lunge_angle_left_to_right': 0.0,
        'lunge_angle_right_to_left': 0.0,

        # Joint Azimuthal angles
        'right_shoulder_azimuth': 0.0,
        'right_shoulder_elevation': 0.0,
        'left_shoulder_azimuth': 0.0,
        'left_shoulder_elevation': 0.0,
        'right_elbow_azimuth': 0.0,
        'right_elbow_elevation': 0.0,
        'left_elbow_azimuth': 0.0,
        'left_elbow_elevation': 0.0,
        'right_hip_azimuth': 0.0,
        'right_hip_elevation': 0.0,
        'left_hip_azimuth': 0.0,
        'left_hip_elevation': 0.0,

        # Distances
        'lunge_distance_lateral': 0.0,
        'lunge_angle_projection': 0.0,
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

        # Center of Gravity
        'body_cog_x': 0.0,
        'body_cog_y': 0.0,
        'body_cog_z': 0.0
    }


def calculate_angle(point1, point2, point3):
    """Helper function to calculate angle between three points."""
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
    return float(angle)  # Ensure float type


def calculate_spherical_rotation(origin, point):
    """Helper function to calculate spherical rotation angles."""
    # Vector from origin to point
    vector = point - origin

    # Calculate azimuth (horizontal angle), elevation (vertical angle) and distance
    distance = np.linalg.norm(vector)

    if distance < 1e-10:
        return {'azimuth': 0.0, 'elevation': 0.0, 'distance': 0.0}

    # Project onto the XZ plane for azimuth
    azimuth = np.degrees(np.arctan2(vector[0], vector[2]))

    # Calculate elevation angle
    elevation = np.degrees(np.arcsin(np.clip(vector[1] / distance, -1.0, 1.0)))

    return {
        'azimuth': float(azimuth),
        'elevation': float(elevation),
        'distance': float(distance)
    }


def calculate_flexion_extension_angles(joints):
    """Calculate joint flexion and extension angles."""
    angles = {}

    # Vertical axis reference
    vertical_axis = np.array([0, 1, 0])

    # Calculate important reference points
    neck = (joints[11] + joints[12]) / 2  # midpoint between shoulders
    mid_hip = (joints[23] + joints[24]) / 2  # midpoint between hips

    # Torso direction vector
    torso_direction = neck - mid_hip

    # Knee angles (0° = complete flexion, 180° = standing straight)
    right_knee_angle = calculate_angle(joints[24], joints[26], joints[28])
    angles['right_knee'] = 180.0 - right_knee_angle  # Convert to described scale

    left_knee_angle = calculate_angle(joints[23], joints[25], joints[27])
    angles['left_knee'] = 180.0 - left_knee_angle  # Convert to described scale

    # Elbow angles (0° = complete flexion, 180° = complete extension)
    right_elbow_angle = calculate_angle(joints[12], joints[14], joints[16])
    angles['right_elbow'] = 180.0 - right_elbow_angle  # Convert to described scale

    left_elbow_angle = calculate_angle(joints[11], joints[13], joints[15])
    angles['left_elbow'] = 180.0 - left_elbow_angle  # Convert to described scale

    # Shoulder Elevation (0° = inline with torso, 90° = perpendicular, 180° = raised arms)
    # Right shoulder
    right_shoulder_vector = joints[14] - joints[12]
    if np.linalg.norm(right_shoulder_vector) > 1e-10 and np.linalg.norm(torso_direction) > 1e-10:
        dot = np.dot(right_shoulder_vector, torso_direction)
        norms = np.linalg.norm(right_shoulder_vector) * np.linalg.norm(torso_direction)
        angle = np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))
        angles['right_shoulder_elevation'] = angle
    else:
        angles['right_shoulder_elevation'] = 0.0

    # Left shoulder
    left_shoulder_vector = joints[13] - joints[11]
    if np.linalg.norm(left_shoulder_vector) > 1e-10 and np.linalg.norm(torso_direction) > 1e-10:
        dot = np.dot(left_shoulder_vector, torso_direction)
        norms = np.linalg.norm(left_shoulder_vector) * np.linalg.norm(torso_direction)
        angle = np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))
        angles['left_shoulder_elevation'] = angle
    else:
        angles['left_shoulder_elevation'] = 0.0

    # Hip Flexion (0° = complete flexion, 180° = standing straight)
    # Right hip
    right_thigh_vector = joints[26] - joints[24]
    if np.linalg.norm(right_thigh_vector) > 1e-10:
        dot = np.dot(right_thigh_vector, vertical_axis)
        norm = np.linalg.norm(right_thigh_vector)
        angle = np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
        angles['right_hip_flexion'] = 180.0 - angle  # Convert to described scale
    else:
        angles['right_hip_flexion'] = 0.0

    # Left hip
    left_thigh_vector = joints[25] - joints[23]
    if np.linalg.norm(left_thigh_vector) > 1e-10:
        dot = np.dot(left_thigh_vector, vertical_axis)
        norm = np.linalg.norm(left_thigh_vector)
        angle = np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
        angles['left_hip_flexion'] = 180.0 - angle  # Convert to described scale
    else:
        angles['left_hip_flexion'] = 0.0

    # Neck rotation and elevation
    nose = joints[0]
    # Head direction vector (from neck to nose)
    head_direction = nose - neck

    # Get shoulder line vector (left to right)
    shoulder_line = joints[12] - joints[11]

    # Project head direction onto horizontal plane (XZ)
    head_direction_xz = np.array([head_direction[0], 0, head_direction[2]])

    # Calculate neck rotation (0° = looking left, 180° = looking right)
    if np.linalg.norm(head_direction_xz) > 1e-10 and np.linalg.norm(shoulder_line) > 1e-10:
        dot = np.dot(head_direction_xz, shoulder_line)
        norms = np.linalg.norm(head_direction_xz) * np.linalg.norm(shoulder_line)
        angle = np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))

        # Determine direction (left or right) using cross product
        cross = np.cross(shoulder_line, head_direction_xz)
        if cross[1] < 0:  # Y component negative means looking right
            angles['neck_rotation'] = 180.0 - angle
        else:  # Looking left
            angles['neck_rotation'] = angle
    else:
        angles['neck_rotation'] = 90.0  # Default to middle position

    # Neck Elevation (0° = looking down, 180° = looking up)
    # Project head direction onto vertical plane (YZ)
    head_direction_yz = np.array([0, head_direction[1], head_direction[2]])

    # Calculate angle with vertical axis
    if np.linalg.norm(head_direction_yz) > 1e-10:
        dot = np.dot(head_direction_yz, vertical_axis)
        norm = np.linalg.norm(head_direction_yz)
        angle = np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))

        # Determine if looking up or down
        if head_direction_yz[2] < 0:  # Negative Z means looking down
            angles['neck_elevation'] = angle
        else:  # Looking up
            angles['neck_elevation'] = 180.0 - angle
    else:
        angles['neck_elevation'] = 90.0  # Default to middle position

    # Ankle measurements
    # Right ankle: Create knee plane normal and check ankle vector direction
    right_knee_plane_normal = np.cross(joints[26] - joints[24], vertical_axis)
    right_ankle_vector = joints[30] - joints[28]

    if np.linalg.norm(right_knee_plane_normal) > 1e-10 and np.linalg.norm(right_ankle_vector) > 1e-10:
        dot = np.dot(right_knee_plane_normal, right_ankle_vector)
        norms = np.linalg.norm(right_knee_plane_normal) * np.linalg.norm(right_ankle_vector)
        angle = np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))

        # Adjust to 0-180 scale based on cross product
        cross = np.cross(right_knee_plane_normal, right_ankle_vector)
        dot_with_vertical = np.dot(cross, vertical_axis)

        if dot_with_vertical < 0:
            angles['right_ankle_direction'] = angle
        else:
            angles['right_ankle_direction'] = 180.0 - angle
    else:
        angles['right_ankle_direction'] = 90.0  # Default middle position

    # Left ankle: Create knee plane normal and check ankle vector direction
    left_knee_plane_normal = np.cross(joints[25] - joints[23], vertical_axis)
    left_ankle_vector = joints[29] - joints[27]

    if np.linalg.norm(left_knee_plane_normal) > 1e-10 and np.linalg.norm(left_ankle_vector) > 1e-10:
        dot = np.dot(left_knee_plane_normal, left_ankle_vector)
        norms = np.linalg.norm(left_knee_plane_normal) * np.linalg.norm(left_ankle_vector)
        angle = np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))

        # Adjust to 0-180 scale based on cross product
        cross = np.cross(left_knee_plane_normal, left_ankle_vector)
        dot_with_vertical = np.dot(cross, vertical_axis)

        if dot_with_vertical < 0:
            angles['left_ankle_direction'] = angle
        else:
            angles['left_ankle_direction'] = 180.0 - angle
    else:
        angles['left_ankle_direction'] = 90.0  # Default middle position

    # Ankle Dorsiflexion (0° = full flexion, 90° = perpendicular to leg)
    # Right ankle
    right_lower_leg = joints[28] - joints[26]  # Ankle to knee
    right_foot = joints[30] - joints[28]  # Toe to ankle

    if np.linalg.norm(right_lower_leg) > 1e-10 and np.linalg.norm(right_foot) > 1e-10:
        dot = np.dot(right_lower_leg, right_foot)
        norms = np.linalg.norm(right_lower_leg) * np.linalg.norm(right_foot)
        angle = np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))
        angles['right_ankle_dorsiflexion'] = angle
    else:
        angles['right_ankle_dorsiflexion'] = 90.0  # Default perpendicular

    # Left ankle
    left_lower_leg = joints[27] - joints[25]  # Ankle to knee
    left_foot = joints[29] - joints[27]  # Toe to ankle

    if np.linalg.norm(left_lower_leg) > 1e-10 and np.linalg.norm(left_foot) > 1e-10:
        dot = np.dot(left_lower_leg, left_foot)
        norms = np.linalg.norm(left_lower_leg) * np.linalg.norm(left_foot)
        angle = np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))
        angles['left_ankle_dorsiflexion'] = angle
    else:
        angles['left_ankle_dorsiflexion'] = 90.0  # Default perpendicular

    return angles


def calculate_torso_parameters(joints):
    """Calculate torso orientation parameters."""
    angles = {}

    # Vertical axis reference
    vertical_axis = np.array([0, 1, 0])

    # Calculate the torso orientation relative to hips
    left_hip = joints[23]
    right_hip = joints[24]
    left_shoulder = joints[11]
    right_shoulder = joints[12]

    # Calculate neck and mid_hip positions
    neck = (left_shoulder + right_shoulder) / 2
    mid_hip = (left_hip + right_hip) / 2

    # Torso direction vector
    torso_direction = neck - mid_hip

    # Hip direction (from right to left hip)
    hip_direction = left_hip - right_hip

    # Shoulder direction (from right to left shoulder)
    shoulder_direction = left_shoulder - right_shoulder

    # Torso Rotation (0° = looking left, 180° = looking right)
    # Project both vectors onto the XZ plane (horizontal)
    hip_direction_xz = np.array([hip_direction[0], 0, hip_direction[2]])
    shoulder_direction_xz = np.array([shoulder_direction[0], 0, shoulder_direction[2]])

    # Calculate the angle between these two directions
    if np.linalg.norm(hip_direction_xz) > 1e-10 and np.linalg.norm(shoulder_direction_xz) > 1e-10:
        hip_shoulder_dot = np.dot(hip_direction_xz, shoulder_direction_xz)
        hip_shoulder_norm = np.linalg.norm(hip_direction_xz) * np.linalg.norm(shoulder_direction_xz)
        torso_rotation_angle = np.degrees(np.arccos(np.clip(hip_shoulder_dot / hip_shoulder_norm, -1.0, 1.0)))

        # Determine rotation direction using cross product
        cross_product = np.cross(hip_direction_xz, shoulder_direction_xz)
        if cross_product[1] < 0:  # Y-component of cross product
            angles['torso_rotation'] = torso_rotation_angle  # Looking left
        else:
            angles['torso_rotation'] = 180.0 - torso_rotation_angle  # Looking right
    else:
        angles['torso_rotation'] = 90.0  # Default middle position

    # Torso Tilt (0° = touching toes, 180° = standing straight)
    # Use the angle between torso and vertical axis
    if np.linalg.norm(torso_direction) > 1e-10:
        torso_vertical_dot = np.dot(torso_direction, vertical_axis)
        torso_vertical_norm = np.linalg.norm(torso_direction)
        torso_tilt_angle = np.degrees(np.arccos(np.clip(torso_vertical_dot / torso_vertical_norm, -1.0, 1.0)))
        angles['torso_tilt'] = 180.0 - torso_tilt_angle  # Convert to described scale
    else:
        angles['torso_tilt'] = 180.0  # Default to standing straight

    # Torso Lateral Tilt (0° = tilted left, 180° = tilted right)
    # Project torso onto YZ plane for side bend
    torso_direction_yz = np.array([0, torso_direction[1], torso_direction[2]])

    if np.linalg.norm(torso_direction_yz) > 1e-10:
        lateral_tilt_dot = np.dot(torso_direction_yz, vertical_axis)
        lateral_tilt_norm = np.linalg.norm(torso_direction_yz)
        lateral_tilt_angle = np.degrees(np.arccos(np.clip(lateral_tilt_dot / lateral_tilt_norm, -1.0, 1.0)))

        # Determine tilt direction (left or right)
        if torso_direction_yz[2] < 0:  # Negative Z means tilted right
            angles['torso_lateral_tilt'] = 180.0 - lateral_tilt_angle
        else:  # Tilted left
            angles['torso_lateral_tilt'] = lateral_tilt_angle
    else:
        angles['torso_lateral_tilt'] = 90.0  # Default middle position (no tilt)

    return angles


def calculate_lunge_angles(joints):
    """Calculate lunge angles between legs."""
    angles = {}

    # Lunge angles between legs projected onto the ground
    right_leg_vector = joints[26] - joints[24]  # Knee to hip
    left_leg_vector = joints[25] - joints[23]  # Knee to hip

    # Project onto XZ plane (ground plane)
    right_leg_xz = np.array([right_leg_vector[0], 0, right_leg_vector[2]])
    left_leg_xz = np.array([left_leg_vector[0], 0, left_leg_vector[2]])

    # Left to right lunge
    if np.linalg.norm(right_leg_xz) > 1e-10 and np.linalg.norm(left_leg_xz) > 1e-10:
        dot = np.dot(right_leg_xz, left_leg_xz)
        norms = np.linalg.norm(right_leg_xz) * np.linalg.norm(left_leg_xz)
        lunge_angle = np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))
        angles['lunge_angle_left_to_right'] = lunge_angle
        angles['lunge_angle_right_to_left'] = lunge_angle  # Same angle, different perspective
    else:
        angles['lunge_angle_left_to_right'] = 0.0
        angles['lunge_angle_right_to_left'] = 0.0

    return angles


def calculate_azimuth_elevation_angles(joints):
    """Calculate joint azimuth and elevation angles."""
    angles = {}

    # Right shoulder spherical coordinates
    right_shoulder_spherical = calculate_spherical_rotation(joints[12], joints[14])
    # Adjust to match descriptions: -90° (back) to 0° (side) to 90° (forward)
    angles['right_shoulder_azimuth'] = right_shoulder_spherical['azimuth']
    # 0° (side) to 180° (up)
    angles['right_shoulder_elevation'] = right_shoulder_spherical['elevation'] + 90.0

    # Left shoulder spherical coordinates
    left_shoulder_spherical = calculate_spherical_rotation(joints[11], joints[13])
    angles['left_shoulder_azimuth'] = left_shoulder_spherical['azimuth']
    angles['left_shoulder_elevation'] = left_shoulder_spherical['elevation'] + 90.0

    # Right elbow spherical coordinates
    right_elbow_spherical = calculate_spherical_rotation(joints[14], joints[16])
    # 0° (down) to 90° (forward) to 180° (up)
    angles['right_elbow_azimuth'] = right_elbow_spherical['azimuth']
    # -90° (touching shoulder) to 0° (perpendicular) to 90° (extended)
    angles['right_elbow_elevation'] = right_elbow_spherical['elevation'] + 90.0

    # Left elbow spherical coordinates
    left_elbow_spherical = calculate_spherical_rotation(joints[13], joints[15])
    angles['left_elbow_azimuth'] = left_elbow_spherical['azimuth']
    angles['left_elbow_elevation'] = left_elbow_spherical['elevation'] + 90.0

    # Hip spherical coordinates
    # Right hip (0° = straight, 90° = sideways, 180° = aligned with torso)
    right_hip_spherical = calculate_spherical_rotation(joints[24], joints[26])
    angles['right_hip_azimuth'] = right_hip_spherical['azimuth']

    # Hip elevation (0° = aligned with torso, 90° = perpendicular, 180° = standing)
    angles['right_hip_elevation'] = right_hip_spherical['elevation'] + 90.0

    # Left hip
    left_hip_spherical = calculate_spherical_rotation(joints[23], joints[25])
    angles['left_hip_azimuth'] = left_hip_spherical['azimuth']
    angles['left_hip_elevation'] = left_hip_spherical['elevation'] + 90.0

    return angles


def calculate_distances(joints):
    """Calculate distances between joints."""
    angles = {}

    # Vertical axis reference
    vertical_axis = np.array([0, 1, 0])

    # Lunge distance (lateral distance between feet)
    left_foot = joints[27]  # Left ankle
    right_foot = joints[28]  # Right ankle

    # Lateral distance (XZ plane)
    foot_distance_vector = np.array([right_foot[0] - left_foot[0], 0, right_foot[2] - left_foot[2]])
    angles['lunge_distance_lateral'] = float(np.linalg.norm(foot_distance_vector))

    # Lunge angle projected on ground
    front_foot = joints[27] if left_foot[2] > right_foot[2] else joints[28]
    back_foot = joints[28] if left_foot[2] > right_foot[2] else joints[27]

    # Ground reference vector (forward direction)
    ground_reference = np.array([0, 0, 1])

    # Vector between feet projected on ground
    feet_vector = np.array([front_foot[0] - back_foot[0], 0, front_foot[2] - back_foot[2]])

    if np.linalg.norm(feet_vector) > 1e-10:
        dot = np.dot(feet_vector, ground_reference)
        norm = np.linalg.norm(feet_vector)
        lunge_projection_angle = np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
        angles['lunge_angle_projection'] = lunge_projection_angle
    else:
        angles['lunge_angle_projection'] = 0.0

    # Elbow to elbow distance
    elbow_distance_vector = joints[14] - joints[13]  # Right to left elbow
    angles['elbow_to_elbow_distance'] = float(np.linalg.norm(elbow_distance_vector))

    return angles


def calculate_heights(joints):
    """Calculate joint heights."""
    angles = {}

    # Use Y coordinate for heights
    angles['right_hip_height'] = float(joints[24][1])
    angles['left_hip_height'] = float(joints[23][1])
    angles['right_wrist_height'] = float(joints[16][1])
    angles['left_wrist_height'] = float(joints[15][1])
    angles['right_shoulder_height'] = float(joints[12][1])
    angles['left_shoulder_height'] = float(joints[11][1])

    return angles


def calculate_velocities(joints_new, joints_old=None, time_delta=1 / 30):
    """
    Calculate joint velocities using temporal data from consecutive frames.

    Parameters:
    - joints_new: Dictionary of current frame joint positions
    - joints_old: Dictionary of previous frame joint positions (optional)
    - time_delta: Time difference between frames in seconds (default: 1/30 for 30fps video)

    Returns:
    - Dictionary of joint velocities in units/second
    """
    velocities = {
        'right_wrist_velocity': 0.0,
        'left_wrist_velocity': 0.0,
        'right_ankle_velocity': 0.0,
        'left_ankle_velocity': 0.0,
        'center_of_mass_velocity': 0.0
    }

    # If no previous joints data, return default values
    if joints_old is None:
        return velocities

    try:
        # Calculate wrist velocities
        if 16 in joints_new and 16 in joints_old:  # Right wrist
            displacement = np.linalg.norm(joints_new[16] - joints_old[16])
            velocities['right_wrist_velocity'] = float(displacement / time_delta)

        if 15 in joints_new and 15 in joints_old:  # Left wrist
            displacement = np.linalg.norm(joints_new[15] - joints_old[15])
            velocities['left_wrist_velocity'] = float(displacement / time_delta)

        # Calculate ankle velocities
        if 28 in joints_new and 28 in joints_old:  # Right ankle
            displacement = np.linalg.norm(joints_new[28] - joints_old[28])
            velocities['right_ankle_velocity'] = float(displacement / time_delta)

        if 27 in joints_new and 27 in joints_old:  # Left ankle
            displacement = np.linalg.norm(joints_new[27] - joints_old[27])
            velocities['left_ankle_velocity'] = float(displacement / time_delta)

        # Calculate center of mass velocity (using hips and shoulders average)
        if all(idx in joints_new for idx in [11, 12, 23, 24]) and all(idx in joints_old for idx in [11, 12, 23, 24]):
            com_new = (joints_new[11] + joints_new[12] + joints_new[23] + joints_new[24]) / 4
            com_old = (joints_old[11] + joints_old[12] + joints_old[23] + joints_old[24]) / 4
            displacement = np.linalg.norm(com_new - com_old)
            velocities['center_of_mass_velocity'] = float(displacement / time_delta)

    except Exception as e:
        print(f"Error calculating velocities: {e}")

    return velocities


def calculate_center_of_gravity(joints):
    """Calculate center of gravity of the body."""
    angles = {}

    # Calculate neck and mid_hip positions
    neck = (joints[11] + joints[12]) / 2  # midpoint between shoulders
    mid_hip = (joints[23] + joints[24]) / 2  # midpoint between hips

    # Weights for different body parts (approximate)
    weights = {
        'head': 0.08,  # 8% of body weight
        'torso': 0.55,  # 55% of body weight
        'upper_arms': 0.06,  # 3% each
        'forearms': 0.04,  # 2% each
        'thighs': 0.2,  # 10% each
        'lower_legs': 0.07  # 3.5% each
    }

    # Calculate CoG
    cog = np.zeros(3)

    # Head (nose)
    cog += weights['head'] * joints[0]

    # Torso (mid-point between neck and mid_hip)
    cog += weights['torso'] * ((neck + mid_hip) / 2)

    # Upper arms
    cog += (weights['upper_arms'] / 2) * joints[12]  # Right shoulder
    cog += (weights['upper_arms'] / 2) * joints[11]  # Left shoulder

    # Forearms (including hands)
    cog += (weights['forearms'] / 2) * joints[16]  # Right wrist
    cog += (weights['forearms'] / 2) * joints[15]  # Left wrist

    # Thighs
    cog += (weights['thighs'] / 2) * joints[26]  # Right knee
    cog += (weights['thighs'] / 2) * joints[25]  # Left knee

    # Lower legs (including feet)
    cog += (weights['lower_legs'] / 2) * joints[28]  # Right ankle
    cog += (weights['lower_legs'] / 2) * joints[27]  # Left ankle

    angles['body_cog_x'] = float(cog[0])
    angles['body_cog_y'] = float(cog[1])
    angles['body_cog_z'] = float(cog[2])

    return angles


def calculate_lunge_distance(keypoints):
    """Calculate lunge distance from pose keypoints."""
    try:
        if keypoints and len(keypoints) >= 33:
            # Get feet positions
            left_foot = np.array([keypoints[31]['x'], keypoints[31]['y'], keypoints[31]['z']])
            right_foot = np.array([keypoints[32]['x'], keypoints[32]['y'], keypoints[32]['z']])

            # Calculate distance between feet
            distance = np.linalg.norm(left_foot - right_foot)
            return float(distance)  # Ensure float type
    except Exception as e:
        print(f"Error calculating lunge distance: {e}")

    return 0.0  # Return 0.0 instead of 0 to ensure float type