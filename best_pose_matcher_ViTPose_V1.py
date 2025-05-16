#!/usr/bin/env python3
"""
Best Pose Matcher - Finds the best matching pose in a video rush

This script identifies the timestamp in a video where the pose most closely
matches a reference image, using the ViTPose model for human pose estimation.
Outputs results to a CSV file with advanced similarity metrics.

Based on code from FETHl/pyscenedetect repository.

Current Date and Time (UTC): 2025-05-13 09:39:08
Current User's Login: FETHl
"""

import os
import cv2
import numpy as np
import argparse
import torch
import time
from datetime import datetime
import subprocess
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import math

# Constants for pose detection
# ViTPose uses COCO format which has 17 keypoints
VITPOSE_KEYPOINTS = 17

# Global metadata
CURRENT_TIMESTAMP = "2025-05-13 09:39:08"
CURRENT_USER = "FETHl"

# Position mapping for similarity calculation
POSITION_MAP = {
    "top-left": (0, 0),
    "top-center": (0, 1),
    "top-right": (0, 2),
    "middle-left": (1, 0),
    "center of image": (1, 1),
    "middle-center": (1, 1),  # Alternate form
    "middle-right": (1, 2),
    "bottom-left": (2, 0),
    "bottom-center": (2, 1),
    "bottom-right": (2, 2)
}

# COCO keypoint indices for direction calculation
KEYPOINT_INDICES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}


class SceneAnalyzer:
    """
    Class responsible for analyzing and extracting information about scenes in a video.
    """
    
    def __init__(self, video_path, threshold=27.0, verbose=True):
        """Initialize the scene analyzer."""
        self.video_path = video_path
        self.threshold = threshold
        self.verbose = verbose
        
        # Check if the file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video properties using FFprobe for accuracy
        video_info = self.get_video_info(video_path)
        
        if video_info:
            self.fps = video_info.get('fps', 30)
            self.total_frames = video_info.get('total_frames', 0)
            self.width = video_info.get('width', 0)
            self.height = video_info.get('height', 0)
            self.duration = video_info.get('duration', 0)
        else:
            # Fallback to OpenCV
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise IOError(f"Could not open video file: {video_path}")
                
            self.fps = video.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            video.release()
        
        # Validate properties
        if self.fps <= 0:
            self.fps = 30.0
            self.log(f"Invalid FPS, using default: {self.fps}")
            
        if self.verbose:
            print(f"Initialized analyzer for {os.path.basename(video_path)}")
            print(f"Video FPS: {self.fps}, Duration: {self.duration} seconds")
            print(f"Resolution: {self.width}x{self.height}")
    
    def log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def get_video_info(self, video_path):
        """Get detailed video information using FFprobe."""
        try:
            cmd = [
                "ffprobe", 
                "-v", "quiet", 
                "-print_format", "json", 
                "-show_format", 
                "-show_streams", 
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            data = json.loads(result.stdout)
            
            info = {}
            
            # Get video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if video_stream:
                # Extract frame rate
                fps_str = video_stream.get('r_frame_rate', '30/1')
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    info['fps'] = num / den
                else:
                    info['fps'] = float(fps_str)
                
                # Get dimensions
                info['width'] = int(video_stream.get('width', 0))
                info['height'] = int(video_stream.get('height', 0))
                
                # Get frames
                info['total_frames'] = int(video_stream.get('nb_frames', 0))
                
                # If nb_frames is missing, calculate from duration
                if info['total_frames'] == 0 and 'duration' in video_stream:
                    duration = float(video_stream.get('duration', 0))
                    info['total_frames'] = int(duration * info['fps'])
            
            # Get duration
            if 'format' in data and 'duration' in data['format']:
                info['duration'] = float(data['format']['duration'])
            elif video_stream and 'duration' in video_stream:
                info['duration'] = float(video_stream['duration'])
            else:
                info['duration'] = 0
                
            return info
            
        except Exception as e:
            self.log(f"Error getting video info: {str(e)}")
            return None


class ViTPoseWrapper:
    """
    Wrapper class for ViTPose model to standardize the interface.
    """
    
    def __init__(self, model_path):
        """
        Initialize the ViTPose model.
        
        Args:
            model_path: Path to the ViTPose model or model name
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Import ViTPose dynamically - make sure mmpose is installed
        try:
            import mmpose
            from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
            from mmpose.apis import process_mmdet_results
            from mmdet.apis import inference_detector, init_detector
        except ImportError:
            raise ImportError("ViTPose requires mmpose and mmdet packages. Please install them with: "
                             "pip install 'mmpose>=0.29.0' 'mmdet>=2.28.0'")
        
        self.mmpose = mmpose
        self.init_pose_model = init_pose_model
        self.inference_top_down_pose_model = inference_top_down_pose_model
        self.process_mmdet_results = process_mmdet_results
        self.init_detector = init_detector
        self.inference_detector = inference_detector
        
        # Initialize person detector (YOLOv3 commonly used with ViTPose)
        detector_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'  # Adjust path as needed
        detector_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        
        try:
            self.detector = self.init_detector(detector_config, detector_checkpoint, device=self.device)
        except Exception as e:
            print(f"Warning: Failed to initialize detector with default config. Using fallback. Error: {e}")
            # Use a common fallback or prompt for specific paths
            self.detector = None
        
        # Initialize ViTPose model - First check if the model is a local file
        if os.path.isfile(model_path):
            print(f"Using local model file: {model_path}")
            try:
                # If it's a local .pth file, we need to determine the appropriate config
                if "huge" in model_path.lower():
                    config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_huge_coco_256x192.py'
                    self.pose_model = self.init_pose_model(config, model_path, device=self.device)
                elif "large" in model_path.lower():
                    config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_large_coco_256x192.py'
                    self.pose_model = self.init_pose_model(config, model_path, device=self.device)
                elif "base" in model_path.lower():
                    config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_base_coco_256x192.py'
                    self.pose_model = self.init_pose_model(config, model_path, device=self.device)
                elif "small" in model_path.lower():
                    config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_small_coco_256x192.py'
                    self.pose_model = self.init_pose_model(config, model_path, device=self.device)
                else:
                    # Default to huge if we can't determine from filename
                    config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_huge_coco_256x192.py'
                    self.pose_model = self.init_pose_model(config, model_path, device=self.device)
            except Exception as e:
                raise ValueError(f"Failed to load local model file {model_path}: {e}")
        else:
            # If not a direct path, try to interpret as a model name or URL
            try:
                # Common ViTPose configs paths
                if "small" in model_path.lower():
                    config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_small_coco_256x192.py'
                    checkpoint = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/vitpose_small_coco_256x192-3d308ffd_20230907.pth'
                elif "base" in model_path.lower():
                    config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_base_coco_256x192.py'
                    checkpoint = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/vitpose_base_coco_256x192-338cdb91_20230607.pth'
                elif "large" in model_path.lower():
                    config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_large_coco_256x192.py'
                    checkpoint = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/vitpose_large_coco_256x192-e3f8264d_20230517.pth'
                elif "huge" in model_path.lower():
                    config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_huge_coco_256x192.py'
                    checkpoint = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/vitpose_huge_coco_256x192-33cc5284_20230517.pth'
                else:
                    # Default to base model
                    config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_base_coco_256x192.py'
                    checkpoint = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/vitpose_base_coco_256x192-338cdb91_20230607.pth'
                
                self.pose_model = self.init_pose_model(config, checkpoint, device=self.device)
            except Exception as e:
                raise ValueError(f"Failed to initialize ViTPose model: {e}")
    
    def __call__(self, img):
        """
        Run inference on an image.
        
        Args:
            img: Input image (RGB)
            
        Returns:
            List of pose results
        """
        # Detect people first if detector is available
        if self.detector:
            try:
                mmdet_results = self.inference_detector(self.detector, img)
                person_results = self.process_mmdet_results(mmdet_results, cat_id=0)
            except Exception as e:
                print(f"Person detection failed: {e}. Using simple full-frame detection.")
                # Fallback: assume the full frame contains one person
                person_results = [{'bbox': [0, 0, img.shape[1], img.shape[0], 1.0]}]
        else:
            # Fallback without detector
            person_results = [{'bbox': [0, 0, img.shape[1], img.shape[0], 1.0]}]
            
        # Run pose estimation
        try:
            pose_results, _ = self.inference_top_down_pose_model(
                self.pose_model,
                img,
                person_results,
                bbox_thr=0.3,
                format='xyxy',
                dataset='coco'
            )
            return pose_results
        except Exception as e:
            print(f"Pose estimation failed: {e}")
            return []


def get_position_in_image(bbox, img_width, img_height):
    """
    Determines the position of a bounding box in the image.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2, (confidence)]
        img_width: Image width
        img_height: Image height
        
    Returns:
        String describing the position (e.g., "top-left", "center of image")
    """
    # Extract coordinates, handling cases where confidence is included
    if len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]
    else:
        return "center of image"  # Default if bbox is invalid
    
    # Get bbox center coordinates
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Determine horizontal position
    if center_x < img_width / 3:
        h_pos = "left"
    elif center_x < 2 * img_width / 3:
        h_pos = "center"
    else:
        h_pos = "right"
    
    # Determine vertical position
    if center_y < img_height / 3:
        v_pos = "top"
    elif center_y < 2 * img_height / 3:
        v_pos = "middle"
    else:
        v_pos = "bottom"
    
    # Combine positions
    if v_pos == "middle" and h_pos == "center":
        return "center of image"
    else:
        return f"{v_pos}-{h_pos}"


def calculate_position_similarity(pos1, pos2):
    """
    Calculate similarity between two positions in an image.
    
    Args:
        pos1: First position string (e.g., "top-left")
        pos2: Second position string (e.g., "center of image")
        
    Returns:
        Similarity score between 0 and 1
    """
    # Get grid positions
    grid1 = POSITION_MAP.get(pos1, (1, 1))  # Default to center if position not found
    grid2 = POSITION_MAP.get(pos2, (1, 1))
    
    # Calculate Euclidean distance in grid space
    distance = math.sqrt((grid1[0] - grid2[0])**2 + (grid1[1] - grid2[1])**2)
    
    # Convert to similarity score (0-1)
    max_distance = math.sqrt(8)  # Maximum distance in 3x3 grid (bottom-right to top-left)
    similarity = 1.0 - (distance / max_distance)
    
    return similarity


def calculate_depth_ratio(bbox, img_width, img_height):
    """
    Calculate a depth ratio based on the size of the bounding box relative to the image.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2, (confidence)]
        img_width: Image width
        img_height: Image height
        
    Returns:
        Depth ratio (0-1), larger value means closer to camera
    """
    try:
        # Extract coordinates, handling cases where confidence is included
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
        else:
            return 0.5  # Default if bbox is invalid
        
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Calculate area ratio
        bbox_area = bbox_width * bbox_height
        img_area = img_width * img_height
        
        # Normalize to 0-1
        area_ratio = min(1.0, bbox_area / (img_area * 0.5))  # A person taking up 50% of frame is considered "close"
        
        return area_ratio
    except Exception as e:
        print(f"Error calculating depth ratio: {e}")
        return 0.5  # Default value


def calculate_depth_similarity(depth1, depth2):
    """
    Calculate similarity between two depth ratios.
    
    Args:
        depth1: First depth ratio (0-1)
        depth2: Second depth ratio (0-1)
        
    Returns:
        Similarity score between 0 and 1
    """
    # Simple absolute difference
    return 1.0 - abs(depth1 - depth2)


def estimate_posture_direction(keypoints):
    """
    Estimate the direction a person is facing based on pose keypoints.
    
    Args:
        keypoints: Array of keypoints [N, 3] where each row is [x, y, confidence]
        
    Returns:
        Direction vector and confidence (tuple)
    """
    try:
        # Check for required keypoints
        if keypoints is None or len(keypoints) < VITPOSE_KEYPOINTS:
            return (0, 0), 0.0
        
        # Get shoulder keypoints
        left_shoulder_idx = KEYPOINT_INDICES['left_shoulder']
        right_shoulder_idx = KEYPOINT_INDICES['right_shoulder']
        left_hip_idx = KEYPOINT_INDICES['left_hip']
        right_hip_idx = KEYPOINT_INDICES['right_hip']
        
        # Check if keypoints exist and have sufficient confidence
        left_shoulder = keypoints[left_shoulder_idx] if left_shoulder_idx < len(keypoints) else None
        right_shoulder = keypoints[right_shoulder_idx] if right_shoulder_idx < len(keypoints) else None
        left_hip = keypoints[left_hip_idx] if left_hip_idx < len(keypoints) else None
        right_hip = keypoints[right_hip_idx] if right_hip_idx < len(keypoints) else None
        
        # Calculate shoulder visibility
        shoulder_vis = 0
        if left_shoulder is not None and len(left_shoulder) > 2:
            shoulder_vis += left_shoulder[2]
        if right_shoulder is not None and len(right_shoulder) > 2:
            shoulder_vis += right_shoulder[2]
        
        # Calculate hip visibility
        hip_vis = 0
        if left_hip is not None and len(left_hip) > 2:
            hip_vis += left_hip[2]
        if right_hip is not None and len(right_hip) > 2:
            hip_vis += right_hip[2]
        
        # Direction vector initialization
        direction_x = 0
        direction_y = 0
        confidence = 0
        
        # Calculate direction vector based on shoulders
        if shoulder_vis > 0.5 and left_shoulder is not None and right_shoulder is not None:
            if len(left_shoulder) > 1 and len(right_shoulder) > 1:
                # Vector from right shoulder to left shoulder
                shoulder_vec_x = left_shoulder[0] - right_shoulder[0]
                shoulder_vec_y = left_shoulder[1] - right_shoulder[1]
                
                # Normalize
                shoulder_mag = math.sqrt(shoulder_vec_x**2 + shoulder_vec_y**2) + 1e-6
                shoulder_vec_x /= shoulder_mag
                shoulder_vec_y /= shoulder_mag
                
                direction_x += shoulder_vec_x
                direction_y += shoulder_vec_y
                confidence += shoulder_vis / 2.0
        
        # Calculate direction vector based on hips
        if hip_vis > 0.5 and left_hip is not None and right_hip is not None:
            if len(left_hip) > 1 and len(right_hip) > 1:
                # Vector from right hip to left hip
                hip_vec_x = left_hip[0] - right_hip[0]
                hip_vec_y = left_hip[1] - right_hip[1]
                
                # Normalize
                hip_mag = math.sqrt(hip_vec_x**2 + hip_vec_y**2) + 1e-6
                hip_vec_x /= hip_mag
                hip_vec_y /= hip_mag
                
                direction_x += hip_vec_x
                direction_y += hip_vec_y
                confidence += hip_vis / 2.0
        
        # Normalize the combined vector
        if confidence > 0:
            direction_mag = math.sqrt(direction_x**2 + direction_y**2) + 1e-6
            direction_x /= direction_mag
            direction_y /= direction_mag
            confidence = min(1.0, confidence)
        
        return (direction_x, direction_y), confidence
    except Exception as e:
        print(f"Error estimating posture direction: {e}")
        return (0, 0), 0.0


def calculate_direction_similarity(dir1, conf1, dir2, conf2):
    """
    Calculate similarity between two direction vectors.
    
    Args:
        dir1: First direction vector (x, y)
        conf1: Confidence of first direction
        dir2: Second direction vector (x, y)
        conf2: Confidence of second direction
        
    Returns:
        Similarity score between 0 and 1
    """
    # If either direction has low confidence, similarity is reduced
    confidence_weight = conf1 * conf2
    
    # If confidence is too low, return neutral score
    if confidence_weight < 0.1:
        return 0.5
    
    # Dot product gives cosine similarity (-1 to 1)
    dot_product = dir1[0] * dir2[0] + dir1[1] * dir2[1]
    
    # Convert to 0-1 range
    direction_sim = (dot_product + 1) / 2
    
    # Weight by confidence
    return direction_sim * confidence_weight


def process_vitpose_results(vitpose_results, image_shape):
    """
    Process ViTPose results into a standardized format.
    
    Args:
        vitpose_results: Results from ViTPose model
        image_shape: Original image shape (height, width)
        
    Returns:
        List of dictionaries containing pose information
    """
    all_poses = []
    
    # Get original dimensions for normalization
    img_height, img_width = image_shape[:2]
    
    for person in vitpose_results:
        try:
            # Extract keypoints
            keypoints = person.get('keypoints', [])
            bbox = person.get('bbox', [0, 0, img_width, img_height, 1.0])
            
            if len(keypoints) == 0:
                continue
            
            # Convert keypoints to normalized form
            landmarks = []
            normalized_keypoints = []
            
            # Filter valid keypoints (with sufficient confidence)
            valid_kpts = []
            for kpt in keypoints:
                if len(kpt) >= 3 and kpt[2] > 0.1:
                    valid_kpts.append(kpt)
            
            if len(valid_kpts) < 5:  # Need at least a few keypoints
                continue
            
            for i, kpt in enumerate(keypoints):
                if i < VITPOSE_KEYPOINTS and len(kpt) >= 3:
                    x, y, conf = kpt
                    # Store normalized coordinates
                    landmarks.append({
                        'x': float(x / img_width),
                        'y': float(y / img_height),
                        'z': 0.0,  # ViTPose doesn't provide Z coordinate
                        'visibility': float(conf)
                    })
                    # Keep original format for some calculations
                    normalized_keypoints.append([x, y, conf])
            
            # Calculate position and other metrics
            position = get_position_in_image(bbox, img_width, img_height)
            depth_ratio = calculate_depth_ratio(bbox, img_width, img_height)
            direction, direction_confidence = estimate_posture_direction(keypoints)
            
            # Calculate confidence based on average keypoint confidence
            confidence = np.mean([kpt[2] for kpt in keypoints if len(kpt) > 2])
            
            # Add this person to our list
            all_poses.append({
                'landmarks': landmarks,
                'keypoints': normalized_keypoints,
                'bbox': bbox[:4] if len(bbox) >= 4 else [0, 0, img_width, img_height],
                'confidence': float(confidence),
                'position': position,
                'depth_ratio': depth_ratio,
                'direction': direction,
                'direction_confidence': direction_confidence
            })
        except Exception as e:
            print(f"Error processing a pose detection: {e}")
            continue
    
    return all_poses


def create_pose_feature(pose_landmarks):
    """
    Creates a normalized feature vector from pose landmarks.
    
    Args:
        pose_landmarks: List of pose landmarks
        
    Returns:
        Normalized feature tensor
    """
    if not pose_landmarks:
        return torch.tensor([])
    
    # Initialize feature vector
    feature_vector = []
    
    # Extract features from landmarks
    for landmark in pose_landmarks:
        if isinstance(landmark, dict):
            if landmark.get('visibility', 0) > 0.1:
                feature_vector.extend([landmark['x'], landmark['y'], landmark['z']])
            else:
                feature_vector.extend([0.0, 0.0, 0.0])
    
    if not feature_vector:
        return torch.tensor([])
    
    # Convert to tensor and normalize
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32)
    if len(feature_tensor) > 0:
        norm = torch.norm(feature_tensor, p=2) + 1e-8  # Avoid division by zero
        feature_tensor = feature_tensor / norm
    
    return feature_tensor


def compute_pose_similarity(feature1, feature2, device):
    """
    Computes cosine similarity between two pose feature vectors.
    
    Args:
        feature1: First feature vector
        feature2: Second feature vector
        device: Computation device (CPU/GPU)
        
    Returns:
        Similarity score between 0 and 1
    """
    if len(feature1) == 0 or len(feature2) == 0:
        return 0.0
    
    # Ensure tensors have the same dimension
    if feature1.shape != feature2.shape:
        min_size = min(feature1.shape[0], feature2.shape[0])
        feature1 = feature1[:min_size]
        feature2 = feature2[:min_size]
    
    feature1 = feature1.to(device)
    feature2 = feature2.to(device)
    
    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        feature1.unsqueeze(0), feature2.unsqueeze(0)
    ).item()
    
    # Return positive value
    return max(0.0, similarity)


def compute_enhanced_similarity(ref_pose, frame_pose, device, weights=None):
    """
    Compute an enhanced similarity score that considers position, depth, and direction.
    
    Args:
        ref_pose: Reference pose data dictionary
        frame_pose: Frame pose data dictionary
        device: Computation device (CPU/GPU)
        weights: Dictionary of weights for each component (optional)
        
    Returns:
        Dictionary with overall score and component scores
    """
    # Default weights if not provided
    if weights is None:
        weights = {
            'posture': 0.6,     # Core posture similarity (keypoints)
            'position': 0.15,   # Position in frame
            'depth': 0.1,       # Depth/size in frame
            'direction': 0.15   # Direction person is facing
        }
    
    # Ensure weights sum to 1.0
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
        for key in weights:
            weights[key] /= total_weight
    
    # Calculate core posture similarity
    ref_feature = create_pose_feature(ref_pose['landmarks'])
    frame_feature = create_pose_feature(frame_pose['landmarks'])
    posture_sim = compute_pose_similarity(ref_feature, frame_feature, device)
    
    # Calculate position similarity (safely)
    try:
        position_sim = calculate_position_similarity(
            ref_pose.get('position', 'center of image'), 
            frame_pose.get('position', 'center of image')
        )
    except:
        position_sim = 0.5  # Default to neutral if calculation fails
    
    # Calculate depth similarity (safely)
    try:
        depth_sim = calculate_depth_similarity(
            ref_pose.get('depth_ratio', 0.5), 
            frame_pose.get('depth_ratio', 0.5)
        )
    except:
        depth_sim = 0.5  # Default to neutral if calculation fails
    
    # Calculate direction similarity (safely)
    try:
        direction_sim = calculate_direction_similarity(
            ref_pose.get('direction', (0, 0)), ref_pose.get('direction_confidence', 0),
            frame_pose.get('direction', (0, 0)), frame_pose.get('direction_confidence', 0)
        )
    except:
        direction_sim = 0.5  # Default to neutral if calculation fails
    
    # Compute weighted score
    overall_sim = (
        weights['posture'] * posture_sim +
        weights['position'] * position_sim +
        weights['depth'] * depth_sim + 
        weights['direction'] * direction_sim
    )
    
    # Return all component scores
    return {
        'overall': overall_sim,
        'posture': posture_sim,
        'position': position_sim,
        'depth': depth_sim,
        'direction': direction_sim
    }


class BestPoseMatcher:
    """
    Main class for finding the best matching pose in a video rush compared to a reference image.
    """
    
    def __init__(self, reference_image_path, pose_model_path="vitpose_huge.pth", verbose=True,
                similarity_weights=None):
        """
        Initializes the pose matcher.
        
        Args:
            reference_image_path: Path to the reference image (pose to detect)
            pose_model_path: Path to the ViTPose model or model variant name
            verbose: Display progress messages
            similarity_weights: Custom weights for similarity calculation
        """
        # Initialize basic parameters
        self.reference_image_path = reference_image_path
        self.pose_model_path = pose_model_path
        self.verbose = verbose
        self.has_reference_pose = False  # Flag to track if reference pose is detected
        
        # Set similarity weights
        self.similarity_weights = similarity_weights or {
            'posture': 0.6,     # Core posture similarity (keypoints)
            'position': 0.15,   # Position in frame
            'depth': 0.1,       # Depth/size in frame
            'direction': 0.15   # Direction person is facing
        }
        
        # Set computation device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"Using device: {self.device}")
        
        # Load reference image
        self.reference_image = cv2.imread(reference_image_path)
        if self.reference_image is None:
            raise ValueError(f"Could not load reference image: {reference_image_path}")
            
        # Store reference image resolution
        self.reference_width = self.reference_image.shape[1]
        self.reference_height = self.reference_image.shape[0]
        self.reference_resolution = (self.reference_width, self.reference_height)
        
        # Initialize ViTPose model
        self.log("Initializing ViTPose model...")
        try:
            self.pose_model = ViTPoseWrapper(pose_model_path)
            self.log(f"ViTPose model loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to initialize ViTPose model: {e}")
        
        # Extract pose from reference image
        self.log(f"Extracting pose from reference image ({self.reference_width}x{self.reference_height})...")
        try:
            # Convert BGR to RGB for ViTPose
            ref_image_rgb = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
            
            # Get pose results
            vitpose_results = self.pose_model(ref_image_rgb)
            
            # Process results
            self.reference_poses = process_vitpose_results(vitpose_results, self.reference_image.shape)
            
            # Check if any poses were detected
            if not self.reference_poses:
                self.log("WARNING: No pose detected in reference image")
                self.reference_pose = None
                self.reference_feature = torch.tensor([])
            else:
                # Use the first detected pose (most prominent)
                self.reference_pose = self.reference_poses[0]
                self.reference_feature = create_pose_feature(self.reference_pose['landmarks'])
                
                if len(self.reference_feature) == 0:
                    self.log("WARNING: Could not extract valid pose features from reference image")
                    self.has_reference_pose = False
                else:
                    self.has_reference_pose = True
                    self.log(f"Reference pose detected at position: {self.reference_pose['position']}")
                    self.log(f"Feature vector dimension: {len(self.reference_feature)}")
                    
                    # Log additional information about reference pose
                    self.log(f"Reference depth ratio: {self.reference_pose['depth_ratio']:.2f}")
                    dir_x, dir_y = self.reference_pose['direction']
                    self.log(f"Reference direction: ({dir_x:.2f}, {dir_y:.2f}), confidence: {self.reference_pose['direction_confidence']:.2f}")
        except Exception as e:
            self.log(f"ERROR extracting pose from reference image: {str(e)}")
            self.reference_poses = []
            self.reference_pose = None
            self.reference_feature = torch.tensor([])
        
        # Create output directory
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def extract_video_frames(self, video_path, start_time=0, duration=None, sample_rate=5):
        """
        Extracts frames from a video at regular intervals.
        
        Args:
            video_path: Path to the video
            start_time: Start time in seconds
            duration: Duration in seconds (if None, uses the entire video)
            sample_rate: Sample rate (frames per second)
            
        Returns:
            Dictionary containing extracted frames and their timestamps
        """
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log(f"Error: Could not open video file: {video_path}")
            return {}
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get total duration if not specified
        if duration is None:
            video_info = SceneAnalyzer(video_path, verbose=False)
            duration = video_info.duration - start_time
        
        # Calculate frame positions to extract
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        
        # Calculate extraction interval
        interval = max(1, int(fps / sample_rate))
            
        result = {
            'frames': [],
            'timestamps': [],
            'frame_indices': []
        }
        
        # Extract frames at specified interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame_idx = start_frame
        
        # Calculate total frames to extract
        total_frames = (end_frame - start_frame) // interval + 1
        
        # Use tqdm for a progress bar if verbose
        frame_iterator = tqdm(range(total_frames), desc="Extracting frames", disable=not self.verbose)
        
        for _ in frame_iterator:
            if current_frame_idx >= end_frame:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
                
            # Store the frame and timestamp
            result['frames'].append(frame)
            result['timestamps'].append(current_frame_idx / fps)
            result['frame_indices'].append(current_frame_idx)
            
            # Skip to next position
            current_frame_idx += interval
            if current_frame_idx < end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        
        cap.release()
        self.log(f"Extracted {len(result['frames'])} frames from {video_path}")
        return result
    
    def compute_frame_pose_similarity(self, frames):
        """
        Computes enhanced pose similarity scores for all frames compared to the reference image.
        
        Args:
            frames: List of frames to analyze
            
        Returns:
            Dictionary with similarity scores and index of best frame
        """
        # If no reference pose, return empty scores
        if not self.has_reference_pose:
            return {'scores': [], 'best_frame_index': -1, 'best_score': 0.0, 'poses': [], 'detailed_scores': []}
            
        self.log("Computing enhanced pose similarity scores...")
        
        overall_scores = []
        component_scores = []
        poses = []
        
        # Process each frame with tqdm progress bar
        frame_iterator = tqdm(enumerate(frames), total=len(frames), desc="Analyzing poses", 
                            disable=not self.verbose)
        
        for i, frame in frame_iterator:
            try:
                # Convert BGR to RGB for ViTPose
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get pose results
                vitpose_results = self.pose_model(frame_rgb)
                
                # Process results
                frame_poses = process_vitpose_results(vitpose_results, frame.shape)
                
                if not frame_poses:
                    # No pose detected
                    overall_scores.append(0.0)
                    component_scores.append({
                        'overall': 0.0,
                        'posture': 0.0,
                        'position': 0.0,
                        'depth': 0.0,
                        'direction': 0.0
                    })
                    poses.append(None)
                    continue
                
                # Compare each detected pose with the reference
                frame_score_tuples = []
                for pose in frame_poses:
                    score_components = compute_enhanced_similarity(
                        self.reference_pose, pose, self.device, weights=self.similarity_weights
                    )
                    
                    if score_components['overall'] > 0:
                        frame_score_tuples.append((score_components['overall'], score_components, pose))
                
                # Use highest similarity score for this frame
                if frame_score_tuples:
                    best_overall, best_components, best_pose = max(frame_score_tuples, key=lambda x: x[0])
                    overall_scores.append(best_overall)
                    component_scores.append(best_components)
                    poses.append(best_pose)
                else:
                    overall_scores.append(0.0)
                    component_scores.append({
                        'overall': 0.0,
                        'posture': 0.0,
                        'position': 0.0,
                        'depth': 0.0,
                        'direction': 0.0
                    })
                    poses.append(None)
            except Exception as e:
                self.log(f"Error processing frame {i}: {str(e)}")
                # Add zero scores for this frame
                overall_scores.append(0.0)
                component_scores.append({
                    'overall': 0.0,
                    'posture': 0.0,
                    'position': 0.0,
                    'depth': 0.0,
                    'direction': 0.0
                })
                poses.append(None)
        
        # Find frame with highest similarity
        if not overall_scores:
            return {
                'scores': [], 
                'best_frame_index': -1, 
                'best_score': 0.0, 
                'poses': [], 
                'detailed_scores': []
            }
            
        best_index = np.argmax(overall_scores)
        best_score = overall_scores[best_index]
        
        return {
            'scores': overall_scores,
            'best_frame_index': int(best_index),
            'best_score': float(best_score),
            'poses': poses,
            'detailed_scores': component_scores
        }
    
    def find_best_pose_match(self, video_path, start_time=0, duration=None, sample_rate=5):
        """
        Finds the best matching pose in the video compared to the reference image.
        
        Args:
            video_path: Path to the video rush
            start_time: Start time in seconds to begin analysis
            duration: Duration in seconds to analyze (if None, uses the entire video)
            sample_rate: Frame sampling rate (frames per second)
            
        Returns:
            Dictionary with information about the best matching frame
        """
        self.log(f"Finding best pose match in {video_path}...")
        
        # Get video info
        video_info = SceneAnalyzer(video_path, verbose=False)
        video_duration = video_info.duration
        
        # Validate inputs
        if start_time >= video_duration:
            raise ValueError(f"Start time ({start_time}s) exceeds video duration ({video_duration}s)")
            
        if duration is not None and start_time + duration > video_duration:
            self.log(f"Warning: Requested duration exceeds video length. Analyzing until end of video.")
            duration = video_duration - start_time
        
        # Check if no reference pose was detected - default to middle of video
        if not self.has_reference_pose:
            self.log("No reference pose detected - defaulting to middle of video.")
            middle_timestamp = video_duration / 2
            return {
                'timestamp': middle_timestamp,
                'frame_index': -1,
                'score': 0.0,
                'frame': None,
                'pose': None,
                'video_duration': video_duration,
                'default_reason': "No reference pose detected",
                'component_scores': {
                    'overall': 0.0,
                    'posture': 0.0,
                    'position': 0.0,
                    'depth': 0.0,
                    'direction': 0.0
                }
            }
        
        # Extract frames from video
        video_frames = self.extract_video_frames(
            video_path,
            start_time=start_time,
            duration=duration,
            sample_rate=sample_rate
        )
        
        if not video_frames['frames']:
            self.log("No frames could be extracted from the video")
            # Default to middle of video
            middle_timestamp = video_duration / 2
            return {
                'timestamp': middle_timestamp,
                'frame_index': -1,
                'score': 0.0,
                'frame': None,
                'pose': None,
                'video_duration': video_duration,
                'default_reason': "No frames extracted",
                'component_scores': {
                    'overall': 0.0,
                    'posture': 0.0,
                    'position': 0.0,
                    'depth': 0.0,
                    'direction': 0.0
                }
            }
        
        # Find best matching frame
        similarity_results = self.compute_frame_pose_similarity(video_frames['frames'])
        
        # No good match found - default to middle of video
        if similarity_results['best_frame_index'] < 0 or similarity_results['best_score'] == 0:
            self.log("No pose matches found in video. Defaulting to middle of video.")
            middle_timestamp = video_duration / 2
            return {
                'timestamp': middle_timestamp,
                'frame_index': -1,
                'score': 0.0,
                'frame': None,
                'pose': None,
                'video_duration': video_duration,
                'default_reason': "No matching pose found",
                'component_scores': {
                    'overall': 0.0,
                    'posture': 0.0,
                    'position': 0.0,
                    'depth': 0.0,
                    'direction': 0.0
                }
            }
        
        # Get information about best match
        best_idx = similarity_results['best_frame_index']
        best_score = similarity_results['best_score']
        best_timestamp = video_frames['timestamps'][best_idx]
        best_frame_idx = video_frames['frame_indices'][best_idx]
        best_frame = video_frames['frames'][best_idx]
        best_pose = similarity_results['poses'][best_idx]
        best_components = similarity_results['detailed_scores'][best_idx]
        
        self.log(f"Best match found at {best_timestamp:.2f}s with overall score {best_score:.4f}")
        self.log(f"Component scores - Posture: {best_components['posture']:.4f}, Position: {best_components['position']:.4f}, " +
                f"Depth: {best_components['depth']:.4f}, Direction: {best_components['direction']:.4f}")
        
        # Create result dictionary
        result = {
            'timestamp': best_timestamp,
            'frame_index': best_frame_idx,
            'score': best_score,
            'frame': best_frame,
            'pose': best_pose,
            'video_duration': video_duration,
            'component_scores': best_components
        }
        
        return result
    
    def create_visualization(self, match_result, output_path=None):
        """
        Creates a visualization comparing the reference image with the best matching frame.
        
        Args:
            match_result: Dictionary with match information
            output_path: Path to save the visualization (optional)
            
        Returns:
            Path to the created visualization
        """
        if not match_result:
            self.log("No match result to visualize")
            return None
        
        # Default output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"pose_match_{timestamp}.png")
        
        # Create comparative figure (2 images side by side with graph below)
        fig = plt.figure(figsize=(12, 9))
        
        # Set up the grid layout
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
        
        # Reference image (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Reference Image ({self.reference_width}x{self.reference_height})")
        
        # Best match (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'frame' in match_result and match_result['frame'] is not None:
            best_frame = match_result['frame']
            ax2.imshow(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB))
            score = match_result.get('score', 0)
            ax2.set_title(f"Best Match (Overall Score: {score:.4f})")
            
            # Plot keypoints on best match if available
            if 'pose' in match_result and match_result['pose'] and 'landmarks' in match_result['pose']:
                h2, w2, _ = best_frame.shape
                for landmark in match_result['pose']['landmarks']:
                    if isinstance(landmark, dict) and landmark.get('visibility', 0) > 0.2:
                        ax2.plot(landmark['x'] * w2, landmark['y'] * h2, 'ro', markersize=4)
                
                # Draw direction vector on best match
                if 'pose' in match_result and 'direction' in match_result['pose']:
                    dir_x, dir_y = match_result['pose']['direction']
                    dir_conf = match_result['pose'].get('direction_confidence', 0)
                    
                    if dir_conf > 0.2:
                        try:
                            # Draw center point
                            center_x = w2 / 2
                            center_y = h2 / 2
                            
                            # Calculate arrow endpoint
                            arrow_len = min(w2, h2) * 0.2  # 20% of the smallest dimension
                            end_x = center_x + dir_x * arrow_len
                            end_y = center_y + dir_y * arrow_len
                            
                            # Draw the arrow
                            ax2.arrow(center_x, center_y, dir_x * arrow_len, dir_y * arrow_len, 
                                    color='blue', width=2, head_width=10, alpha=0.7)
                        except Exception:
                            # Skip drawing arrow if there's an error
                            pass
        else:
            # If no match frame, show blank image
            ax2.imshow(np.ones((self.reference_height, self.reference_width, 3), dtype=np.uint8) * 200)
            ax2.set_title("No Match Found")
        
        # Plot keypoints and direction on reference image
        if self.has_reference_pose and self.reference_pose and 'landmarks' in self.reference_pose:
            h1, w1, _ = self.reference_image.shape
            
            # Plot keypoints
            for landmark in self.reference_pose['landmarks']:
                if isinstance(landmark, dict) and landmark.get('visibility', 0) > 0.2:
                    ax1.plot(landmark['x'] * w1, landmark['y'] * h1, 'ro', markersize=4)
            
            # Draw direction vector on reference image
            if 'direction' in self.reference_pose:
                dir_x, dir_y = self.reference_pose['direction']
                dir_conf = self.reference_pose.get('direction_confidence', 0)
                
                if dir_conf > 0.2:
                    try:
                        # Draw center point
                        center_x = w1 / 2
                        center_y = h1 / 2
                        
                        # Calculate arrow endpoint
                        arrow_len = min(w1, h1) * 0.2  # 20% of the smallest dimension
                        end_x = center_x + dir_x * arrow_len
                        end_y = center_y + dir_y * arrow_len
                        
                        # Draw the arrow
                        ax1.arrow(center_x, center_y, dir_x * arrow_len, dir_y * arrow_len, 
                                color='blue', width=2, head_width=10, alpha=0.7)
                    except Exception:
                        # Skip drawing arrow if there's an error
                        pass
        
        # Component score chart (bottom row)
        ax3 = fig.add_subplot(gs[1, :])
        
        # Prepare component scores
        if 'component_scores' in match_result and match_result['component_scores']:
            components = match_result['component_scores']
            component_names = ['Overall', 'Posture', 'Position', 'Depth', 'Direction']
            component_values = [
                components.get('overall', 0),
                components.get('posture', 0),
                components.get('position', 0),
                components.get('depth', 0),
                components.get('direction', 0)
            ]
            
            # Bar colors
            colors = ['green', 'blue', 'orange', 'purple', 'red']
            
            # Create bar chart
            bars = ax3.bar(component_names, component_values, color=colors, alpha=0.7)
            
            # Add value labels on top of each bar
            for bar, value in zip(bars, component_values):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.2f}",
                    ha='center', va='bottom',
                    fontsize=9
                )
            
            # Set chart properties
            ax3.set_ylim(0, 1.1)
            ax3.set_ylabel('Score')
            ax3.set_title('Similarity Component Scores')
            ax3.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            ax3.text(0.5, 0.5, "No component scores available", 
                    ha='center', va='center', fontsize=12)
        
        # Add basic information
        plt.figtext(0.01, 0.01, f"Best match timestamp: {match_result['timestamp']:.2f}s", fontsize=9)
        
        # If default middle was used, show reason
        if match_result['score'] == 0:
            default_reason = match_result.get('default_reason', "No match found")
            plt.figtext(0.01, 0.04, f"Defaulted to middle of video - {default_reason}", fontsize=9, color='red')
        else:
            plt.figtext(0.01, 0.04, f"Frame index: {match_result['frame_index']}", fontsize=9)
        
        # Meta-information
        plt.figtext(0.01, 0.97, f"Generated: {CURRENT_TIMESTAMP} by {CURRENT_USER}", fontsize=8, color='gray')
        
        # Remove axes
        ax1.axis('off')
        ax2.axis('off')
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        self.log(f"Visualization saved: {output_path}")
        return output_path

    def write_to_csv(self, results, csv_path):
        """
        Writes match results to a CSV file.
        
        Args:
            results: List of dictionaries containing match results
            csv_path: Path to the output CSV file
            
        Returns:
            Path to the created CSV file
        """
        # Create header with component scores
        header = [
            'Rush_Name', 
            'Reference_Image', 
            'Best_Pose_Match_Found_At', 
            'Similarity_Score',
            'Posture_Score',
            'Position_Score',
            'Depth_Score',
            'Direction_Score'
        ]
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            
            for result in results:
                rush_name = os.path.basename(result['video_path'])
                ref_image = os.path.basename(result['reference_path'])
                timestamp = result['timestamp']
                score = result['score']
                
                # Get component scores
                if 'component_scores' in result:
                    components = result['component_scores']
                    posture_score = components.get('posture', 0)
                    position_score = components.get('position', 0)
                    depth_score = components.get('depth', 0)
                    direction_score = components.get('direction', 0)
                else:
                    posture_score = position_score = depth_score = direction_score = 0
                
                writer.writerow([
                    rush_name, 
                    ref_image, 
                    f"{timestamp:.2f}", 
                    f"{score:.4f}",
                    f"{posture_score:.4f}",
                    f"{position_score:.4f}",
                    f"{depth_score:.4f}",
                    f"{direction_score:.4f}"
                ])
        
        self.log(f"Results saved to CSV: {csv_path}")
        return csv_path


def process_videos(reference_image, video_paths, pose_model="vitpose_huge.pth", 
                  sample_rate=5, output_csv=None, verbose=True,
                  similarity_weights=None):
    """
    Process multiple videos with the same reference image and output to CSV.
    
    Args:
        reference_image: Path to the reference image
        video_paths: List of paths to video rushes
        pose_model: Path to ViTPose model or model variant name
        sample_rate: Frame sampling rate
        output_csv: Path to output CSV (defaults to 'pose_matches_YYYYMMDD_HHMMSS.csv')
        verbose: Display progress messages
        similarity_weights: Custom weights for similarity calculation
        
    Returns:
        Path to the created CSV file
    """
    # Create matcher with reference image
    try:
        matcher = BestPoseMatcher(
            reference_image,
            pose_model_path=pose_model,
            verbose=verbose,
            similarity_weights=similarity_weights
        )
        
        # Default CSV output path
        if output_csv is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = os.path.join(matcher.output_dir, f"pose_matches_{timestamp}.csv")
        
        # Process each video
        results = []
        for video_path in video_paths:
            try:
                if verbose:
                    print(f"\n{'='*50}")
                    print(f"Processing video: {os.path.basename(video_path)}")
                    print(f"{'='*50}")
                
                # Measure processing time
                start_time = time.time()
                
                # Find best match
                match_result = matcher.find_best_pose_match(
                    video_path,
                    sample_rate=sample_rate
                )
                
                # Check if we need to default to middle of video
                if match_result['score'] == 0:
                    if verbose:
                        print(f"No match found - defaulting to middle of video ({match_result['timestamp']:.2f}s)")
                        if 'default_reason' in match_result:
                            print(f"Reason: {match_result['default_reason']}")
                
                # Create visualization
                vis_path = matcher.create_visualization(
                    match_result, 
                    output_path=os.path.join(matcher.output_dir, f"match_{os.path.basename(video_path)}_{int(time.time())}.png")
                )
                
                # Add to results
                result_dict = {
                    'video_path': video_path,
                    'reference_path': reference_image,
                    'timestamp': match_result['timestamp'],
                    'score': match_result['score'],
                    'component_scores': match_result.get('component_scores', {}),
                    'visualization': vis_path,
                    'video_duration': match_result.get('video_duration', 0)
                }
                
                results.append(result_dict)
                
                elapsed_time = time.time() - start_time
                if verbose:
                    print(f"Processing completed in {elapsed_time:.2f} seconds")
                    print(f"Best pose match: {match_result['timestamp']:.2f}s (Score: {match_result['score']:.4f})")
                    
                    # Print component scores if available
                    if 'component_scores' in match_result:
                        comp = match_result['component_scores']
                        print(f"Component scores:")
                        print(f"- Posture: {comp.get('posture', 0):.4f}")
                        print(f"- Position: {comp.get('position', 0):.4f}")
                        print(f"- Depth: {comp.get('depth', 0):.4f}")
                        print(f"- Direction: {comp.get('direction', 0):.4f}")
                    
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Try to get video duration
                try:
                    video_info = SceneAnalyzer(video_path, verbose=False)
                    video_duration = video_info.duration
                    middle_timestamp = video_duration / 2
                except:
                    video_duration = 0
                    middle_timestamp = 0
                
                # Add a failed result with middle timestamp
                results.append({
                    'video_path': video_path,
                    'reference_path': reference_image,
                    'timestamp': middle_timestamp,
                    'score': 0,
                    'component_scores': {
                        'overall': 0.0,
                        'posture': 0.0,
                        'position': 0.0,
                        'depth': 0.0,
                        'direction': 0.0
                    },
                    'visualization': None,
                    'video_duration': video_duration
                })
        
        # Write all results to CSV
        csv_path = matcher.write_to_csv(results, output_csv)
        
        if verbose:
            print(f"\nAll results saved to CSV: {csv_path}")
        
        return csv_path
    
    except Exception as e:
        print(f"Fatal error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Create a minimal CSV with default values if everything fails
        if output_csv is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_csv = os.path.join(output_dir, f"pose_matches_error_{timestamp}.csv")
            
        # Write basic CSV with middle timestamps
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Rush_Name', 'Reference_Image', 'Best_Pose_Match_Found_At', 'Similarity_Score', 
                            'Posture_Score', 'Position_Score', 'Depth_Score', 'Direction_Score'])
            
            for video_path in video_paths:
                try:
                    # Try to get video duration
                    video_info = SceneAnalyzer(video_path, verbose=False)
                    middle_timestamp = video_info.duration / 2
                except:
                    middle_timestamp = 0
                
                writer.writerow([
                    os.path.basename(video_path),
                    os.path.basename(reference_image),
                    f"{middle_timestamp:.2f}",
                    "0.0000",
                    "0.0000",
                    "0.0000",
                    "0.0000",
                    "0.0000"
                ])
                
        print(f"Created fallback CSV with default middle timestamps: {output_csv}")
        return output_csv


def main():
    """
    Main entry point with command-line interface.
    
    Example usage:
    python best_pose_matcher.py --reference reference.jpg --video rush1.mp4 rush2.mp4 --output-csv results.csv
    """
    parser = argparse.ArgumentParser(description="Find the best matching pose in videos compared to a reference image")
    
    # Required arguments
    parser.add_argument("--reference", required=True, 
                       help="Path to reference image (pose to match)")
    parser.add_argument("--video", required=True, nargs='+',
                       help="Path to one or more video rushes")
    
    # Optional arguments
    parser.add_argument("--pose-model", default="vitpose_huge.pth", 
                       help="Path to ViTPose model or model variant (default: vitpose_huge.pth)")
    parser.add_argument("--sample-rate", type=float, default=5, 
                       help="Frame sampling rate in fps (default: 5)")
    parser.add_argument("--output-csv", 
                       help="Path for CSV output (optional)")
    parser.add_argument("--posture-weight", type=float, default=0.6,
                       help="Weight for posture similarity (default: 0.6)")
    parser.add_argument("--position-weight", type=float, default=0.15,
                       help="Weight for position similarity (default: 0.15)")
    parser.add_argument("--depth-weight", type=float, default=0.1,
                       help="Weight for depth similarity (default: 0.1)")
    parser.add_argument("--direction-weight", type=float, default=0.15,
                       help="Weight for direction similarity (default: 0.15)")
    parser.add_argument("--quiet", action="store_true", 
                       help="Suppress progress messages")
    
    args = parser.parse_args()
    
    try:
        # Configure similarity weights
        similarity_weights = {
            'posture': args.posture_weight,
            'position': args.position_weight,
            'depth': args.depth_weight,
            'direction': args.direction_weight
        }
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(similarity_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            if not args.quiet:
                print(f"Normalizing weights (sum was {total_weight:.2f})")
            for key in similarity_weights:
                similarity_weights[key] /= total_weight
        
        # Process all videos
        csv_path = process_videos(
            args.reference,
            args.video,
            pose_model=args.pose_model,
            sample_rate=args.sample_rate,
            output_csv=args.output_csv,
            verbose=not args.quiet,
            similarity_weights=similarity_weights
        )
        
        print(f"CSV results saved to: {csv_path}")
        return 0
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Print metadata
    print(f"Current Date and Time (UTC): 2025-05-13 09:53:22")
    print(f"Current User's Login: FETHl")
    print("-" * 50)
    main()