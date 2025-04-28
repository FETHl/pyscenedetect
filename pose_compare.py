#!/usr/bin/env python3
"""
Pose Comparison Tool - Analyzes and compares human postures and tool handling in images.

This module uses pose estimation and object detection to compare two images,
focusing on human posture, hand gestures, and tool handling.

Date: 2025-04-28
Author: FETHl
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple, Union, List, Optional
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import mediapipe as mp

# Check if ultralytics is installed for YOLOv11 support
ULTRALYTICS_AVAILABLE = False
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    pass

# Current timestamp and user
CURRENT_TIMESTAMP = "2025-04-28 12:02:00"
CURRENT_USER = "FETHl"


def get_all_poses_mediapipe(image: np.ndarray) -> List[dict]:
    """
    Extract poses for all people in an image using MediaPipe.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of pose dictionaries for each person detected
    """
    # MediaPipe Pose only detects one person at a time, so we need a workaround
    # Let's use a simpler approach with MediaPipe's detector first
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
    results = pose_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    all_poses = []
    
    if results.pose_landmarks:
        # Single person detected
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        all_poses.append({
            'landmarks': landmarks,
            'bbox': estimate_bbox_from_landmarks(landmarks, image.shape[1], image.shape[0]),
            'confidence': np.mean([lm['visibility'] for lm in landmarks if 'visibility' in lm])
        })
    
    return all_poses


def get_all_poses_yolo(image: np.ndarray, pose_model) -> List[dict]:
    """
    Extract poses for all people in an image using YOLOv11 pose model.
    
    Args:
        image: Input image as numpy array
        pose_model: YOLOv11 pose model instance
        
    Returns:
        List of pose dictionaries for each person detected
    """
    # Convert BGR to RGB for YOLO
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference with YOLOv11
    results = pose_model(img_rgb)
    
    all_poses = []
    
    # Check if we have valid results with keypoints
    if len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
        # Get height and width for normalization
        h, w = image.shape[:2]
        
        # Get all detections (all persons)
        for i in range(len(results[0].keypoints.data)):
            # Get keypoints for this person
            kpts = results[0].keypoints.data[i].cpu().numpy()
            
            # Get bounding box for this person
            if hasattr(results[0], 'boxes') and len(results[0].boxes.data) > i:
                bbox = results[0].boxes.data[i].cpu().numpy()
                confidence = bbox[4] if len(bbox) > 4 else 0.0
                bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
            else:
                # Estimate bbox from keypoints if not available
                x_coords = [kpt[0] for kpt in kpts]
                y_coords = [kpt[1] for kpt in kpts]
                bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                confidence = np.mean([kpt[2] for kpt in kpts])
            
            # Convert keypoints to our standardized format
            landmarks = []
            for j, kpt in enumerate(kpts):
                x, y, conf = kpt
                landmarks.append({
                    'x': float(x / w),
                    'y': float(y / h),
                    'z': 0.0,  # YOLOv11 doesn't provide Z-coordinate
                    'visibility': float(conf)
                })
            
            # Add this person to our list
            all_poses.append({
                'landmarks': landmarks,
                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                'confidence': float(confidence)
            })
    
    return all_poses


def estimate_bbox_from_landmarks(landmarks, img_width, img_height):
    """
    Estimate a bounding box from pose landmarks.
    
    Args:
        landmarks: List of pose landmarks
        img_width: Image width
        img_height: Image height
        
    Returns:
        [x1, y1, x2, y2] bounding box
    """
    x_coords = [lm['x'] * img_width for lm in landmarks]
    y_coords = [lm['y'] * img_height for lm in landmarks]
    
    # Add padding to the bounding box
    padding = 0.1  # 10% padding
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    
    x1 = max(0, min(x_coords) - width * padding)
    y1 = max(0, min(y_coords) - height * padding)
    x2 = min(img_width, max(x_coords) + width * padding)
    y2 = min(img_height, max(y_coords) + height * padding)
    
    return [x1, y1, x2, y2]


def get_hand_landmarks(image: np.ndarray, hands_model) -> List[dict]:
    """
    Extract hand landmarks from an image using MediaPipe.
    
    Args:
        image: Input image as numpy array
        hands_model: MediaPipe hands model instance
        
    Returns:
        List of hand landmarks dictionaries
    """
    results = hands_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    all_hands = []
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
            
            # Get the hand label if available
            handedness = ""
            if results.multi_handedness and i < len(results.multi_handedness):
                handedness = results.multi_handedness[i].classification[0].label
            
            all_hands.append({
                'landmarks': landmarks,
                'handedness': handedness,
                'confidence': np.mean([landmark.z for landmark in hand_landmarks.landmark])
            })
    
    return all_hands


def detect_tool(image: np.ndarray, detection_model, device: torch.device, 
                tool_classes: List[int] = None) -> Tuple[Optional[Dict], np.ndarray]:
    """
    Detect tools in an image using YOLOv5.
    
    Args:
        image: Input image as numpy array
        detection_model: YOLOv5 model instance
        device: Torch device (CPU or CUDA)
        tool_classes: List of class indices representing tools (default: common tools)
        
    Returns:
        Tuple containing (detection dict or None if no tool, cropped tool region or None)
    """
    if tool_classes is None:
        # Common tool classes in COCO dataset 
        # (scissors, knife, fork, spoon, cell phone, keyboard, remote, hair drier)
        tool_classes = [76, 77, 78, 44, 67, 73, 75, 78]
    
    # YOLOv5 expects images in PIL format or numpy array
    # Convert BGR (OpenCV format) to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference directly with numpy array
    results = detection_model(img_rgb)
    
    # Process results
    pred = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]
    
    # Filter for tool classes
    tool_detections = [p for p in pred if int(p[5]) in tool_classes]
    
    if not tool_detections:
        return None, None
    
    # Get highest confidence tool detection
    best_tool = tool_detections[np.argmax([t[4] for t in tool_detections])]
    x1, y1, x2, y2, confidence, class_id = best_tool
    
    # Create detection info dict
    detection = {
        'bbox': [float(x1), float(y1), float(x2), float(y2)],
        'confidence': float(confidence),
        'class_id': int(class_id),
        'class_name': results.names[int(class_id)]
    }
    
    # Crop tool region
    tool_crop = image[int(y1):int(y2), int(x1):int(x2)]
    
    return detection, tool_crop


def create_pose_feature(pose_landmarks: List, hand_landmarks: List) -> torch.Tensor:
    """
    Create a normalized feature vector from pose and hand landmarks.
    
    Args:
        pose_landmarks: List of pose landmarks
        hand_landmarks: List of hand landmarks
        
    Returns:
        Torch tensor containing normalized feature vector
    """
    if not pose_landmarks:
        return torch.tensor([])
    
    # Extract pose keypoints
    pose_coords = []
    for landmark in pose_landmarks:
        if isinstance(landmark, dict):
            pose_coords.extend([landmark['x'], landmark['y'], landmark['z']])
        else:
            # Handle other formats if needed
            pass
    
    # Extract hand keypoints if available
    hand_coords = []
    if hand_landmarks:
        for hand in hand_landmarks:
            if isinstance(hand, dict) and 'landmarks' in hand:
                for landmark in hand['landmarks']:
                    hand_coords.extend([landmark['x'], landmark['y'], landmark['z']])
            elif isinstance(hand, list):
                for landmark in hand:
                    if isinstance(landmark, dict):
                        hand_coords.extend([landmark['x'], landmark['y'], landmark['z']])
    
    # Concatenate features
    feature_vector = pose_coords + hand_coords
    
    # Convert to tensor and normalize
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32)
    
    # Normalize the feature vector
    if len(feature_tensor) > 0:
        feature_tensor = F.normalize(feature_tensor, p=2, dim=0)
    
    return feature_tensor


def compute_pose_similarity(feature1: torch.Tensor, feature2: torch.Tensor, device: torch.device) -> float:
    """
    Compute cosine similarity between two pose feature vectors.
    
    Args:
        feature1: First feature tensor
        feature2: Second feature tensor
        device: Torch device (CPU or CUDA)
        
    Returns:
        Similarity score [0-1]
    """
    if len(feature1) == 0 or len(feature2) == 0:
        return 0.0
    
    feature1 = feature1.to(device)
    feature2 = feature2.to(device)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(feature1.unsqueeze(0), feature2.unsqueeze(0)).item()
    
    # Return positive similarity value [0-1]
    return max(0.0, similarity)


def compute_region_similarity(region1: np.ndarray, region2: np.ndarray) -> float:
    """
    Compute structural similarity between two image regions (tool or background).
    
    Args:
        region1: First image region
        region2: Second image region
        
    Returns:
        SSIM similarity score [0-1]
    """
    if region1 is None or region2 is None:
        return 0.0
    
    # Convert to grayscale
    if len(region1.shape) == 3:
        region1_gray = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
    else:
        region1_gray = region1
        
    if len(region2.shape) == 3:
        region2_gray = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)
    else:
        region2_gray = region2
    
    # Resize to match dimensions (use smaller dimensions)
    h1, w1 = region1_gray.shape
    h2, w2 = region2_gray.shape
    target_h, target_w = min(h1, h2), min(w1, w2)
    
    if target_h == 0 or target_w == 0:
        return 0.0
    
    region1_resized = cv2.resize(region1_gray, (target_w, target_h))
    region2_resized = cv2.resize(region2_gray, (target_w, target_h))
    
    # Compute SSIM
    ssim_score, _ = ssim(region1_resized, region2_resized, full=True)
    
    return max(0.0, float(ssim_score))


def compute_combined_similarity(pose_sim: float, region_sim: float, pose_weight: float = 0.7) -> float:
    """
    Compute weighted combination of pose and region similarity scores.
    
    Args:
        pose_sim: Pose similarity score
        region_sim: Region (tool/background) similarity score
        pose_weight: Weight for pose score (1-pose_weight for region)
        
    Returns:
        Combined similarity score [0-1]
    """
    return pose_sim * pose_weight + region_sim * (1.0 - pose_weight)


def create_visualization(image1: np.ndarray, image2: np.ndarray, 
                        best_poses: Tuple[dict, dict],
                        tool_bbox1: Optional[List], tool_bbox2: Optional[List], 
                        filename: str) -> None:
    """
    Create visualization of pose landmarks and tool bounding boxes.
    
    Args:
        image1: First input image
        image2: Second input image
        best_poses: Tuple of (best_pose1, best_pose2) that match best
        tool_bbox1: Tool bounding box for first image (optional)
        tool_bbox2: Tool bounding box for second image (optional)
        filename: Output file path
    """
    best_pose1, best_pose2 = best_poses
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display images
    ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    
    # Plot poses with red markers
    # Plot best matching pose in first image
    for landmark in best_pose1['landmarks']:
        ax1.plot(landmark['x'] * image1.shape[1], landmark['y'] * image1.shape[0], 
                'ro', markersize=4)
    
    # Plot best matching pose in second image
    for landmark in best_pose2['landmarks']:
        ax2.plot(landmark['x'] * image2.shape[1], landmark['y'] * image2.shape[0], 
                'ro', markersize=4)
    
    # Plot pose bounding boxes
    if 'bbox' in best_pose1:
        x1, y1, x2, y2 = best_pose1['bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=2, 
                            edgecolor='b', 
                            facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x1, y1-5, "Best Match", color='b', fontsize=10, fontweight='bold')
    
    if 'bbox' in best_pose2:
        x1, y1, x2, y2 = best_pose2['bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=2, 
                            edgecolor='b', 
                            facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x1, y1-5, "Best Match", color='b', fontsize=10, fontweight='bold')
    
    # Plot tool bounding boxes
    if tool_bbox1:
        x1, y1, x2, y2 = tool_bbox1
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='y', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x1, y2+15, "Tool", color='y', fontsize=10)
    
    if tool_bbox2:
        x1, y1, x2, y2 = tool_bbox2
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='y', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x1, y2+15, "Tool", color='y', fontsize=10)
    
    # Set titles
    ax1.set_title("Image 1")
    ax2.set_title("Image 2")
    
    # Remove axes
    ax1.axis('off')
    ax2.axis('off')
    
    # Add timestamp and user info in the figure
    plt.figtext(0.01, 0.01, f"Generated: {CURRENT_TIMESTAMP} | User: {CURRENT_USER}", fontsize=8)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(description="Compare poses and tool handling in two images")
    parser.add_argument("--image1", required=True, help="Path to first image")
    parser.add_argument("--image2", required=True, help="Path to second image")
    parser.add_argument("--model_pose", default="yolo", 
                        choices=["mediapipe", "yolo"], 
                        help="Pose estimation model to use")
    parser.add_argument("--pose_model_path", default="yolo11x-pose.pt",
                        help="Path to custom YOLOv11 pose model file (if using 'yolo' pose model)")
    parser.add_argument("--model_detect", default="yolov5s", 
                        choices=["yolov5s", "yolov5m", "yolov5l", "yolov5x"],
                        help="Object detection model to use")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"], help="Device to run models on")
    parser.add_argument("--output", default="comparison_results", 
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Generate timestamp-based filenames to prevent overwriting
    timestamp_str = CURRENT_TIMESTAMP.replace(" ", "_").replace(":", "-")
    output_dir = os.path.join(args.output, timestamp_str)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"[{CURRENT_TIMESTAMP}] Using device: {device}")
    
    # Load images
    image1 = cv2.imread(args.image1)
    image2 = cv2.imread(args.image2)
    
    if image1 is None or image2 is None:
        print(f"[{CURRENT_TIMESTAMP}] Error: Unable to load input images: {args.image1}, {args.image2}")
        return
    
    # Initialize models
    print(f"[{CURRENT_TIMESTAMP}] Initializing models...")
    
    # YOLOv5 object detection model
    detection_model = torch.hub.load('ultralytics/yolov5', args.model_detect, pretrained=True)
    detection_model.to(device)
    
    # Initialize pose model based on selection
    if args.model_pose == "mediapipe":
        print(f"[{CURRENT_TIMESTAMP}] Using MediaPipe for pose estimation")
        mp_hands = mp.solutions.hands
        hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        
        # Extract poses for all people in the images
        print(f"[{CURRENT_TIMESTAMP}] Extracting poses with MediaPipe...")
        all_poses1 = get_all_poses_mediapipe(image1)
        all_poses2 = get_all_poses_mediapipe(image2)
        
        # Extract hand landmarks
        hand_landmarks1 = get_hand_landmarks(image1, hands_model)
        hand_landmarks2 = get_hand_landmarks(image2, hands_model)
        
    elif args.model_pose == "yolo":
        if not ULTRALYTICS_AVAILABLE:
            print(f"[{CURRENT_TIMESTAMP}] Error: To use YOLOv11, you need to install the ultralytics package:")
            print("pip install ultralytics")
            return
            
        print(f"[{CURRENT_TIMESTAMP}] Using YOLOv11 model for pose estimation: {args.pose_model_path}")
        
        # Check if model file exists
        if not os.path.isfile(args.pose_model_path):
            print(f"[{CURRENT_TIMESTAMP}] Error: YOLOv11 pose model file not found: {args.pose_model_path}")
            return
            
        # Load YOLOv11 pose model using ultralytics
        pose_model = YOLO(args.pose_model_path)
        
        # Extract all poses from both images
        print(f"[{CURRENT_TIMESTAMP}] Extracting poses with YOLOv11...")
        all_poses1 = get_all_poses_yolo(image1, pose_model)
        all_poses2 = get_all_poses_yolo(image2, pose_model)
        
        # No hand detection with YOLOv11 pose model
        hand_landmarks1 = []
        hand_landmarks2 = []
    
    # Report number of people detected
    print(f"[{CURRENT_TIMESTAMP}] Detected {len(all_poses1)} people in first image")
    print(f"[{CURRENT_TIMESTAMP}] Detected {len(all_poses2)} people in second image")
    
    if len(all_poses1) == 0 or len(all_poses2) == 0:
        print(f"[{CURRENT_TIMESTAMP}] Error: No people detected in at least one image")
        return
    
    # Compute pose features for all people
    print(f"[{CURRENT_TIMESTAMP}] Computing pose features...")
    features1 = []
    features2 = []
    
    for pose in all_poses1:
        feature = create_pose_feature(pose['landmarks'], hand_landmarks1)
        features1.append(feature)
    
    for pose in all_poses2:
        feature = create_pose_feature(pose['landmarks'], hand_landmarks2)
        features2.append(feature)
    
    # Compare all poses and find best match
    print(f"[{CURRENT_TIMESTAMP}] Finding best matching poses...")
    best_similarity = -1
    best_pair = (0, 0)  # indices of best matching poses
    
    for i, feature1 in enumerate(features1):
        for j, feature2 in enumerate(features2):
            similarity = compute_pose_similarity(feature1, feature2, device)
            if similarity > best_similarity:
                best_similarity = similarity
                best_pair = (i, j)
    
    best_pose1 = all_poses1[best_pair[0]]
    best_pose2 = all_poses2[best_pair[1]]
    
    if len(all_poses1) > 1 or len(all_poses2) > 1:
        print(f"[{CURRENT_TIMESTAMP}] Best matching poses: Person {best_pair[0]+1} in image 1 and Person {best_pair[1]+1} in image 2")
    
    print(f"[{CURRENT_TIMESTAMP}] Pose similarity: {best_similarity:.4f}")
    
    # Detect tools
    print(f"[{CURRENT_TIMESTAMP}] Detecting tools...")
    tool_detection1, tool_region1 = detect_tool(image1, detection_model, device)
    tool_detection2, tool_region2 = detect_tool(image2, detection_model, device)
    
    # Handle background comparison if no tool detected
    if tool_region1 is None or tool_region2 is None:
        print(f"[{CURRENT_TIMESTAMP}] No tool detected in at least one image, using background regions")
        # If tools not detected, use person regions instead
        if tool_region1 is None and 'bbox' in best_pose1:
            x1, y1, x2, y2 = best_pose1['bbox']
            tool_region1 = image1[int(y1):int(y2), int(x1):int(x2)]
            
        if tool_region2 is None and 'bbox' in best_pose2:
            x1, y1, x2, y2 = best_pose2['bbox']
            tool_region2 = image2[int(y1):int(y2), int(x1):int(x2)]
    
    # Compute region similarity
    print(f"[{CURRENT_TIMESTAMP}] Computing region similarity...")
    region_similarity = compute_region_similarity(tool_region1, tool_region2)
    combined_similarity = compute_combined_similarity(best_similarity, region_similarity)
    
    # Print results with timestamp
    print(f"\n[{CURRENT_TIMESTAMP}] Comparison Results (User: {CURRENT_USER}):")
    print(f"Pose Similarity: {best_similarity:.4f}")
    print(f"Region Similarity: {region_similarity:.4f}")
    print(f"Combined Similarity: {combined_similarity:.4f}")
    
    # Create visualization
    visualization_path = os.path.join(output_dir, f"pose_comparison_{timestamp_str}.png")
    tool_bbox1 = tool_detection1['bbox'] if tool_detection1 else None
    tool_bbox2 = tool_detection2['bbox'] if tool_detection2 else None
    
    create_visualization(
        image1, image2, 
        (best_pose1, best_pose2),
        tool_bbox1, tool_bbox2, 
        visualization_path
    )
    print(f"[{CURRENT_TIMESTAMP}] Visualization saved to: {visualization_path}")
    
    # Save JSON report
    report = {
        "pose_similarity": best_similarity,
        "region_similarity": region_similarity,
        "combined_similarity": combined_similarity,
        "image1": {
            "path": args.image1,
            "people_detected": len(all_poses1),
            "best_match_index": best_pair[0]
        },
        "image2": {
            "path": args.image2,
            "people_detected": len(all_poses2),
            "best_match_index": best_pair[1]
        },
        "metadata": {
            "date": CURRENT_TIMESTAMP,
            "user": CURRENT_USER,
            "pose_model": args.model_pose,
            "detect_model": args.model_detect,
            "model_version": "YOLOv11" if args.model_pose == "yolo" else "MediaPipe"
        }
    }
    
    json_path = os.path.join(output_dir, f"comparison_report_{timestamp_str}.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[{CURRENT_TIMESTAMP}] Report saved to: {json_path}")
    
    # Create a summary file with basic results
    summary_path = os.path.join(output_dir, f"summary_{timestamp_str}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Pose Comparison Summary\n")
        f.write(f"=====================\n\n")
        f.write(f"Date/Time: {CURRENT_TIMESTAMP}\n")
        f.write(f"User: {CURRENT_USER}\n\n")
        f.write(f"Image 1: {args.image1} ({len(all_poses1)} people detected)\n")
        f.write(f"Image 2: {args.image2} ({len(all_poses2)} people detected)\n\n")
        
        if len(all_poses1) > 1 or len(all_poses2) > 1:
            f.write(f"Best Match: Person {best_pair[0]+1} in Image 1 with Person {best_pair[1]+1} in Image 2\n\n")
        
        f.write(f"Pose Model: {args.model_pose.upper()} {'(YOLOv11)' if args.model_pose == 'yolo' else ''}\n")
        f.write(f"Detection Model: {args.model_detect}\n\n")
        f.write(f"Similarity Scores:\n")
        f.write(f"- Pose: {best_similarity:.4f}\n")
        f.write(f"- Region: {region_similarity:.4f}\n")
        f.write(f"- Combined: {combined_similarity:.4f}\n")
    
    print(f"[{CURRENT_TIMESTAMP}] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()