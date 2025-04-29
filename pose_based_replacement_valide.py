#!/usr/bin/env python3
"""
Outil de Remplacement Vidéo Basé sur la Posture

Ce script combine l'analyse de posture et le remplacement de scènes pour trouver des segments vidéo
avec des poses similaires à une image de référence, et les utiliser pour remplacer des scènes dans une vidéo cible.

Fonctionnalités:
- Trouve les images avec des poses les plus similaires à une image de référence
- Ajuste la résolution des images extraites pour correspondre à l'image de référence
- Positionne précisément l'image correspondante au même endroit relatif que la référence
- Étend ou raccourcit les segments tout en maintenant la position de l'image de référence
- Préserve l'audio original lors du remplacement des scènes

Auteur: FETHl
Date: 2025-04-29
"""

import os
import cv2
import numpy as np
import argparse
import torch
import time
import subprocess
from typing import Dict, Tuple, List, Optional
from datetime import timedelta, datetime
import json
import matplotlib.pyplot as plt

# Import components from existing scripts
from audio_preserving_replacement import (
    SceneAnalyzer, AudioPreservingReplacer, CudaMovementMatcher
)

# Check if ultralytics is installed
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Configuration globale
CURRENT_TIMESTAMP = "2025-04-29 06:41:15"
CURRENT_USER = "FETHl"

# Nombre total de points de repère dans le modèle de pose YOLOv11
YOLO_POSE_KEYPOINTS = 17  # Le modèle YOLOv11 a 17 points de repère corps
FEATURE_DIM_PER_KEYPOINT = 3  # x, y, z (z est toujours 0 dans YOLOv11)


def get_position_in_image(bbox, img_width, img_height):
    """
    Determine the position of a bounding box in the image.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box
        img_width: Image width
        img_height: Image height
        
    Returns:
        Position description (e.g., "top-left", "center", etc.)
    """
    # Get center coordinates of the bbox
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Determine horizontal position
    if center_x < img_width / 3:
        h_pos = "gauche"
    elif center_x < 2 * img_width / 3:
        h_pos = "centre"
    else:
        h_pos = "droite"
    
    # Determine vertical position
    if center_y < img_height / 3:
        v_pos = "haut"
    elif center_y < 2 * img_height / 3:
        v_pos = "milieu"
    else:
        v_pos = "bas"
    
    # Combine positions
    if v_pos == "milieu" and h_pos == "centre":
        return "centre de l'image"
    else:
        return f"{v_pos}-{h_pos} de l'image"


def get_all_poses_yolo(image: np.ndarray, pose_model, target_resolution=None) -> List[dict]:
    """
    Extract poses for all people in an image using YOLOv11 pose model.
    
    Args:
        image: Input image as numpy array
        pose_model: YOLOv11 pose model instance
        target_resolution: Tuple (width, height) to resize image before detection
        
    Returns:
        List of pose dictionaries for each person detected
    """
    # Resize image if target_resolution is provided
    if target_resolution is not None:
        h, w = image.shape[:2]
        
        # Only resize if dimensions are different
        if (w, h) != target_resolution:
            image = cv2.resize(image, target_resolution)
    
    # Convert BGR to RGB for YOLO
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference with YOLOv11
    results = pose_model(img_rgb)
    
    all_poses = []
    
    # Get original dimensions for normalization
    orig_h, orig_w = image.shape[:2]
    
    # Check if we have valid results with keypoints
    if len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
        # Get all detections (all persons)
        for i in range(len(results[0].keypoints.data)):
            # Get keypoints for this person
            kpts = results[0].keypoints.data[i].cpu().numpy()
            
            # Skip if no valid keypoints
            if kpts.size == 0:
                continue
                
            # Extract valid coordinates (non-zero confidence)
            valid_kpts = kpts[kpts[:, 2] > 0.1]
            
            # Skip if no valid keypoints after filtering
            if len(valid_kpts) == 0:
                continue
                
            # Get bounding box for this person
            if hasattr(results[0], 'boxes') and len(results[0].boxes.data) > i:
                bbox = results[0].boxes.data[i].cpu().numpy()
                confidence = float(bbox[4]) if len(bbox) > 4 else 0.0
                bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]  # x1, y1, x2, y2
            else:
                # Estimate bbox from keypoints if not available
                # Make sure we have valid coordinates
                x_coords = [kpt[0] for kpt in valid_kpts]
                y_coords = [kpt[1] for kpt in valid_kpts]
                
                # Skip if no valid coordinates
                if not x_coords or not y_coords:
                    continue
                    
                bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                confidence = float(np.mean([kpt[2] for kpt in kpts]))
            
            # Get position in image
            position = get_position_in_image(bbox, orig_w, orig_h)
            
            # Convertir à un dictionnaire de points de repère indexés par position (0-16 pour YOLOv11)
            landmarks_dict = {}
            
            # Initialiser tous les points de repère avec des valeurs par défaut
            for j in range(YOLO_POSE_KEYPOINTS):
                landmarks_dict[j] = {
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0,
                    'visibility': 0.0
                }
            
            # Remplir les points de repère détectés
            for j, kpt in enumerate(kpts):
                if j < YOLO_POSE_KEYPOINTS:  # Protection contre les index hors limites
                    x, y, conf = kpt
                    landmarks_dict[j] = {
                        'x': float(x / orig_w),
                        'y': float(y / orig_h),
                        'z': 0.0,  # YOLOv11 doesn't provide Z-coordinate
                        'visibility': float(conf)
                    }
            
            # Convertir le dictionnaire en liste pour compatibilité
            landmarks = [landmarks_dict[j] for j in range(YOLO_POSE_KEYPOINTS)]
            
            # Add this person to our list
            all_poses.append({
                'landmarks': landmarks,
                'landmarks_dict': landmarks_dict,  # Store also as dictionary for easier access
                'bbox': bbox,
                'confidence': confidence,
                'position': position
            })
    
    return all_poses


def create_pose_feature(pose_landmarks: List, hand_landmarks: List = None) -> torch.Tensor:
    """
    Create a normalized feature vector from pose landmarks, ensuring consistent dimensionality.
    
    Args:
        pose_landmarks: List of pose landmarks (should be standardized to contain all keypoints)
        hand_landmarks: List of hand landmarks (optional)
        
    Returns:
        Torch tensor containing normalized feature vector
    """
    if not pose_landmarks:
        return torch.tensor([])
    
    # Initialize feature vector with zeros for all keypoints
    feature_vector = []
    
    # Iterate through landmarks in order to create consistent feature vector
    for landmark in pose_landmarks:
        if isinstance(landmark, dict):
            # Only use landmarks with reasonable visibility
            if landmark.get('visibility', 0) > 0.1:
                feature_vector.extend([landmark['x'], landmark['y'], landmark['z']])
            else:
                # Add zeros for low-visibility landmarks to maintain consistent dimensionality
                feature_vector.extend([0.0, 0.0, 0.0])
    
    # If we have no valid features, return empty tensor
    if not feature_vector:
        return torch.tensor([])
    
    # Convert to tensor
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32)
    
    # Normalize the feature vector
    if len(feature_tensor) > 0:
        # Add a small epsilon to avoid division by zero
        norm = torch.norm(feature_tensor, p=2) + 1e-8
        feature_tensor = feature_tensor / norm
    
    return feature_tensor


def compute_pose_similarity(feature1: torch.Tensor, feature2: torch.Tensor, device: torch.device) -> float:
    """
    Compute cosine similarity between two pose feature vectors, handling dimension mismatches.
    
    Args:
        feature1: First feature tensor
        feature2: Second feature tensor
        device: Torch device (CPU or CUDA)
        
    Returns:
        Similarity score [0-1]
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
    
    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(feature1.unsqueeze(0), feature2.unsqueeze(0)).item()
    
    # Return positive similarity value [0-1]
    return max(0.0, similarity)


class PoseBasedSceneReplacement:
    """
    Main class for performing pose-based video scene replacement.
    Finds segments in a source video with poses similar to a reference image,
    and uses them to replace scenes in a target video.
    """
    
    def __init__(self, target_video_path, reference_image_path, 
                 pose_model_path="yolo11x-pose.pt", verbose=True):
        """
        Initialize the pose-based scene replacement tool.
        
        Args:
            target_video_path: Path to the target video (to be edited)
            reference_image_path: Path to the reference image (pose to match)
            pose_model_path: Path to YOLOv11 pose model
            verbose: Whether to print detailed progress messages
        """
        self.target_video_path = target_video_path
        self.reference_image_path = reference_image_path
        self.pose_model_path = pose_model_path
        self.verbose = verbose
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load reference image and extract features
        self.reference_image = cv2.imread(reference_image_path)
        if self.reference_image is None:
            raise ValueError(f"Could not load reference image: {reference_image_path}")
            
        # Store reference image resolution
        self.reference_width = self.reference_image.shape[1]
        self.reference_height = self.reference_image.shape[0]
        self.reference_resolution = (self.reference_width, self.reference_height)
        
        # Create output directories
        self.temp_dir = "temp_files"
        self.output_dir = "output_videos"
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.replacer = AudioPreservingReplacer(target_video_path, verbose=verbose)
        self.analyzer = self.replacer.analyzer
        
        # Initialize models
        self.log("Initializing pose model...")
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Please install ultralytics package: pip install ultralytics")
            
        self.pose_model = YOLO(pose_model_path)
        
        # Extract pose from reference image
        self.log(f"Extracting pose from reference image ({self.reference_width}x{self.reference_height})...")
        self.reference_poses = get_all_poses_yolo(self.reference_image, self.pose_model)
        
        if not self.reference_poses:
            raise ValueError("No poses detected in reference image")
            
        # For now, use the first detected pose (most prominent)
        self.reference_pose = self.reference_poses[0]
        self.reference_feature = create_pose_feature(self.reference_pose['landmarks'])
        
        if len(self.reference_feature) == 0:
            raise ValueError("Could not extract valid pose features from reference image")
            
        self.log(f"Reference pose detected at position: {self.reference_pose['position']}")
        self.log(f"Feature vector dimension: {len(self.reference_feature)}")
    
    def log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def extract_video_frames(self, video_path, start_time=0, duration=None, sample_rate=5):
        """
        Extract frames from a video at regular intervals.
        
        Args:
            video_path: Path to the video
            start_time: Start time in seconds
            duration: Duration to extract in seconds (None for entire video)
            sample_rate: Number of frames per second to extract
            
        Returns:
            Dictionary with frames and timestamps
        """
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
        
        # Calculate extraction interval based on sample rate
        interval = int(fps / sample_rate)
        if interval < 1:
            interval = 1
            
        result = {
            'frames': [],
            'timestamps': [],
            'frame_indices': []
        }
        
        # Set to start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames at the specified interval
        current_frame_idx = start_frame
        
        while current_frame_idx < end_frame:
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
        Compute pose similarity scores for all frames relative to the reference image.
        Adjusts resolution of each frame to match the reference image before comparison.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary with similarity scores and indices of best matching frames
        """
        self.log("Computing pose similarity scores with resolution adjustment...")
        
        scores = []
        poses = []
        
        # Process each frame
        for i, frame in enumerate(frames):
            # Extract poses from this frame - resize to match reference image resolution
            frame_poses = get_all_poses_yolo(frame, self.pose_model, target_resolution=self.reference_resolution)
            
            if not frame_poses:
                # No poses detected in this frame
                scores.append(0.0)
                poses.append(None)
                continue
            
            # Compare each detected pose to the reference
            frame_scores = []
            for pose in frame_poses:
                feature = create_pose_feature(pose['landmarks'])
                # Skip if we couldn't extract valid features
                if len(feature) == 0:
                    continue
                    
                # Debug: print feature dimensions for comparison
                if i == 0:
                    self.log(f"Reference feature dimension: {len(self.reference_feature)}, Frame feature dimension: {len(feature)}")
                
                similarity = compute_pose_similarity(self.reference_feature, feature, self.device)
                frame_scores.append((similarity, pose))
            
            # Use highest similarity score for this frame
            if frame_scores:
                best_score, best_pose = max(frame_scores, key=lambda x: x[0])
                scores.append(best_score)
                poses.append(best_pose)
            else:
                scores.append(0.0)
                poses.append(None)
                
            # Show progress
            if i % 10 == 0 and i > 0:
                self.log(f"Processed {i+1}/{len(frames)} frames")
        
        # Find frame with highest similarity
        if not scores:
            return {'scores': [], 'best_frame_index': -1, 'best_score': 0.0, 'poses': []}
            
        best_index = np.argmax(scores)
        best_score = scores[best_index]
        
        return {
            'scores': scores,
            'best_frame_index': int(best_index),
            'best_score': float(best_score),
            'poses': poses
        }
    
    def get_reference_frame_position(self, target_scene_info):
        """
        Extraire et analyser l'image de référence dans la scène cible pour obtenir sa position relative.
        
        Args:
            target_scene_info: Informations sur la scène cible
            
        Returns:
            Position relative de l'image de référence dans la scène (0-1) et timestamp absolu
        """
        # Extraire des frames de la scène cible
        target_frames = self.extract_video_frames(
            self.target_video_path, 
            start_time=target_scene_info['start_time'], 
            duration=target_scene_info['duration'],
            sample_rate=10  # Taux d'échantillonnage plus élevé pour une meilleure précision
        )
        
        if not target_frames['frames']:
            # Si nous ne pouvons pas extraire des frames, on place l'image au milieu par défaut
            self.log("Impossible d'extraire des frames de la scène cible, utilisation de la position par défaut (milieu)")
            relative_position = 0.5
            reference_timestamp = target_scene_info['start_time'] + (target_scene_info['duration'] / 2)
            return relative_position, reference_timestamp
        
        # Trouver la frame la plus similaire à notre image de référence (avec ajustement de résolution)
        similarity_results = self.compute_frame_pose_similarity(target_frames['frames'])
        
        if similarity_results['best_frame_index'] < 0:
            # Si nous ne trouvons pas de bonne correspondance, on place l'image au milieu par défaut
            self.log("Aucune correspondance trouvée dans la scène cible, utilisation de la position par défaut (milieu)")
            relative_position = 0.5
            reference_timestamp = target_scene_info['start_time'] + (target_scene_info['duration'] / 2)
        else:
            # Calculer la position relative et le timestamp absolu
            best_index = similarity_results['best_frame_index']
            best_timestamp = target_frames['timestamps'][best_index]
            reference_timestamp = target_scene_info['start_time'] + best_timestamp
            
            # Position relative (0-1) dans la scène
            relative_position = best_timestamp / target_scene_info['duration']
            
            self.log(f"Image de référence trouvée à {reference_timestamp:.2f}s dans la vidéo cible " + 
                     f"(position relative dans la scène: {relative_position:.2f})")
            
        return relative_position, reference_timestamp
    
    def find_best_segment(self, source_video_path, target_scene_info, step_size=5.0, segment_overlap=0.5):
        """
        Trouver le meilleur segment dans la vidéo source correspondant à l'image de référence,
        et le positionner correctement par rapport à la position de référence dans la scène cible.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            target_scene_info: Informations sur la scène cible
            step_size: Longueur des morceaux vidéo à analyser à la fois en secondes
            segment_overlap: Chevauchement entre les morceaux (0-1)
            
        Returns:
            Dictionnaire avec les informations sur le meilleur segment
        """
        self.log(f"Recherche du meilleur segment dans {source_video_path}...")
        
        # Obtenir les informations sur la vidéo source
        source_info = SceneAnalyzer(source_video_path, verbose=False)
        source_duration = source_info.duration
        
        # Obtenir la durée cible
        target_duration = target_scene_info['duration']
        
        # Si la durée cible dépasse la longueur de la vidéo source, retourner une erreur
        if target_duration > source_duration:
            self.log(f"Erreur: Durée cible ({target_duration:.2f}s) dépasse la longueur " +
                     f"de la vidéo source ({source_duration:.2f}s)")
            return None
            
        # Déterminer la position de l'image de référence dans la scène cible
        ref_position_rel, ref_timestamp = self.get_reference_frame_position(target_scene_info)
        
        # Initialiser les variables pour le suivi du meilleur segment
        best_score = -1
        best_segment = None
        best_frame_timestamp = None
        best_frame_index = None
        best_frame_poses = None
        
        # Analyser la vidéo par morceaux pour éviter de charger toute la vidéo en mémoire
        chunk_size = step_size
        overlap = segment_overlap * chunk_size
        
        current_time = 0
        while current_time + chunk_size <= source_duration:
            # Extraire les frames de ce morceau
            chunk_frames = self.extract_video_frames(
                source_video_path, 
                start_time=current_time, 
                duration=chunk_size
            )
            
            if not chunk_frames['frames']:
                current_time += chunk_size - overlap
                continue
                
            # Calculer la similarité de pose pour les frames de ce morceau
            # Avec ajustement de résolution pour correspondre à l'image de référence
            similarity_results = self.compute_frame_pose_similarity(chunk_frames['frames'])
            
            # Vérifier si nous avons une nouvelle meilleure correspondance
            if similarity_results['best_score'] > best_score:
                best_score = similarity_results['best_score']
                best_frame_index = similarity_results['best_frame_index']
                
                # Convertir l'index du morceau en timestamp global et index de frame
                best_frame_timestamp = current_time + chunk_frames['timestamps'][best_frame_index]
                best_frame_index_global = chunk_frames['frame_indices'][best_frame_index]
                
                # Calculer les limites du segment en respectant la position relative
                # L'image correspondante doit être placée à la même position relative que l'image de référence
                
                # Calculer le temps avant et après l'image dans le segment final
                time_before = ref_position_rel * target_duration
                time_after = (1 - ref_position_rel) * target_duration
                
                # Calculer les bornes du segment
                segment_start = max(0, best_frame_timestamp - time_before)
                segment_end = min(source_duration, best_frame_timestamp + time_after)
                
                # Ajuster si on atteint les limites
                if segment_start == 0:
                    # Si on est au début, essayer d'étendre la fin
                    segment_end = min(source_duration, segment_start + target_duration)
                    
                if segment_end == source_duration:
                    # Si on est à la fin, essayer d'étendre le début
                    segment_start = max(0, segment_end - target_duration)
                
                # Stocker les informations sur le meilleur segment
                best_segment = {
                    'start_time': segment_start,
                    'end_time': segment_end,
                    'duration': segment_end - segment_start,
                    'score': best_score,
                    'reference_time': best_frame_timestamp,
                    'reference_frame_index': best_frame_index_global,
                    'target_ref_position': ref_position_rel,
                    'target_ref_timestamp': ref_timestamp
                }
                
                # Stocker les informations de pose pour la visualisation
                best_frame_poses = similarity_results['poses'][best_frame_index]
                
                self.log(f"Nouvelle meilleure correspondance à {best_frame_timestamp:.2f}s avec score {best_score:.4f}")
                self.log(f"Segment provisoire: {segment_start:.2f}s - {segment_end:.2f}s (durée: {segment_end - segment_start:.2f}s)")
            
            # Passer au morceau suivant avec chevauchement
            current_time += chunk_size - overlap
            
            # Arrêter si on trouve une très bonne correspondance (score > 0.85)
            if best_score > 0.85:
                self.log(f"Excellente correspondance trouvée, arrêt anticipé de la recherche")
                break
        
        if best_segment is None:
            self.log("Erreur: Impossible de trouver un segment approprié")
            return None
            
        # Extraire la meilleure frame pour référence
        cap = cv2.VideoCapture(source_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_segment['reference_frame_index'])
        ret, best_frame = cap.read()
        cap.release()
        
        if ret:
            # Redimensionner la frame pour qu'elle corresponde à la résolution de référence (pour la visualisation)
            best_frame_resized = cv2.resize(best_frame, self.reference_resolution)
            best_segment['reference_frame'] = best_frame_resized
            best_segment['pose'] = best_frame_poses
        
        # Ajuster la durée du segment si nécessaire
        adjusted_segment = self.adjust_segment_duration(
            source_video_path, best_segment, target_duration
        )
        
        return adjusted_segment
    
    def adjust_segment_duration(self, source_video_path, segment, target_duration):
        """
        Ajuster la durée du segment pour correspondre à la durée cible,
        tout en maintenant l'image de meilleure correspondance à la même position relative.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            segment: Dictionnaire avec les informations sur le segment
            target_duration: Durée requise en secondes
            
        Returns:
            Segment ajusté (dictionnaire)
        """
        current_duration = segment['duration']
        
        # Si les durées sont très proches, pas besoin d'ajuster
        if abs(current_duration - target_duration) < 0.1:
            self.log("La durée du segment correspond à la cible, aucun ajustement nécessaire")
            return segment
        
        # Créer une copie du segment à ajuster
        adjusted = segment.copy()
        
        # Position relative de l'image de référence dans le segment actuel
        ref_time = segment['reference_time']
        start_time = segment['start_time']
        end_time = segment['end_time']
        
        # Calculer la position relative actuelle
        current_rel_pos = (ref_time - start_time) / current_duration
        
        # Position relative cible (celle déterminée dans la scène originale)
        target_rel_pos = segment.get('target_ref_position', 0.5)
        
        if current_duration < target_duration:
            # Le segment est trop court, besoin d'étendre
            self.log(f"Durée du segment ({current_duration:.2f}s) inférieure à la cible ({target_duration:.2f}s)")
            self.log("Extension en ajoutant des frames de manière fluide")
            
            # Calculer combien étendre
            extension_needed = target_duration - current_duration
            
            # Calculer combien étendre au début et à la fin pour maintenir la position relative
            time_before_ref = ref_time - start_time
            time_after_ref = end_time - ref_time
            
            # Calculer le temps avant/après souhaité pour la nouvelle durée
            target_time_before = target_rel_pos * target_duration
            target_time_after = (1 - target_rel_pos) * target_duration
            
            # Calculer l'extension nécessaire avant et après
            extend_start = target_time_before - time_before_ref
            extend_end = target_time_after - time_after_ref
            
            # Ajuster les limites du segment
            new_start = max(0, start_time - extend_start)
            # Utiliser source_info pour obtenir la durée réelle
            source_info = SceneAnalyzer(source_video_path, verbose=False)
            new_end = min(end_time + extend_end, source_info.duration)
            
            # Si on ne peut pas étendre suffisamment d'un côté, compenser de l'autre
            if new_start == 0 and new_end - new_start < target_duration:
                new_end = min(source_info.duration, new_start + target_duration)
            
            if new_end == source_info.duration and new_end - new_start < target_duration:
                new_start = max(0, new_end - target_duration)
            
            # Mettre à jour le segment
            adjusted['start_time'] = new_start
            adjusted['end_time'] = new_end
            adjusted['duration'] = new_end - new_start
            
            # Calculer la nouvelle position relative de l'image de référence
            new_rel_pos = (ref_time - new_start) / (new_end - new_start)
            
            self.log(f"Segment étendu: {adjusted['start_time']:.2f}s à {adjusted['end_time']:.2f}s " +
                    f"({adjusted['duration']:.2f}s)")
            self.log(f"Position relative de l'image de référence: {new_rel_pos:.4f}")
            
        elif current_duration > target_duration:
            # Le segment est trop long, besoin de couper
            self.log(f"Durée du segment ({current_duration:.2f}s) supérieure à la cible ({target_duration:.2f}s)")
            self.log("Coupe tout en préservant la position relative de l'image de référence")
            
            # Calculer le temps avant/après souhaité pour la nouvelle durée
            target_time_before = target_rel_pos * target_duration
            target_time_after = (1 - target_rel_pos) * target_duration
            
            # Calculer les nouvelles limites du segment
            new_start = ref_time - target_time_before
            new_end = ref_time + target_time_after
            
            # Mettre à jour le segment
            adjusted['start_time'] = new_start
            adjusted['end_time'] = new_end
            adjusted['duration'] = target_duration
            
            self.log(f"Segment coupé: {adjusted['start_time']:.2f}s à {adjusted['end_time']:.2f}s " +
                    f"({adjusted['duration']:.2f}s)")
        
        return adjusted
    
    def create_visualizations(self, segment, target_scene, output_path=None):
        """
        Créer une visualisation de la meilleure frame correspondante et de l'image de référence.
        
        Args:
            segment: Informations sur le meilleur segment
            target_scene: Informations sur la scène cible
            output_path: Chemin pour sauvegarder la visualisation (optionnel)
            
        Returns:
            Chemin vers la visualisation sauvegardée
        """
        # Chemin de sortie par défaut
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"visualization_{timestamp}.png")
        
        # Créer la figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Afficher l'image de référence
        ax1.imshow(cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Image de référence ({self.reference_width}x{self.reference_height})")
        
        # Afficher la meilleure frame correspondante
        if 'reference_frame' in segment:
            ax2.imshow(cv2.cvtColor(segment['reference_frame'], cv2.COLOR_BGR2RGB))
            ax2.set_title(f"Meilleure correspondance (Score: {segment['score']:.4f})")
        
        # Tracer les points clés de pose de référence
        if self.reference_pose and 'landmarks' in self.reference_pose:
            h1, w1, _ = self.reference_image.shape
            for landmark in self.reference_pose['landmarks']:
                if landmark.get('visibility', 0) > 0.2:  # Ne tracer que les points visibles
                    ax1.plot(landmark['x'] * w1, landmark['y'] * h1, 'ro', markersize=4)
        
        # Tracer les points clés de la meilleure pose correspondante
        if 'pose' in segment and segment['pose'] and 'landmarks' in segment['pose']:
            h2, w2 = self.reference_resolution
            for landmark in segment['pose']['landmarks']:
                if landmark.get('visibility', 0) > 0.2:  # Ne tracer que les points visibles
                    ax2.plot(landmark['x'] * w2, landmark['y'] * h2, 'ro', markersize=4)
        
        # Ajouter des informations textuelles
        plt.figtext(0.02, 0.02, f"Temps de l'image de référence: {segment['reference_time']:.2f}s", fontsize=9)
        plt.figtext(0.02, 0.05, f"Segment: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s ({segment['duration']:.2f}s)", fontsize=9)
        plt.figtext(0.02, 0.08, f"Scène cible: {target_scene['start_time']:.2f}s - {target_scene['end_time']:.2f}s ({target_scene['duration']:.2f}s)", fontsize=9)
        
        if 'target_ref_position' in segment:
            rel_pos = segment['target_ref_position']
            plt.figtext(0.02, 0.11, f"Position relative cible: {rel_pos:.4f}", fontsize=9)
            
            # Calculer la position relative actuelle
            current_rel_pos = (segment['reference_time'] - segment['start_time']) / segment['duration']
            plt.figtext(0.02, 0.14, f"Position relative actuelle: {current_rel_pos:.4f}", fontsize=9)
        
        # Ajouter timestamp et utilisateur
        plt.figtext(0.02, 0.01, f"Généré: {CURRENT_TIMESTAMP} | Utilisateur: {CURRENT_USER}", fontsize=8, color='gray')
        
        # Supprimer les axes
        ax1.axis('off')
        ax2.axis('off')
        
        # Sauvegarder la figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        self.log(f"Visualisation sauvegardée à {output_path}")
        return output_path
    
    def replace_scene_with_best_pose_match(self, source_video_path, target_scene_number, output_path=None):
        """
        Remplacer une scène dans la vidéo cible par le meilleur segment correspondant de la vidéo source.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            target_scene_number: Numéro de la scène à remplacer dans la vidéo cible
            output_path: Chemin pour la vidéo de sortie (optionnel)
            
        Returns:
            Chemin vers la vidéo de sortie
        """
        # Obtenir les informations sur la scène cible
        target_info = self.replacer.get_target_scene_info(target_scene_number)
        if not target_info:
            self.log(f"Erreur: Scène {target_scene_number} non trouvée dans la vidéo cible")
            return None
        
        # Trouver le meilleur segment correspondant dans la vidéo source
        best_segment = self.find_best_segment(
            source_video_path, 
            target_info
        )
        
        if not best_segment:
            self.log("Impossible de trouver un segment approprié pour le remplacement")
            return None
        
        # Créer une visualisation
        vis_path = self.create_visualizations(best_segment, target_info)
        
        # Remplacer la scène avec le meilleur segment
        self.log(f"Remplacement de la scène {target_scene_number} par le segment de {best_segment['start_time']:.2f}s " +
                f"à {best_segment['end_time']:.2f}s")
        
        result_path = self.replacer.replace_scene_preserving_audio(
            source_video_path,
            target_scene_number,
            output_path=output_path,
            best_match_start=best_segment['start_time']
        )
        
        if result_path:
            self.log(f"Remplacement de scène réussi")
            self.log(f"Vidéo de sortie sauvegardée à: {result_path}")
            
            # Générer des informations détaillées sur le remplacement
            info_path = os.path.join(self.output_dir, "replacement_info.txt")
            with open(info_path, 'a') as f:
                f.write("\n" + "="*50 + "\n")
                f.write(f"Date: {CURRENT_TIMESTAMP}\n")
                f.write(f"Utilisateur: {CURRENT_USER}\n")
                f.write(f"Vidéo cible: {self.target_video_path}\n")
                f.write(f"Vidéo source: {source_video_path}\n")
                f.write(f"Image de référence: {self.reference_image_path}\n")
                f.write(f"Scène remplacée: {target_scene_number} ({target_info['start_time']:.2f}s - {target_info['end_time']:.2f}s)\n")
                f.write(f"Segment utilisé: {best_segment['start_time']:.2f}s - {best_segment['end_time']:.2f}s\n")
                f.write(f"Score de similarité: {best_segment['score']:.4f}\n")
                f.write(f"Position relative cible: {best_segment.get('target_ref_position', 0.5):.4f}\n")
                f.write(f"Visualisation: {vis_path}\n")
                f.write(f"Vidéo de sortie: {result_path}\n")
            
            self.log(f"Informations détaillées sauvegardées dans {info_path}")
        else:
            self.log("Échec du remplacement de scène")
        
        return result_path


def main():
    """Point d'entrée principal avec interface CLI."""
    parser = argparse.ArgumentParser(description="Outil de remplacement de scènes vidéo basé sur la pose")
    parser.add_argument("--target-video", required=True, help="Chemin vers la vidéo cible (à éditer)")
    parser.add_argument("--source-video", required=True, help="Chemin vers la vidéo source (pour extraire des segments)")
    parser.add_argument("--reference-image", required=True, help="Chemin vers l'image de référence (pose à correspondre)")
    parser.add_argument("--target-scene", type=int, required=True, help="Numéro de la scène à remplacer dans la vidéo cible")
    parser.add_argument("--pose-model", default="yolo11x-pose.pt", help="Chemin vers le fichier du modèle YOLOv11 pose")
    parser.add_argument("--output", help="Chemin pour la vidéo de sortie")
    parser.add_argument("--quiet", action="store_true", help="Supprimer les messages de progression")
    
    args = parser.parse_args()
    
    try:
        # Créer l'outil de remplacement basé sur la pose
        replacer = PoseBasedSceneReplacement(
            args.target_video,
            args.reference_image,
            pose_model_path=args.pose_model,
            verbose=not args.quiet
        )
        
        # Traiter le remplacement
        start_time = time.time()
        output_path = replacer.replace_scene_with_best_pose_match(
            args.source_video,
            args.target_scene,
            output_path=args.output
        )
        
        elapsed_time = time.time() - start_time
        
        if not args.quiet:
            print(f"\nTraitement terminé en {elapsed_time:.2f} secondes")
            if output_path:
                print(f"Résultat sauvegardé à: {output_path}")
                
    except Exception as e:
        print(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    main()