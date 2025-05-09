#!/usr/bin/env python3
"""
Outil de Remplacement Vidéo Basé sur la Posture - Version Multiple Scènes (Corrigé)

Ce script combine l'analyse de posture et le remplacement de scènes pour trouver des segments vidéo
avec des poses similaires à une image de référence, et les utiliser pour remplacer plusieurs scènes
identiques dans une vidéo cible.

Auteur: FETHl
Date: 2025-05-05 14:19:45
Version: 2.0

CORRECTION: Résolution du problème d'image fixe lors du remplacement de plusieurs scènes
"""

import os
import cv2
import numpy as np
import argparse
import torch
import time
import subprocess
import math
from typing import Dict, Tuple, List, Optional
from datetime import timedelta, datetime
import json
import matplotlib.pyplot as plt

# Import des composants depuis le script existant
from audio_preserving_replacement import (
    SceneAnalyzer, AudioPreservingReplacer, CudaMovementMatcher
)

# Vérification si ultralytics est installé
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Configuration globale
CURRENT_TIMESTAMP = "2025-05-05 14:19:45"
CURRENT_USER = "FETHl"

# Nombre total de points de repère dans le modèle de pose YOLOv11
YOLO_POSE_KEYPOINTS = 17  # Le modèle YOLOv11 a 17 points de repère corps
FEATURE_DIM_PER_KEYPOINT = 3  # x, y, z (z est toujours 0 dans YOLOv11)


def get_position_in_image(bbox, img_width, img_height):
    """
    Détermine la position d'une boîte englobante dans l'image.
    """
    # Obtenir les coordonnées du centre de la bbox
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Déterminer la position horizontale
    if center_x < img_width / 3:
        h_pos = "gauche"
    elif center_x < 2 * img_width / 3:
        h_pos = "centre"
    else:
        h_pos = "droite"
    
    # Déterminer la position verticale
    if center_y < img_height / 3:
        v_pos = "haut"
    elif center_y < 2 * img_height / 3:
        v_pos = "milieu"
    else:
        v_pos = "bas"
    
    # Combiner les positions
    if v_pos == "milieu" and h_pos == "centre":
        return "centre de l'image"
    else:
        return f"{v_pos}-{h_pos} de l'image"


def get_all_poses_yolo(image: np.ndarray, pose_model, target_resolution=None) -> List[dict]:
    """
    Extrait les poses de toutes les personnes dans une image en utilisant le modèle YOLOv11.
    """
    # Redimensionner l'image si target_resolution est fourni
    if target_resolution is not None:
        h, w = image.shape[:2]
        
        # Redimensionner uniquement si les dimensions sont différentes
        if (w, h) != target_resolution:
            image = cv2.resize(image, target_resolution)
    
    # Convertir BGR en RGB pour YOLO
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Lancer l'inférence avec YOLOv11
    results = pose_model(img_rgb)
    
    all_poses = []
    
    # Obtenir les dimensions originales pour normalisation
    orig_h, orig_w = image.shape[:2]
    
    # Vérifier si nous avons des résultats valides avec des keypoints
    if len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
        # Obtenir toutes les détections (toutes les personnes)
        for i in range(len(results[0].keypoints.data)):
            # Obtenir les keypoints pour cette personne
            kpts = results[0].keypoints.data[i].cpu().numpy()
            
            # Ignorer si pas de keypoints valides
            if kpts.size == 0:
                continue
                
            # Extraire les coordonnées valides (confiance non nulle)
            valid_kpts = kpts[kpts[:, 2] > 0.1]
            
            # Ignorer si pas de keypoints valides après filtrage
            if len(valid_kpts) == 0:
                continue
                
            # Obtenir la boîte englobante pour cette personne
            if hasattr(results[0], 'boxes') and len(results[0].boxes.data) > i:
                bbox = results[0].boxes.data[i].cpu().numpy()
                confidence = float(bbox[4]) if len(bbox) > 4 else 0.0
                bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]  # x1, y1, x2, y2
            else:
                # Estimer la bbox à partir des keypoints si non disponible
                # S'assurer que nous avons des coordonnées valides
                x_coords = [kpt[0] for kpt in valid_kpts]
                y_coords = [kpt[1] for kpt in valid_kpts]
                
                # Ignorer si pas de coordonnées valides
                if not x_coords or not y_coords:
                    continue
                    
                bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                confidence = float(np.mean([kpt[2] for kpt in kpts]))
            
            # Obtenir la position dans l'image
            position = get_position_in_image(bbox, orig_w, orig_h)
            
            # Convertir en dictionnaire de points de repère indexés par position (0-16 pour YOLOv11)
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
                        'z': 0.0,  # YOLOv11 ne fournit pas de coordonnée Z
                        'visibility': float(conf)
                    }
            
            # Convertir le dictionnaire en liste pour compatibilité
            landmarks = [landmarks_dict[j] for j in range(YOLO_POSE_KEYPOINTS)]
            
            # Ajouter cette personne à notre liste
            all_poses.append({
                'landmarks': landmarks,
                'landmarks_dict': landmarks_dict,  # Stocker aussi comme dictionnaire pour un accès plus facile
                'bbox': bbox,
                'confidence': confidence,
                'position': position
            })
    
    return all_poses


def create_pose_feature(pose_landmarks: List, hand_landmarks: List = None) -> torch.Tensor:
    """
    Crée un vecteur de caractéristiques normalisé à partir des points de repère de pose.
    """
    if not pose_landmarks:
        return torch.tensor([])
    
    # Initialiser le vecteur de caractéristiques avec des zéros pour tous les keypoints
    feature_vector = []
    
    # Itérer à travers les landmarks pour créer un vecteur de caractéristiques cohérent
    for landmark in pose_landmarks:
        if isinstance(landmark, dict):
            # Utiliser uniquement les landmarks avec une visibilité raisonnable
            if landmark.get('visibility', 0) > 0.1:
                feature_vector.extend([landmark['x'], landmark['y'], landmark['z']])
            else:
                # Ajouter des zéros pour les landmarks à faible visibilité pour maintenir une dimensionnalité cohérente
                feature_vector.extend([0.0, 0.0, 0.0])
    
    # Si nous n'avons pas de caractéristiques valides, retourner un tenseur vide
    if not feature_vector:
        return torch.tensor([])
    
    # Convertir en tenseur
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32)
    
    # Normaliser le vecteur de caractéristiques
    if len(feature_tensor) > 0:
        # Ajouter un petit epsilon pour éviter la division par zéro
        norm = torch.norm(feature_tensor, p=2) + 1e-8
        feature_tensor = feature_tensor / norm
    
    return feature_tensor


def compute_pose_similarity(feature1: torch.Tensor, feature2: torch.Tensor, device: torch.device) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs de caractéristiques de pose.
    """
    if len(feature1) == 0 or len(feature2) == 0:
        return 0.0
    
    # S'assurer que les tenseurs ont la même dimension
    if feature1.shape != feature2.shape:
        min_size = min(feature1.shape[0], feature2.shape[0])
        feature1 = feature1[:min_size]
        feature2 = feature2[:min_size]
    
    feature1 = feature1.to(device)
    feature2 = feature2.to(device)
    
    # Calculer la similarité cosinus
    similarity = torch.nn.functional.cosine_similarity(feature1.unsqueeze(0), feature2.unsqueeze(0)).item()
    
    # Retourner une valeur de similarité positive [0-1]
    return max(0.0, similarity)


class PoseBasedSceneReplacement:
    """
    Classe principale pour effectuer le remplacement de scènes basé sur la pose.
    """
    
    def __init__(self, target_video_path, reference_image_path, 
                 pose_model_path="yolo11x-pose.pt", verbose=True):
        """
        Initialise l'outil de remplacement de scènes basé sur la pose.
        """
        self.target_video_path = target_video_path
        self.reference_image_path = reference_image_path
        self.pose_model_path = pose_model_path
        self.verbose = verbose
        
        # Définir le périphérique
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Charger l'image de référence et extraire les caractéristiques
        self.reference_image = cv2.imread(reference_image_path)
        if self.reference_image is None:
            raise ValueError(f"Impossible de charger l'image de référence: {reference_image_path}")
            
        # Stocker la résolution de l'image de référence
        self.reference_width = self.reference_image.shape[1]
        self.reference_height = self.reference_image.shape[0]
        self.reference_resolution = (self.reference_width, self.reference_height)
        
        # Créer les répertoires de sortie
        self.temp_dir = "temp_files"
        self.output_dir = "output_videos"
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialiser les composants
        self.replacer = AudioPreservingReplacer(target_video_path, verbose=verbose)
        self.analyzer = self.replacer.analyzer
        
        # Initialiser les modèles
        self.log("Initialisation du modèle de pose...")
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Veuillez installer le package ultralytics: pip install ultralytics")
            
        self.pose_model = YOLO(pose_model_path)
        
        # Extraire la pose de l'image de référence
        self.log(f"Extraction de la pose depuis l'image de référence ({self.reference_width}x{self.reference_height})...")
        self.reference_poses = get_all_poses_yolo(self.reference_image, self.pose_model)
        
        if not self.reference_poses:
            raise ValueError("Aucune pose détectée dans l'image de référence")
            
        # Pour l'instant, utiliser la première pose détectée (la plus proéminente)
        self.reference_pose = self.reference_poses[0]
        self.reference_feature = create_pose_feature(self.reference_pose['landmarks'])
        
        if len(self.reference_feature) == 0:
            raise ValueError("Impossible d'extraire des caractéristiques de pose valides de l'image de référence")
            
        self.log(f"Pose de référence détectée à la position: {self.reference_pose['position']}")
        self.log(f"Dimension du vecteur de caractéristiques: {len(self.reference_feature)}")
    
    def log(self, message):
        """Affiche un message si le mode verbose est activé"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def extract_video_frames(self, video_path, start_time=0, duration=None, sample_rate=5):
        """
        Extrait des frames d'une vidéo à intervalles réguliers.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log(f"Erreur: Impossible d'ouvrir le fichier vidéo: {video_path}")
            return {}
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Obtenir la durée totale si non spécifiée
        if duration is None:
            video_info = SceneAnalyzer(video_path, verbose=False)
            duration = video_info.duration - start_time
        
        # Calculer les positions des frames à extraire
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        
        # Calculer l'intervalle d'extraction basé sur le taux d'échantillonnage
        interval = int(fps / sample_rate)
        if interval < 1:
            interval = 1
            
        result = {
            'frames': [],
            'timestamps': [],
            'frame_indices': []
        }
        
        # Régler à la position de départ
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extraire les frames à l'intervalle spécifié
        current_frame_idx = start_frame
        
        while current_frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Stocker la frame et le timestamp
            result['frames'].append(frame)
            result['timestamps'].append(current_frame_idx / fps)
            result['frame_indices'].append(current_frame_idx)
            
            # Passer à la position suivante
            current_frame_idx += interval
            if current_frame_idx < end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        
        cap.release()
        self.log(f"Extraction de {len(result['frames'])} frames depuis {video_path}")
        return result
    
    def compute_frame_pose_similarity(self, frames):
        """
        Calcule les scores de similarité de pose pour toutes les frames par rapport à l'image de référence.
        """
        self.log("Calcul des scores de similarité de pose avec ajustement de résolution...")
        
        scores = []
        poses = []
        
        # Traiter chaque frame
        for i, frame in enumerate(frames):
            # Extraire les poses de cette frame - redimensionner pour correspondre à la résolution de l'image de référence
            frame_poses = get_all_poses_yolo(frame, self.pose_model, target_resolution=self.reference_resolution)
            
            if not frame_poses:
                # Aucune pose détectée dans cette frame
                scores.append(0.0)
                poses.append(None)
                continue
            
            # Comparer chaque pose détectée avec la référence
            frame_scores = []
            for pose in frame_poses:
                feature = create_pose_feature(pose['landmarks'])
                # Ignorer si nous ne pouvons pas extraire des caractéristiques valides
                if len(feature) == 0:
                    continue
                    
                # Debug: imprimer les dimensions des caractéristiques pour comparaison
                if i == 0:
                    self.log(f"Dimension des caractéristiques de référence: {len(self.reference_feature)}, Dimension des caractéristiques de frame: {len(feature)}")
                
                similarity = compute_pose_similarity(self.reference_feature, feature, self.device)
                frame_scores.append((similarity, pose))
            
            # Utiliser le score de similarité le plus élevé pour cette frame
            if frame_scores:
                best_score, best_pose = max(frame_scores, key=lambda x: x[0])
                scores.append(best_score)
                poses.append(best_pose)
            else:
                scores.append(0.0)
                poses.append(None)
                
            # Afficher la progression
            if i % 10 == 0 and i > 0:
                self.log(f"Traitement de {i+1}/{len(frames)} frames")
        
        # Trouver la frame avec la plus grande similarité
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
            
            # S'assurer que la position relative est bien entre 0 et 1
            if relative_position > 1:
                self.log(f"ATTENTION: Position relative calculée > 1 ({relative_position:.4f}), normalisation à 1.0")
                relative_position = 1.0
            
            self.log(f"Image de référence trouvée à {reference_timestamp:.2f}s dans la vidéo cible " + 
                    f"(position relative dans la scène: {relative_position:.4f})")
        
        return relative_position, reference_timestamp
    
    def find_best_segment(self, source_video_path, target_scene_info, step_size=2.0, segment_overlap=0.7, scan_entire_video=True):
        """
        Trouver le meilleur segment dans la vidéo source correspondant à l'image de référence.
        """
        self.log(f"Recherche du meilleur segment dans {source_video_path} (SCAN COMPLET)")
    
        # Obtenir les informations sur la vidéo source
        source_info = SceneAnalyzer(source_video_path, verbose=False)
        source_duration = source_info.duration
        
        # Obtenir la durée cible
        target_duration = target_scene_info['duration']
        self.log(f"Durée de la scène cible: {target_duration:.2f}s")
        
        # Déterminer la position de l'image de référence dans la scène cible
        ref_position_rel, ref_timestamp = self.get_reference_frame_position(target_scene_info)
        
        # Vérifier si ref_position_rel est bien une valeur entre 0 et 1
        if ref_position_rel > 1:
            self.log(f"ATTENTION: Position relative anormale ({ref_position_rel}), conversion en valeur normalisée")
            # Convertir secondes en position relative (0-1)
            ref_position_rel = min(1.0, max(0.0, ref_position_rel / target_duration))
            self.log(f"Position relative corrigée: {ref_position_rel:.4f}")
        
        self.log(f"Position relative de référence dans la scène cible: {ref_position_rel:.4f}")
        
        # Liste pour stocker TOUS les résultats
        all_matches = []
        
        # Analyser la vidéo par morceaux
        chunk_size = step_size
        overlap = segment_overlap * chunk_size
        
        # Afficher la progression
        progress_step = 10  # Afficher tous les 10%
        progress_marker = progress_step
        
        self.log(f"Durée totale de la vidéo source: {source_duration:.2f}s")
        
        current_time = 0
        while current_time + chunk_size <= source_duration:
            # Afficher la progression
            progress = (current_time / source_duration) * 100
            if progress >= progress_marker:
                self.log(f"Progression: {progress_marker}% de la vidéo analysée")
                progress_marker += progress_step
                
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
            similarity_results = self.compute_frame_pose_similarity(chunk_frames['frames'])
            
            # Si on a trouvé une correspondance dans ce chunk
            if similarity_results['best_frame_index'] >= 0:
                # Convertir l'index du morceau en timestamp global
                frame_index = similarity_results['best_frame_index']
                frame_timestamp = current_time + chunk_frames['timestamps'][frame_index]
                frame_index_global = chunk_frames['frame_indices'][frame_index]
                frame_score = similarity_results['best_score']
                frame_pose = similarity_results['poses'][frame_index]
                
                # Ajouter TOUS les bons scores à notre liste de résultats
                # (pas seulement le meilleur)
                if frame_score > 0.2:  # Seuil minimal pour éviter le bruit
                    all_matches.append({
                        'timestamp': frame_timestamp,
                        'frame_index': frame_index_global,
                        'score': frame_score,
                        'pose': frame_pose,
                        'chunk_time': current_time
                    })
                    
                    self.log(f"Match trouvé à {frame_timestamp:.2f}s (position {current_time:.1f}s dans la vidéo) avec score {frame_score:.4f}")
            
            # Passer au morceau suivant avec chevauchement
            current_time += chunk_size - overlap
        
        self.log(f"Scan complet de la vidéo terminé. {len(all_matches)} correspondances trouvées.")
        
        # Aucun match trouvé
        if not all_matches:
            self.log("Aucune correspondance trouvée dans la vidéo source.")
            return None
        
        # Trier TOUS les matches par score et prendre le meilleur
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Afficher les 5 meilleures correspondances
        self.log("\nTop 5 des meilleures correspondances:")
        for i, match in enumerate(all_matches[:5]):
            self.log(f"  {i+1}. Position: {match['timestamp']:.2f}s, Score: {match['score']:.4f}, Chunk: {match['chunk_time']:.1f}s")
        
        # Une fois la meilleure correspondance trouvée:
        best_match = all_matches[0]
        best_frame_timestamp = best_match['timestamp']
        best_frame_index = best_match['frame_index']
        best_score = best_match['score']
        best_frame_poses = best_match['pose']
        
        self.log(f"\nMeilleure correspondance globale trouvée à {best_frame_timestamp:.2f}s (score: {best_score:.4f})")
        
        # Calculer les bornes du segment en respectant la position relative
        # Combien de temps il faut avant le point clé dans la scène cible
        time_before = ref_position_rel * target_duration
        # Combien de temps il faut après le point clé (en secondes)
        time_after = (1 - ref_position_rel) * target_duration
        
        self.log(f"Pour respecter la position relative {ref_position_rel:.4f}:")
        self.log(f"- Besoin de {time_before:.2f}s avant le point clé")
        self.log(f"- Besoin de {time_after:.2f}s après le point clé")
        self.log(f"- Durée totale: {time_before + time_after:.2f}s")
        
        # Calculer les bornes du segment en se basant sur ces temps
        segment_start = max(0, best_frame_timestamp - time_before)
        segment_end = min(source_duration, best_frame_timestamp + time_after)
        segment_duration = segment_end - segment_start
        
        # Afficher les limites initiales
        self.log(f"Bornes initiales du segment: {segment_start:.2f}s - {segment_end:.2f}s (durée: {segment_duration:.2f}s)")
        
        # Vérifier si les limites dépassent les bornes de la vidéo
        if segment_start == 0:
            self.log("Le segment commence au début de la vidéo source, ajustement nécessaire")
            segment_end = min(source_duration, segment_start + target_duration)
        
        if segment_end == source_duration:
            self.log("Le segment atteint la fin de la vidéo source, ajustement nécessaire")
            segment_start = max(0, segment_end - target_duration)
        
        # Recalculer la durée après ajustements
        adjusted_duration = segment_end - segment_start
        
        # Vérifier si la durée obtenue correspond à la cible
        if abs(adjusted_duration - target_duration) > 0.1:
            self.log(f"ATTENTION: La durée obtenue ({adjusted_duration:.2f}s) diffère de la cible ({target_duration:.2f}s)")
            self.log(f"Ajustement supplémentaire requis pour respecter la durée cible")
        
        # Calculer la position relative réelle du point clé dans le segment extrait
        actual_rel_pos = (best_frame_timestamp - segment_start) / adjusted_duration
        self.log(f"Position relative réelle du point clé dans le segment extrait: {actual_rel_pos:.4f}")
        
        # Créer le segment final
        best_segment = {
            'start_time': segment_start,
            'end_time': segment_end,
            'duration': segment_end - segment_start,
            'score': best_score,
            'reference_time': best_frame_timestamp,
            'reference_frame_index': best_frame_index,
            'target_ref_position': ref_position_rel,
            'target_ref_timestamp': ref_timestamp,
            'actual_ref_position': actual_rel_pos
        }
        
        # Extraire la meilleure frame pour référence
        cap = cv2.VideoCapture(source_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_index)
        ret, best_frame = cap.read()
        cap.release()
        
        if ret:
            best_frame_resized = cv2.resize(best_frame, self.reference_resolution)
            best_segment['reference_frame'] = best_frame_resized
            best_segment['pose'] = best_frame_poses
        
        # Ajuster la durée du segment si nécessaire
        self.log("\nAjustement fin de la durée du segment...")
        adjusted_segment = self.adjust_segment_duration(
            source_video_path, best_segment, target_duration
        )
        
        return adjusted_segment
    
    def adjust_segment_duration(self, source_video_path, segment, target_duration):
        """
        Ajuster la durée du segment pour correspondre exactement à la durée cible.
        """
        current_duration = segment['duration']
        
        # Position relative cible (celle déterminée dans la scène originale)
        target_rel_pos = segment.get('target_ref_position', 0.5)
        
        # Position actuelle du point clé
        ref_time = segment['reference_time']
        start_time = segment['start_time']
        end_time = segment['end_time']
        
        # Position relative actuelle
        current_rel_pos = (ref_time - start_time) / current_duration
        
        self.log(f"Ajustement de durée:")
        self.log(f"- Durée actuelle: {current_duration:.2f}s, Durée cible: {target_duration:.2f}s")
        self.log(f"- Position relative cible: {target_rel_pos:.4f}, Position actuelle: {current_rel_pos:.4f}")
        
        # Si les durées sont très proches, pas besoin d'ajuster
        if abs(current_duration - target_duration) < 0.1:
            self.log("La durée du segment correspond déjà à la cible, aucun ajustement nécessaire")
            return segment
        
        # Créer une copie du segment à ajuster
        adjusted = segment.copy()
        
        if current_duration < target_duration:
            # Le segment est trop court, besoin d'étendre
            self.log(f"Segment trop court ({current_duration:.2f}s), extension nécessaire (+{target_duration - current_duration:.2f}s)")
            
            # Calcul plus précis des temps avant/après le point clé
            time_before_ref = ref_time - start_time
            time_after_ref = end_time - ref_time
            
            # Calcul des temps cibles avant/après
            target_time_before = target_rel_pos * target_duration
            target_time_after = (1 - target_rel_pos) * target_duration
            
            # Calcul des extensions nécessaires
            extend_before = target_time_before - time_before_ref
            extend_after = target_time_after - time_after_ref
            
            self.log(f"- Extension avant: {extend_before:.2f}s, Extension après: {extend_after:.2f}s")
            
            # Nouvelles bornes
            source_info = SceneAnalyzer(source_video_path, verbose=False)
            new_start = max(0, start_time - extend_before)
            new_end = min(source_info.duration, end_time + extend_after)
            
            # Si on ne peut pas étendre suffisamment d'un côté, compenser de l'autre
            if new_start == 0 and new_end - new_start < target_duration:
                self.log("Impossible d'étendre suffisamment vers le début, compensation vers la fin")
                new_end = min(source_info.duration, new_start + target_duration)
            
            if new_end == source_info.duration and new_end - new_start < target_duration:
                self.log("Impossible d'étendre suffisamment vers la fin, compensation vers le début")
                new_start = max(0, new_end - target_duration)
            
            # Mise à jour du segment
            adjusted['start_time'] = new_start
            adjusted['end_time'] = new_end
            adjusted['duration'] = new_end - new_start
            
            # Calculer la nouvelle position relative
            new_rel_pos = (ref_time - new_start) / (new_end - new_start)
            adjusted['actual_ref_position'] = new_rel_pos
            
            self.log(f"Segment étendu: {new_start:.2f}s - {new_end:.2f}s (durée: {new_end - new_start:.2f}s)")
            self.log(f"Nouvelle position relative du point clé: {new_rel_pos:.4f}")
            
        elif current_duration > target_duration:
            # Le segment est trop long, besoin de couper
            self.log(f"Segment trop long ({current_duration:.2f}s), découpage nécessaire (-{current_duration - target_duration:.2f}s)")
            
            # Calcul précis des nouvelles bornes
            target_time_before = target_rel_pos * target_duration
            target_time_after = (1 - target_rel_pos) * target_duration
            
            # Nouvelles bornes centrées sur le point clé
            new_start = ref_time - target_time_before
            new_end = ref_time + target_time_after
            
            # Mise à jour du segment
            adjusted['start_time'] = new_start
            adjusted['end_time'] = new_end
            adjusted['duration'] = target_duration
            adjusted['actual_ref_position'] = target_rel_pos  # Devrait être exact après découpage
            
            self.log(f"Segment découpé: {new_start:.2f}s - {new_end:.2f}s (durée exacte: {target_duration:.2f}s)")
            self.log(f"Position relative préservée: {target_rel_pos:.4f}")
        
        # Vérification finale
        final_duration = adjusted['end_time'] - adjusted['start_time']
        self.log(f"Durée finale du segment: {final_duration:.2f}s (cible: {target_duration:.2f}s)")
        
        if abs(final_duration - target_duration) > 0.1:
            self.log("AVERTISSEMENT: La durée finale diffère encore légèrement de la cible")
        else:
            self.log("✓ Durée ajustée avec succès")
        
        return adjusted
    
    def create_visualizations(self, segment, target_scene, output_path=None):
        """
        Créer une visualisation de la meilleure frame correspondante et de l'image de référence.
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
        
        # Ajouter des informations textuelles plus détaillées
        plt.figtext(0.02, 0.02, f"Meilleure correspondance à: {segment['reference_time']:.2f}s dans la source", fontsize=9)
        plt.figtext(0.02, 0.05, f"Segment extrait: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s ({segment['duration']:.2f}s)", fontsize=9)
        plt.figtext(0.02, 0.08, f"Scène cible: {target_scene['start_time']:.2f}s - {target_scene['end_time']:.2f}s ({target_scene['duration']:.2f}s)", fontsize=9)
        
        # Afficher les positions relatives
        if 'target_ref_position' in segment:
            target_rel_pos = segment['target_ref_position']
            actual_rel_pos = segment.get('actual_ref_position', 
                                        (segment['reference_time'] - segment['start_time']) / segment['duration'])
            
            plt.figtext(0.02, 0.11, f"Position relative cible: {target_rel_pos:.4f} ({target_rel_pos * target_scene['duration']:.2f}s)", fontsize=9)
            plt.figtext(0.02, 0.14, f"Position relative obtenue: {actual_rel_pos:.4f} ({actual_rel_pos * segment['duration']:.2f}s)", fontsize=9)
            
            # Afficher l'écart
            rel_pos_diff = abs(target_rel_pos - actual_rel_pos)
            plt.figtext(0.02, 0.17, f"Écart de position: {rel_pos_diff:.4f} ({rel_pos_diff * segment['duration']:.2f}s)", 
                    fontsize=9, color='red' if rel_pos_diff > 0.05 else 'green')
        
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
 ########################################################################################
 # 
    def extract_segment_with_guaranteed_motion(self, source_video_path, segment, output_path=None):
        """
        Extrait un segment vidéo avec garantie de mouvement en utilisant une méthode spéciale
        pour résoudre les problèmes d'image fixe.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            segment: Dictionnaire avec les informations du segment
            output_path: Chemin pour sauvegarder le segment extrait
                
        Returns:
            Chemin vers le segment extrait
        """
        # Générer un nom de fichier unique
        timestamp = int(time.time())
        base_name = os.path.splitext(os.path.basename(source_video_path))[0]
        
        if output_path is None:
            output_path = os.path.join(self.temp_dir, f"{base_name}_segment_{timestamp}.mp4")
        
        # Créer un dossier temporaire pour cette extraction
        temp_extract_dir = os.path.join(self.temp_dir, f"extract_{timestamp}")
        os.makedirs(temp_extract_dir, exist_ok=True)
        frames_dir = os.path.join(temp_extract_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        start_time = segment['start_time']
        duration = segment['duration']
        
        self.log(f"EXTRACTION SPÉCIALE: Segment à {start_time:.2f}s (durée: {duration:.2f}s)")
        
        try:
            # Obtenir les informations vidéo
            source_info = SceneAnalyzer(source_video_path, verbose=False)
            
            # FIX: Utiliser un framerate standard arrondi pour éviter les problèmes de décimales
            original_fps = source_info.fps
            if abs(original_fps - 30) < 0.1:
                source_fps = 30
            elif abs(original_fps - 29.97) < 0.1:
                source_fps = 29.97
            elif abs(original_fps - 25) < 0.1:
                source_fps = 25
            elif abs(original_fps - 24) < 0.1:
                source_fps = 24
            else:
                # Arrondir à un nombre entier pour éviter les problèmes de décimales
                source_fps = round(original_fps)
            
            # MÉTHODE 1: EXTRACTION FRAME PAR FRAME
            self.log("Extraction des frames individuelles à haute fréquence...")
            extract_frames_cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.6f}",
                "-i", source_video_path,
                "-t", f"{duration:.6f}",
                "-vsync", "0",              # Mode précis
                "-q:v", "1",                # Qualité maximale
                "-vf", f"fps={source_fps}", # Extraire au framerate déterminé
                os.path.join(frames_dir, "frame_%08d.png")
            ]
            
            subprocess.run(extract_frames_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Vérifier combien de frames ont été extraites
            frames = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
            frame_count = len(frames)
            
            self.log(f"Frames extraites: {frame_count}")
            
            # Si moins de 10 frames extraites, générer des frames intermédiaires
            if frame_count < 10:
                self.log("ATTENTION: Moins de 10 frames extraites, création de frames intermédiaires...")
                
                if frame_count > 0:
                    # Utiliser la première frame comme base
                    base_frame_path = os.path.join(frames_dir, frames[0])
                    base_frame = cv2.imread(base_frame_path)
                    
                    if base_frame is not None:
                        h, w = base_frame.shape[:2]
                        
                        # Générer 60 frames intermédiaires avec variations
                        for i in range(60):
                            # Créer un effet de mouvement
                            scale = 1.0 + 0.001 * i
                            dx = int(w * 0.001 * i)
                            dy = int(h * 0.0005 * i)
                            
                            M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
                            M[0, 2] += dx
                            M[1, 2] += dy
                            
                            warped = cv2.warpAffine(base_frame, M, (w, h))
                            
                            brightness = 1.0 + 0.005 * i
                            adjusted = cv2.convertScaleAbs(warped, alpha=brightness, beta=0)
                            
                            output_path_frame = os.path.join(frames_dir, f"synth_{i:08d}.png")
                            cv2.imwrite(output_path_frame, adjusted)
                            
                        # Mettre à jour le compteur de frames
                        frames = [f for f in os.listdir(frames_dir) if f.endswith(".png")]
                        frame_count = len(frames)
                        self.log(f"Après génération: {frame_count} frames disponibles")
            
            # Renommer les frames dans l'ordre pour assurer la séquence correcte
            all_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
            for i, old_name in enumerate(all_frames):
                os.rename(
                    os.path.join(frames_dir, old_name),
                    os.path.join(frames_dir, f"final_{i:08d}.png")
                )
            
            # FIX: Construire la vidéo avec des paramètres plus standards et fiables
            rebuild_cmd = [
                "ffmpeg", "-y",
                "-framerate", f"{source_fps}",
                "-i", os.path.join(frames_dir, "final_%08d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "17",                # Très haute qualité
                "-preset", "medium",
                "-g", "15",                  # GOP size plus standard
                "-vsync", "cfr",             # Framerate constant
                "-r", f"{source_fps}",       # Forcer le framerate
                # FIX: Retrait du paramètre problématique movflags
                output_path
            ]
            
            self.log("Construction de la vidéo à partir des frames avec keyframes fréquentes...")
            
            try:
                subprocess.run(rebuild_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as err:
                self.log(f"Erreur avec la commande principale: {str(err)}")
                self.log("Tentative avec méthode alternative...")
                
                # MÉTHODE ALTERNATIVE: utiliser des paramètres plus simples
                alt_rebuild_cmd = [
                    "ffmpeg", "-y",
                    "-framerate", "30",      # Framerate fixe à 30
                    "-i", os.path.join(frames_dir, "final_%08d.png"),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "18",
                    "-preset", "veryfast",
                    output_path
                ]
                
                subprocess.run(alt_rebuild_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
            # Vérifier le résultat
            if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                # Vérifier les propriétés de la vidéo générée
                probe_cmd = [
                    "ffprobe", 
                    "-v", "error", 
                    "-select_streams", "v:0",
                    "-show_entries", "stream=nb_frames,duration,r_frame_rate",
                    "-of", "json",
                    output_path
                ]
                
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                if probe_result.stdout:
                    try:
                        probe_data = json.loads(probe_result.stdout)
                        frames = int(probe_data['streams'][0].get('nb_frames', '0'))
                        duration_sec = float(probe_data['streams'][0].get('duration', '0'))
                        framerate = probe_data['streams'][0].get('r_frame_rate', '0/1')
                        
                        self.log(f"Segment extrait avec succès: {frames} frames, {duration_sec:.2f}s, {framerate}")
                        return output_path
                    except:
                        pass
                    
                self.log("Segment extrait avec succès!")
                return output_path
            
            # Si la méthode principale échoue, utiliser une approche simple
            self.log("Échec de la méthode principale, utilisation de l'approche de secours...")
            
            # MÉTHODE 2: EXTRACTION DIRECTE AVEC KEYFRAMES FORCÉES
            direct_output = os.path.join(temp_extract_dir, "direct_extract.mp4")
            
            direct_cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.6f}",
                "-i", source_video_path,
                "-t", f"{duration:.6f}",
                "-c:v", "libx264",
                "-crf", "17",
                "-preset", "veryfast",
                "-tune", "film",
                "-g", "15",                 # GOP size plus standard
                "-keyint_min", "1",        # Forcer des keyframes
                "-force_key_frames", "expr:gte(t,0)",
                "-vsync", "cfr",           # Framerate constant
                "-r", f"{source_fps}",      # Forcer le framerate original
                "-an",                      # Pas d'audio
                direct_output
            ]
            
            self.log("Extraction directe avec keyframes forcées...")
            subprocess.run(direct_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(direct_output) and os.path.getsize(direct_output) > 10000:
                # Copier vers le chemin de sortie
                import shutil
                shutil.copy(direct_output, output_path)
                self.log("Extraction directe réussie!")
                return output_path
            
            self.log("Échec de toutes les méthodes d'extraction!")
            return None
            
        except Exception as e:
            self.log(f"ERREUR lors de l'extraction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

#######################################################################################

    def replace_multiple_scenes_with_best_pose_match(self, source_video_path, target_scene_numbers,
                                            output_path=None, scan_entire_video=False):
        """
        Remplacer plusieurs scènes dans la vidéo cible par le même segment source trouvé.
        Version corrigée pour résoudre le problème d'image fixe et préserver l'audio original.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            target_scene_numbers: Liste des numéros de scènes à remplacer
            output_path: Chemin pour la vidéo de sortie finale (optionnel)
            scan_entire_video: Si True, analyser toute la vidéo pour la recherche du meilleur segment
            
        Returns:
            Chemin vers la vidéo de sortie finale
        """
        timestamp = int(time.time())
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.log(f"=== DEBUT REMPLACEMENT MULTIPLE [{current_date}] par {CURRENT_USER} ===")
        
        if not target_scene_numbers:
            self.log("Erreur: Aucune scène spécifiée pour le remplacement")
            return None
            
        # Vérifier que toutes les scènes cibles existent et collecter leurs informations
        scene_infos = []
        for scene_num in target_scene_numbers:
            info = self.replacer.get_target_scene_info(scene_num)
            if not info:
                self.log(f"Erreur: Scène {scene_num} non trouvée dans la vidéo cible")
                return None
            scene_infos.append(info)
            
        self.log(f"Traitement de {len(target_scene_numbers)} scènes à remplacer: {target_scene_numbers}")
        
        # Afficher les détails de chaque scène
        for i, scene in enumerate(scene_infos):
            self.log(f"  Scène {target_scene_numbers[i]}: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s (durée: {scene['duration']:.2f}s)")
        
        # Créer un dossier temporaire pour cette session
        session_dir = os.path.join(self.temp_dir, f"multi_replace_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Extraction de l'audio original de la vidéo cible
        self.log("Extraction de l'audio original de la vidéo cible...")
        original_audio_path = os.path.join(session_dir, f"original_audio_{timestamp}.aac")
        original_audio_path = self.replacer.extract_audio(self.target_video_path, output_path=original_audio_path)
        if not original_audio_path:
            self.log("AVERTISSEMENT: Échec de l'extraction de l'audio. La vidéo finale n'aura pas d'audio.")
        else:
            self.log(f"Audio extrait avec succès: {original_audio_path}")
        
        # ÉTAPE 1: Trouver le meilleur segment pour la première scène
        primary_scene = scene_infos[0]
        primary_scene_num = target_scene_numbers[0]
        
        self.log(f"Recherche du segment source initial basé sur la scène {primary_scene_num}...")
        
        original_segment = self.find_best_segment(
            source_video_path, 
            primary_scene,
            scan_entire_video=scan_entire_video
        )
        
        if not original_segment:
            self.log("Impossible de trouver un segment approprié pour le remplacement")
            return None
            
        # Créer une visualisation pour la première scène
        vis_path = self.create_visualizations(
            original_segment, 
            primary_scene, 
            output_path=os.path.join(self.output_dir, f"vis_scene{primary_scene_num}_{timestamp}.png")
        )
        
        # ÉTAPE 2: Traiter chaque scène séparément
        # Traiter les scènes en ordre inverse pour éviter les problèmes de timecode
        scene_pairs = sorted(zip(target_scene_numbers, scene_infos), 
                        key=lambda pair: pair[1]['start_time'], 
                        reverse=True)
        
        # Vidéo de travail actuelle (commencer avec la vidéo originale)
        current_video = self.target_video_path
        
        for i, (scene_num, scene_info) in enumerate(scene_pairs):
            self.log(f"\n=== TRAITEMENT SCÈNE {i+1}/{len(scene_pairs)}: Scène {scene_num} ===")
            
            # Chemin de sortie pour cette étape
            if i == len(scene_pairs) - 1 and output_path:  # Dernière scène (première dans la vidéo)
                step_output = output_path
            else:
                # Chemin temporaire pour cette étape
                step_output = os.path.join(
                    session_dir, 
                    f"step{i+1}_scene{scene_num}_{timestamp}.mp4"
                )
            
            # ÉTAPE CRUCIALE: Extraire le segment spécifique pour cette scène avec garantie de mouvement
            # Ajuster le segment si nécessaire pour cette scène spécifique
            adjusted_segment = original_segment.copy()
            if abs(scene_info['duration'] - primary_scene['duration']) > 0.1:
                self.log(f"Ajustement du segment pour correspondre à la durée de la scène {scene_num} ({scene_info['duration']:.2f}s)")
                adjusted_segment = self.adjust_segment_duration(
                    source_video_path,
                    original_segment,
                    scene_info['duration']
                )
            
            # Extraire le segment avec la méthode spéciale pour garantir le mouvement
            segment_path = os.path.join(session_dir, f"segment_scene{scene_num}_{timestamp}.mp4")
            segment_path = self.extract_segment_with_guaranteed_motion(
                source_video_path,
                adjusted_segment,
                output_path=segment_path
            )
            
            if not segment_path or not os.path.exists(segment_path):
                self.log(f"Échec de l'extraction du segment pour la scène {scene_num}")
                if i == 0:  # Si c'est la première scène à traiter, abandonner
                    return None
                else:
                    # Sinon continuer avec le résultat précédent
                    return current_video
            
            # ÉTAPE 3: Remplacer la scène actuelle
            # Créer des fichiers temporaires pour cette opération
            before_path = os.path.join(session_dir, f"before_scene{scene_num}_{timestamp}.mp4")
            after_path = os.path.join(session_dir, f"after_scene{scene_num}_{timestamp}.mp4")
            concat_list = os.path.join(session_dir, f"concat_list_scene{scene_num}_{timestamp}.txt")
            video_only_path = os.path.join(session_dir, f"video_only_scene{scene_num}_{timestamp}.mp4")
            
            try:
                # 3.1 Extraire la partie avant la scène
                if scene_info['start_time'] > 0:
                    before_cmd = [
                        "ffmpeg", "-y",
                        "-i", current_video,
                        "-ss", "0",
                        "-to", f"{scene_info['start_time']:.6f}",
                        "-c:v", "libx264",  # Réencodage pour meilleures transitions
                        "-crf", "18",
                        "-preset", "medium",
                        "-pix_fmt", "yuv420p",
                        "-an",  # Pas d'audio
                        before_path
                    ]
                    
                    self.log(f"Extraction du segment avant la scène {scene_num}...")
                    subprocess.run(before_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # 3.2 Extraire la partie après la scène
                target_analyzer = SceneAnalyzer(current_video, verbose=False)
                if scene_info['end_time'] < target_analyzer.duration:
                    after_cmd = [
                        "ffmpeg", "-y",
                        "-i", current_video,
                        "-ss", f"{scene_info['end_time']:.6f}",
                        "-c:v", "libx264",  # Réencodage pour meilleures transitions
                        "-crf", "18",
                        "-preset", "medium",
                        "-pix_fmt", "yuv420p",
                        "-an",  # Pas d'audio
                        after_path
                    ]
                    
                    self.log(f"Extraction du segment après la scène {scene_num}...")
                    subprocess.run(after_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # 3.3 Pre-process segments for concatenation (CRITICAL FIX)
                before_processed = None
                if os.path.exists(before_path) and os.path.getsize(before_path) > 10000:
                    before_processed = os.path.join(session_dir, f"before_processed_{timestamp}.mp4")
                    preprocess_cmd = [
                        "ffmpeg", "-y",
                        "-i", before_path,
                        "-c:v", "libx264",
                        "-crf", "18",
                        "-preset", "medium",
                        "-vsync", "cfr",     # Constant framerate
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        "-an",               # No audio
                        before_processed
                    ]
                    subprocess.run(preprocess_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Process segment_path to ensure it has proper motion
                segment_processed = os.path.join(session_dir, f"segment_processed_{timestamp}.mp4")
                preprocess_cmd = [
                    "ffmpeg", "-y",
                    "-i", segment_path,
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-preset", "fast",      # Use fast preset for replacement segment
                    "-vsync", "cfr",        # Constant framerate 
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an",                  # No audio
                    segment_processed
                ]
                subprocess.run(preprocess_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                after_processed = None
                if os.path.exists(after_path) and os.path.getsize(after_path) > 10000:
                    after_processed = os.path.join(session_dir, f"after_processed_{timestamp}.mp4")
                    preprocess_cmd = [
                        "ffmpeg", "-y",
                        "-i", after_path,
                        "-c:v", "libx264", 
                        "-crf", "18",
                        "-preset", "medium",
                        "-vsync", "cfr",     # Constant framerate
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        "-an",               # No audio
                        after_processed
                    ]
                    subprocess.run(preprocess_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # 3.4 Prepare concat list with processed segments
                segment_list = []
                if before_processed:
                    segment_list.append(before_processed)
                segment_list.append(segment_processed)  # Use processed segment
                if after_processed:
                    segment_list.append(after_processed)

                # Write concat list
                with open(concat_list, 'w') as f:
                    for path in segment_list:
                        f.write(f"file '{os.path.abspath(path)}'\n")

                # 3.5 Concatenate with special flags to ensure motion preservation
                concat_cmd = ["ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list,
                "-c:v", "libx264",      # Réencodage pour garantir la compatibilité
                "-crf", "18",
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
                "-g", "15",             # GOP size standard
                "-vsync", "1",          # Synchronisation adaptée
                "-an",                  # Pas d'audio
                video_only_path
                ]

                self.log(f"Concaténation des segments vidéo pour la scène {scene_num}...")
                subprocess.run(concat_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
                # 3.6 Combiner la vidéo et l'audio original
                if original_audio_path and os.path.exists(original_audio_path) and os.path.getsize(original_audio_path) > 1000:
                    final_cmd = [
                        "ffmpeg", "-y",
                        "-i", video_only_path,
                        "-i", original_audio_path,
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-strict", "experimental",
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        "-shortest",
                        step_output
                    ]
                    
                    self.log("Combinaison de la vidéo avec l'audio original...")
                    result = subprocess.run(final_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    # Vérifier si l'audio a bien été intégré
                    audio_check_cmd = [
                        "ffprobe", "-v", "quiet", "-select_streams", "a", 
                        "-show_entries", "stream=codec_type", "-of", "json", 
                        step_output
                    ]
                    
                    try:
                        audio_check = subprocess.run(audio_check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        audio_data = json.loads(audio_check.stdout)
                        if 'streams' in audio_data and len(audio_data['streams']) > 0:
                            self.log("✓ Audio correctement intégré dans la vidéo finale")
                        else:
                            self.log("⚠️ AVERTISSEMENT: La vidéo finale ne semble pas contenir d'audio")
                    except:
                        self.log("⚠️ Impossible de vérifier l'intégration de l'audio")
                else:
                    # Si l'extraction audio a échoué, utiliser la vidéo sans audio
                    import shutil
                    shutil.copy(video_only_path, step_output)
                    self.log("⚠️ ATTENTION: Utilisation de la vidéo sans audio (extraction audio échouée)")
                
                # 3.7 Vérifier le résultat
                if not os.path.exists(step_output) or os.path.getsize(step_output) < 10000:
                    raise Exception("Échec de la création de la vidéo finale")
                
                self.log(f"Remplacement de la scène {scene_num} terminé avec succès!")
                
                # Nettoyer les fichiers intermédiaires
                if i > 0 and current_video != self.target_video_path and os.path.exists(current_video):
                    try:
                        os.remove(current_video)
                        self.log("Fichier intermédiaire supprimé")
                    except:
                        self.log("Impossible de supprimer le fichier intermédiaire")
                
                # Mettre à jour la vidéo actuelle pour l'étape suivante
                current_video = step_output
                
            except Exception as e:
                self.log(f"ERREUR lors du remplacement de la scène {scene_num}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Si c'est la première scène, abandonner
                if i == 0:
                    return None
                
                # Sinon continuer avec le résultat précédent
                return current_video
        
        # ÉTAPE FINALE: Vérifier la durée de la vidéo
        if os.path.exists(current_video):
            try:
                final_info = SceneAnalyzer(current_video, verbose=False)
                original_info = SceneAnalyzer(self.target_video_path, verbose=False)
                
                self.log(f"\nVérification finale:")
                self.log(f"Durée originale: {original_info.duration:.2f}s")
                self.log(f"Durée finale: {final_info.duration:.2f}s")
                self.log(f"Différence: {abs(final_info.duration - original_info.duration):.2f}s")
                
                # Vérifier si l'audio est présent dans la vidéo finale
                audio_check_cmd = [
                    "ffprobe", "-v", "quiet", "-select_streams", "a", 
                    "-show_entries", "stream=codec_type", "-of", "json", 
                    current_video
                ]
                
                try:
                    audio_check = subprocess.run(audio_check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    audio_data = json.loads(audio_check.stdout)
                    if 'streams' in audio_data and len(audio_data['streams']) > 0:
                        self.log("✓ Confirmation finale: Audio présent dans la vidéo finale")
                    else:
                        self.log("⚠️ Vérification finale: La vidéo finale ne contient PAS d'audio")
                except:
                    self.log("⚠️ Impossible de vérifier la présence d'audio dans la vidéo finale")
                    
            except:
                self.log("Impossible d'analyser la durée de la vidéo finale")
        
        # ÉTAPE SUCCÈS: Journalisation
        log_path = os.path.join(self.output_dir, "multi_scene_log.txt")
        with open(log_path, 'a') as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"Date: {current_date}\n")
            f.write(f"Utilisateur: {CURRENT_USER}\n")
            f.write(f"Vidéo cible: {self.target_video_path}\n")
            f.write(f"Vidéo source: {source_video_path}\n")
            f.write(f"Image de référence: {self.reference_image_path}\n")
            f.write(f"Scènes remplacées: {', '.join(map(str, target_scene_numbers))}\n")
            f.write(f"Segment source original: {original_segment['start_time']:.2f}s - {original_segment['end_time']:.2f}s\n")
            f.write(f"Score de similarité: {original_segment['score']:.4f}\n")
            f.write(f"Visualisation: {vis_path}\n")
            f.write(f"Vidéo finale: {current_video}\n")
            f.write(f"Audio préservé: {'Oui' if original_audio_path else 'Non'}\n")
        
        self.log(f"\nRemplacement de {len(target_scene_numbers)} scènes terminé avec succès!")
        self.log(f"Vidéo finale: {current_video}")
        
        return current_video
    def replace_scene_with_best_pose_match(self, source_video_path, target_scene_number, output_path=None):
        """
        Remplacer une scène dans la vidéo cible par le meilleur segment correspondant de la vidéo source.
        Cette méthode utilise maintenant la même fonction que pour les scènes multiples pour plus de cohérence.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            target_scene_number: Numéro de la scène à remplacer dans la vidéo cible
            output_path: Chemin pour la vidéo de sortie (optionnel)
            
        Returns:
            Chemin vers la vidéo de sortie
        """
        return self.replace_multiple_scenes_with_best_pose_match(
            source_video_path,
            [target_scene_number],  # Liste avec un seul élément
            output_path=output_path
        )


def main():
    """Point d'entrée principal avec interface CLI."""
    parser = argparse.ArgumentParser(description="Outil de remplacement de scènes vidéo basé sur la pose - Version Multi-Scènes")
    parser.add_argument("--target-video", required=True, help="Chemin vers la vidéo cible (à éditer)")
    parser.add_argument("--source-video", required=True, help="Chemin vers la vidéo source (pour extraire des segments)")
    parser.add_argument("--reference-image", required=True, help="Chemin vers l'image de référence (pose à correspondre)")
    parser.add_argument("--target-scenes", type=int, nargs='+', required=True, 
                        help="Numéro(s) de la/des scène(s) à remplacer dans la vidéo cible. Pour remplacer plusieurs scènes avec le même rush, spécifier plusieurs numéros.")
    parser.add_argument("--pose-model", default="yolo11x-pose.pt", help="Chemin vers le fichier du modèle YOLOv8 pose")
    parser.add_argument("--output", help="Chemin pour la vidéo de sortie")
    parser.add_argument("--scan-entire", action="store_true", help="Scanner toute la vidéo source pour trouver la meilleure correspondance")
    parser.add_argument("--quiet", action="store_true", help="Supprimer les messages de progression")
    
    args = parser.parse_args()
    
    try:
        # Mettre à jour la date et l'utilisateur actuels
        global CURRENT_TIMESTAMP, CURRENT_USER
        CURRENT_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Créer l'outil de remplacement basé sur la pose
        replacer = PoseBasedSceneReplacement(
            args.target_video,
            args.reference_image,
            pose_model_path=args.pose_model,
            verbose=not args.quiet
        )
        
        # Traiter le remplacement
        start_time = time.time()
        
        # Utiliser la fonction multi-scènes pour tous les cas (même une seule scène)
        output_path = replacer.replace_multiple_scenes_with_best_pose_match(
            args.source_video,
            args.target_scenes,
            output_path=args.output,
            scan_entire_video=args.scan_entire
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
    # Mettre à jour le timestamp courant
    CURRENT_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    main()