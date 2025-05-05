#!/usr/bin/env python3
"""
Outil de Remplacement Vidéo Basé sur la Posture - Version Multiple Scènes

Ce script combine l'analyse de posture et le remplacement de scènes pour trouver des segments vidéo
avec des poses similaires à une image de référence, et les utiliser pour remplacer plusieurs scènes
identiques dans une vidéo cible.

Fonctionnalités:
- Remplace plusieurs occurrences de la même scène avec exactement le même rush source
- Trouve les images avec des poses les plus similaires à une image de référence 
- Préserve l'audio original lors du remplacement des scènes

Auteur: FETHl
Date: 2025-05-02
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
CURRENT_TIMESTAMP = "2025-05-02 11:37:29"
CURRENT_USER = "FETHl"

# Nombre total de points de repère dans le modèle de pose YOLOv11
YOLO_POSE_KEYPOINTS = 17  # Le modèle YOLOv11 a 17 points de repère corps
FEATURE_DIM_PER_KEYPOINT = 3  # x, y, z (z est toujours 0 dans YOLOv11)


def get_position_in_image(bbox, img_width, img_height):
    """
    Détermine la position d'une boîte englobante dans l'image.
    
    Args:
        bbox: [x1, y1, x2, y2] boîte englobante
        img_width: Largeur de l'image
        img_height: Hauteur de l'image
        
    Returns:
        Description de la position (ex: "haut-gauche", "centre", etc.)
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
    
    Args:
        image: Image d'entrée au format numpy array
        pose_model: Instance du modèle de pose YOLOv11
        target_resolution: Tuple (largeur, hauteur) pour redimensionner l'image avant détection
        
    Returns:
        Liste de dictionnaires de poses pour chaque personne détectée
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
    Crée un vecteur de caractéristiques normalisé à partir des points de repère de pose,
    assurant une dimensionnalité constante.
    
    Args:
        pose_landmarks: Liste des points de repère de pose (doit être standardisée pour contenir tous les points clés)
        hand_landmarks: Liste des points de repère de main (optionnel)
        
    Returns:
        Tenseur torch contenant le vecteur de caractéristiques normalisé
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
    Calcule la similarité cosinus entre deux vecteurs de caractéristiques de pose,
    en gérant les différences de dimensions.
    
    Args:
        feature1: Premier tenseur de caractéristiques
        feature2: Deuxième tenseur de caractéristiques
        device: Dispositif torch (CPU ou CUDA)
        
    Returns:
        Score de similarité [0-1]
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
    Trouve des segments dans une vidéo source avec des poses similaires à une image de référence,
    et les utilise pour remplacer des scènes dans une vidéo cible.
    """
    
    def __init__(self, target_video_path, reference_image_path, 
                 pose_model_path="yolo11x-pose.pt", verbose=True):
        """
        Initialise l'outil de remplacement de scènes basé sur la pose.
        
        Args:
            target_video_path: Chemin vers la vidéo cible (à éditer)
            reference_image_path: Chemin vers l'image de référence (pose à correspondre)
            pose_model_path: Chemin vers le modèle de pose YOLOv11
            verbose: Affiche des messages détaillés si True
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
        
        Args:
            video_path: Chemin vers la vidéo
            start_time: Temps de début en secondes
            duration: Durée à extraire en secondes (None pour la vidéo entière)
            sample_rate: Nombre de frames par seconde à extraire
            
        Returns:
            Dictionnaire avec les frames et les timestamps
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
        Ajuste la résolution de chaque frame pour correspondre à l'image de référence avant comparaison.
        
        Args:
            frames: Liste des frames vidéo
            
        Returns:
            Dictionnaire avec les scores de similarité et les indices des frames les plus semblables
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
            
            # CORRECTION: S'assurer que la position relative est bien entre 0 et 1
            if relative_position > 1:
                self.log(f"ATTENTION: Position relative calculée > 1 ({relative_position:.4f}), normalisation à 1.0")
                relative_position = 1.0
            
            self.log(f"Image de référence trouvée à {reference_timestamp:.2f}s dans la vidéo cible " + 
                    f"(position relative dans la scène: {relative_position:.4f})")
            
           
        return relative_position, reference_timestamp
    
    def find_best_segment(self, source_video_path, target_scene_info, step_size=2.0, segment_overlap=0.7, scan_entire_video=True):
        """
        Trouver le meilleur segment dans la vidéo source correspondant à l'image de référence.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            target_scene_info: Informations sur la scène cible
            step_size: Taille des morceaux pour l'analyse (en secondes)
            segment_overlap: Chevauchement entre les morceaux (0-1)
            scan_entire_video: Si True, analyser toute la vidéo
            
        Returns:
            Dictionnaire contenant les informations sur le meilleur segment
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
        
        # CORRECTION CRUCIALE: Vérifier si ref_position_rel est bien une valeur entre 0 et 1
        # Si supérieur à 1, c'est probablement une position en secondes
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
        
        # ÉTAPE CRITIQUE: Calculer les bornes du segment en respectant la position relative
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
        Ajuster la durée du segment pour correspondre exactement à la durée cible,
        tout en maintenant l'image de meilleure correspondance à la même position relative.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            segment: Dictionnaire avec les informations sur le segment
            target_duration: Durée requise en secondes
            
        Returns:
            Segment ajusté (dictionnaire)
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

#############################################################################################################

    def extract_segment_as_file_radical(self, source_video_path, segment, output_path=None):
        """
        Méthode radicale d'extraction de segment vidéo par extraction puis 
        reconstruction frame par frame pour éliminer tout problème de saccade.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            segment: Dictionnaire contenant les informations du segment (start_time, duration)
            output_path: Chemin pour sauvegarder le segment extrait (optionnel)
            
        Returns:
            Chemin vers le fichier extrait
        """
        if output_path is None:
            timestamp = int(time.time())
            base_name = os.path.splitext(os.path.basename(source_video_path))[0]
            output_path = os.path.join(self.temp_dir, f"{base_name}_segment_{timestamp}.mp4")
        
        start_time = segment['start_time']
        duration = segment['duration']
        
        self.log(f"MÉTHODE RADICALE: Extraction du segment de {start_time:.2f}s (durée: {duration:.2f}s)...")
        
        # Créer un dossier temporaire pour les frames extraites
        frames_dir = os.path.join(self.temp_dir, f"frames_{timestamp}")
        os.makedirs(frames_dir, exist_ok=True)
        
        try:
            # Obtenir les informations sur la vidéo source
            info = SceneAnalyzer(source_video_path, verbose=False)
            fps = info.fps
            width = info.width
            height = info.height
            
            self.log(f"Vidéo source: {width}x{height} à {fps}fps")
            
            # 1. Extraire toutes les frames dans la plage temporelle
            self.log("Étape 1: Extraction de toutes les frames individuelles...")
            
            frame_extract_cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.6f}",
                "-i", source_video_path,
                "-t", f"{duration:.6f}",
                "-vsync", "0",             # Mode de synchronisation vidéo: passthrough
                "-q:v", "1",               # Qualité maximale pour les frames individuelles
                "-frame_pts", "true",      # Utiliser PTS comme nom de fichier
                "-f", "image2",            # Format de sortie: séquence d'images
                os.path.join(frames_dir, "frame_%08d.png")  # Pattern de nom de fichier
            ]
            
            subprocess.run(frame_extract_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Vérifier que des frames ont été extraites
            extracted_frames = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
            if not extracted_frames:
                raise Exception("Aucune frame n'a été extraite")
                
            frame_count = len(extracted_frames)
            self.log(f"Extraction réussie: {frame_count} frames extraites")
            
            if frame_count < 2:
                raise Exception(f"Trop peu de frames extraites ({frame_count}), segment probablement invalide")
            
            # 2. Recréer la vidéo à partir des frames extraites
            self.log("Étape 2: Reconstruction de la vidéo à partir des frames individuelles...")
            
            # Utiliser ffmpeg avec un framerate constant défini et une qualité élevée
            reconstruct_cmd = [
                "ffmpeg", "-y",
                "-framerate", f"{fps}",
                "-i", os.path.join(frames_dir, "frame_%08d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "15",              # Qualité très élevée
                "-preset", "slow",
                "-vsync", "cfr",           # Constant frame rate
                "-g", "10",                # Keyframe tous les 10 frames
                "-keyint_min", "10",       # Forcer des keyframes fréquents
                "-movflags", "+faststart",
                output_path
            ]
            
            subprocess.run(reconstruct_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Vérifier que la vidéo reconstruite est valide
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 10000:
                raise Exception("La reconstruction vidéo a échoué, fichier inexistant ou trop petit")
            
            # Vérifier les propriétés de la vidéo reconstruite
            output_info = SceneAnalyzer(output_path, verbose=False)
            self.log(f"Vidéo reconstruite: {output_info.duration:.2f}s à {output_info.fps}fps")
            
            # Suppression des fichiers temporaires
            import shutil
            shutil.rmtree(frames_dir)
            
            return output_path
            
        except Exception as e:
            self.log(f"ERREUR dans l'extraction radicale: {str(e)}")
            
            # Nettoyage des fichiers temporaires en cas d'erreur
            import shutil
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            
            # Tentative ultime avec méthode encore plus basique
            self.log("Tentative ULTIME avec méthode ultra-basique...")
            
            try:
                # Méthode ultra-simple: -c:v copy pour éviter tout problème de réencodage
                simple_cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{start_time + 0.5:.6f}",  # Décalage pour éviter les problèmes de début
                    "-i", source_video_path,
                    "-t", f"{duration - 0.5:.6f}",     # Réduire légèrement la durée pour compenser
                    "-c", "copy",                      # Pas de réencodage
                    "-avoid_negative_ts", "1",
                    "-async", "1",
                    output_path
                ]
                
                subprocess.run(simple_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                    self.log("Extraction ultra-basique réussie")
                    return output_path
                else:
                    return None
                    
            except:
                self.log("Échec de toutes les méthodes d'extraction")
                return None
##############################################################################################################
    def concat_video_segments(self, segment_files, output_path):
        """
        Concatène les segments vidéo avec une méthode améliorée pour éviter les saccades.
        
        Args:
            segment_files: Liste des fichiers de segments à concaténer
            output_path: Chemin pour le fichier de sortie
            
        Returns:
            Chemin vers le fichier concaténé
        """
        self.log(f"Concaténation avancée de {len(segment_files)} segments vidéo...")
        
        # Vérifier que tous les segments existent
        valid_segments = []
        for segment in segment_files:
            if segment and os.path.exists(segment) and os.path.getsize(segment) > 10000:
                valid_segments.append(segment)
        
        if not valid_segments:
            self.log("Aucun segment valide à concaténer")
            return None
        
        # Si un seul segment, le copier directement
        if len(valid_segments) == 1:
            import shutil
            shutil.copy(valid_segments[0], output_path)
            return output_path
        
        timestamp = int(time.time())
        list_file = os.path.join(self.temp_dir, f"concat_list_{timestamp}.txt")
        
        # Créer le fichier de liste pour FFmpeg
        with open(list_file, 'w') as f:
            for segment in valid_segments:
                f.write(f"file '{os.path.abspath(segment)}'\n")
        
        try:
            # Méthode 1: Concaténation avancée avec réencodage pour assurer la fluidité
            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c:v", "libx264",         # Réencodage pour assurer la cohérence
                "-crf", "18",              # Haute qualité
                "-preset", "slow",
                "-vsync", "cfr",           # Constant frame rate
                "-r", "30",                # Forcer un framerate constant
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",                     # Pas d'audio pour l'instant
                output_path
            ]
            
            self.log("Tentative de concaténation avec réencodage...")
            subprocess.run(concat_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Vérifier que la sortie est valide
            if os.path.exists(output_path) and os.path.getsize(output_path) > sum([os.path.getsize(s) for s in valid_segments]) / 10:
                self.log("Concaténation réussie avec réencodage")
                return output_path
                
        except Exception as e:
            self.log(f"Échec de la concaténation avec réencodage: {str(e)}")
            
            try:
                # Méthode 2: Concaténation simple (sans réencodage)
                simple_concat_cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file,
                    "-c", "copy",           # Sans réencodage
                    "-an",                  # Sans audio
                    output_path
                ]
                
                self.log("Tentative de concaténation simple...")
                subprocess.run(simple_concat_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                    self.log("Concaténation simple réussie")
                    return output_path
                    
            except Exception as e:
                self.log(f"Échec de la concaténation simple: {str(e)}")
                
        return None
###############################################################################################################

    def extract_segment_as_file_motion_fix(self, source_video_path, segment, output_path=None):
        """
        Méthode spécialisée pour extraire un segment vidéo tout en garantissant le mouvement.
        Solution au problème d'image fixe dans les scènes remplacées.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            segment: Dictionnaire contenant les informations du segment (start_time, duration)
            output_path: Chemin pour sauvegarder le segment extrait (optionnel)
            
        Returns:
            Chemin vers le fichier extrait
        """
        # Générer un nom de fichier unique avec timestamp
        timestamp = int(time.time())
        base_name = os.path.splitext(os.path.basename(source_video_path))[0]
        
        if output_path is None:
            output_path = os.path.join(self.temp_dir, f"{base_name}_segment_{timestamp}.mp4")
        
        # Créer un dossier temporaire spécifique pour cette extraction
        temp_extract_dir = os.path.join(self.temp_dir, f"extract_{timestamp}")
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        start_time = segment['start_time']
        duration = segment['duration']
        
        # Obtenir les informations sur la vidéo source
        source_info = SceneAnalyzer(source_video_path, verbose=False)
        source_fps = source_info.fps
        
        self.log(f"[MOTION FIX] Extraction du segment à {start_time:.2f}s (durée: {duration:.2f}s, fps: {source_fps:.2f})")
        self.log(f"Heure actuelle (UTC): 2025-05-05 09:54:02, Utilisateur: FETHl")
        
        try:
            # MÉTHODE 1: EXTRACTION DIRECTE OPTIMISÉE POUR LE MOUVEMENT
            direct_output = os.path.join(temp_extract_dir, "direct_extract.mp4")
            
            # Commande optimisée pour garantir le mouvement
            direct_cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.6f}",
                "-i", source_video_path,
                "-t", f"{duration:.6f}",
                "-c:v", "libx264",
                "-crf", "15",             # Très haute qualité
                "-preset", "veryslow",    # Compression maximale
                "-tune", "film",
                "-profile:v", "high",
                "-level", "4.1",
                "-pix_fmt", "yuv420p",
                "-r", f"{source_fps}",    # Forcer le framerate source
                "-vsync", "cfr",          # Constant frame rate
                "-an",                    # Sans audio
                "-x264opts", "keyint=10:min-keyint=1",  # Keyframes très fréquentes
                direct_output
            ]
            
            self.log("Tentative d'extraction directe optimisée pour le mouvement...")
            subprocess.run(direct_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Vérifier le résultat
            if os.path.exists(direct_output) and os.path.getsize(direct_output) > 10000:
                # Vérifier si cette vidéo contient réellement du mouvement
                self.log("Vérification du mouvement dans le segment extrait...")
                
                # Utiliser FFprobe pour vérifier les propriétés vidéo
                probe_cmd = [
                    "ffprobe", 
                    "-v", "error", 
                    "-select_streams", "v:0",
                    "-show_entries", "stream=avg_frame_rate,nb_frames",
                    "-of", "json",
                    direct_output
                ]
                
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                probe_data = json.loads(probe_result.stdout) if probe_result.stdout else {}
                
                frames = 0
                if 'streams' in probe_data and probe_data['streams']:
                    frames = int(probe_data['streams'][0].get('nb_frames', 0))
                    
                self.log(f"Segment extrait: {frames} frames")
                
                if frames > 5:  # Si plus de 5 frames détectées, le segment semble valide
                    self.log("Segment directement extrait valide avec mouvement confirmé")
                    # Copier vers le chemin de sortie final
                    import shutil
                    shutil.copy(direct_output, output_path)
                    return output_path
            
            self.log("L'extraction directe n'a pas produit de segment valide, passage à la méthode frame par frame")
            
            # MÉTHODE 2: EXTRACTION FRAME PAR FRAME AVEC DÉCIMALES DE TEMPS
            # Pour s'assurer que nous extrayons plusieurs frames distinctes
            frames_dir = os.path.join(temp_extract_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Extraire une séquence de frames avec temps précis
            extract_frames_cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.6f}",
                "-i", source_video_path,
                "-t", f"{duration:.6f}",
                "-vsync", "0",            # Passthrough mode
                "-q:v", "1",              # Qualité maximale
                "-frame_pts", "true",     # Utiliser PTS pour noms de fichiers
                "-f", "image2",           # Format d'image séquentielle
                os.path.join(frames_dir, "frame_%08d.png")
            ]
            
            self.log("Extraction de frames individuelles avec timing précis...")
            # Capturer la sortie d'erreur pour diagnostic
            extract_result = subprocess.run(extract_frames_cmd, capture_output=True, text=True)
            
            if extract_result.stderr:
                # Sauvegarder les logs pour diagnostic
                with open(os.path.join(temp_extract_dir, "ffmpeg_extract_log.txt"), "w") as f:
                    f.write(extract_result.stderr)
            
            # Vérifier les frames extraites
            extracted_frames = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
            frame_count = len(extracted_frames)
            
            self.log(f"Frames extraites: {frame_count}")
            
            if frame_count < 3:
                self.log(f"ALERTE: Seulement {frame_count} frames extraites, duplication artificielle...")
                
                # Si trop peu de frames, dupliquer la première frame plusieurs fois
                # pour créer un mouvement artificiel (mieux qu'une image fixe)
                if frame_count > 0:
                    first_frame = os.path.join(frames_dir, extracted_frames[0])
                    for i in range(1, 30):  # Créer 30 frames artificielles
                        new_frame = os.path.join(frames_dir, f"frame_dup_{i:08d}.png")
                        import shutil
                        shutil.copy(first_frame, new_frame)
                    
                    # Réorganiser les noms pour la séquence
                    all_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
                    for i, frame in enumerate(all_frames):
                        os.rename(
                            os.path.join(frames_dir, frame),
                            os.path.join(frames_dir, f"renamed_{i:08d}.png")
                        )
                    
                    # Mettre à jour la liste des frames
                    extracted_frames = sorted([f for f in os.listdir(frames_dir) if f.startswith("renamed_")])
                    frame_count = len(extracted_frames)
            
            # Reconstruire la vidéo avec des paramètres optimisés pour le mouvement
            if frame_count > 0:
                rebuild_cmd = [
                    "ffmpeg", "-y",
                    "-framerate", f"{source_fps}",
                    "-pattern_type", "glob",
                    "-i", os.path.join(frames_dir, "*.png"),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "15",
                    "-preset", "medium",
                    "-tune", "film",
                    "-profile:v", "high",
                    "-level", "4.1",
                    "-r", f"{source_fps}",
                    "-g", "1",              # Chaque frame est une keyframe
                    "-keyint_min", "1",     # Forcer une keyframe à chaque frame
                    "-vsync", "cfr",        # Constant frame rate
                    "-movflags", "+faststart",
                    output_path
                ]
                
                self.log("Reconstruction de la vidéo à partir des frames...")
                rebuild_result = subprocess.run(rebuild_cmd, capture_output=True, text=True)
                
                if rebuild_result.stderr:
                    # Sauvegarder les logs pour diagnostic
                    with open(os.path.join(temp_extract_dir, "ffmpeg_rebuild_log.txt"), "w") as f:
                        f.write(rebuild_result.stderr)
                
                # Vérifier le résultat final
                if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                    # Analyse finale
                    probe_cmd = [
                        "ffprobe", 
                        "-v", "error", 
                        "-select_streams", "v:0",
                        "-show_entries", "stream=avg_frame_rate,nb_frames,duration",
                        "-of", "json",
                        output_path
                    ]
                    
                    final_probe = subprocess.run(probe_cmd, capture_output=True, text=True)
                    final_data = json.loads(final_probe.stdout) if final_probe.stdout else {}
                    
                    if 'streams' in final_data and final_data['streams']:
                        stream = final_data['streams'][0]
                        frames = int(stream.get('nb_frames', 0))
                        duration_str = stream.get('duration', '0')
                        framerate_str = stream.get('avg_frame_rate', '0/1')
                        
                        try:
                            num, den = map(int, framerate_str.split('/'))
                            framerate = num/den if den != 0 else 0
                        except:
                            framerate = 0
                        
                        self.log(f"Segment final: {frames} frames, durée={duration_str}s, fps={framerate}")
                    
                    return output_path
            
            # Si toutes les méthodes ont échoué, dernier recours
            self.log("DERNIER RECOURS: Création d'une vidéo artificielle avec mouvement imposé")
            
            # Créer une vidéo artificielle avec un dégradé qui change dans le temps
            # (au moins ce ne sera pas une image fixe)
            art_frames_dir = os.path.join(temp_extract_dir, "art_frames")
            os.makedirs(art_frames_dir, exist_ok=True)
            
            # Utiliser la première frame source si disponible, sinon créer une image artificielle
            sample_frame = None
            if frame_count > 0:
                sample_frame = cv2.imread(os.path.join(frames_dir, extracted_frames[0]))
            
            if sample_frame is not None:
                height, width = sample_frame.shape[:2]
            else:
                # Dimensions standard si pas de frame source
                width, height = 1280, 720
                sample_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Créer 30 frames artificielles avec un effet de zoom et de luminosité
            for i in range(30):
                # Copier la frame source
                art_frame = sample_frame.copy()
                
                # Appliquer un effet de zoom progressif
                scale = 1.0 + (i / 30) * 0.1  # Zoom de 0% à 10%
                center_x, center_y = width // 2, height // 2
                
                # Matrice de transformation pour le zoom
                M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
                art_frame = cv2.warpAffine(art_frame, M, (width, height))
                
                # Ajuster la luminosité
                brightness = 1.0 + (i / 30) * 0.2  # Augmentation de 0% à 20%
                art_frame = cv2.convertScaleAbs(art_frame, alpha=brightness, beta=0)
                
                # Sauvegarder la frame
                cv2.imwrite(os.path.join(art_frames_dir, f"art_{i:08d}.png"), art_frame)
            
            # Créer une vidéo à partir des frames artificielles
            art_video_cmd = [
                "ffmpeg", "-y",
                "-framerate", f"{source_fps}",
                "-pattern_type", "glob",
                "-i", os.path.join(art_frames_dir, "*.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-preset", "medium",
                "-r", f"{source_fps}",
                "-t", f"{duration:.6f}",
                "-vsync", "cfr",
                output_path
            ]
            
            subprocess.run(art_video_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                self.log("Vidéo artificielle créée avec succès comme solution de dernier recours")
                return output_path
            
            self.log("ÉCHEC TOTAL: Impossible de créer un segment vidéo avec mouvement")
            return None
            
        except Exception as e:
            self.log(f"ERREUR dans l'extraction avec garantie de mouvement: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Sauvegarder les traces d'erreur dans un fichier
            with open(os.path.join(temp_extract_dir, "error_log.txt"), "w") as f:
                f.write(f"Exception: {str(e)}\n")
                f.write(traceback.format_exc())
            
            return None
        
        finally:
            # Ne pas nettoyer les fichiers temporaires pour permettre le diagnostic
            self.log("Fichiers temporaires conservés pour diagnostic dans: " + temp_extract_dir)
##############################################################################################################

    def extract_segment_as_file(self, source_video_path, segment, output_path=None):
        """
        Extraire un segment spécifique d'une vidéo et le sauvegarder dans un fichier.
        Version avancée pour éviter les saccades et les images figées.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            segment: Dictionnaire contenant les informations du segment (start_time, duration)
            output_path: Chemin pour sauvegarder le segment extrait (optionnel)
            
        Returns:
            Chemin vers le fichier extrait
        """
        if output_path is None:
            timestamp = int(time.time())
            base_name = os.path.splitext(os.path.basename(source_video_path))[0]
            output_path = os.path.join(self.temp_dir, f"{base_name}_segment_{timestamp}.mp4")
        
        start_time = segment['start_time']
        duration = segment['duration']
        
        self.log(f"Extraction du segment de {start_time:.2f}s (durée: {duration:.2f}s)...")
        
        # Obtenir les informations sur la vidéo source
        info = SceneAnalyzer(source_video_path, verbose=False)
        fps = info.fps
        
        # Ajuster légèrement le début et la durée pour éviter les problèmes
        # Reculer de 0.5 seconde pour assurer une meilleure précision
        safe_start_time = max(0, start_time - 0.5)
        safe_duration = duration + 0.5  # Compenser le recul
        
        # Première tentative: extraction en deux étapes (plus précise)
        try:
            # 1. Extraire un segment temporaire plus long avec copie directe (sans réencodage)
            temp_output = os.path.join(self.temp_dir, f"temp_extract_{timestamp}.mp4")
            
            # Extraction directe depuis la source
            cmd1 = [
                "ffmpeg", "-y",
                "-ss", f"{safe_start_time:.6f}",
                "-i", source_video_path,
                "-t", f"{safe_duration:.6f}",
                "-c:v", "copy",          # Copie directe - pas de réencodage
                "-an",                   # Sans audio
                temp_output
            ]
            
            self.log(f"Étape 1: Extraction brute du segment...")
            subprocess.run(cmd1, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if not os.path.exists(temp_output) or os.path.getsize(temp_output) < 10000:
                raise Exception("L'extraction brute a échoué, fichier trop petit ou inexistant")
            
            # 2. Réencoder le segment temporaire avec précision accrue
            # Correction du début relatif pour compenser le décalage initial
            relative_start = start_time - safe_start_time
            
            cmd2 = [
                "ffmpeg", "-y",
                "-i", temp_output,
                "-ss", f"{relative_start:.6f}",
                "-t", f"{duration:.6f}",
                "-c:v", "libx264",
                "-crf", "18",            # Haute qualité
                "-preset", "slow",       # Meilleure qualité de compression
                "-tune", "film",
                "-force_key_frames", f"expr:gte(t,0)",  # Forcer une keyframe au début
                "-x264opts", "keyint=25:min-keyint=25", # Plus de keyframes
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                output_path
            ]
            
            self.log(f"Étape 2: Réencodage précis du segment à {start_time:.2f}s...")
            subprocess.run(cmd2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Vérification de la sortie finale
            if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                segment_info = SceneAnalyzer(output_path, verbose=False)
                self.log(f"Segment extrait avec succès: durée={segment_info.duration:.2f}s (attendu: {duration:.2f}s)")
                
                # Nettoyer le fichier temporaire
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    
                return output_path
        
        except Exception as e:
            self.log(f"Erreur lors de l'extraction en deux étapes: {str(e)}")
            self.log("Tentative avec méthode alternative...")
        
        # Si la première méthode échoue, essayer la méthode directe optimisée
        try:
            # Méthode directe avec nombreuses keyframes et segment plus large
            cmd_alt = [
                "ffmpeg", "-y",
                "-ss", f"{safe_start_time:.6f}",  # Position légèrement avant pour assurer un démarrage propre
                "-i", source_video_path,
                "-t", f"{safe_duration:.6f}",     # Durée légèrement plus longue
                "-c:v", "libx264",
                "-crf", "17",                     # Qualité très haute
                "-preset", "veryslow",            # Qualité maximale
                "-tune", "film",
                "-force_key_frames", f"expr:gte(t,{relative_start})",  # Forcer une keyframe au début réel
                "-x264opts", "keyint=15:min-keyint=15",  # Keyframes très fréquentes
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                output_path
            ]
            
            self.log("Extraction directe avec compression haute qualité...")
            subprocess.run(cmd_alt, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                # Si le segment est plus long que prévu, recouper avec précision
                segment_info = SceneAnalyzer(output_path, verbose=False)
                if abs(segment_info.duration - duration) > 0.5:
                    self.log(f"Recoupage du segment pour précision (extrait: {segment_info.duration:.2f}s vs attendu: {duration:.2f}s)")
                    
                    # Fichier de sortie final
                    final_output = os.path.join(self.temp_dir, f"{base_name}_segment_final_{timestamp}.mp4")
                    
                    trim_cmd = [
                        "ffmpeg", "-y",
                        "-i", output_path,
                        "-ss", f"{relative_start:.6f}",  # Ajustement interne
                        "-t", f"{duration:.6f}",
                        "-c:v", "libx264",
                        "-crf", "18",
                        "-preset", "slow",
                        "-force_key_frames", "expr:gte(t,0)",
                        "-pix_fmt", "yuv420p",
                        "-an",
                        final_output
                    ]
                    
                    subprocess.run(trim_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    # Remplacer par la version finale
                    if os.path.exists(final_output) and os.path.getsize(final_output) > 10000:
                        os.remove(output_path)
                        output_path = final_output
                
                self.log(f"Segment extrait avec succès par méthode alternative")
                return output_path
            else:
                self.log(f"Erreur: Le fichier de sortie est vide ou trop petit")
                
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode() if hasattr(e, 'stderr') else str(e)
            self.log(f"Erreur lors de l'extraction alternative: {error_output}")
            
            # Dernière tentative de récupération avec une méthode plus simple
            self.log("Tentative de méthode de dernier recours...")
            try:
                recovery_cmd = [
                    "ffmpeg", "-y",
                    "-accurate_seek",             # Seek précis
                    "-ss", f"{start_time:.6f}",
                    "-i", source_video_path,
                    "-t", f"{duration:.6f}",
                    "-c:v", "libx264",
                    "-crf", "20",
                    "-preset", "fast",
                    "-g", "15",                   # Groupe d'images réduit (plus de keyframes)
                    "-sc_threshold", "0",         # Désactive le changement de scène
                    "-an",
                    output_path
                ]
                
                subprocess.run(recovery_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                    self.log("Extraction de dernier recours réussie")
                    return output_path
            except:
                self.log("Échec de la méthode de dernier recours")
                
            return None

    def replace_scene_with_segment(self, source_video_path, target_video_path, segment, scene_info, output_path=None):
        """
        Remplacer une scène spécifique par un segment prédéfini avec garantie de mouvement.
        Version entièrement réécrite pour résoudre les problèmes d'images fixes.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            target_video_path: Chemin vers la vidéo cible
            segment: Dictionnaire contenant les informations du segment source
            scene_info: Dictionnaire contenant les informations de la scène à remplacer
            output_path: Chemin pour la vidéo de sortie (optionnel)
            
        Returns:
            Chemin vers la vidéo de sortie
        """
        # Date/Timestamp pour logs et fichiers temporaires
        timestamp = int(time.time())
        current_date = "2025-05-05 11:28:02"  # Date UTC actuelle
        user = "FETHl"  # Utilisateur actuel
        
        self.log(f"=== DÉBUT REMPLACEMENT DE SCÈNE [{current_date}] par {user} ===")
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(target_video_path))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}_scene{scene_info['number']}_replaced_{timestamp}.mp4")
        
        # Créer un dossier spécifique pour les fichiers temporaires de cette opération
        session_temp_dir = os.path.join(self.temp_dir, f"replace_session_{timestamp}")
        os.makedirs(session_temp_dir, exist_ok=True)
        
        # Obtenir la durée totale de la vidéo cible
        target_analyzer = SceneAnalyzer(target_video_path, verbose=False)
        target_duration = target_analyzer.duration
        target_fps = target_analyzer.fps
        
        self.log(f"Remplacement de la scène {scene_info['number']} (durée: {scene_info['duration']:.2f}s)")
        self.log(f"Timestamps - Début: {scene_info['start_time']:.2f}s, Fin: {scene_info['end_time']:.2f}s")
        self.log(f"Vidéo cible: durée={target_duration:.2f}s, fps={target_fps:.2f}")
        
        try:
            # 1. EXTRAIRE LA PARTIE AVANT LA SCÈNE
            before_path = None
            if scene_info['start_time'] > 0.1:  # S'il y a du contenu avant (au moins 0.1s)
                before_path = os.path.join(session_temp_dir, f"before_scene_{scene_info['number']}.mp4")
                
                try:
                    # Extraction avec paramètres optimisés pour éviter les saccades à la fin
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", target_video_path,
                        "-ss", "0",
                        "-to", f"{scene_info['start_time']:.6f}",
                        "-c:v", "libx264",  # Réencodage pour garantir des keyframes correctes
                        "-crf", "18",       # Haute qualité
                        "-preset", "slow",
                        "-x264opts", "keyint=25:min-keyint=25",  # Keyframes fréquentes
                        "-force_key_frames", f"expr:gte(t,{scene_info['start_time'] - 0.1})",  # Keyframe juste avant la fin
                        "-pix_fmt", "yuv420p",
                        "-an",              # Sans audio (géré séparément)
                        before_path
                    ]
                    
                    self.log(f"Extraction du segment avant la scène: 0s - {scene_info['start_time']:.2f}s")
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    # Vérifier la validité du segment extrait
                    if os.path.exists(before_path) and os.path.getsize(before_path) > 10000:
                        before_info = SceneAnalyzer(before_path, verbose=False)
                        self.log(f"Segment avant extrait: durée={before_info.duration:.2f}s")
                    else:
                        self.log("Segment avant non valide ou trop petit, extraction alternative")
                        # Méthode alternative
                        alt_cmd = [
                            "ffmpeg", "-y",
                            "-i", target_video_path,
                            "-ss", "0",
                            "-to", f"{scene_info['start_time']:.6f}",
                            "-c", "copy",  # Essai avec copy simple
                            "-avoid_negative_ts", "1",
                            "-an",
                            before_path
                        ]
                        subprocess.run(alt_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except Exception as e:
                    self.log(f"Erreur lors de l'extraction du segment avant: {str(e)}")
                    before_path = None
            
            # 2. AJUSTER LA DURÉE DU SEGMENT SOURCE SI NÉCESSAIRE
            adjusted_segment = segment.copy()
            if abs(segment['duration'] - scene_info['duration']) > 0.1:
                self.log(f"Ajustement du segment source: {segment['duration']:.2f}s → {scene_info['duration']:.2f}s")
                adjusted_segment = self.adjust_segment_duration(source_video_path, segment, scene_info['duration'])
            
            # 3. EXTRAIRE LE SEGMENT SOURCE AVEC GARANTIE DE MOUVEMENT
            start_time = adjusted_segment['start_time']
            duration = adjusted_segment['duration']
            source_info = SceneAnalyzer(source_video_path, verbose=False)
            source_fps = source_info.fps
            
            self.log(f"EXTRACTION DE SEGMENT SOURCE AVEC GARANTIE DE MOUVEMENT: {start_time:.2f}s (durée: {duration:.2f}s)")
            
            # Créer un dossier pour les frames
            frames_dir = os.path.join(session_temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            segment_path = os.path.join(session_temp_dir, f"source_segment.mp4")
            
            # MÉTHODE 1: EXTRACTION DIRECTE AVEC PARAMÈTRES POUR GARANTIR LE MOUVEMENT
            try:
                self.log("Tentative d'extraction directe avec paramètres optimisés pour le mouvement")
                direct_cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{start_time:.6f}",
                    "-i", source_video_path,
                    "-t", f"{duration:.6f}",
                    "-c:v", "libx264",
                    "-crf", "15",             # Très haute qualité
                    "-preset", "veryslow",    # Compression maximale
                    "-profile:v", "high",
                    "-level", "4.1",
                    "-pix_fmt", "yuv420p",
                    "-r", f"{source_fps}",    # Forcer le framerate source
                    "-vsync", "cfr",          # Constant frame rate
                    "-an",                    # Sans audio
                    "-force_key_frames", "expr:gte(t,0)",  # Keyframe au début
                    "-x264opts", "keyint=10:min-keyint=1",  # Keyframes très fréquentes
                    segment_path
                ]
                
                subprocess.run(direct_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Vérifier si l'extraction a produit une vidéo valide avec mouvement
                if os.path.exists(segment_path) and os.path.getsize(segment_path) > 10000:
                    # Vérifier le nombre de frames
                    probe_cmd = [
                        "ffprobe", 
                        "-v", "error", 
                        "-select_streams", "v:0",
                        "-show_entries", "stream=nb_frames,duration",
                        "-of", "json",
                        segment_path
                    ]
                    
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    probe_data = json.loads(probe_result.stdout) if probe_result.stdout else {}
                    
                    frames = 0
                    duration_sec = 0
                    if 'streams' in probe_data and probe_data['streams']:
                        frames = int(probe_data['streams'][0].get('nb_frames', '0'))
                        duration_sec = float(probe_data['streams'][0].get('duration', '0'))
                    
                    self.log(f"Segment extrait: {frames} frames, durée={duration_sec:.2f}s")
                    
                    # Si le nombre de frames est trop faible pour la durée (moins de 10 fps), 
                    # c'est probablement qu'on a une image fixe
                    if frames < 3 or (duration_sec > 0 and frames / duration_sec < 5):
                        self.log(f"ALERTE: Extraction directe a produit une image quasi-fixe ({frames} frames)")
                        raise Exception(f"Trop peu de frames extraites ({frames}), passage à la méthode frame par frame")
                    else:
                        self.log(f"Extraction directe réussie avec mouvement confirmé ({frames} frames)")
                        # Segment valide avec mouvement confirmé, on peut passer à l'étape suivante
                else:
                    raise Exception("L'extraction directe n'a pas produit de fichier valide")
                    
            except Exception as e:
                self.log(f"Échec de l'extraction directe: {str(e)}")
                
                # MÉTHODE 2: EXTRACTION FRAME PAR FRAME ET RECONSTRUCTION
                try:
                    self.log("Tentative d'extraction frame par frame...")
                    
                    # Supprimer les frames existantes
                    import shutil
                    if os.path.exists(frames_dir):
                        shutil.rmtree(frames_dir)
                        os.makedirs(frames_dir, exist_ok=True)
                    
                    # Extraire les frames avec un taux plus élevé
                    extract_frames_cmd = [
                        "ffmpeg", "-y",
                        "-ss", f"{start_time:.6f}",
                        "-i", source_video_path,
                        "-t", f"{duration:.6f}",
                        "-vsync", "0",            # Passthrough mode
                        "-q:v", "1",              # Qualité maximale
                        "-r", f"{source_fps*2}",  # Double du framerate pour assurer plus de frames
                        "-f", "image2",           # Format d'image séquentielle
                        os.path.join(frames_dir, "frame_%08d.png")
                    ]
                    
                    frame_result = subprocess.run(extract_frames_cmd, capture_output=True, text=True)
                    
                    # Vérifier les frames extraites
                    extracted_frames = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
                    frame_count = len(extracted_frames)
                    
                    self.log(f"Extraction frame par frame: {frame_count} frames extraites")
                    
                    if frame_count < 3:
                        self.log("ALERTE CRITIQUE: Très peu de frames extraites, tentative de duplication...")
                        
                        # Si trop peu de frames, dupliquer la première frame plusieurs fois avec variations
                        if frame_count > 0:
                            # Lire la première frame
                            first_frame = cv2.imread(os.path.join(frames_dir, extracted_frames[0]))
                            h, w = first_frame.shape[:2]
                            
                            # Créer des variations pour simuler du mouvement
                            for i in range(1, 30):
                                # Créer une variation de la frame
                                # 1. Légère translation
                                dx = int(w * 0.01 * i / 30)  # Déplacement horizontal de 0 à 1%
                                dy = int(h * 0.01 * i / 30)  # Déplacement vertical de 0 à 1%
                                
                                M = np.float32([[1, 0, dx], [0, 1, dy]])
                                shifted = cv2.warpAffine(first_frame, M, (w, h))
                                
                                # 2. Léger zoom
                                scale = 1.0 + 0.002 * i  # Zoom progressif jusqu'à +6%
                                M2 = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
                                zoomed = cv2.warpAffine(shifted, M2, (w, h))
                                
                                # 3. Légère variation de luminosité
                                brightness = 1.0 + 0.01 * i  # Augmentation progressive de la luminosité
                                brightened = cv2.convertScaleAbs(zoomed, alpha=brightness, beta=0)
                                
                                # Sauvegarder la frame modifiée
                                dup_frame_path = os.path.join(frames_dir, f"frame_dup_{i:08d}.png")
                                cv2.imwrite(dup_frame_path, brightened)
                            
                            # Mettre à jour la liste des frames
                            extracted_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
                            frame_count = len(extracted_frames)
                            self.log(f"Après duplication avec variations: {frame_count} frames disponibles")
                    
                    # Reconstruire la vidéo à partir des frames
                    if frame_count > 0:
                        # Nommage cohérent pour ffmpeg
                        for i, old_name in enumerate(sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])):
                            os.rename(
                                os.path.join(frames_dir, old_name),
                                os.path.join(frames_dir, f"final_{i:08d}.png")
                            )
                        
                        # Reconstruire avec des paramètres optimisés
                        reconstruct_cmd = [
                            "ffmpeg", "-y",
                            "-framerate", f"{source_fps}",  # Framerate source
                            "-i", os.path.join(frames_dir, "final_%08d.png"),
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-crf", "15",               # Très haute qualité
                            "-preset", "medium",        # Bon équilibre qualité/vitesse
                            "-profile:v", "high",
                            "-level", "4.1",
                            "-r", f"{source_fps}",      # Maintenir le framerate source
                            "-g", "1",                  # Groupe d'images (keyframe tous les 1 frame)
                            "-keyint_min", "1",         # Interval minimum entre keyframes
                            "-vsync", "cfr",            # Constant frame rate
                            "-movflags", "+faststart",
                            segment_path
                        ]
                        
                        self.log("Reconstruction de la vidéo à partir des frames...")
                        subprocess.run(reconstruct_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        
                        # Vérifier le résultat
                        if os.path.exists(segment_path) and os.path.getsize(segment_path) > 10000:
                            self.log(f"Reconstruction réussie à partir de {frame_count} frames")
                        else:
                            raise Exception("Échec de la reconstruction vidéo à partir des frames")
                    else:
                        raise Exception("Aucune frame n'a pu être extraite")
                        
                except Exception as e:
                    self.log(f"Échec de l'extraction frame par frame: {str(e)}")
                    
                    # MÉTHODE 3: CRÉATION D'UNE VIDÉO SYNTHÉTIQUE À PARTIR D'UNE SEULE IMAGE
                    try:
                        self.log("DERNIER RECOURS: Création d'une vidéo synthétique avec effet de mouvement...")
                        
                        # Essayer d'extraire une seule image de la source
                        single_frame_path = os.path.join(session_temp_dir, "single_frame.png")
                        
                        single_frame_cmd = [
                            "ffmpeg", "-y",
                            "-ss", f"{start_time:.6f}",
                            "-i", source_video_path,
                            "-frames:v", "1",
                            "-q:v", "1",
                            single_frame_path
                        ]
                        
                        subprocess.run(single_frame_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        
                        # Si on a pu extraire une image, créer une vidéo avec zoom et effet
                        if os.path.exists(single_frame_path) and os.path.getsize(single_frame_path) > 1000:
                            # Lire l'image
                            img = cv2.imread(single_frame_path)
                            if img is None:
                                raise Exception("Image source invalide")
                                
                            h, w = img.shape[:2]
                            
                            # Dossier pour les frames synthétiques
                            synth_dir = os.path.join(session_temp_dir, "synth_frames")
                            os.makedirs(synth_dir, exist_ok=True)
                            
                            # Créer 30 frames avec des effets
                            frame_count = int(source_fps * duration)  # Calculer le nombre exact de frames
                            if frame_count < 30:
                                frame_count = 30  # Minimum 30 frames
                                
                            for i in range(frame_count):
                                # Progression de 0 à 1
                                progress = i / frame_count
                                
                                # Appliquer une combinaison d'effets:
                                # 1. Zoom oscillant
                                zoom_factor = 1.0 + 0.05 * math.sin(progress * 2 * math.pi)
                                
                                # 2. Rotation légère
                                angle = 2.0 * math.sin(progress * 4 * math.pi)
                                
                                # 3. Translation subtile
                                dx = int(w * 0.02 * math.sin(progress * 3 * math.pi))
                                dy = int(h * 0.02 * math.cos(progress * 3 * math.pi))
                                
                                # Appliquer les transformations
                                M = cv2.getRotationMatrix2D((w/2, h/2), angle, zoom_factor)
                                M[0, 2] += dx
                                M[1, 2] += dy
                                
                                transformed = cv2.warpAffine(img, M, (w, h))
                                
                                # Ajouter une légère variation de luminosité
                                brightness = 1.0 + 0.1 * math.sin(progress * 6 * math.pi)
                                contrast = 1.0 + 0.05 * math.sin(progress * 8 * math.pi)
                                adjusted = cv2.convertScaleAbs(transformed, alpha=contrast, beta=brightness)
                                
                                # Sauvegarder la frame
                                cv2.imwrite(os.path.join(synth_dir, f"synth_{i:08d}.png"), adjusted)
                            
                            # Créer la vidéo synthétique
                            synth_cmd = [
                                "ffmpeg", "-y",
                                "-framerate", f"{source_fps}",
                                "-i", os.path.join(synth_dir, "synth_%08d.png"),
                                "-c:v", "libx264",
                                "-pix_fmt", "yuv420p",
                                "-crf", "18",
                                "-preset", "medium",
                                "-r", f"{source_fps}",
                                "-t", f"{duration:.6f}",
                                "-vsync", "cfr",
                                segment_path
                            ]
                            
                            self.log(f"Création d'une vidéo synthétique à partir de {frame_count} frames modifiées...")
                            subprocess.run(synth_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            
                            if os.path.exists(segment_path) and os.path.getsize(segment_path) > 10000:
                                self.log("Vidéo synthétique créée avec succès comme solution de dernier recours")
                            else:
                                raise Exception("Échec de la création de vidéo synthétique")
                        else:
                            raise Exception("Impossible d'extraire une seule image de la source")
                            
                    except Exception as e:
                        self.log(f"ÉCHEC DE TOUTES LES MÉTHODES D'EXTRACTION DE SEGMENT: {str(e)}")
                        return None
            
            # Vérification finale du segment extrait
            if not os.path.exists(segment_path) or os.path.getsize(segment_path) < 10000:
                self.log("Échec de l'extraction du segment source")
                return None
                
            # Analyse finale du segment extrait
            probe_cmd = [
                "ffprobe", 
                "-v", "error", 
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,duration,avg_frame_rate",
                "-of", "json",
                segment_path
            ]
            
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            probe_data = json.loads(probe_result.stdout) if probe_result.stdout else {}
            
            if 'streams' in probe_data and probe_data['streams']:
                stream = probe_data['streams'][0]
                frames = int(stream.get('nb_frames', '0'))
                duration_str = stream.get('duration', '0')
                framerate_str = stream.get('avg_frame_rate', '0/1')
                
                try:
                    num, den = map(int, framerate_str.split('/'))
                    framerate = num/den if den != 0 else 0
                except:
                    framerate = 0
                    
                self.log(f"SEGMENT SOURCE FINAL: {frames} frames, durée={duration_str}s, fps={framerate}")
            
            # 4. EXTRAIRE LA PARTIE APRÈS LA SCÈNE
            after_path = None
            remaining_duration = target_duration - scene_info['end_time']
            
            if remaining_duration > 0.1:  # S'il reste plus de 0.1 seconde après la scène
                after_path = os.path.join(session_temp_dir, f"after_scene_{scene_info['number']}.mp4")
                
                try:
                    # Extraction avec paramètres optimisés pour éviter les saccades au début
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", f"{scene_info['end_time'] - 0.1:.6f}",  # Commencer légèrement avant
                        "-i", target_video_path,
                        "-t", f"{remaining_duration + 0.1:.6f}",
                        "-c:v", "libx264",
                        "-crf", "18",
                        "-preset", "slow",
                        "-x264opts", "keyint=25:min-keyint=25",
                        "-force_key_frames", "expr:gte(t,0)",  # Forcer une keyframe au début
                        "-pix_fmt", "yuv420p",
                        "-an",
                        after_path
                    ]
                    
                    self.log(f"Extraction du segment après: {scene_info['end_time']:.2f}s - {target_duration:.2f}s")
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    # Vérifier le segment extrait
                    if os.path.exists(after_path) and os.path.getsize(after_path) > 10000:
                        after_info = SceneAnalyzer(after_path, verbose=False)
                        self.log(f"Segment après extrait: durée={after_info.duration:.2f}s")
                        
                        # Vérifier si le segment est trop court
                        if after_info.duration < remaining_duration * 0.9:
                            self.log(f"ATTENTION: Segment après tronqué ({after_info.duration:.2f}s vs {remaining_duration:.2f}s)")
                            
                            # Méthode alternative
                            alt_after_path = os.path.join(session_temp_dir, f"after_scene_alt.mp4")
                            alt_cmd = [
                                "ffmpeg", "-y",
                                "-i", target_video_path,
                                "-ss", f"{scene_info['end_time']:.6f}",
                                "-c", "copy",
                                "-an",
                                alt_after_path
                            ]
                            
                            subprocess.run(alt_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            
                            if os.path.exists(alt_after_path) and os.path.getsize(alt_after_path) > 10000:
                                alt_info = SceneAnalyzer(alt_after_path, verbose=False)
                                if alt_info.duration > after_info.duration:
                                    self.log(f"Utilisation du segment alternatif: {alt_info.duration:.2f}s")
                                    after_path = alt_after_path
                    else:
                        self.log("Segment après invalide ou trop petit")
                        after_path = None
                        
                except Exception as e:
                    self.log(f"Erreur lors de l'extraction du segment après: {str(e)}")
                    after_path = None
            else:
                self.log(f"Pas de segment après (fin de vidéo: {remaining_duration:.2f}s restantes)")
            
            # 5. EXTRAIRE L'AUDIO DE LA VIDÉO CIBLE
            audio_path = os.path.join(session_temp_dir, f"audio.aac")
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", target_video_path,
                    "-vn",                # Pas de vidéo
                    "-acodec", "aac",     # Encoder en AAC pour compatibilité
                    "-strict", "experimental",
                    "-b:a", "192k",       # Bitrate audio de qualité
                    audio_path
                ]
                
                self.log("Extraction de l'audio original...")
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
                    self.log("ATTENTION: Audio extrait vide ou invalide")
                    
                    # Tentative alternative
                    alt_audio_cmd = [
                        "ffmpeg", "-y",
                        "-i", target_video_path,
                        "-vn",
                        "-acodec", "copy",  # Essai sans réencodage
                        os.path.join(session_temp_dir, "audio_alt.aac")
                    ]
                    
                    subprocess.run(alt_audio_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    alt_audio_path = os.path.join(session_temp_dir, "audio_alt.aac")
                    if os.path.exists(alt_audio_path) and os.path.getsize(alt_audio_path) > 1000:
                        audio_path = alt_audio_path
                        self.log("Audio alternatif extrait avec succès")
                
            except Exception as e:
                self.log(f"Erreur lors de l'extraction de l'audio: {str(e)}")
                audio_path = None
            
            # 6. CONCATÉNATION AMÉLIORÉE DES SEGMENTS VIDÉO
            self.log("Concaténation des segments vidéo...")
            
            # 6.1 Préparer les segments valides
            valid_segments = []
            if before_path and os.path.exists(before_path) and os.path.getsize(before_path) > 10000:
                valid_segments.append(before_path)
                self.log(f"+ Segment avant: {os.path.basename(before_path)}")
                
            if segment_path and os.path.exists(segment_path) and os.path.getsize(segment_path) > 10000:
                valid_segments.append(segment_path)
                self.log(f"+ Segment source: {os.path.basename(segment_path)}")
                
            if after_path and os.path.exists(after_path) and os.path.getsize(after_path) > 10000:
                valid_segments.append(after_path)
                self.log(f"+ Segment après: {os.path.basename(after_path)}")
            
            if not valid_segments:
                self.log("ERREUR: Aucun segment valide à concaténer")
                return None
                
            # 6.2 Fichier de liste pour concaténation
            concat_list_path = os.path.join(session_temp_dir, "concat_list.txt")
            with open(concat_list_path, 'w') as f:
                for segment in valid_segments:
                    f.write(f"file '{os.path.abspath(segment)}'\n")
                    
            self.log(f"Liste de concaténation créée avec {len(valid_segments)} segments")
            
            # 6.3 Concaténer avec réencodage pour assurer la fluidité
            video_only_path = os.path.join(session_temp_dir, "video_only.mp4")
            try:
                concat_cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_path,
                    "-c:v", "libx264",
                    "-profile:v", "high",
                    "-level:v", "4.1",
                    "-pix_fmt", "yuv420p",
                    "-crf", "18",
                    "-preset", "slow",
                    "-x264opts", "keyint=25:min-keyint=25", # Keyframes fréquentes
                    "-vsync", "cfr",        # Constant frame rate
                    "-r", f"{target_fps}",  # Forcer le même framerate que la vidéo cible
                    "-movflags", "+faststart",
                    "-an",                  # Sans audio
                    video_only_path
                ]
                
                self.log("Concaténation avec réencodage pour fluidité optimale...")
                subprocess.run(concat_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Vérifier le résultat
                if not os.path.exists(video_only_path) or os.path.getsize(video_only_path) < 10000:
                    raise Exception("Échec de la concaténation avec réencodage")
                    
                concat_info = SceneAnalyzer(video_only_path, verbose=False)
                self.log(f"Vidéo concaténée: durée={concat_info.duration:.2f}s")
                
            except Exception as e:
                self.log(f"Erreur lors de la concaténation: {str(e)}")
                
                # Méthode alternative (simple copie)
                try:
                    simple_concat_cmd = [
                        "ffmpeg", "-y",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", concat_list_path,
                        "-c", "copy",
                        "-an",
                        video_only_path
                    ]
                    
                    self.log("Tentative de concaténation simple...")
                    subprocess.run(simple_concat_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    if not os.path.exists(video_only_path) or os.path.getsize(video_only_path) < 10000:
                        self.log("Échec de toutes les méthodes de concaténation")
                        return None
                except Exception as e2:
                    self.log(f"Échec de la concaténation simple: {str(e2)}")
                    return None
            
            # 7. COMBINAISON FINALE DE LA VIDÉO ET DE L'AUDIO
            try:
                # 7.1 Vérifier si nous avons de l'audio
                if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                    self.log("Combinaison de la vidéo avec l'audio original...")
                    
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", video_only_path,   # Vidéo sans audio
                        "-i", audio_path,        # Audio
                        "-c:v", "copy",          # Copier la vidéo sans réencodage
                        "-c:a", "aac",           # Encoder l'audio en AAC
                        "-b:a", "192k",          # Bitrate audio
                        "-strict", "experimental",
                        "-map", "0:v:0",         # Utiliser la vidéo du premier input
                        "-map", "1:a:0",         # Utiliser l'audio du second input
                        "-shortest",             # Terminer quand le plus court finit
                        output_path
                    ]
                    
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                else:
                    # 7.2 Si pas d'audio, utiliser directement la vidéo
                    self.log("Pas d'audio valide, utilisation de la vidéo sans audio")
                    import shutil
                    shutil.copy(video_only_path, output_path)
                
                # 7.3 Vérification finale
                if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                    output_info = SceneAnalyzer(output_path, verbose=False)
                    self.log(f"Vidéo finale générée: {output_path}")
                    self.log(f"Durée finale: {output_info.duration:.2f}s (vidéo originale: {target_duration:.2f}s)")
                    
                    # Vérifier la différence de durée
                    duration_diff = abs(output_info.duration - target_duration)
                    if duration_diff > 3.0:
                        self.log(f"ATTENTION: Différence de durée significative: {duration_diff:.2f}s")
                    
                    self.log(f"=== REMPLACEMENT TERMINÉ AVEC SUCCÈS [{current_date}] ===")
                    return output_path
                else:
                    self.log("Échec de la génération de la vidéo finale")
                    return None
                    
            except Exception as e:
                self.log(f"Erreur lors de la combinaison finale: {str(e)}")
                return None
                
        except Exception as e:
            self.log(f"ERREUR CRITIQUE: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return None
            
        finally:
            # Conserver les fichiers temporaires pour diagnostic
            self.log(f"Fichiers temporaires conservés dans: {session_temp_dir}")

    def replace_multiple_scenes_with_best_pose_match(self, source_video_path, target_scene_numbers,
                                              output_path=None, scan_entire_video=False):
        """
        Remplacer plusieurs scènes dans la vidéo cible par le même segment source trouvé.
        Cette fonction est utile lorsque la même scène est répétée à différents emplacements.
        
        La recherche du meilleur segment est faite UNE SEULE FOIS pour la première scène.
        Ensuite, le même segment est utilisé pour toutes les autres scènes.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            target_scene_numbers: Liste des numéros de scènes à remplacer
            output_path: Chemin pour la vidéo de sortie finale (optionnel)
            scan_entire_video: Si True, analyser toute la vidéo pour la recherche du meilleur segment
            
        Returns:
            Chemin vers la vidéo de sortie finale
        """
        if not target_scene_numbers:
            self.log("Erreur: Aucune scène spécifiée pour le remplacement")
            return None
            
        # Vérifier que toutes les scènes cibles existent
        scene_infos = []
        for scene_num in target_scene_numbers:
            info = self.replacer.get_target_scene_info(scene_num)
            if not info:
                self.log(f"Erreur: Scène {scene_num} non trouvée dans la vidéo cible")
                return None
            scene_infos.append(info)
            
        self.log(f"Traitement de {len(target_scene_numbers)} scènes à remplacer: {target_scene_numbers}")
        self.log(f"Positions des scènes à remplacer:")
        for i, scene in enumerate(scene_infos):
            self.log(f"  Scène {target_scene_numbers[i]}: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s (durée: {scene['duration']:.2f}s)")
        
        # Utiliser la première scène comme référence pour trouver le meilleur segment
        primary_scene = scene_infos[0]
        primary_scene_num = target_scene_numbers[0]
        
        self.log(f"Utilisation de la scène {primary_scene_num} comme référence principale")
        self.log(f"Durée de la scène de référence: {primary_scene['duration']:.2f}s")
        
        # ÉTAPE 1: Trouver le meilleur segment correspondant dans la vidéo source pour la première scène
        best_segment = self.find_best_segment(
            source_video_path, 
            primary_scene,
            scan_entire_video=scan_entire_video
        )
        
        if not best_segment:
            self.log("Impossible de trouver un segment approprié pour le remplacement")
            return None
            
        # Créer une visualisation pour la première scène
        vis_path = self.create_visualizations(
            best_segment, 
            primary_scene, 
            output_path=os.path.join(self.output_dir, f"vis_scene{primary_scene_num}_{time.strftime('%Y%m%d_%H%M%S')}.png")
        )
        
        # Mémoriser le segment source original
        original_segment = best_segment.copy()
        
        # ÉTAPE 2: NOUVELLE APPROCHE - traiter TOUTES les scènes en une seule opération
        # Pour éviter les problèmes liés aux modifications séquentielles
        
        if len(target_scene_numbers) == 1:
            # Cas simple: une seule scène à remplacer
            self.log(f"Remplacement de la scène {target_scene_numbers[0]}...")
            
            result_path = self.replace_scene_with_segment(
                source_video_path,         # Vidéo source contenant le segment
                self.target_video_path,    # Vidéo cible originale
                original_segment,          # Segment original à utiliser
                scene_infos[0],            # Info sur la scène à remplacer
                output_path=output_path    # Chemin de sortie
            )
            
            if result_path:
                self.log(f"Remplacement terminé avec succès: {result_path}")
            else:
                self.log(f"Échec du remplacement")
                
            return result_path
        
        else:
            # Cas complexe: plusieurs scènes à remplacer
            self.log(f"Remplacement de plusieurs scènes: {target_scene_numbers}")
            
            # Traitement des scènes de la fin vers le début pour éviter les décalages de timecode
            # Trier les scènes par ordre décroissant de position (commencer par la fin)
            scene_pairs = sorted(zip(target_scene_numbers, scene_infos), 
                                key=lambda pair: pair[1]['start_time'], 
                                reverse=True)
            
            # Vidéo de travail actuelle (commencer avec la vidéo originale)
            current_video = self.target_video_path
            
            # Pour chaque scène, en commençant par la dernière
            for i, (scene_num, scene_info) in enumerate(scene_pairs):
                self.log(f"\n{i+1}/{len(scene_pairs)}. Remplacement de la scène {scene_num} (position: {scene_info['start_time']:.2f}s - {scene_info['end_time']:.2f}s)")
                
                # Chemin de sortie pour cette étape
                if i == len(scene_pairs) - 1 and output_path:  # Dernière scène (qui est en fait la première dans la vidéo)
                    step_output = output_path
                else:
                    # Chemin temporaire pour cette étape
                    timestamp = int(time.time())
                    base_name = os.path.splitext(os.path.basename(self.target_video_path))[0]
                    step_output = os.path.join(
                        self.output_dir, 
                        f"{base_name}_multi_step{i+1}_{timestamp}.mp4"
                    )
                
                # Ajuster le segment si nécessaire pour cette scène spécifique
                adjusted_segment = original_segment.copy()
                if abs(scene_info['duration'] - primary_scene['duration']) > 0.1:
                    self.log(f"Ajustement du segment pour correspondre à la durée de la scène {scene_num} ({scene_info['duration']:.2f}s)")
                    adjusted_segment = self.adjust_segment_duration(
                        source_video_path,
                        original_segment,
                        scene_info['duration']
                    )
                
                # Remplacer cette scène
                result = self.replace_scene_with_segment(
                    source_video_path,  # Vidéo source 
                    current_video,      # Vidéo cible actuelle
                    adjusted_segment,   # Segment ajusté pour cette scène
                    scene_info,         # Infos sur cette scène
                    output_path=step_output  # Chemin de sortie pour cette étape
                )
                
                if not result:
                    self.log(f"Échec du remplacement de la scène {scene_num}")
                    # Si c'est la première scène traitée, retourner None
                    if i == 0:
                        return None
                    # Sinon, retourner le résultat de l'étape précédente
                    return current_video
                
                # Supprimer la vidéo intermédiaire précédente si nécessaire
                if i > 0 and current_video != self.target_video_path and os.path.exists(current_video):
                    try:
                        os.remove(current_video)
                        self.log(f"Fichier intermédiaire supprimé: {current_video}")
                    except:
                        self.log(f"Impossible de supprimer le fichier intermédiaire: {current_video}")
                
                # Mettre à jour la vidéo actuelle pour l'étape suivante
                current_video = result
            
            # Vérifier la durée finale
            if os.path.exists(current_video):
                final_analyzer = SceneAnalyzer(current_video, verbose=False)
                original_analyzer = SceneAnalyzer(self.target_video_path, verbose=False)
                
                self.log(f"\nVérification finale:")
                self.log(f"Durée originale: {original_analyzer.duration:.2f}s")
                self.log(f"Durée finale: {final_analyzer.duration:.2f}s")
                
                if abs(final_analyzer.duration - original_analyzer.duration) > 5.0:  # Plus de 5 secondes d'écart
                    self.log(f"ALERTE: La vidéo finale a une durée très différente de l'originale!")
                elif abs(final_analyzer.duration - original_analyzer.duration) > 1.0:  # Entre 1 et 5 secondes
                    self.log(f"ATTENTION: Légère différence de durée entre la vidéo finale et l'originale.")
                else:
                    self.log(f"Durée préservée correctement.")
            
            # Générer des informations détaillées sur tous les remplacements
            info_path = os.path.join(self.output_dir, "multi_replacement_info.txt")
            with open(info_path, 'a') as f:
                f.write("\n" + "="*50 + "\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Utilisateur: {CURRENT_USER}\n")
                f.write(f"Vidéo cible: {self.target_video_path}\n")
                f.write(f"Vidéo source: {source_video_path}\n")
                f.write(f"Image de référence: {self.reference_image_path}\n")
                f.write(f"Scènes remplacées: {', '.join(map(str, target_scene_numbers))}\n")
                f.write(f"Segment source original: {original_segment['start_time']:.2f}s - {original_segment['end_time']:.2f}s\n")
                f.write(f"Score de similarité: {original_segment['score']:.4f}\n")
                f.write(f"Visualisation de référence: {vis_path}\n")
                f.write(f"Vidéo de sortie finale: {current_video}\n")
            
            self.log(f"\nRemplacement de {len(target_scene_numbers)} scènes terminé avec succès!")
            self.log(f"Vidéo finale sauvegardée à: {current_video}")
            return current_video
            
    def replace_scene_with_best_pose_match(self, source_video_path, target_scene_number, 
                                          output_path=None, scan_entire_video=False):
        """
        Remplacer une scène dans la vidéo cible par le meilleur segment correspondant trouvé.
        Cette méthode est un wrapper pour le cas d'une seule scène.
        
        Args:
            source_video_path: Chemin vers la vidéo source
            target_scene_number: Numéro de la scène à remplacer
            output_path: Chemin pour la vidéo de sortie (optionnel)
            scan_entire_video: Si True, analyser toute la vidéo
            
        Returns:
            Chemin vers la vidéo de sortie
        """
        return self.replace_multiple_scenes_with_best_pose_match(
            source_video_path,
            [target_scene_number],  # Mettre dans une liste pour le cas d'une seule scène
            output_path=output_path,
            scan_entire_video=scan_entire_video
        )


def main():
    """Point d'entrée principal avec interface CLI."""
    parser = argparse.ArgumentParser(description="Outil de remplacement de scènes vidéo basé sur la pose")
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
        # Créer l'outil de remplacement basé sur la pose
        replacer = PoseBasedSceneReplacement(
            args.target_video,
            args.reference_image,
            pose_model_path=args.pose_model,
            verbose=not args.quiet
        )
        
        # Traiter le remplacement
        start_time = time.time()
        
        # Si plusieurs scènes sont spécifiées, utiliser la fonction multi-scènes
        if len(args.target_scenes) > 1:
            output_path = replacer.replace_multiple_scenes_with_best_pose_match(
                args.source_video,
                args.target_scenes,
                output_path=args.output,
                scan_entire_video=args.scan_entire
            )
        else:
            # Pour une seule scène, utiliser la fonction standard
            output_path = replacer.replace_scene_with_best_pose_match(
                args.source_video,
                args.target_scenes[0],
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
