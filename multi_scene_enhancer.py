#!/usr/bin/env python3
"""
EnhancedVideoReplacer - Outil avancé d'édition vidéo avec remplacement de scènes multiples

Fonctionnalités:
- Remplace plusieurs scènes dans une vidéo template avec le même segment source
- Ajoute des calques PNG (logo, numéros de scène, contours)
- Insère des pauses à des moments précis
- Garantit que les scènes aient exactement la même durée
- Permet de mettre à l'échelle l'image extraite

Auteur: FETHl
Date: 2025-04-30
"""

import os
import cv2
import numpy as np
import argparse
import torch
import time
import subprocess
import shutil
from typing import Dict, Tuple, List, Optional
from datetime import timedelta, datetime
import matplotlib.pyplot as plt

# Import du script de remplacement basé sur la pose
from pose_based_replacement import PoseBasedSceneReplacement, get_all_poses_yolo, create_pose_feature

class EnhancedVideoReplacer:
    """
    Classe principale pour l'édition vidéo avancée avec remplacement de scènes multiples,
    superposition de calques et insertion de pauses.
    """
    
    def __init__(self, template_video, source_video, reference_image, 
                 logo_path, contour_path, pose_model_path="yolo11x-pose.pt", 
                 verbose=True, image_scale=1.0):
        """
        Initialise l'outil d'édition vidéo avancé.
        
        Args:
            template_video: Chemin vers la vidéo template
            source_video: Chemin vers la vidéo source (rush)
            reference_image: Chemin vers l'image de référence pour la détection de pose
            logo_path: Chemin vers l'image du logo à superposer
            contour_path: Chemin vers l'image du contour à superposer
            pose_model_path: Chemin vers le modèle de pose
            verbose: Activer les messages de log détaillés
            image_scale: Facteur d'échelle pour l'image extraite
        """
        self.template_video = template_video
        self.source_video = source_video
        self.reference_image = reference_image
        self.logo_path = logo_path
        self.contour_path = contour_path
        self.pose_model_path = pose_model_path
        self.verbose = verbose
        self.image_scale = image_scale
        
        # Initialiser le remplaceur de scènes basé sur la pose
        self.pose_replacer = PoseBasedSceneReplacement(
            template_video,
            reference_image,
            pose_model_path=pose_model_path,
            verbose=verbose
        )
        
        # Créer les répertoires de travail
        self.temp_dir = "temp_enhanced"
        self.output_dir = "output_enhanced"
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Charger les images de superposition
        self.logo_image = self.load_image_with_alpha(logo_path)
        self.contour_image = self.load_image_with_alpha(contour_path)
        
        self.log("EnhancedVideoReplacer initialisé avec succès")
        self.log(f"Facteur d'échelle pour l'image extraite: {self.image_scale}")
    
    def log(self, message):
        """Affiche un message de log si le mode verbose est activé."""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def load_image_with_alpha(self, image_path):
        """
        Charge une image PNG avec son canal alpha.
        
        Args:
            image_path: Chemin vers l'image PNG
            
        Returns:
            Image chargée avec canal alpha ou None en cas d'erreur
        """
        if not os.path.isfile(image_path):
            self.log(f"Erreur: Image non trouvée: {image_path}")
            return None
            
        try:
            # Charger l'image avec son canal alpha
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                self.log(f"Erreur: Impossible de charger l'image: {image_path}")
                return None
                
            # Vérifier si l'image a un canal alpha
            if image.shape[2] < 4:
                self.log(f"Attention: L'image {image_path} n'a pas de canal alpha, ajout d'un canal alpha complet")
                # Ajouter un canal alpha complet (255)
                b, g, r = cv2.split(image)
                alpha = np.ones(b.shape, dtype=b.dtype) * 255
                image = cv2.merge((b, g, r, alpha))
            
            return image
        except Exception as e:
            self.log(f"Erreur lors du chargement de l'image {image_path}: {str(e)}")
            return None
    
    def overlay_image(self, background, overlay, x, y, scale=1.0):
        """
        Superpose une image avec transparence sur une autre.
        
        Args:
            background: Image d'arrière-plan (BGR)
            overlay: Image à superposer (BGRA)
            x, y: Position de l'overlay (coin supérieur gauche)
            scale: Facteur d'échelle pour l'overlay
            
        Returns:
            Image combinée
        """
        # Vérifier que les entrées sont valides
        if background is None or overlay is None:
            return background
            
        # Redimensionner l'overlay si nécessaire
        if scale != 1.0:
            h, w = overlay.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            overlay = cv2.resize(overlay, (new_w, new_h))
        
        # Obtenir les dimensions
        h, w = overlay.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # Calculer la région de superposition (gérer les dépassements)
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(bg_w, x + w)
        y_end = min(bg_h, y + h)
        
        # Ajuster les coordonnées pour l'overlay
        o_x_start = max(0, -x)
        o_y_start = max(0, -y)
        o_x_end = o_x_start + (x_end - x_start)
        o_y_end = o_y_start + (y_end - y_start)
        
        # Vérifier si la région est valide
        if x_end <= x_start or y_end <= y_start or o_x_end <= o_x_start or o_y_end <= o_y_start:
            return background
        
        # Extraire la région et le canal alpha
        overlay_region = overlay[o_y_start:o_y_end, o_x_start:o_x_end]
        alpha = overlay_region[:, :, 3] / 255.0
        
        # Créer une copie modifiable du fond
        result = background.copy()
        
        # Appliquer l'alpha blending
        for c in range(3):  # Canaux BGR
            result[y_start:y_end, x_start:x_end, c] = (
                overlay_region[:, :, c] * alpha + 
                result[y_start:y_end, x_start:x_end, c] * (1 - alpha)
            ).astype(np.uint8)
        
        return result
    
    def add_text_overlay(self, image, text, position, font_scale=1.0, 
                         color=(255, 255, 255), thickness=2, outline=True):
        """
        Ajoute du texte sur une image avec option de contour.
        
        Args:
            image: Image sur laquelle ajouter le texte
            text: Texte à ajouter
            position: Position (x, y) du texte
            font_scale: Échelle de la police
            color: Couleur du texte (BGR)
            thickness: Épaisseur du texte
            outline: Ajouter un contour noir autour du texte
            
        Returns:
            Image avec le texte ajouté
        """
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Ajouter un contour noir si demandé
        if outline:
            # Dessiner le texte avec contour noir
            cv2.putText(image, text, (x-1, y-1), font, font_scale, (0, 0, 0), thickness+1)
            cv2.putText(image, text, (x+1, y-1), font, font_scale, (0, 0, 0), thickness+1)
            cv2.putText(image, text, (x-1, y+1), font, font_scale, (0, 0, 0), thickness+1)
            cv2.putText(image, text, (x+1, y+1), font, font_scale, (0, 0, 0), thickness+1)
        
        # Dessiner le texte principal
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
        
        return image
    
    def scale_source_video(self, input_path, output_path, scale_factor):
        """
        Redimensionne la vidéo source selon le facteur d'échelle spécifié.
        
        Args:
            input_path: Chemin vers la vidéo d'entrée
            output_path: Chemin pour la vidéo redimensionnée
            scale_factor: Facteur d'échelle pour le redimensionnement
            
        Returns:
            Chemin vers la vidéo redimensionnée
        """
        if scale_factor == 1.0:
            # Pas besoin de redimensionner, utiliser la vidéo originale
            if input_path != output_path:
                shutil.copyfile(input_path, output_path)
            return output_path
        
        self.log(f"Redimensionnement de la vidéo source avec un facteur d'échelle de {scale_factor}...")
        
        # Obtenir les dimensions de la vidéo source
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Calculer les nouvelles dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        self.log(f"Redimensionnement de {width}x{height} à {new_width}x{new_height}")
        
        # Utiliser FFmpeg pour redimensionner la vidéo
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"scale={new_width}:{new_height}",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-c:a", "copy",
            output_path
        ]
        
        self.log(f"Exécution de la commande: {' '.join(ffmpeg_cmd)}")
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            self.log(f"Erreur lors du redimensionnement: {result.stderr.decode()}")
            return None
        
        self.log(f"Vidéo redimensionnée créée: {output_path}")
        return output_path
    
    def find_best_segment_for_scenes(self, scene_numbers):
        """
        Trouve le meilleur segment source pour remplacer plusieurs scènes.
        
        Args:
            scene_numbers: Liste des numéros de scènes à remplacer
            
        Returns:
            Informations sur le meilleur segment et les scènes cibles
        """
        self.log(f"Recherche du meilleur segment pour les scènes: {scene_numbers}")
        
        # Vérifier que toutes les scènes existent
        target_scenes = []
        for scene_num in scene_numbers:
            scene_info = self.pose_replacer.replacer.get_target_scene_info(scene_num)
            if not scene_info:
                self.log(f"Erreur: Scène {scene_num} non trouvée dans la vidéo template")
                return None
            target_scenes.append(scene_info)
        
        # Utiliser la première scène comme référence pour trouver le meilleur segment
        reference_scene = target_scenes[0]
        self.log(f"Utilisation de la scène {scene_numbers[0]} comme référence")
        self.log(f"Durée de la scène de référence: {reference_scene['duration']:.2f}s")
        
        # Redimensionner la vidéo source si nécessaire
        source_video = self.source_video
        if self.image_scale != 1.0:
            scaled_source = os.path.join(self.temp_dir, "scaled_source.mp4")
            source_video = self.scale_source_video(self.source_video, scaled_source, self.image_scale)
            if not source_video:
                self.log("Erreur lors du redimensionnement de la vidéo source")
                return None
        
        # Trouver le meilleur segment pour la scène de référence
        best_segment = self.pose_replacer.find_best_segment(
            source_video, 
            reference_scene,
            scan_entire_video=True
        )
        
        if not best_segment:
            self.log("Aucun segment approprié trouvé dans la vidéo source")
            return None
        
        self.log(f"Meilleur segment trouvé: {best_segment['start_time']:.2f}s - " +
                f"{best_segment['end_time']:.2f}s (score: {best_segment['score']:.4f})")
        
        # Extraire la frame de meilleure correspondance
        best_frame_index = best_segment['reference_frame_index']
        cap = cv2.VideoCapture(source_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_index)
        ret, best_frame = cap.read()
        cap.release()
        
        # Enregistrer les informations sur le meilleur segment et les scènes cibles
        result = {
            'best_segment': best_segment,
            'target_scenes': target_scenes,
            'scene_numbers': scene_numbers,
            'best_frame': best_frame if ret else None,
            'best_frame_time': best_segment['reference_time'],
            'source_video': source_video
        }
        
        return result
    
    def create_enhanced_video(self, scene_numbers, pause_duration=2.0, output_path=None):
        """
        Crée une vidéo améliorée avec remplacement de scènes multiples,
        ajout de calques et pauses.
        
        Args:
            scene_numbers: Liste des numéros de scènes à remplacer
            pause_duration: Durée de la pause en secondes pour la deuxième scène
            output_path: Chemin pour la vidéo de sortie
            
        Returns:
            Chemin vers la vidéo de sortie
        """
        if len(scene_numbers) != 2:
            self.log("Erreur: Exactement 2 numéros de scène sont requis")
            return None
        
        # Générer un nom de fichier par défaut si non spécifié
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"enhanced_video_{timestamp}.mp4")
        
        # Trouver le meilleur segment pour les deux scènes
        segment_info = self.find_best_segment_for_scenes(scene_numbers)
        if not segment_info:
            return None
        
        best_segment = segment_info['best_segment']
        target_scenes = segment_info['target_scenes']
        best_frame = segment_info['best_frame']
        source_video = segment_info['source_video']
        
        # Créer une visualisation du meilleur segment
        vis_path = self.pose_replacer.create_visualizations(
            best_segment, target_scenes[0])
        self.log(f"Visualisation créée: {vis_path}")
        
        # 1. Remplacer les deux scènes individuellement
        scene1_path = os.path.join(self.temp_dir, f"scene{scene_numbers[0]}_replaced.mp4")
        scene2_path = os.path.join(self.temp_dir, f"scene{scene_numbers[1]}_replaced.mp4")
        
        # Remplacer la première scène
        self.log(f"Remplacement de la scène {scene_numbers[0]}...")
        result1 = self.pose_replacer.replacer.replace_scene_preserving_audio(
            source_video,
            scene_numbers[0],
            output_path=scene1_path,
            best_match_start=best_segment['start_time']
        )
        
        # Remplacer la deuxième scène
        self.log(f"Remplacement de la scène {scene_numbers[1]}...")
        result2 = self.pose_replacer.replacer.replace_scene_preserving_audio(
            source_video,
            scene_numbers[1],
            output_path=scene2_path,
            best_match_start=best_segment['start_time']
        )
        
        if not result1 or not result2:
            self.log("Erreur lors du remplacement des scènes")
            return None
        
        # 2. Ajouter les calques et pauses pour chaque scène
        scene1_enhanced = os.path.join(self.temp_dir, f"scene{scene_numbers[0]}_enhanced.mp4")
        scene2_enhanced = os.path.join(self.temp_dir, f"scene{scene_numbers[1]}_enhanced.mp4")
        
        # CORRECTION: Pour la première scène: Logo uniquement
        self.log(f"Amélioration de la scène {scene_numbers[0]} avec uniquement le logo...")
        self.enhance_scene_with_overlays(
            scene1_path, 
            scene1_enhanced,
            scene_num=scene_numbers[0],
            add_logo=True,
            add_contour=False,  # Pas de contour sur la première scène
            add_scene_number=True,
            pause_frame_index=None,  # Pas de pause sur la première scène
            pause_duration=0.0,
            best_segment=best_segment
        )
        
        # CORRECTION: Pour la deuxième scène: Logo + Contour + Pause
        self.log(f"Amélioration de la scène {scene_numbers[1]} avec logo, contour et pause...")
        self.enhance_scene_with_overlays(
            scene2_path, 
            scene2_enhanced,
            scene_num=scene_numbers[1],
            add_logo=True,
            add_contour=True,  # Ajouter le contour sur la deuxième scène
            add_scene_number=True,
            pause_frame_index=best_segment['reference_frame_index'],  # Ajouter la pause sur la deuxième scène
            pause_duration=pause_duration,
            best_segment=best_segment
        )
        
        # 3. Assembler la vidéo finale en remplaçant les deux scènes
        self.log("Assemblage de la vidéo finale...")
        self.assemble_final_video(
            template_video=self.template_video,
            enhanced_scenes=[
                {"scene_num": scene_numbers[0], "video_path": scene1_enhanced},
                {"scene_num": scene_numbers[1], "video_path": scene2_enhanced}
            ],
            output_path=output_path
        )
        
        self.log(f"Vidéo améliorée créée avec succès: {output_path}")
        return output_path
    
    def enhance_scene_with_overlays(self, input_path, output_path, scene_num, 
                                    add_logo=True, add_contour=False, add_scene_number=True,
                                    pause_frame_index=None, pause_duration=0.0, best_segment=None):
        """
        Améliore une scène en ajoutant des calques et une pause.
        
        Args:
            input_path: Chemin vers la vidéo d'entrée
            output_path: Chemin pour la vidéo de sortie
            scene_num: Numéro de la scène
            add_logo: Ajouter le logo
            add_contour: Ajouter le contour autour de la pose
            add_scene_number: Ajouter le numéro de scène
            pause_frame_index: Index de frame pour la pause
            pause_duration: Durée de la pause en secondes
            best_segment: Informations sur le meilleur segment
            
        Returns:
            Chemin vers la vidéo améliorée
        """
        # Vérifier si la vidéo d'entrée existe
        if not os.path.isfile(input_path):
            self.log(f"Erreur: Fichier vidéo non trouvé: {input_path}")
            return None
        
        # Créer un répertoire temporaire pour les frames
        frames_dir = os.path.join(self.temp_dir, f"frames_scene{scene_num}")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extraire les frames de la vidéo
        self.log(f"Extraction des frames de {input_path}...")
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculer la frame où se trouve le point clé
        key_frame = None
        if best_segment and pause_frame_index is not None:
            # Conversion de l'index de frame dans la vidéo source vers l'index dans la vidéo extraite
            scene_start_time = best_segment['start_time']
            key_time = best_segment['reference_time']
            key_time_offset = key_time - scene_start_time
            key_frame = int(key_time_offset * fps)
            self.log(f"Frame clé calculée: {key_frame} (temps: {key_time_offset:.2f}s)")
        
        # Positions pour le logo et le texte
        logo_scale = 0.15
        logo_position = (width - int(self.logo_image.shape[1] * logo_scale) - 20, 20)
        text_position = (20, height - 30)
        
        # Position pour le contour (centré sur la pose)
        contour_scale = 0.8
        if add_contour and best_segment and 'reference_frame' in best_segment:
            # Calculer le centrage du contour sur la pose
            ref_frame = best_segment['reference_frame']
            ref_h, ref_w = ref_frame.shape[:2]
            
            # Ajuster l'échelle du contour si nécessaire
            scaled_contour_w = int(self.contour_image.shape[1] * contour_scale)
            scaled_contour_h = int(self.contour_image.shape[0] * contour_scale)
            
            # Centrer le contour sur l'image
            contour_x = (width - scaled_contour_w) // 2
            contour_y = (height - scaled_contour_h) // 2
            
            contour_position = (contour_x, contour_y)
        else:
            # Position par défaut si pas d'information sur la pose
            contour_position = (width // 4, height // 4)
        
        # Variable pour suivre si nous sommes en pause
        in_pause = False
        pause_end_frame = -1
        if pause_duration > 0 and key_frame is not None:
            pause_end_frame = key_frame + int(pause_duration * fps)
            self.log(f"Pause planifiée: frames {key_frame} à {pause_end_frame}")
        
        # Traiter chaque frame
        frames_with_overlays = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Si nous sommes en pause, dupliquer la frame clé
            if pause_duration > 0 and key_frame is not None:
                if frame_idx == key_frame:
                    in_pause = True
                    pause_frame = frame.copy()
                    
                    # Ajouter les calques à la frame de pause
                    if add_logo:
                        pause_frame = self.overlay_image(
                            pause_frame, self.logo_image, 
                            *logo_position, scale=logo_scale
                        )
                    
                    if add_scene_number:
                        pause_frame = self.add_text_overlay(
                            pause_frame, f"Scène {scene_num}", 
                            text_position, font_scale=1.2
                        )
                    
                    if add_contour:
                        pause_frame = self.overlay_image(
                            pause_frame, self.contour_image, 
                            *contour_position, scale=contour_scale
                        )
                    
                    # Sauvegarder la frame clé multiple fois pour créer la pause
                    pause_frames_count = int(pause_duration * fps)
                    for p in range(pause_frames_count):
                        pause_frame_path = os.path.join(frames_dir, f"frame_{frame_idx + p:05d}.jpg")
                        cv2.imwrite(pause_frame_path, pause_frame)
                        frames_with_overlays.append(pause_frame_path)
                    
                    # Ajuster l'index de frame
                    frame_idx += pause_frames_count
                    continue
                
                # Sauter les frames jusqu'à la fin de la pause
                if in_pause and frame_idx < pause_end_frame:
                    frame_idx += 1
                    continue
                    
                if frame_idx == pause_end_frame:
                    in_pause = False
            
            # Appliquer les calques à la frame courante
            if add_logo:
                frame = self.overlay_image(
                    frame, self.logo_image, 
                    *logo_position, scale=logo_scale
                )
            
            if add_scene_number:
                frame = self.add_text_overlay(
                    frame, f"Scène {scene_num}", 
                    text_position, font_scale=1.2
                )
            
            # Ajouter le contour si demandé
            if add_contour:
                frame = self.overlay_image(
                    frame, self.contour_image, 
                    *contour_position, scale=contour_scale
                )
            
            # Sauvegarder la frame
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames_with_overlays.append(frame_path)
            
            frame_idx += 1
        
        cap.release()
        
        # Créer la vidéo à partir des frames modifiées
        self.log(f"Création de la vidéo améliorée à partir de {len(frames_with_overlays)} frames...")
        
        # Utiliser FFmpeg pour assembler les frames
        if frames_with_overlays:
            # Le premier frame pour le format
            sample_frame = cv2.imread(frames_with_overlays[0])
            h, w = sample_frame.shape[:2]
            
            # Créer la vidéo avec FFmpeg
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(frames_dir, "frame_%05d.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",
                "-preset", "fast",
                output_path
            ]
            
            self.log(f"Exécution de la commande: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                self.log(f"Erreur lors de la création de la vidéo: {result.stderr.decode()}")
                return None
            
            self.log(f"Vidéo améliorée créée: {output_path}")
            return output_path
        else:
            self.log("Aucune frame générée pour la vidéo améliorée")
            return None
    
    def assemble_final_video(self, template_video, enhanced_scenes, output_path):
        """
        Assemble la vidéo finale en remplaçant plusieurs scènes dans le template.
        
        Args:
            template_video: Chemin vers la vidéo template
            enhanced_scenes: Liste de dictionnaires {scene_num, video_path} pour les scènes améliorées
            output_path: Chemin pour la vidéo finale
            
        Returns:
            Chemin vers la vidéo finale
        """
        # Créer une liste de segments vidéo temporaires
        segments = []
        
        # Trier les scènes par ordre croissant
        enhanced_scenes.sort(key=lambda x: x["scene_num"])
        
        # Obtenir les informations sur toutes les scènes du template
        scene_analyzer = self.pose_replacer.analyzer
        all_scenes = scene_analyzer.scene_info
        
        # Créer une liste de segments à concaténer
        start_time = 0
        
        for i, scene in enumerate(all_scenes):
            scene_num = i + 1  # Les numéros de scène commencent à 1
            scene_start = scene["start_time"]
            scene_end = scene["end_time"]
            
            # Vérifier si cette scène doit être remplacée
            enhanced_scene = next((s for s in enhanced_scenes if s["scene_num"] == scene_num), None)
            
            if scene_start > start_time:
                # Ajouter le segment du template avant cette scène
                segment_path = os.path.join(self.temp_dir, f"template_segment_{start_time:.2f}s_{scene_start:.2f}s.mp4")
                
                self.log(f"Extraction du segment template: {start_time:.2f}s - {scene_start:.2f}s")
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", template_video,
                    "-ss", str(start_time),
                    "-to", str(scene_start),
                    "-c", "copy",
                    segment_path
                ]
                
                subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                segments.append(segment_path)
            
            if enhanced_scene:
                # Utiliser la scène améliorée
                self.log(f"Utilisation de la scène améliorée {scene_num}")
                segments.append(enhanced_scene["video_path"])
            else:
                # Utiliser la scène originale du template
                segment_path = os.path.join(self.temp_dir, f"template_scene_{scene_num}.mp4")
                
                self.log(f"Extraction de la scène originale {scene_num}: {scene_start:.2f}s - {scene_end:.2f}s")
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", template_video,
                    "-ss", str(scene_start),
                    "-to", str(scene_end),
                    "-c", "copy",
                    segment_path
                ]
                
                subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                segments.append(segment_path)
            
            start_time = scene_end
        
        # Ajouter le segment final après la dernière scène
        video_info = scene_analyzer.get_video_info()
        video_duration = video_info["duration"]
        
        if start_time < video_duration:
            segment_path = os.path.join(self.temp_dir, f"template_segment_{start_time:.2f}s_end.mp4")
            
            self.log(f"Extraction du segment final: {start_time:.2f}s - {video_duration:.2f}s")
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", template_video,
                "-ss", str(start_time),
                "-to", str(video_duration),
                "-c", "copy",
                segment_path
            ]
            
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            segments.append(segment_path)
        
        # Créer un fichier de liste pour FFmpeg
        concat_list_path = os.path.join(self.temp_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for segment in segments:
                f.write(f"file '{os.path.abspath(segment)}'\n")
        
        # Concaténer tous les segments
        self.log(f"Concaténation de {len(segments)} segments vidéo...")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            self.log(f"Erreur lors de la concaténation: {result.stderr.decode()}")
            return None
        
        self.log(f"Vidéo finale assemblée: {output_path}")
        return output_path
    
    def cleanup(self):
        """Nettoie les fichiers temporaires."""
        if os.path.exists(self.temp_dir):
            self.log("Nettoyage des fichiers temporaires...")
            shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)


def main():
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(description="Outil avancé d'édition vidéo avec remplacement de scènes multiples")
    parser.add_argument("--template-video", required=True, help="Chemin vers la vidéo template")
    parser.add_argument("--source-video", required=True, help="Chemin vers la vidéo source (rush)")
    parser.add_argument("--reference-image", required=True, help="Chemin vers l'image de référence pour la détection de pose")
    parser.add_argument("--scene1", type=int, required=True, help="Numéro de la première scène à remplacer (avec logo uniquement)")
    parser.add_argument("--scene2", type=int, required=True, help="Numéro de la deuxième scène à remplacer (avec logo, contour et pause)")
    parser.add_argument("--logo", required=True, help="Chemin vers l'image du logo (PNG avec transparence)")
    parser.add_argument("--contour", required=True, help="Chemin vers l'image du contour (PNG avec transparence)")
    parser.add_argument("--pose-model", default="yolo11x-pose.pt", help="Chemin vers le modèle de pose YOLOv11")
    parser.add_argument("--pause-duration", type=float, default=2.0, help="Durée de la pause en secondes")
    parser.add_argument("--image-scale", type=float, default=1.0, help="Facteur d'échelle pour l'image extraite (0.5-2.0)")
    parser.add_argument("--output", help="Chemin pour la vidéo de sortie")
    parser.add_argument("--cleanup", action="store_true", help="Nettoyer les fichiers temporaires après traitement")
    parser.add_argument("--quiet", action="store_true", help="Mode silencieux (moins de messages)")
    
    args = parser.parse_args()
    
    try:
        print(f"Date et heure actuelles: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Utilisateur: {os.environ.get('USER', 'FETHl')}")
        
        # Vérifier que les fichiers existent
        for file_path in [args.template_video, args.source_video, args.reference_image, args.logo, args.contour]:
            if not os.path.isfile(file_path):
                print(f"Erreur: Fichier non trouvé: {file_path}")
                return 1
        
        # Vérifier que le facteur d'échelle est valide
        if args.image_scale <= 0 or args.image_scale > 2.0:
            print(f"Erreur: Le facteur d'échelle doit être entre 0.1 et 2.0 (valeur actuelle: {args.image_scale})")
            return 1
        
        # Créer l'outil d'édition vidéo avancé
        enhancer = EnhancedVideoReplacer(
            args.template_video,
            args.source_video,
            args.reference_image,
            args.logo,
            args.contour,
            pose_model_path=args.pose_model,
            verbose=not args.quiet,
            image_scale=args.image_scale
        )
        
        # Créer la vidéo améliorée
        start_time = time.time()
        output_path = enhancer.create_enhanced_video(
            scene_numbers=[args.scene1, args.scene2],
            pause_duration=args.pause_duration,
            output_path=args.output
        )
        
        elapsed_time = time.time() - start_time
        
        if output_path:
            print(f"\nVidéo améliorée créée avec succès en {elapsed_time:.2f} secondes:")
            print(f"- Chemin: {output_path}")
            print(f"- Scène {args.scene1}: Avec logo uniquement")
            print(f"- Scène {args.scene2}: Avec logo, contour et pause de {args.pause_duration}s")
            print(f"- Facteur d'échelle de l'image: {args.image_scale}")
        else:
            print("\nErreur lors de la création de la vidéo améliorée")
        
        # Nettoyer les fichiers temporaires si demandé
        if args.cleanup:
            enhancer.cleanup()
        
    except Exception as e:
        print(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()