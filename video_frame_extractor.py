import cv2
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from skimage.metrics import structural_similarity as ssim

class VideoFrameExtractor:
    def __init__(self, video_path):
        """
        Initialize the VideoFrameExtractor with a video file.
        
        Args:
            video_path (str): Path to the video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.sharp_frames = []
        self.selected_index = None
        
        print(f"Vidéo chargée: {self.total_frames} frames, {self.fps} FPS, {self.width}x{self.height}")
        print(f"Durée totale: {self.total_frames / self.fps:.2f} secondes")
    
    def extract_quality_frames(self, sharpness_threshold=100, stability_threshold=0.8, 
                               detail_threshold=50, sample_interval=10, max_frames=100):
        """
        Extract high quality frames from the video based on sharpness, stability and detail.
        
        Args:
            sharpness_threshold (int): Minimum laplacian variance to consider a frame sharp
            stability_threshold (float): Minimum SSIM score for stability (0-1)
            detail_threshold (int): Minimum detail score for high resolution quality
            sample_interval (int): Sample every Nth frame to speed up processing
            max_frames (int): Maximum number of quality frames to extract
            
        Returns:
            list: List of tuples containing (frame_index, frame_image, quality_score)
        """
        self.sharp_frames = []
        quality_frames = []
        prev_frame = None
        
        print(f"\n🔍 Extraction des images de haute qualité (échantillonnage tous les {sample_interval} frames)...")
        print("Critères d'analyse: netteté, stabilité et richesse de détails")
        
        try:
            # Sequential processing (safe, no multithreading)
            for i in range(0, self.total_frames, sample_interval):
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = self.cap.read()
                    if not ret:
                        print(f"Impossible de lire la frame {i}. Arrêt de l'extraction.")
                        break
                    
                    # Convert to grayscale for analysis
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # 1. Calculate sharpness using Laplacian variance
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    sharpness_score = laplacian.var()
                    
                    # 2. Calculate stability (compared to previous frame)
                    stability_score = 0
                    if prev_frame is not None:
                        # Calculate SSIM between current and previous frame
                        # Higher SSIM = more similar = more stable
                        try:
                            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                            stability_score = ssim(gray, prev_gray)
                        except Exception as e:
                            print(f"Erreur lors du calcul de stabilité: {e}")
                            stability_score = 0
                    
                    # Save current frame for next iteration
                    prev_frame = frame.copy()
                    
                    # 3. Calculate detail score using gradient magnitude
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                    detail_score = np.mean(gradient_magnitude)
                    
                    # 4. Calculate noise level (lower is better)
                    blur = cv2.GaussianBlur(gray, (7, 7), 0)
                    noise_level = np.mean(np.abs(gray.astype(np.float32) - blur.astype(np.float32)))
                    noise_score = 1.0 / (1.0 + noise_level)  # Invert so higher is better
                    
                    # Normalize scores to 0-100 range (approximate based on typical values)
                    normalized_sharpness = min(100, sharpness_score / 5)
                    normalized_stability = stability_score * 100
                    normalized_detail = min(100, detail_score / 2)
                    normalized_noise = noise_score * 100
                    
                    # Create weighted total score
                    # Weight sharpness more heavily as it's critical
                    total_score = (normalized_sharpness * 0.5 + 
                                  normalized_stability * 0.3 + 
                                  normalized_detail * 0.1 +
                                  normalized_noise * 0.1)
                    
                    # Create score dictionary for detailed analysis
                    score_dict = {
                        'sharpness': normalized_sharpness,
                        'stability': normalized_stability,
                        'detail': normalized_detail,
                        'noise': normalized_noise,
                        'total': total_score
                    }
                    
                    # If total score is good enough, add to our collection
                    if total_score > sharpness_threshold:
                        quality_frames.append((i, frame, score_dict))
                    
                    # Progress update
                    if i % (sample_interval * 50) == 0:
                        print(f"Traitement frame {i}/{self.total_frames} - Trouvé {len(quality_frames)} images de qualité")
                        
                    # Stop if we've found enough frames
                    if len(quality_frames) >= max_frames:
                        print(f"Nombre maximum d'images atteint ({max_frames})")
                        break
                        
                except Exception as e:
                    print(f"Erreur lors de l'analyse de la frame {i}: {e}")
                    continue
                
            # Sort frames by quality score (best first)
            quality_frames.sort(key=lambda x: x[2]['total'], reverse=True)
            
            # Store only frame_index, frame, and total_score for compatibility
            self.sharp_frames = [(idx, frame, scores['total']) for idx, frame, scores in quality_frames]
            
            print(f"\n✅ Extraction terminée: {len(self.sharp_frames)} images de haute qualité trouvées")
            
            # Display top 3 scores for reference
            if self.sharp_frames:
                print("\nMeilleures images (scores de qualité):")
                for i in range(min(3, len(quality_frames))):
                    idx, _, scores = quality_frames[i]
                    time = idx / self.fps
                    print(f"  {i+1}. Frame {idx} (temps: {time:.2f}s)")
                    print(f"     Netteté: {scores['sharpness']:.2f}, Stabilité: {scores['stability']:.2f}, Détails: {scores['detail']:.2f}")
                    print(f"     Score total: {scores['total']:.2f}")
        
        except Exception as e:
            print(f"Erreur critique pendant l'extraction des images: {e}")
            # En cas d'erreur, essayons une méthode plus simple
            print("Tentative avec une méthode de repli plus simple...")
            self._extract_simple_sharp_frames(sample_interval, max_frames)
            
        return self.sharp_frames
    
    def _extract_simple_sharp_frames(self, sample_interval=10, max_frames=100):
        """
        Fallback method to extract frames using a simpler approach.
        """
        self.sharp_frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print("Extraction des images avec méthode simplifiée...")
        
        for i in range(0, self.total_frames, sample_interval):
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Simple sharpness calculation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = laplacian.var()
                
                self.sharp_frames.append((i, frame, sharpness))
                
                if len(self.sharp_frames) % 10 == 0:
                    print(f"Traitement: {i}/{self.total_frames} - Trouvé {len(self.sharp_frames)} images")
                
                if len(self.sharp_frames) >= max_frames:
                    break
                    
            except Exception as e:
                print(f"Erreur à la frame {i}: {e}")
                continue
        
        # Sort by sharpness
        self.sharp_frames.sort(key=lambda x: x[2], reverse=True)
        print(f"Extraction terminée: {len(self.sharp_frames)} images trouvées")
    
    def select_frame_interactive(self, num_frames=10, frames_per_page=10):
        """
        Display sharp frames and allow interactive selection.
        
        Args:
            num_frames (int): Number of frames to display
            frames_per_page (int): Number of frames to show per page
            
        Returns:
            int: Index of selected frame in sharp_frames list
        """
        if not self.sharp_frames:
            print("Aucune image de qualité extraite. Exécutez extract_quality_frames() d'abord.")
            return None
            
        # Limit to the number of available frames
        frames_to_show = min(num_frames, len(self.sharp_frames))
        
        # Create an interactive frame selector
        selector = FrameSelector(self.sharp_frames[:frames_to_show], self.fps, frames_per_page)
        self.selected_index = selector.select_frame()
        
        if self.selected_index is not None:
            frame_idx = self.sharp_frames[self.selected_index][0]
            time_sec = frame_idx / self.fps
            print(f"✅ Image sélectionnée {self.selected_index} (frame vidéo {frame_idx}, temps: {time_sec:.2f}s)")
        else:
            print("❌ Aucune image n'a été sélectionnée.")
        
        return self.selected_index
    
    def extract_centered_clip(self, reference_index=None, total_duration=10, output_path=None):
        """
        Extract a video clip centered around a reference frame, adjusting for video boundaries.
        
        Args:
            reference_index (int): Index in the sharp_frames list to center around (None to use selected_index)
            total_duration (float): Total duration of the clip in seconds
            output_path (str): Path to save the output video (default: auto-generated)
            
        Returns:
            str: Path to the extracted video clip
        """
        # Use the interactively selected index if no reference_index is provided
        if reference_index is None:
            reference_index = self.selected_index
            
        if reference_index is None or not self.sharp_frames or reference_index >= len(self.sharp_frames):
            raise ValueError("Aucune image de référence valide. Veuillez d'abord sélectionner une image.")
            
        frame_index = self.sharp_frames[reference_index][0]
        frame_time = frame_index / self.fps
        
        # Calculate before and after durations based on position in video
        half_duration = total_duration / 2
        video_duration = self.total_frames / self.fps
        
        # Initialize with balanced durations
        before_seconds = half_duration
        after_seconds = half_duration
        
        # Adjust if reference frame is close to video boundaries
        if frame_time < half_duration:
            # Near the beginning of the video
            before_seconds = frame_time
            after_seconds = total_duration - before_seconds
        elif frame_time > video_duration - half_duration:
            # Near the end of the video
            after_seconds = video_duration - frame_time
            before_seconds = total_duration - after_seconds
        
        # Calculate start and end times
        start_time = max(0, frame_time - before_seconds)
        end_time = min(video_duration, frame_time + after_seconds)
        
        # If we hit a boundary, try to extend in the other direction to maintain total duration
        actual_duration = end_time - start_time
        if actual_duration < total_duration:
            if start_time == 0:
                # We hit the beginning, extend the end if possible
                end_time = min(video_duration, end_time + (total_duration - actual_duration))
            elif end_time == video_duration:
                # We hit the end, extend the beginning if possible
                start_time = max(0, start_time - (total_duration - actual_duration))
        
        # Calculate start and end frames
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        # Generate output path if not provided
        if output_path is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            output_path = f"clip_{base_name}_{timestamp_str}.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # Write frames to the output video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print("\n📋 RÉSUMÉ DE L'EXTRACTION:")
        print(f"• Position de l'image de référence: {frame_time:.2f}s (frame {frame_index})")
        print(f"• Durée totale demandée: {total_duration:.2f}s")
        print(f"• Durée avant l'image de référence: {frame_time-start_time:.2f}s")
        print(f"• Durée après l'image de référence: {end_time-frame_time:.2f}s")
        print(f"• Segment extrait: {start_time:.2f}s à {end_time:.2f}s (durée totale: {end_time-start_time:.2f}s)")
        print(f"• Frames extraites: {start_frame} à {end_frame} ({end_frame-start_frame+1} frames)")
        
        frames_processed = 0
        total_frames = end_frame - start_frame + 1
        
        print("\n🎬 EXTRACTION EN COURS...")
        for i in range(start_frame, end_frame + 1):
            ret, frame = self.cap.read()
            if not ret:
                print(f"Erreur de lecture à la frame {i}, arrêt de l'extraction")
                break
                
            # Draw a marker on the reference frame
            if i == frame_index:
                # Add a red border around the reference frame
                border_thickness = 10
                frame[:border_thickness, :] = [0, 0, 255]  # Top
                frame[-border_thickness:, :] = [0, 0, 255]  # Bottom
                frame[:, :border_thickness] = [0, 0, 255]  # Left
                frame[:, -border_thickness:] = [0, 0, 255]  # Right
                
                # Add text label
                cv2.putText(frame, "IMAGE DE REFERENCE", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out.write(frame)
            
            # Progress update
            frames_processed += 1
            if frames_processed % 100 == 0 or frames_processed == total_frames:
                progress = (frames_processed / total_frames) * 100
                print(f"Traitement: {frames_processed}/{total_frames} frames ({progress:.1f}%)")
        
        out.release()
        print(f"\n✅ Clip sauvegardé: {output_path}")
        
        return output_path
    
    def process_video_auto(self, total_duration=10, quality_threshold=50, sample_interval=15, max_frames=50):
        """
        Process a video automatically by extracting quality frames, showing them for selection,
        and then extracting a clip around the selected frame.
        
        Args:
            total_duration (float): Total duration of the clip in seconds
            quality_threshold (float): Minimum quality score to consider a frame
            sample_interval (int): Sample every Nth frame to speed up processing
            max_frames (int): Maximum number of quality frames to extract
            
        Returns:
            str: Path to the extracted video clip
        """
        print("🔍 PHASE 1: EXTRACTION DES IMAGES DE HAUTE QUALITÉ...")
        self.extract_quality_frames(
            sharpness_threshold=quality_threshold,
            sample_interval=sample_interval,
            max_frames=max_frames
        )
        
        print("\n🖱️ PHASE 2: SÉLECTION DE L'IMAGE DE RÉFÉRENCE...")
        print("Cliquez sur une image pour la sélectionner comme référence.")
        self.select_frame_interactive(num_frames=max_frames, frames_per_page=9)
        
        if self.selected_index is None:
            print("❌ Opération annulée: aucune image n'a été sélectionnée.")
            return None
        
        print("\n✂️ PHASE 3: EXTRACTION DU CLIP VIDÉO...")
        return self.extract_centered_clip(total_duration=total_duration)
    
    def save_sharp_frame(self, index=None, output_path=None):
        """
        Save a specific sharp frame as an image.
        
        Args:
            index (int): Index in the sharp_frames list (None to use selected_index)
            output_path (str): Path to save the image (default: auto-generated)
            
        Returns:
            str: Path to the saved image
        """
        # Use the interactively selected index if no index is provided
        if index is None:
            index = self.selected_index
            
        if index is None or not self.sharp_frames or index >= len(self.sharp_frames):
            raise ValueError("Invalid index or no sharp frames extracted")
        
        frame_index, frame, _ = self.sharp_frames[index]
        
        if output_path is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            output_path = f"frame_{base_name}_{timestamp_str}.jpg"
        
        cv2.imwrite(output_path, frame)
        print(f"Image sauvegardée: {output_path}")
        
        return output_path
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


class FrameSelector:
    """Helper class to display and select frames interactively"""
    
    def __init__(self, frames, fps, frames_per_page=9):
        """
        Initialize the frame selector.
        
        Args:
            frames: List of (frame_index, frame_image, sharpness) tuples
            fps: Frames per second of the video
            frames_per_page: Number of frames to show per page
        """
        self.frames = frames
        self.fps = fps
        self.frames_per_page = frames_per_page
        self.selected_index = None
        self.current_page = 0
        self.total_pages = (len(frames) + frames_per_page - 1) // frames_per_page
        
    def select_frame(self):
        """
        Display frames and allow user to select one.
        
        Returns:
            int: Index of selected frame or None if canceled
        """
        # Setup the figure
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title('Sélection d\'image de référence')
        
        # Draw the initial page
        self._draw_page()
        
        # Start the interactive loop
        plt.show()
        
        return self.selected_index
    
    def _draw_page(self):
        """Draw the current page of frames"""
        # Clear the figure
        self.fig.clf()
        
        # Calculate start and end indices for the current page
        start_idx = self.current_page * self.frames_per_page
        end_idx = min(start_idx + self.frames_per_page, len(self.frames))
        num_frames_on_page = end_idx - start_idx
        
        # Set the title with page info
        self.fig.suptitle(f"Cliquez sur une image pour la sélectionner (Page {self.current_page+1}/{self.total_pages})", 
                          fontsize=16)
        
        # Calculate grid layout
        cols = 3
        rows = (num_frames_on_page + cols - 1) // cols
        
        # Create subplots for each frame
        self.axes = []
        for i in range(num_frames_on_page):
            idx = start_idx + i
            frame_idx, frame, quality = self.frames[idx]
            time_sec = frame_idx / self.fps
            
            # Create subplot
            ax = self.fig.add_subplot(rows, cols, i+1)
            self.axes.append(ax)
            
            # Convert from BGR to RGB for matplotlib
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the image
            ax.imshow(rgb_frame)
            
            # Use quality score if it's a number, otherwise use as a label
            if isinstance(quality, (int, float)):
                quality_text = f"Qualité: {quality:.1f}"
            else:
                quality_text = str(quality)
                
            ax.set_title(f"Index: {idx}, Frame: {frame_idx}\nTime: {time_sec:.2f}s, {quality_text}")
            ax.axis('off')
            
            # Store the frame index in the axis for later retrieval
            ax.frame_index = idx
        
        # Add navigation buttons (only if there are multiple pages)
        if self.total_pages > 1:
            self.fig.subplots_adjust(bottom=0.15)
            
            # Previous button
            prev_ax = self.fig.add_axes([0.3, 0.05, 0.1, 0.05])
            self.prev_button = Button(prev_ax, 'Précédent')
            self.prev_button.on_clicked(self._on_prev)
            
            # Next button
            next_ax = self.fig.add_axes([0.6, 0.05, 0.1, 0.05])
            self.next_button = Button(next_ax, 'Suivant')
            self.next_button.on_clicked(self._on_next)
        
        # Connect the click event for image selection
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Adjust layout
        self.fig.tight_layout(rect=[0, 0.1, 1, 0.95])
        self.fig.canvas.draw_idle()
    
    def _on_click(self, event):
        """Handle click events on the figure"""
        # Check if click was on an image axis
        if event.inaxes in self.axes:
            # Get the frame index from the axis
            self.selected_index = event.inaxes.frame_index
            plt.close(self.fig)
    
    def _on_prev(self, event):
        """Go to previous page"""
        self.current_page = (self.current_page - 1) % self.total_pages
        self._draw_page()
    
    def _on_next(self, event):
        """Go to next page"""
        self.current_page = (self.current_page + 1) % self.total_pages
        self._draw_page()