import cv2
import numpy as np
import os
import argparse
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from datetime import timedelta
import time

class VideoSceneFrameExtractor:
    """
    A class for detecting scenes in a video and extracting the best frame
    from each scene based on sharpness and stability.
    """
    
    def __init__(self, video_path, output_dir="extracted_frames", 
                 threshold=27.0, frame_window=3, verbose=True):
        """
        Initialize the scene detector and frame extractor.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            threshold: Threshold for scene detection sensitivity
            frame_window: Window in seconds around middle point to analyze
            verbose: Whether to print progress information
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.threshold = threshold
        self.frame_window = frame_window
        self.verbose = verbose
        self.scenes = []
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get video properties
        video = cv2.VideoCapture(video_path)
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        video.release()
            
        if self.verbose:
            print(f"Initialized extractor for {os.path.basename(video_path)}")
            print(f"Video FPS: {self.fps}, Duration: {timedelta(seconds=self.duration)}")
    
    def log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def detect_scenes(self):
        """
        Detect scenes in the video file using PySceneDetect.
        
        Returns:
            List of scene boundaries (start, end) in seconds
        """
        self.log(f"Detecting scenes in {self.video_path}...")
        
        # Create video & scene managers
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        
        # Add content detector
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))
        
        # Improve processing speed by downscaling
        video_manager.set_downscale_factor()
        
        # Start video manager
        video_manager.start()
        
        # Detect scenes
        scene_manager.detect_scenes(frame_source=video_manager)
        
        # Get scene list
        scene_list = scene_manager.get_scene_list()
        
        # Convert to seconds
        self.scenes = []
        
        for scene in scene_list:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            self.scenes.append((start_time, end_time))
            
        video_manager.release()
        
        self.log(f"Detected {len(self.scenes)} scenes")
        return self.scenes
    
    def calculate_sharpness(self, frame):
        """
        Calculate the sharpness of a frame using Laplacian variance.
        Higher value = sharper image.
        """
        if frame is None:
            return 0
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def calculate_stability(self, prev_frame, curr_frame):
        """
        Calculate the stability between two frames using optical flow.
        Lower value = more stable.
        """
        if prev_frame is None or curr_frame is None:
            return float('inf')
            
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate magnitude of flow vectors
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Return average magnitude as a measure of instability
        return np.mean(mag)
    
    def extract_best_frame(self, scene_start, scene_end):
        """
        Extract the best frame from a scene based on sharpness and stability.
        
        Args:
            scene_start: Start time of the scene in seconds
            scene_end: End time of the scene in seconds
            
        Returns:
            The best frame and its timestamp
        """
        # Calculate the middle of the scene
        mid_time = (scene_start + scene_end) / 2
        
        # Define window around middle point
        window_start = max(scene_start, mid_time - self.frame_window / 2)
        window_end = min(scene_end, mid_time + self.frame_window / 2)
        
        self.log(f"Analyzing window {window_start:.2f}s - {window_end:.2f}s")
        
        cap = cv2.VideoCapture(self.video_path)
        
        # Convert time to frame numbers
        start_frame = int(window_start * self.fps)
        end_frame = int(window_end * self.fps)
        
        # Set video to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        best_score = -float('inf')
        best_frame = None
        best_timestamp = None
        prev_frame = None
        
        # Analyze frames in the window
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate metrics
            sharpness = self.calculate_sharpness(frame)
            stability = float('inf') if prev_frame is None else self.calculate_stability(prev_frame, frame)
            
            # Normalize stability (lower is better, so invert it)
            normalized_stability = 1 / (stability + 1e-6)  # Add small epsilon to avoid division by zero
            
            # Combined score (weighted sum)
            # You can adjust these weights based on what's more important
            score = 0.7 * sharpness + 0.3 * normalized_stability
            
            timestamp = frame_idx / self.fps
            
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_timestamp = timestamp
                
            prev_frame = frame.copy()
            
        cap.release()
        
        return best_frame, best_timestamp
    
    def process_video(self):
        """
        Process the entire video: detect scenes and extract the best frame from each.
        
        Returns:
            List of tuples (scene_number, frame_timestamp, frame_path)
        """
        if not self.scenes:
            self.detect_scenes()
            
        results = []
        
        for i, (start_time, end_time) in enumerate(self.scenes):
            scene_num = i + 1
            self.log(f"Processing scene {scene_num}/{len(self.scenes)}: {start_time:.2f}s - {end_time:.2f}s")
            
            best_frame, timestamp = self.extract_best_frame(start_time, end_time)
            
            if best_frame is not None:
                # Define output path for this frame
                frame_path = os.path.join(
                    self.output_dir, 
                    f"scene_{scene_num:03d}_frame_{timestamp:.2f}.jpg"
                )
                
                # Save the frame
                cv2.imwrite(frame_path, best_frame)
                self.log(f"Saved best frame at {timestamp:.2f}s to {frame_path}")
                
                results.append((scene_num, timestamp, frame_path))
            else:
                self.log(f"Failed to extract frame for scene {scene_num}")
                
        return results


def main():
    """Command-line interface for the video scene frame extractor"""
    parser = argparse.ArgumentParser(description='Extract the best frame from each scene in a video')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--output-dir', default='extracted_frames', help='Directory to save extracted frames')
    parser.add_argument('--threshold', type=float, default=27.0, help='Threshold for scene detection')
    parser.add_argument('--frame-window', type=float, default=3.0, help='Window in seconds around middle point to analyze')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = VideoSceneFrameExtractor(
        args.video_path, 
        output_dir=args.output_dir,
        threshold=args.threshold,
        frame_window=args.frame_window,
        verbose=not args.quiet
    )
    
    # Process video
    start_time = time.time()
    results = extractor.process_video()
    elapsed_time = time.time() - start_time
    
    # Print results
    if not args.quiet:
        print("\nExtraction completed in {:.2f} seconds".format(elapsed_time))
        print(f"Extracted {len(results)} frames:")
        for scene_num, timestamp, path in results:
            print(f"Scene {scene_num}: Frame at {timestamp:.2f}s - {path}")


if __name__ == "__main__":
    main()
