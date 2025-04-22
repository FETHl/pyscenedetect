import cv2
import numpy as np
import os
import argparse
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from datetime import timedelta
import time

class SceneExtractor:
    """
    Class responsible for detecting scene boundaries in a video file.
    """
    
    def __init__(self, video_path, threshold=27.0, verbose=True):
        """
        Initialize the scene detector.
        
        Args:
            video_path: Path to the video file
            threshold: Threshold for scene detection sensitivity
            verbose: Whether to print progress information
        """
        self.video_path = video_path
        self.threshold = threshold
        self.verbose = verbose
        self.scenes = []
        
        # Get video properties
        video = cv2.VideoCapture(video_path)
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        video.release()
            
        if self.verbose:
            print(f"Initialized scene extractor for {os.path.basename(video_path)}")
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
    
    def get_video_properties(self):
        """
        Returns video properties.
        
        Returns:
            Dictionary containing video properties
        """
        return {
            'path': self.video_path,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration': self.duration
        }


class EnhancedFrameExtractor:
    """
    Class responsible for extracting and enhancing frames from scenes
    with CUDA acceleration and 4K resolution output.
    """
    
    def __init__(self, video_properties, output_dir="enhanced_frames", 
                 frame_window=3, verbose=True):
        """
        Initialize the frame extractor.
        
        Args:
            video_properties: Dictionary with video properties
            output_dir: Directory to save extracted frames
            frame_window: Window in seconds around middle point to analyze
            verbose: Whether to print progress information
        """
        self.video_path = video_properties['path']
        self.fps = video_properties['fps']
        self.frame_window = frame_window
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Check if CUDA is available
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        if self.cuda_available:
            self.log("CUDA is available for GPU acceleration")
            # Initialize CUDA streams for parallel processing
            self.cuda_stream = cv2.cuda_Stream()
        else:
            self.log("WARNING: CUDA is not available, falling back to CPU processing")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def calculate_sharpness(self, frame):
        """
        Calculate the sharpness of a frame using Laplacian variance.
        Uses GPU if available.
        
        Args:
            frame: Input frame
            
        Returns:
            Sharpness score (higher is better)
        """
        if frame is None:
            return 0
            
        if self.cuda_available:
            # GPU implementation
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Convert to grayscale
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian
            gpu_laplacian = cv2.cuda.createLaplacianFilter(cv2.CV_64F, 1)
            gpu_result = gpu_laplacian.apply(gpu_gray)
            
            # Download result
            laplacian = gpu_result.download()
            return laplacian.var()
        else:
            # CPU implementation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return laplacian.var()
    
    def calculate_stability(self, prev_frame, curr_frame):
        """
        Calculate the stability between two frames using optical flow.
        Uses GPU if available.
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            
        Returns:
            Stability score (lower is better)
        """
        if prev_frame is None or curr_frame is None:
            return float('inf')
        
        if self.cuda_available:
            # GPU implementation
            gpu_prev = cv2.cuda_GpuMat()
            gpu_curr = cv2.cuda_GpuMat()
            
            # Upload frames to GPU
            gpu_prev.upload(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
            gpu_curr.upload(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY))
            
            # Create optical flow object
            gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(
                5, 0.5, False, 15, 3, 5, 1.2, 0
            )
            
            # Calculate flow
            flow_gpu = gpu_flow.calc(gpu_prev, gpu_curr, None)
            
            # Download flow
            flow = flow_gpu.download()
            
            # Calculate magnitude
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            return np.mean(mag)
        else:
            # CPU implementation
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Calculate magnitude
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            return np.mean(mag)
    
    def enhance_image(self, frame):
        """
        Enhance the image quality with sharpening, contrast improvement,
        and dark area enhancement using GPU acceleration if available.
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame (4K resolution)
        """
        if frame is None:
            return None
            
        # Target 4K resolution
        target_width, target_height = 3840, 2160
        
        if self.cuda_available:
            # GPU implementation
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Resize to 4K if needed
            h, w = frame.shape[:2]
            if w != target_width or h != target_height:
                gpu_frame = cv2.cuda.resize(gpu_frame, (target_width, target_height), 
                                           interpolation=cv2.INTER_LANCZOS4)
            
            # Convert to Lab color space for better enhancement
            gpu_lab = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2Lab)
            
            # Split into channels
            gpu_l, gpu_a, gpu_b = cv2.cuda.split(gpu_lab)
            
            # Apply CLAHE to L channel for contrast enhancement
            clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gpu_l = clahe.apply(gpu_l, stream=self.cuda_stream)
            
            # Merge back
            gpu_lab = cv2.cuda.merge([gpu_l, gpu_a, gpu_b])
            
            # Convert back to BGR
            gpu_enhanced = cv2.cuda.cvtColor(gpu_lab, cv2.COLOR_Lab2BGR)
            
            # Apply unsharp mask for sharpening
            gpu_blurred = cv2.cuda.GaussianBlur(gpu_frame, (0, 0), 3)
            gpu_sharpened = cv2.cuda.addWeighted(gpu_frame, 1.5, gpu_blurred, -0.5, 0)
            
            # Download result
            enhanced_frame = gpu_sharpened.download()
            
            return enhanced_frame
        else:
            # CPU implementation
            
            # Resize to 4K if needed
            h, w = frame.shape[:2]
            if w != target_width or h != target_height:
                frame = cv2.resize(frame, (target_width, target_height), 
                                  interpolation=cv2.INTER_LANCZOS4)
            
            # Convert to Lab color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            lab = cv2.merge([l, a, b])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
            
            # Apply unsharp mask for sharpening
            gaussian = cv2.GaussianBlur(frame, (0, 0), 3)
            enhanced = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
            
            return enhanced
    
    def extract_best_frame(self, scene_start, scene_end):
        """
        Extract the best frame from a scene based on sharpness and stability.
        
        Args:
            scene_start: Start time of the scene in seconds
            scene_end: End time of the scene in seconds
            
        Returns:
            The best enhanced frame and its timestamp
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
            score = 0.7 * sharpness + 0.3 * normalized_stability
            
            timestamp = frame_idx / self.fps
            
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_timestamp = timestamp
                
            prev_frame = frame.copy()
            
        cap.release()
        
        # Enhance the best frame if found
        if best_frame is not None:
            self.log(f"Enhancing best frame at {best_timestamp:.2f}s")
            enhanced_frame = self.enhance_image(best_frame)
            return enhanced_frame, best_timestamp
        else:
            return None, None
    
    def process_scenes(self, scenes):
        """
        Process scenes and extract the best frame from each.
        
        Args:
            scenes: List of scene boundaries (start, end) in seconds
            
        Returns:
            List of tuples (scene_number, frame_timestamp, frame_path)
        """
        results = []
        
        for i, (start_time, end_time) in enumerate(scenes):
            scene_num = i + 1
            self.log(f"Processing scene {scene_num}/{len(scenes)}: {start_time:.2f}s - {end_time:.2f}s")
            
            best_frame, timestamp = self.extract_best_frame(start_time, end_time)
            
            if best_frame is not None:
                # Define output path for this frame
                frame_path = os.path.join(
                    self.output_dir, 
                    f"scene_{scene_num:03d}_frame_{timestamp:.2f}.jpg"
                )
                
                # Save the frame (use imwrite quality parameter for JPEG)
                cv2.imwrite(frame_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.log(f"Saved enhanced frame at {timestamp:.2f}s to {frame_path}")
                
                results.append((scene_num, timestamp, frame_path))
            else:
                self.log(f"Failed to extract frame for scene {scene_num}")
                
        return results


def main():
    """Command-line interface for the enhanced video scene frame extractor"""
    parser = argparse.ArgumentParser(
        description='Extract and enhance the best frame from each scene in a video')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--output-dir', default='enhanced_frames', 
                        help='Directory to save extracted frames')
    parser.add_argument('--threshold', type=float, default=27.0, 
                        help='Threshold for scene detection')
    parser.add_argument('--frame-window', type=float, default=3.0, 
                        help='Window in seconds around middle point to analyze')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Step 1: Create scene extractor and detect scenes
    scene_extractor = SceneExtractor(
        args.video_path, 
        threshold=args.threshold,
        verbose=not args.quiet
    )
    
    start_time = time.time()
    scenes = scene_extractor.detect_scenes()
    
    if not scenes:
        print("No scenes detected. Exiting.")
        return
    
    # Step 2: Create frame extractor and process each scene
    frame_extractor = EnhancedFrameExtractor(
        scene_extractor.get_video_properties(),
        output_dir=args.output_dir,
        frame_window=args.frame_window,
        verbose=not args.quiet
    )
    
    results = frame_extractor.process_scenes(scenes)
    elapsed_time = time.time() - start_time
    
    # Print results
    if not args.quiet:
        print("\nExtraction completed in {:.2f} seconds".format(elapsed_time))
        print(f"Extracted and enhanced {len(results)} frames:")
        for scene_num, timestamp, path in results:
            print(f"Scene {scene_num}: Frame at {timestamp:.2f}s - {path}")


if __name__ == "__main__":
    main()