import cv2
import numpy as np
import os
import argparse
import subprocess
import time
from datetime import timedelta
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim




class CudaMovementMatcher:
    """
    GPU-accelerated video matcher for detecting movement patterns and scene similarity.
    """
    
    def __init__(self, source_video_path, verbose=True, use_cuda=True):
        """Initialize the CUDA-accelerated matcher."""
        self.source_path = source_video_path
        self.verbose = verbose
        self.analyzer = SceneAnalyzer(source_video_path, verbose=verbose)
        
        # Default region of interest - full frame
        self.roi = None
        # Movement importance weight
        self.movement_weight = 0.7
        
        # Check CUDA availability
        self.use_cuda = use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_cuda:
            self.log(f"CUDA enabled with {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
            device_info = cv2.cuda.DeviceInfo()
            self.log(f"Using GPU: {device_info.name()}, Total memory: {device_info.totalMemory() / (1024**2):.1f} MB")
        else:
            self.log("CUDA not available or disabled, using CPU processing")
    
    def log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def set_region_of_interest(self, x, y, width, height):
        """
        Set a specific region of interest to focus on for movement detection.
        
        Args:
            x, y: Top-left coordinates of ROI
            width, height: Dimensions of ROI
        """
        self.roi = (x, y, width, height)
        self.log(f"Set ROI to ({x}, {y}, {width}, {height})")
        
    def set_movement_importance(self, weight=0.7):
        """
        Set how important movement matching is compared to other factors.
        
        Args:
            weight: Value between 0 and 1 (higher means more emphasis on movement)
        """
        self.movement_weight = max(0, min(1, weight))
        self.log(f"Set movement importance to {self.movement_weight}")
    
    def cuda_optical_flow(self, prev_gray, current_gray):
        """
        Calculate optical flow using CUDA acceleration.
        
        Args:
            prev_gray: Previous frame in grayscale
            current_gray: Current frame in grayscale
            
        Returns:
            Optical flow field
        """
        if self.use_cuda:
            # Upload images to GPU
            prev_gpu = cv2.cuda_GpuMat()
            curr_gpu = cv2.cuda_GpuMat()
            prev_gpu.upload(prev_gray)
            curr_gpu.upload(current_gray)
            
            # Create CUDA Farneback optical flow
            flow_gpu = cv2.cuda_FarnebackOpticalFlow.create(
                numLevels=3,
                pyrScale=0.5,
                fastPyramids=False,
                winSize=15,
                numIters=3,
                polyN=5,
                polySigma=1.2,
                flags=0
            )
            
            # Calculate flow
            flow_gpu = flow_gpu.calc(prev_gpu, curr_gpu, None)
            
            # Download result from GPU
            flow = flow_gpu.download()
            return flow
        else:
            # Fall back to CPU implementation
            return cv2.calcOpticalFlowFarneback(
                prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
    
    def cuda_edge_detection(self, gray_frame):
        """
        Perform edge detection using CUDA acceleration.
        
        Args:
            gray_frame: Input grayscale frame
            
        Returns:
            Edge map
        """
        if self.use_cuda:
            # Upload image to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(gray_frame)
            
            # Apply blur to reduce noise
            gpu_blur = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 1.0
            )
            gpu_blurred = gpu_blur.apply(gpu_frame)
            
            # Create CUDA Canny detector
            detector = cv2.cuda.createCannyEdgeDetector(100, 200)
            
            # Detect edges
            edges_gpu = detector.detect(gpu_blurred)
            
            # Download result
            edges = edges_gpu.download()
            return edges
        else:
            # Fall back to CPU implementation
            return cv2.Canny(gray_frame, 100, 200)
    
    def extract_frames_with_features(self, video_path, start_time, duration, sample_rate=4.0):
        """
        Extract frames and compute features with CUDA acceleration when available.
        
        Args:
            video_path: Path to the video
            start_time: Start time in seconds
            duration: Duration to extract in seconds
            sample_rate: Frames per second to extract (increased to 4.0)
            
        Returns:
            Dictionary with frames and features
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log(f"Error: Could not open video file: {video_path}")
            return {}
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame positions to extract
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        
        # Calculate extraction interval based on sample rate
        interval = int(fps / sample_rate)
        if interval < 1:
            interval = 1
            
        result = {
            'frames': [],
            'gray_frames': [],
            'optical_flow': [],
            'edges': [],
            'timestamps': [],
            'keypoints': [],
            'descriptors': [],
            'movement_intensity': []
        }
        
        # Set to start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # For optical flow calculation
        prev_gray = None
        
        # Create feature detector - CUDA-accelerated if available
        if self.use_cuda:
            orb = cv2.cuda.ORB.create(nfeatures=500)
        else:
            orb = cv2.ORB_create(nfeatures=500)
        
        # Extract frames at the specified interval
        current_frame = start_frame
        frame_idx = 0
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Apply ROI if specified
            if self.roi:
                x, y, w, h = self.roi
                # Ensure ROI is within frame boundaries
                x = max(0, min(x, width-1))
                y = max(0, min(y, height-1))
                w = max(1, min(w, width-x))
                h = max(1, min(h, height-y))
                roi_frame = frame[y:y+h, x:x+w]
                # Store the original frame
                result['frames'].append(frame)
                # Convert ROI to grayscale for processing
                gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            else:
                # Store the original frame
                result['frames'].append(frame)
                # Convert to grayscale for processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            result['gray_frames'].append(gray)
            
            # Store timestamp
            result['timestamps'].append(current_frame / fps)
            
            # Compute edges using accelerated edge detection
            edges = self.cuda_edge_detection(gray)
            result['edges'].append(edges)
            
            # Extract keypoints and descriptors for feature matching
            if self.use_cuda:
                # GPU version
                gpu_gray = cv2.cuda_GpuMat()
                gpu_gray.upload(gray)
                
                keypoints, descriptors = orb.detectAndComputeAsync(gpu_gray, None)
                if descriptors is not None:
                    descriptors = descriptors.download()
            else:
                # CPU version
                keypoints, descriptors = orb.detectAndCompute(gray, None)
                
            result['keypoints'].append(keypoints)
            result['descriptors'].append(descriptors)
            
            # Compute optical flow (good for detecting movement patterns)
            if prev_gray is not None:
                # Calculate optical flow with CUDA acceleration
                flow = self.cuda_optical_flow(prev_gray, gray)
                result['optical_flow'].append(flow)
                
                # Calculate movement intensity (magnitude of optical flow)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                movement_intensity = np.mean(mag)
                result['movement_intensity'].append(movement_intensity)
            
            prev_gray = gray.copy()
            
            # Skip to next position
            current_frame += interval
            frame_idx += 1
            if current_frame < end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        cap.release()
        return result
    
    # Rest of the methods remain the same as in MovementFocusedMatcher...
    # Include all the other methods from MovementFocusedMatcher:
    # - detect_key_movements
    # - compute_movement_pattern_score 
    # - compute_flow_similarity
    # - compare_segments_advanced
    # - compute_position_score
    # - compute_visual_similarity
    # - compute_feature_matching_score
    # - find_best_match_advanced
    
    def detect_key_movements(self, movement_data):
        """
        Detect frames with significant movements using intensity thresholds.
        
        Args:
            movement_data: List of movement intensity values
            
        Returns:
            List of indices where key movements occur
        """
        if not movement_data:
            return []
            
        # Convert to numpy array for processing
        intensities = np.array(movement_data)
        
        # Calculate statistics
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)
        
        # Identify frames with movement intensity above threshold
        # (mean + 0.75 standard deviations)
        threshold = mean_intensity + 0.75 * std_intensity
        key_frames = np.where(intensities > threshold)[0]
        
        # Group consecutive frames
        key_movements = []
        if len(key_frames) > 0:
            current_group = [key_frames[0]]
            
            for i in range(1, len(key_frames)):
                if key_frames[i] - key_frames[i-1] <= 2:  # Consider consecutive if within 2 frames
                    current_group.append(key_frames[i])
                else:
                    # Add the middle frame of current group as the key moment
                    if current_group:
                        middle_idx = current_group[len(current_group)//2]
                        key_movements.append(middle_idx)
                    current_group = [key_frames[i]]
            
            # Add the last group
            if current_group:
                middle_idx = current_group[len(current_group)//2]
                key_movements.append(middle_idx)
                
        return key_movements
    
    def compute_movement_pattern_score(self, target_data, source_data):
        """
        Compare movement patterns with focus on key movements.
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if not target_data or not source_data:
            return 0
            
        # Extract movement intensities
        target_intensities = target_data.get('movement_intensity', [])
        source_intensities = source_data.get('movement_intensity', [])
        
        if not target_intensities or not source_intensities:
            return 0
            
        # Detect key movements
        target_key_movements = self.detect_key_movements(target_intensities)
        
        if not target_key_movements:
            self.log("No key movements detected in target video")
            # Fall back to regular movement comparison
            target_pattern = np.array(target_intensities)
            source_pattern = np.array(source_intensities[:len(target_intensities)])
            
            if len(source_pattern) < len(target_pattern):
                # Pad source pattern if necessary
                source_pattern = np.pad(source_pattern, 
                                        (0, len(target_pattern) - len(source_pattern)), 
                                        'constant')
                
            # Calculate correlation of movement patterns
            correlation = np.corrcoef(target_pattern, source_pattern)[0, 1]
            if np.isnan(correlation):
                return 0
                
            return (correlation + 1) / 2
        
        # For each key movement in the target, find the best match in the source
        key_movement_scores = []
        target_flows = target_data.get('optical_flow', [])
        source_flows = source_data.get('optical_flow', [])
        
        if not target_flows or not source_flows:
            return 0
            
        for key_idx in target_key_movements:
            if key_idx >= len(target_flows):
                continue
                
            target_flow = target_flows[key_idx]
            
            # Find best matching flow in source
            best_score = -1
            for i in range(len(source_flows)):
                score = self.compute_flow_similarity(target_flow, source_flows[i])
                if score > best_score:
                    best_score = score
                    
            if best_score > 0:
                key_movement_scores.append(best_score)
        
        # Calculate average score for key movements
        if key_movement_scores:
            avg_key_score = np.mean(key_movement_scores)
            self.log(f"Key movement match score: {avg_key_score:.4f}")
            return avg_key_score
        else:
            return 0
    
    def compute_flow_similarity(self, flow1, flow2):
        """
        Compare two optical flow fields for movement similarity.
        
        Returns:
            Similarity score between 0 and 1
        """
        if flow1.shape != flow2.shape:
            # Resize if dimensions don't match
            flow2 = cv2.resize(flow2, (flow1.shape[1], flow1.shape[0]))
        
        # Compute magnitude and angle of flow vectors
        mag1, ang1 = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
        mag2, ang2 = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])
        
        # Normalize magnitudes
        norm_mag1 = mag1 / (np.max(mag1) + 1e-10)
        norm_mag2 = mag2 / (np.max(mag2) + 1e-10)
        
        # Compare both magnitude and direction
        mag_sim = 1 - np.mean(np.abs(norm_mag1 - norm_mag2))
        
        # Convert angles to unit vectors for better comparison
        x1, y1 = np.cos(ang1), np.sin(ang1)
        x2, y2 = np.cos(ang2), np.sin(ang2)
        
        # Dot product measures direction similarity (1 = same direction, -1 = opposite)
        # Only consider directions where there is significant movement
        mask = (norm_mag1 > 0.25) & (norm_mag2 > 0.25)
        if np.sum(mask) > 0:
            dot_product = (x1[mask] * x2[mask] + y1[mask] * y2[mask])
            dir_sim = (np.mean(dot_product) + 1) / 2  # Normalize to 0-1
        else:
            dir_sim = 0.5  # Neutral if no significant movement
        
        # Combined similarity (70% magnitude, 30% direction)
        return 0.7 * mag_sim + 0.3 * dir_sim
    
    def compare_segments_advanced(self, target_data, source_data):
        """
        Compare two video segments with emphasis on movement patterns.
        
        Args:
            target_data: Features from target segment
            source_data: Features from source segment
            
        Returns:
            Similarity score (higher is better)
        """
        if not target_data or not source_data:
            return 0
            
        # Check if we have enough frames
        if len(target_data.get('gray_frames', [])) < 2 or len(source_data.get('gray_frames', [])) < 2:
            return 0
            
        # Movement pattern comparison (with key movement focus)
        movement_score = self.compute_movement_pattern_score(target_data, source_data)
        
        # Equipment/object positioning comparison (edges)
        position_score = self.compute_position_score(target_data.get('edges', []), 
                                                     source_data.get('edges', []))
        
        # Visual similarity comparison
        visual_score = self.compute_visual_similarity(target_data.get('gray_frames', []), 
                                                     source_data.get('gray_frames', []))
        
        # Feature points matching score
        feature_score = self.compute_feature_matching_score(target_data.get('keypoints', []),
                                                           target_data.get('descriptors', []),
                                                           source_data.get('keypoints', []),
                                                           source_data.get('descriptors', []))
        
        # Weight factors (adjusted to emphasize movement more)
        # Movement is now 70% of the total score
        weights = {
            'movement': self.movement_weight,       # Movement pattern similarity (most important)
            'position': 0.15,                       # Equipment/object positioning similarity
            'visual': 0.1,                          # Overall visual similarity
            'features': 0.05                        # Feature point matching
        }
        
        # Ensure weights sum to 1
        weight_sum = sum(weights.values())
        if weight_sum != 1.0:
            for key in weights:
                weights[key] /= weight_sum
        
        # Calculate weighted total score
        total_score = (weights['movement'] * movement_score + 
                       weights['position'] * position_score + 
                       weights['visual'] * visual_score +
                       weights['features'] * feature_score)
        
        # Print detailed scores if verbose
        if self.verbose:
            self.log(f"Movement: {movement_score:.4f}, Position: {position_score:.4f}, " +
                     f"Visual: {visual_score:.4f}, Features: {feature_score:.4f}")
        
        return total_score
    
    def compute_position_score(self, target_edges, source_edges):
        """
        Compare equipment/object positioning using edge detection.
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if not target_edges or not source_edges:
            return 0
            
        min_frames = min(len(target_edges), len(source_edges))
        scores = []
        
        for i in range(min_frames):
            edges1 = target_edges[i]
            edges2 = source_edges[i % len(source_edges)]
            
            if edges1.shape != edges2.shape:
                # Resize if dimensions don't match
                edges2 = cv2.resize(edges2, (edges1.shape[1], edges1.shape[0]))
            
            # Calculate SSIM between edge maps
            score = ssim(edges1, edges2)
            scores.append((score + 1) / 2)  # Normalize to 0-1
            
        return np.mean(scores) if scores else 0
    
    def compute_visual_similarity(self, target_frames, source_frames):
        """
        Compare overall visual similarity between frames.
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if not target_frames or not source_frames:
            return 0
            
        min_frames = min(len(target_frames), len(source_frames))
        scores = []
        
        for i in range(min_frames):
            frame1 = target_frames[i]
            frame2 = source_frames[i % len(source_frames)]
            
            if frame1.shape != frame2.shape:
                # Resize if dimensions don't match
                frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
            
            # Calculate SSIM between frames
            score = ssim(frame1, frame2)
            scores.append((score + 1) / 2)  # Normalize to 0-1
            
        return np.mean(scores) if scores else 0
    
    def compute_feature_matching_score(self, target_keypoints, target_descriptors, 
                                      source_keypoints, source_descriptors):
        """
        Compare feature points to track specific objects or equipment.
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if (not target_keypoints or not target_descriptors or 
            not source_keypoints or not source_descriptors):
            return 0
            
        # We'll use a sampling of frames for computational efficiency
        sample_indices = np.linspace(0, min(len(target_keypoints), len(source_keypoints))-1, 
                                     5, dtype=int)
        
        scores = []
        for idx in sample_indices:
            if (idx >= len(target_keypoints) or idx >= len(source_keypoints) or
                target_descriptors[idx] is None or source_descriptors[idx] is None):
                continue
                
            # Create feature matcher - using CUDA if available
            if self.use_cuda:
                matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
                
                # Upload descriptors to GPU
                gpu_desc1 = cv2.cuda_GpuMat()
                gpu_desc2 = cv2.cuda_GpuMat()
                gpu_desc1.upload(target_descriptors[idx])
                gpu_desc2.upload(source_descriptors[idx])
                
                # Match descriptors
                gpu_matches = matcher.match(gpu_desc1, gpu_desc2)
                matches = matcher.matchDownload([gpu_matches])[0]
            else:
                # CPU matcher
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(target_descriptors[idx], source_descriptors[idx])
            
            if not matches:
                continue
                
            # Sort matches by distance (lower is better)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate matching score based on number and quality of matches
            # Normalize distance (lower is better)
            avg_distance = np.mean([m.distance for m in matches[:20]]) if len(matches) >= 20 else 0
            if avg_distance > 0:
                norm_distance = max(0, min(1, 1 - (avg_distance / 100)))
                
                # Normalize match count (higher is better)
                match_count = min(1.0, len(matches) / 50)
                
                # Combined score
                score = 0.6 * norm_distance + 0.4 * match_count
                scores.append(score)
                
        return np.mean(scores) if scores else 0
    
    def find_best_match_advanced(self, target_video_path, target_start_time, target_duration, step=0.25):
        """
        Find the best matching segment in the source video with advanced movement detection.
        Uses GPU acceleration when available.
        
        Args:
            target_video_path: Path to the target video
            target_start_time: Start time of the target segment in seconds
            target_duration: Duration of the target segment in seconds
            step: Step size in seconds for sliding window (smaller = more precise but slower)
            
        Returns:
            Tuple of (best_start_time, best_score)
        """
        # Make sure we don't try to search beyond the source video
        max_start_time = self.analyzer.duration - target_duration
        if max_start_time < 0:
            self.log(f"Source video ({self.analyzer.duration:.2f}s) is shorter than target segment ({target_duration:.2f}s)")
            return (0, 0)
            
        # Extract target frames and features
        self.log(f"Extracting frames and features from target segment ({target_duration:.2f}s)...")
        start_time = time.time()
        target_data = self.extract_frames_with_features(target_video_path, target_start_time, target_duration)
        extract_time = time.time() - start_time
        self.log(f"Extracted features in {extract_time:.2f}s using {'GPU' if self.use_cuda else 'CPU'}")
        
        if not target_data or not target_data.get('frames'):
            self.log("Failed to extract target frames")
            return (0, 0)
            
        self.log(f"Extracted {len(target_data.get('frames', []))} frames from target segment")
        
        # Search for the best match
        best_start_time = 0
        best_score = -1
        
        self.log(f"Searching for best match in source video (step: {step:.2f}s)...")
        
        # Try different starting positions
        current_time = 0
        comparison_count = 0
        
        # Store intermediate results for visualization/debugging
        results = []
        
        search_start_time = time.time()
        while current_time <= max_start_time:
            # Extract frames and features from the source segment
            source_data = self.extract_frames_with_features(self.source_path, current_time, target_duration)
            
            if not source_data or not source_data.get('frames'):
                current_time += step
                continue
                
            # Compare segments
            score = self.compare_segments_advanced(target_data, source_data)
            results.append((current_time, score))
            comparison_count += 1
            
            if comparison_count % 5 == 0:
                self.log(f"Comparing position {current_time:.2f}s: score = {score:.4f}")
                
            if score > best_score:
                best_score = score
                best_start_time = current_time
                self.log(f"New best match at {best_start_time:.2f}s with score {best_score:.4f}")
                
            current_time += step
        
        search_time = time.time() - search_start_time
        self.log(f"Search completed in {search_time:.2f}s")
        self.log(f"Best match found at {best_start_time:.2f}s with score {best_score:.4f}")
        
        return (best_start_time, best_score)



class SceneAnalyzer:
    """
    Class responsible for analyzing and extracting information about scenes in a video.
    """
    
    def __init__(self, video_path, threshold=27.0, verbose=True):
        """Initialize the scene analyzer."""
        self.video_path = video_path
        self.threshold = threshold
        self.verbose = verbose
        self.scenes = []
        
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
            print(f"Video FPS: {self.fps}, Duration: {timedelta(seconds=self.duration)}")
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
    
    def detect_scenes(self):
        """Detect scenes in the video file using PySceneDetect."""
        self.log(f"Detecting scenes in {self.video_path}...")
        
        try:
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
            
            # Convert to seconds with frame accuracy
            self.scenes = []
            
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                
                # Calculate frame-precise times
                start_frame = int(start_time * self.fps)
                end_frame = int(end_time * self.fps)
                precise_start = start_frame / self.fps
                precise_end = end_frame / self.fps
                
                self.scenes.append((precise_start, precise_end))
                
            video_manager.release()
            
            self.log(f"Detected {len(self.scenes)} scenes")
            return self.scenes
            
        except Exception as e:
            self.log(f"Error during scene detection: {str(e)}")
            # Create one scene for the whole video
            self.log("Creating a single scene for the entire video")
            self.scenes = [(0, self.duration)]
            return self.scenes
    
    def get_scene_info(self):
        """Returns information about all detected scenes."""
        if not self.scenes:
            self.detect_scenes()
            
        scene_info = []
        for i, (start_time, end_time) in enumerate(self.scenes):
            scene_num = i + 1
            duration = end_time - start_time
            
            # Calculate frame-precise information
            start_frame = int(start_time * self.fps)
            end_frame = int(end_time * self.fps)
            frame_count = end_frame - start_frame
            
            scene_info.append({
                'number': scene_num,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'frame_count': frame_count
            })
        
        return scene_info




class EnhancedVideoMatcher:
    """
    Enhanced class for finding the best matching segment in a source video,
    with improved detection of movement patterns and equipment positioning.
    """
    
    def __init__(self, source_video_path, verbose=True):
        """Initialize the enhanced video matcher."""
        self.source_path = source_video_path
        self.verbose = verbose
        self.analyzer = SceneAnalyzer(source_video_path, verbose=verbose)
    
    def log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def extract_frames_with_features(self, video_path, start_time, duration, sample_rate=2.0):
        """
        Extract frames and compute features relevant for movement and position matching.
        
        Args:
            video_path: Path to the video
            start_time: Start time in seconds
            duration: Duration to extract in seconds
            sample_rate: Frames per second to extract (default: 2.0)
            
        Returns:
            Dictionary with frames and features
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log(f"Error: Could not open video file: {video_path}")
            return {}
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame positions to extract
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        
        # Calculate extraction interval based on sample rate
        interval = int(fps / sample_rate)
        if interval < 1:
            interval = 1
            
        result = {
            'frames': [],
            'gray_frames': [],
            'optical_flow': [],
            'edges': [],
            'timestamps': []
        }
        
        # Set to start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # For optical flow calculation
        prev_gray = None
        
        # Extract frames at the specified interval
        current_frame = start_frame
        frame_idx = 0
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Store the original frame
            result['frames'].append(frame)
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result['gray_frames'].append(gray)
            
            # Store timestamp
            result['timestamps'].append(current_frame / fps)
            
            # Compute edges using Canny (good for detecting equipment positioning)
            edges = cv2.Canny(gray, 100, 200)
            result['edges'].append(edges)
            
            # Compute optical flow (good for detecting movement patterns)
            if prev_gray is not None:
                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                result['optical_flow'].append(flow)
            
            prev_gray = gray.copy()
            
            # Skip to next position
            current_frame += interval
            frame_idx += 1
            if current_frame < end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        cap.release()
        return result
    
    def compute_movement_score(self, flow1, flow2):
        """
        Compare movement patterns using optical flow.
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if flow1.shape != flow2.shape:
            # Resize if dimensions don't match
            flow2 = cv2.resize(flow2, (flow1.shape[1], flow1.shape[0]))
        
        # Compute the magnitude of flow vectors
        mag1, _ = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
        mag2, _ = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])
        
        # Flatten and normalize
        mag1_flat = mag1.flatten() / (np.max(mag1) + 1e-10)
        mag2_flat = mag2.flatten() / (np.max(mag2) + 1e-10)
        
        # Compute correlation coefficient
        correlation = np.corrcoef(mag1_flat, mag2_flat)[0, 1]
        if np.isnan(correlation):
            return 0
        
        # Normalize to 0-1 range
        return (correlation + 1) / 2
    
    def compute_position_score(self, edges1, edges2):
        """
        Compare equipment/object positioning using edge detection.
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if edges1.shape != edges2.shape:
            # Resize if dimensions don't match
            edges2 = cv2.resize(edges2, (edges1.shape[1], edges1.shape[0]))
        
        # Calculate SSIM between edge maps
        score = ssim(edges1, edges2)
        
        # Normalize to 0-1 range
        return (score + 1) / 2
    
    def compute_frame_similarity(self, frame1, frame2):
        """
        Compare overall visual similarity between frames.
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if frame1.shape != frame2.shape:
            # Resize if dimensions don't match
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # Calculate SSIM between frames
        score = ssim(frame1, frame2)
        
        # Also calculate histogram similarity as a backup measure
        hist1 = cv2.calcHist([frame1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([frame2], [0], None, [64], [0, 256])
        
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Combine scores (weighted average)
        combined_score = 0.7 * score + 0.3 * hist_score
        
        return combined_score
    
    def compare_segments_advanced(self, target_data, source_data):
        """
        Compare two video segments using multiple features.
        
        Args:
            target_data: Features from target segment
            source_data: Features from source segment
            
        Returns:
            Similarity score (higher is better)
        """
        if not target_data or not source_data:
            return 0
            
        # Check if we have enough frames
        if len(target_data['gray_frames']) < 2 or len(source_data['gray_frames']) < 2:
            return 0
            
        # Use the smaller number of frames
        min_frames = min(len(target_data['gray_frames']), len(source_data['gray_frames']))
        
        # Weight factors for different aspects of similarity
        weights = {
            'visual': 0.3,      # Overall visual similarity
            'movement': 0.9,    # Movement pattern similarity (most important)
            'position': 0.8     # Equipment/object positioning similarity
        }
        
        total_score = 0
        movement_scores = []
        position_scores = []
        visual_scores = []
        
        # Skip first frame for optical flow comparison
        for i in range(1, min_frames):
            # Frame similarity score
            frame1 = target_data['gray_frames'][i]
            frame2 = source_data['gray_frames'][i % len(source_data['gray_frames'])]
            visual_score = self.compute_frame_similarity(frame1, frame2)
            visual_scores.append(visual_score)
            
            # Movement pattern score (using optical flow)
            if i < len(target_data['optical_flow']) and i-1 < len(source_data['optical_flow']):
                flow1 = target_data['optical_flow'][i-1]
                flow2 = source_data['optical_flow'][(i-1) % len(source_data['optical_flow'])]
                movement_score = self.compute_movement_score(flow1, flow2)
                movement_scores.append(movement_score)
            
            # Equipment/object positioning score (using edges)
            edges1 = target_data['edges'][i]
            edges2 = source_data['edges'][i % len(source_data['edges'])]
            position_score = self.compute_position_score(edges1, edges2)
            position_scores.append(position_score)
        
        # Calculate average scores
        avg_visual = np.mean(visual_scores) if visual_scores else 0
        avg_movement = np.mean(movement_scores) if movement_scores else 0
        avg_position = np.mean(position_scores) if position_scores else 0
        
        # Calculate weighted total score
        total_score = (weights['visual'] * avg_visual + 
                       weights['movement'] * avg_movement + 
                       weights['position'] * avg_position)
        
        # Print detailed scores for debugging
        if self.verbose:
            self.log(f"Visual similarity: {avg_visual:.4f}, Movement: {avg_movement:.4f}, Position: {avg_position:.4f}")
        
        return total_score
    
    def find_best_match_advanced(self, target_video_path, target_start_time, target_duration, step=0.5):
        """
        Find the best matching segment in the source video with advanced features.
        
        Args:
            target_video_path: Path to the target video
            target_start_time: Start time of the target segment in seconds
            target_duration: Duration of the target segment in seconds
            step: Step size in seconds for sliding window (smaller = more precise but slower)
            
        Returns:
            Tuple of (best_start_time, best_score)
        """
        # Make sure we don't try to search beyond the source video
        max_start_time = self.analyzer.duration - target_duration
        if max_start_time < 0:
            self.log(f"Source video ({self.analyzer.duration:.2f}s) is shorter than target segment ({target_duration:.2f}s)")
            return (0, 0)
            
        # Extract target frames and features
        self.log(f"Extracting frames and features from target segment ({target_duration:.2f}s)...")
        target_data = self.extract_frames_with_features(target_video_path, target_start_time, target_duration)
        
        if not target_data or not target_data['frames']:
            self.log("Failed to extract target frames")
            return (0, 0)
            
        self.log(f"Extracted {len(target_data['frames'])} frames from target segment")
        
        # Search for the best match
        best_start_time = 0
        best_score = -1
        
        self.log(f"Searching for best match in source video (step: {step:.2f}s)...")
        
        # Try different starting positions
        current_time = 0
        comparison_count = 0
        
        # Store intermediate results for visualization/debugging
        results = []
        
        while current_time <= max_start_time:
            # Extract frames and features from the source segment
            source_data = self.extract_frames_with_features(self.source_path, current_time, target_duration)
            
            if not source_data or not source_data['frames']:
                current_time += step
                continue
                
            # Compare segments
            score = self.compare_segments_advanced(target_data, source_data)
            results.append((current_time, score))
            comparison_count += 1
            
            if comparison_count % 5 == 0:
                self.log(f"Comparing position {current_time:.2f}s: score = {score:.4f}")
                
            if score > best_score:
                best_score = score
                best_start_time = current_time
                self.log(f"New best match at {best_start_time:.2f}s with score {best_score:.4f}")
                
            current_time += step
        
        self.log(f"Best match found at {best_start_time:.2f}s with score {best_score:.4f}")
        
        # For visualization/debugging, you can save the results
        # self.save_matching_results(results, target_duration)
        
        return (best_start_time, best_score)
    
    def save_matching_results(self, results, segment_duration):
        """
        Save matching results for visualization.
        
        Args:
            results: List of (time, score) tuples
            segment_duration: Duration of the segment being matched
        """
        import matplotlib.pyplot as plt
        
        times = [t for t, _ in results]
        scores = [s for _, s in results]
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, scores)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Matching Score')
        plt.title(f'Segment Matching Scores (duration: {segment_duration:.2f}s)')
        plt.grid(True)
        
        # Find best match point
        best_time = times[np.argmax(scores)]
        best_score = max(scores)
        plt.plot(best_time, best_score, 'ro', markersize=10)
        plt.annotate(f'Best match: {best_time:.2f}s, score: {best_score:.4f}', 
                    (best_time, best_score), 
                    xytext=(best_time+0.5, best_score-0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        
        # Save plot
        output_dir = "matching_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'matching_scores_{time.time()}.png'))
        plt.close()


class MovementFocusedMatcher:
    """
    Enhanced matcher that focuses on detecting key movements between videos
    with support for region-of-interest focusing and movement pattern analysis.
    """
    
    def __init__(self, source_video_path, verbose=True):
        """Initialize the movement-focused matcher."""
        self.source_path = source_video_path
        self.verbose = verbose
        self.analyzer = SceneAnalyzer(source_video_path, verbose=verbose)
        # Default region of interest - full frame
        self.roi = None
        # Movement importance weight
        self.movement_weight = 0.7  # Increased from 0.5 to 0.7



    def log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)    
    def set_region_of_interest(self, x, y, width, height):
        """
        Set a specific region of interest to focus on for movement detection.
        
        Args:
            x, y: Top-left coordinates of ROI
            width, height: Dimensions of ROI
        """
        self.roi = (x, y, width, height)
        self.log(f"Set ROI to ({x}, {y}, {width}, {height})")
        
    def set_movement_importance(self, weight=0.7):
        """
        Set how important movement matching is compared to other factors.
        
        Args:
            weight: Value between 0 and 1 (higher means more emphasis on movement)
        """
        self.movement_weight = max(0, min(1, weight))
        self.log(f"Set movement importance to {self.movement_weight}")
    
    def extract_frames_with_features(self, video_path, start_time, duration, sample_rate=4.0):
        """
        Extract frames and compute features with higher sample rate for better movement tracking.
        
        Args:
            video_path: Path to the video
            start_time: Start time in seconds
            duration: Duration to extract in seconds
            sample_rate: Frames per second to extract (increased to 4.0)
            
        Returns:
            Dictionary with frames and features
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log(f"Error: Could not open video file: {video_path}")
            return {}
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame positions to extract
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        
        # Calculate extraction interval based on sample rate
        interval = int(fps / sample_rate)
        if interval < 1:
            interval = 1
            
        result = {
            'frames': [],
            'gray_frames': [],
            'optical_flow': [],
            'edges': [],
            'timestamps': [],
            'keypoints': [],
            'descriptors': [],
            'movement_intensity': []
        }
        
        # Set to start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # For optical flow calculation
        prev_gray = None
        
        # Initialize feature detector
        orb = cv2.ORB_create(nfeatures=500)
        
        # Extract frames at the specified interval
        current_frame = start_frame
        frame_idx = 0
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Apply ROI if specified
            if self.roi:
                x, y, w, h = self.roi
                # Ensure ROI is within frame boundaries
                x = max(0, min(x, width-1))
                y = max(0, min(y, height-1))
                w = max(1, min(w, width-x))
                h = max(1, min(h, height-y))
                roi_frame = frame[y:y+h, x:x+w]
                # Store the original frame
                result['frames'].append(frame)
                # Convert ROI to grayscale for processing
                gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            else:
                # Store the original frame
                result['frames'].append(frame)
                # Convert to grayscale for processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            result['gray_frames'].append(gray)
            
            # Store timestamp
            result['timestamps'].append(current_frame / fps)
            
            # Compute edges using Canny (good for detecting equipment positioning)
            edges = cv2.Canny(gray, 100, 200)
            result['edges'].append(edges)
            
            # Extract keypoints and descriptors for feature matching
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            result['keypoints'].append(keypoints)
            result['descriptors'].append(descriptors)
            
            # Compute optical flow (good for detecting movement patterns)
            if prev_gray is not None:
                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                result['optical_flow'].append(flow)
                
                # Calculate movement intensity (magnitude of optical flow)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                movement_intensity = np.mean(mag)
                result['movement_intensity'].append(movement_intensity)
            
            prev_gray = gray.copy()
            
            # Skip to next position
            current_frame += interval
            frame_idx += 1
            if current_frame < end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        cap.release()
        return result
    
    def detect_key_movements(self, movement_data):
        """
        Detect frames with significant movements using intensity thresholds.
        
        Args:
            movement_data: List of movement intensity values
            
        Returns:
            List of indices where key movements occur
        """
        if not movement_data:
            return []
            
        # Convert to numpy array for processing
        intensities = np.array(movement_data)
        
        # Calculate statistics
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)
        
        # Identify frames with movement intensity above threshold
        # (mean + 0.75 standard deviations)
        threshold = mean_intensity + 0.75 * std_intensity
        key_frames = np.where(intensities > threshold)[0]
        
        # Group consecutive frames
        key_movements = []
        if len(key_frames) > 0:
            current_group = [key_frames[0]]
            
            for i in range(1, len(key_frames)):
                if key_frames[i] - key_frames[i-1] <= 2:  # Consider consecutive if within 2 frames
                    current_group.append(key_frames[i])
                else:
                    # Add the middle frame of current group as the key moment
                    if current_group:
                        middle_idx = current_group[len(current_group)//2]
                        key_movements.append(middle_idx)
                    current_group = [key_frames[i]]
            
            # Add the last group
            if current_group:
                middle_idx = current_group[len(current_group)//2]
                key_movements.append(middle_idx)
                
        return key_movements
    
    def compute_movement_pattern_score(self, target_data, source_data):
        """
        Compare movement patterns with focus on key movements.
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if not target_data or not source_data:
            return 0
            
        # Extract movement intensities
        target_intensities = target_data.get('movement_intensity', [])
        source_intensities = source_data.get('movement_intensity', [])
        
        if not target_intensities or not source_intensities:
            return 0
            
        # Detect key movements
        target_key_movements = self.detect_key_movements(target_intensities)
        
        if not target_key_movements:
            self.log("No key movements detected in target video")
            # Fall back to regular movement comparison
            target_pattern = np.array(target_intensities)
            source_pattern = np.array(source_intensities[:len(target_intensities)])
            
            if len(source_pattern) < len(target_pattern):
                # Pad source pattern if necessary
                source_pattern = np.pad(source_pattern, 
                                        (0, len(target_pattern) - len(source_pattern)), 
                                        'constant')
                
            # Calculate correlation of movement patterns
            correlation = np.corrcoef(target_pattern, source_pattern)[0, 1]
            if np.isnan(correlation):
                return 0
                
            return (correlation + 1) / 2
        
        # For each key movement in the target, find the best match in the source
        key_movement_scores = []
        target_flows = target_data.get('optical_flow', [])
        source_flows = source_data.get('optical_flow', [])
        
        if not target_flows or not source_flows:
            return 0
            
        for key_idx in target_key_movements:
            if key_idx >= len(target_flows):
                continue
                
            target_flow = target_flows[key_idx]
            
            # Find best matching flow in source
            best_score = -1
            for i in range(len(source_flows)):
                score = self.compute_flow_similarity(target_flow, source_flows[i])
                if score > best_score:
                    best_score = score
                    
            if best_score > 0:
                key_movement_scores.append(best_score)
        
        # Calculate average score for key movements
        if key_movement_scores:
            avg_key_score = np.mean(key_movement_scores)
            self.log(f"Key movement match score: {avg_key_score:.4f}")
            return avg_key_score
        else:
            return 0
    
    def compute_flow_similarity(self, flow1, flow2):
        """
        Compare two optical flow fields for movement similarity.
        
        Returns:
            Similarity score between 0 and 1
        """
        if flow1.shape != flow2.shape:
            # Resize if dimensions don't match
            flow2 = cv2.resize(flow2, (flow1.shape[1], flow1.shape[0]))
        
        # Compute magnitude and angle of flow vectors
        mag1, ang1 = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
        mag2, ang2 = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])
        
        # Normalize magnitudes
        norm_mag1 = mag1 / (np.max(mag1) + 1e-10)
        norm_mag2 = mag2 / (np.max(mag2) + 1e-10)
        
        # Compare both magnitude and direction
        mag_sim = 1 - np.mean(np.abs(norm_mag1 - norm_mag2))
        
        # Convert angles to unit vectors for better comparison
        x1, y1 = np.cos(ang1), np.sin(ang1)
        x2, y2 = np.cos(ang2), np.sin(ang2)
        
        # Dot product measures direction similarity (1 = same direction, -1 = opposite)
        # Only consider directions where there is significant movement
        mask = (norm_mag1 > 0.25) & (norm_mag2 > 0.25)
        if np.sum(mask) > 0:
            dot_product = (x1[mask] * x2[mask] + y1[mask] * y2[mask])
            dir_sim = (np.mean(dot_product) + 1) / 2  # Normalize to 0-1
        else:
            dir_sim = 0.5  # Neutral if no significant movement
        
        # Combined similarity (70% magnitude, 30% direction)
        return 0.7 * mag_sim + 0.3 * dir_sim
    
    def compare_segments_advanced(self, target_data, source_data):
        """
        Compare two video segments with emphasis on movement patterns.
        
        Args:
            target_data: Features from target segment
            source_data: Features from source segment
            
        Returns:
            Similarity score (higher is better)
        """
        if not target_data or not source_data:
            return 0
            
        # Check if we have enough frames
        if len(target_data.get('gray_frames', [])) < 2 or len(source_data.get('gray_frames', [])) < 2:
            return 0
            
        # Movement pattern comparison (with key movement focus)
        movement_score = self.compute_movement_pattern_score(target_data, source_data)
        
        # Equipment/object positioning comparison (edges)
        position_score = self.compute_position_score(target_data.get('edges', []), 
                                                     source_data.get('edges', []))
        
        # Visual similarity comparison
        visual_score = self.compute_visual_similarity(target_data.get('gray_frames', []), 
                                                     source_data.get('gray_frames', []))
        
        # Feature points matching score
        feature_score = self.compute_feature_matching_score(target_data.get('keypoints', []),
                                                           target_data.get('descriptors', []),
                                                           source_data.get('keypoints', []),
                                                           source_data.get('descriptors', []))
        
        # Weight factors (adjusted to emphasize movement more)
        # Movement is now 70% of the total score
        weights = {
            'movement': self.movement_weight,       # Movement pattern similarity (most important)
            'position': 0.15,                       # Equipment/object positioning similarity
            'visual': 0.1,                          # Overall visual similarity
            'features': 0.05                        # Feature point matching
        }
        
        # Ensure weights sum to 1
        weight_sum = sum(weights.values())
        if weight_sum != 1.0:
            for key in weights:
                weights[key] /= weight_sum
        
        # Calculate weighted total score
        total_score = (weights['movement'] * movement_score + 
                       weights['position'] * position_score + 
                       weights['visual'] * visual_score +
                       weights['features'] * feature_score)
        
        # Print detailed scores if verbose
        if self.verbose:
            self.log(f"Movement: {movement_score:.4f}, Position: {position_score:.4f}, " +
                     f"Visual: {visual_score:.4f}, Features: {feature_score:.4f}")
        
        return total_score
    
    def compute_position_score(self, target_edges, source_edges):
        """
        Compare equipment/object positioning using edge detection.
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if not target_edges or not source_edges:
            return 0
            
        min_frames = min(len(target_edges), len(source_edges))
        scores = []
        
        for i in range(min_frames):
            edges1 = target_edges[i]
            edges2 = source_edges[i % len(source_edges)]
            
            if edges1.shape != edges2.shape:
                # Resize if dimensions don't match
                edges2 = cv2.resize(edges2, (edges1.shape[1], edges1.shape[0]))
            
            # Calculate SSIM between edge maps
            score = ssim(edges1, edges2)
            scores.append((score + 1) / 2)  # Normalize to 0-1
            
        return np.mean(scores) if scores else 0
    
    def compute_visual_similarity(self, target_frames, source_frames):
        """
        Compare overall visual similarity between frames.
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if not target_frames or not source_frames:
            return 0
            
        min_frames = min(len(target_frames), len(source_frames))
        scores = []
        
        for i in range(min_frames):
            frame1 = target_frames[i]
            frame2 = source_frames[i % len(source_frames)]
            
            if frame1.shape != frame2.shape:
                # Resize if dimensions don't match
                frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
            
            # Calculate SSIM between frames
            score = ssim(frame1, frame2)
            scores.append((score + 1) / 2)  # Normalize to 0-1
            
        return np.mean(scores) if scores else 0
    
    def compute_feature_matching_score(self, target_keypoints, target_descriptors, 
                                      source_keypoints, source_descriptors):
        """
        Compare feature points to track specific objects or equipment.
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if (not target_keypoints or not target_descriptors or 
            not source_keypoints or not source_descriptors):
            return 0
            
        # We'll use a sampling of frames for computational efficiency
        sample_indices = np.linspace(0, min(len(target_keypoints), len(source_keypoints))-1, 
                                     5, dtype=int)
        
        scores = []
        for idx in sample_indices:
            if (idx >= len(target_keypoints) or idx >= len(source_keypoints) or
                target_descriptors[idx] is None or source_descriptors[idx] is None):
                continue
                
            # Create feature matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            # Match descriptors
            matches = bf.match(target_descriptors[idx], source_descriptors[idx])
            
            if not matches:
                continue
                
            # Sort matches by distance (lower is better)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate matching score based on number and quality of matches
            # Normalize distance (lower is better)
            avg_distance = np.mean([m.distance for m in matches[:20]]) if len(matches) >= 20 else 0
            if avg_distance > 0:
                norm_distance = max(0, min(1, 1 - (avg_distance / 100)))
                
                # Normalize match count (higher is better)
                match_count = min(1.0, len(matches) / 50)
                
                # Combined score
                score = 0.6 * norm_distance + 0.4 * match_count
                scores.append(score)
                
        return np.mean(scores) if scores else 0
    
    def find_best_match_advanced(self, target_video_path, target_start_time, target_duration, step=0.25):
        """
        Find the best matching segment in the source video with advanced movement detection.
        
        Args:
            target_video_path: Path to the target video
            target_start_time: Start time of the target segment in seconds
            target_duration: Duration of the target segment in seconds
            step: Step size in seconds for sliding window (smaller = more precise but slower)
            
        Returns:
            Tuple of (best_start_time, best_score)
        """
        # Make sure we don't try to search beyond the source video
        max_start_time = self.analyzer.duration - target_duration
        if max_start_time < 0:
            self.log(f"Source video ({self.analyzer.duration:.2f}s) is shorter than target segment ({target_duration:.2f}s)")
            return (0, 0)
            
        # Extract target frames and features
        self.log(f"Extracting frames and features from target segment ({target_duration:.2f}s)...")
        target_data = self.extract_frames_with_features(target_video_path, target_start_time, target_duration)
        
        if not target_data or not target_data.get('frames'):
            self.log("Failed to extract target frames")
            return (0, 0)
            
        self.log(f"Extracted {len(target_data.get('frames', []))} frames from target segment")
        
        # Search for the best match
        best_start_time = 0
        best_score = -1
        
        self.log(f"Searching for best match in source video (step: {step:.2f}s)...")
        
        # Try different starting positions
        current_time = 0
        comparison_count = 0
        
        # Store intermediate results for visualization/debugging
        results = []
        
        while current_time <= max_start_time:
            # Extract frames and features from the source segment
            source_data = self.extract_frames_with_features(self.source_path, current_time, target_duration)
            
            if not source_data or not source_data.get('frames'):
                current_time += step
                continue
                
            # Compare segments
            score = self.compare_segments_advanced(target_data, source_data)
            results.append((current_time, score))
            comparison_count += 1
            
            if comparison_count % 5 == 0:
                self.log(f"Comparing position {current_time:.2f}s: score = {score:.4f}")
                
            if score > best_score:
                best_score = score
                best_start_time = current_time
                self.log(f"New best match at {best_start_time:.2f}s with score {best_score:.4f}")
                
            current_time += step
        
        self.log(f"Best match found at {best_start_time:.2f}s with score {best_score:.4f}")
        
        return (best_start_time, best_score)


class AudioPreservingReplacer:
    """
    Class for replacing video segments while preserving the original audio.
    """
    
    def __init__(self, training_video_path, verbose=True):
        """Initialize the audio-preserving replacer."""
        self.training_path = training_video_path
        self.verbose = verbose
        self.analyzer = SceneAnalyzer(training_video_path, verbose=verbose)
        
        # Get video properties
        self.fps = self.analyzer.fps
        self.width = self.analyzer.width
        self.height = self.analyzer.height
        
        # Create temporary directories
        self.temp_dir = "temp_files"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def extract_audio(self, video_path, output_path=None):
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the audio file (optional)
            
        Returns:
            Path to the extracted audio file
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self.temp_dir, f"{base_name}_audio.aac")
            
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "copy",  # Copy audio codec (no re-encoding)
                output_path
            ]
            
            self.log(f"Extracting audio from {video_path}...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                self.log(f"Audio extracted to {output_path}")
                return output_path
            else:
                self.log(f"Failed to extract audio")
                return None
                
        except subprocess.CalledProcessError as e:
            self.log(f"Error extracting audio: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            return None
    
    def extract_video_segment(self, video_path, start_time, duration, output_path=None):
        """
        Extract a video segment without audio.
        
        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            duration: Duration in seconds
            output_path: Path to save the video segment (optional)
            
        Returns:
            Path to the extracted video segment
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self.temp_dir, f"{base_name}_segment_{start_time:.2f}s_{duration:.2f}s.mp4")
            
        try:
            # Calculate exact frame count for precise duration
            info = SceneAnalyzer(video_path, verbose=False)
            frame_count = int(duration * info.fps)
            
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.6f}",
                "-i", video_path,
                "-frames:v", str(frame_count),
                "-an",  # No audio
                "-c:v", "libx264",  # Use H.264 codec
                "-crf", "18",  # High quality
                "-preset", "medium",
                output_path
            ]
            
            self.log(f"Extracting video segment from {start_time:.2f}s (duration: {duration:.2f}s)...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                self.log(f"Video segment extracted to {output_path}")
                return output_path
            else:
                self.log(f"Failed to extract video segment")
                return None
                
        except subprocess.CalledProcessError as e:
            self.log(f"Error extracting video segment: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            return None
    
    def get_target_scene_info(self, target_scene):
        """Get information about the target scene."""
        scene_info = self.analyzer.get_scene_info()
        
        # Find target scene
        target_info = None
        for scene in scene_info:
            if scene['number'] == target_scene:
                target_info = scene
                break
                
        if target_info is None:
            self.log(f"Error: Scene {target_scene} not found in training video")
            return None
            
        return target_info
   
    

    def replace_scene_preserving_audio(self, source_video_path, target_scene, output_path=None, 
                                      best_match_start=None, roi=None, movement_weight=0.7,
                                      use_cuda=True):
        """
        Replace a scene in the training video with a segment from the source video,
        while preserving the original audio. Uses GPU acceleration when available.
        
        Args:
            source_video_path: Path to the source video
            target_scene: Scene number to replace
            output_path: Path to save the result video (optional)
            best_match_start: Start time of the best match in source video (if None, will auto-detect)
            roi: Region of interest (x, y, width, height) to focus on for movement detection
            movement_weight: Weight (0-1) to assign to movement matching importance
            use_cuda: Whether to use CUDA acceleration when available
            
        Returns:
            Path to the result video
        """
        # Get target scene info
        target_info = self.get_target_scene_info(target_scene)
        if not target_info:
            return None
            
        # Extract the original audio
        audio_path = self.extract_audio(self.training_path)
        if not audio_path:
            self.log("Failed to extract audio from training video")
            return None
            
        # Find best matching segment if not provided
        if best_match_start is None:
            self.log("Finding best matching segment in source video...")
            # Use CudaMovementMatcher for GPU acceleration
            matcher = CudaMovementMatcher(source_video_path, verbose=self.verbose, use_cuda=use_cuda)
            
            # Set region of interest if provided
            if roi:
                matcher.set_region_of_interest(*roi)
                
            # Set movement importance weight
            matcher.set_movement_importance(movement_weight)
            
            best_match_start, score = matcher.find_best_match_advanced(
                self.training_path,
                target_info['start_time'],
                target_info['duration'],
                step=0.25  # Use smaller step for more precise matching
            )
            
            if score < 0.5:
                self.log(f"Warning: Low match score ({score:.4f}). Match quality might be poor.")
                
        # Extract video segments
        before_path = None
        if target_info['start_time'] > 0:
            before_path = self.extract_video_segment(
                self.training_path,
                0,
                target_info['start_time']
            )
            
        # Extract the matching segment from source video
        source_segment_path = self.extract_video_segment(
            source_video_path,
            best_match_start,
            target_info['duration']
        )
        
        if not source_segment_path:
            self.log("Failed to extract source video segment")
            return None
            
        # Extract the after segment
        after_path = None
        if target_info['end_time'] < self.analyzer.duration:
            after_path = self.extract_video_segment(
                self.training_path,
                target_info['end_time'],
                self.analyzer.duration - target_info['end_time']
            )
            
        # Determine output path
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.training_path))[0]
            output_dir = "replaced_videos"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"{base_name}_scene{target_scene}_replaced_with_audio.mp4")
            
        # Combine video segments
        video_list_path = os.path.join(self.temp_dir, "video_segments.txt")
        with open(video_list_path, 'w') as f:
            if before_path:
                f.write(f"file '{os.path.abspath(before_path)}'\n")
            f.write(f"file '{os.path.abspath(source_segment_path)}'\n")
            if after_path:
                f.write(f"file '{os.path.abspath(after_path)}'\n")
                
        # Create a video-only file by concatenating segments
        video_only_path = os.path.join(self.temp_dir, "video_only.mp4")
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", video_list_path,
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium",
                "-an",  # No audio
                video_only_path
            ]
            
            self.log("Concatenating video segments...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if not os.path.exists(video_only_path) or os.path.getsize(video_only_path) == 0:
                self.log("Failed to concatenate video segments")
                return None
                
            # Combine video with original audio
            cmd = [
                "ffmpeg", "-y",
                "-i", video_only_path,
                "-i", audio_path,
                "-c:v", "copy",  # Copy the video (no re-encoding)
                "-c:a", "aac",   # Make sure audio is AAC format
                "-strict", "experimental",
                "-map", "0:v:0",  # Use video from first input
                "-map", "1:a:0",  # Use audio from second input
                "-shortest",      # End when shortest input ends
                output_path
            ]
            
            self.log("Combining video with original audio...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                self.log(f"Scene replacement completed, result saved to {output_path}")
                return output_path
            else:
                self.log("Failed to create final output")
                return None
                
        except subprocess.CalledProcessError as e:
            self.log(f"Error during video processing: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            return None


def main():
    """Command-line interface for the audio-preserving scene replacement tool"""
  
    parser = argparse.ArgumentParser(description='Replace scenes in a video while preserving original audio')
    parser.add_argument('training_video', help='Path to the training video file')
    parser.add_argument('--source-video', help='Path to the source video file')
    parser.add_argument('--target-scene', type=int, help='Scene number to replace')
    parser.add_argument('--analyze', action='store_true', help='Analyze the training video')
    parser.add_argument('--start-time', type=float, help='Start time in source video (if known)')
    parser.add_argument('--output', help='Path to save the output video')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
                       help='Region of interest for movement detection (x y width height)')
    parser.add_argument('--movement-weight', type=float, default=0.7,
                       help='Weight of movement matching (0-1, default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA acceleration even if available')
    
    args = parser.parse_args()
    
    try:
        # Create the scene analyzer
        analyzer = SceneAnalyzer(
            args.training_video,
            verbose=not args.quiet
        )
        
        if args.analyze:
            # Analyze the training video
            scenes = analyzer.get_scene_info()
            
            if not args.quiet:
                print("\nTraining Video Analysis Results:")
                print(f"Detected {len(scenes)} scenes:")
                
                for scene in scenes:
                    print(f"Scene {scene['number']}: {scene['start_time']:.3f}s - {scene['end_time']:.3f}s " + 
                          f"(Duration: {scene['duration']:.3f}s, Frames: {scene['frame_count']})")
                
        elif args.source_video and args.target_scene:
            # Create the replacer
            replacer = AudioPreservingReplacer(
                args.training_video,
                verbose=not args.quiet
            )
            
            # Replace the scene
            start_time = time.time()
            result_path = replacer.replace_scene_preserving_audio(
                args.source_video,
                args.target_scene,
                output_path=args.output,
                best_match_start=args.start_time,
                roi=args.roi,
                movement_weight=args.movement_weight,
                use_cuda=not args.no_cuda
            )
            
            elapsed_time = time.time() - start_time
            
            if not args.quiet:
                print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
                if result_path:
                    print(f"Result saved to: {result_path}")
        else:
            print("Error: Must provide --analyze or both --source-video and --target-scene")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    main()