import cv2
import numpy as np
import os
import argparse
import subprocess
import time
from datetime import timedelta
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

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


class VideoMatcher:
    """
    Class for finding the best matching segment in a source video.
    """
    
    def __init__(self, source_video_path, verbose=True):
        """Initialize the video matcher."""
        self.source_path = source_video_path
        self.verbose = verbose
        self.analyzer = SceneAnalyzer(source_video_path, verbose=verbose)
    
    def log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def extract_frames(self, video_path, start_time, duration, sample_rate=1.0):
        """
        Extract frames from a video segment for comparison.
        
        Args:
            video_path: Path to the video
            start_time: Start time in seconds
            duration: Duration to extract in seconds
            sample_rate: Frames per second to extract (default: 1.0)
            
        Returns:
            List of frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log(f"Error: Could not open video file: {video_path}")
            return []
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame positions to extract
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        
        # Calculate extraction interval based on sample rate
        interval = int(fps / sample_rate)
        if interval < 1:
            interval = 1
            
        frames = []
        
        # Set to start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames at the specified interval
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale for more efficient comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            
            # Skip to next position
            current_frame += interval
            if current_frame < end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        cap.release()
        return frames
    
    def compare_segments(self, frames1, frames2):
        """
        Compare two sets of frames and calculate similarity score.
        
        Args:
            frames1: First set of frames
            frames2: Second set of frames
            
        Returns:
            Similarity score (higher is better)
        """
        if not frames1 or not frames2:
            return 0
            
        # Use the smaller number of frames
        min_frames = min(len(frames1), len(frames2))
        
        if min_frames == 0:
            return 0
            
        total_score = 0
        
        # Compare frames using SSIM or histogram comparison
        for i in range(min_frames):
            frame1 = frames1[i % len(frames1)]
            frame2 = frames2[i % len(frames2)]
            
            # Resize if dimensions don't match
            if frame1.shape != frame2.shape:
                frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
            
            # Calculate histogram similarity
            hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
            
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            total_score += score
            
        # Return average similarity score
        return total_score / min_frames
    
    def find_best_match(self, target_video_path, target_start_time, target_duration, window_duration=None, step=1.0):
        """
        Find the best matching segment in the source video.
        
        Args:
            target_video_path: Path to the target video
            target_start_time: Start time of the target segment in seconds
            target_duration: Duration of the target segment in seconds
            window_duration: Duration to search in source (default: 2x target_duration)
            step: Step size in seconds for sliding window
            
        Returns:
            Tuple of (best_start_time, best_score)
        """
        if window_duration is None:
            # Default to searching in a window twice the target duration
            window_duration = min(target_duration * 2, self.analyzer.duration)
        
        # Make sure we don't try to search beyond the source video
        max_start_time = self.analyzer.duration - target_duration
        if max_start_time < 0:
            self.log(f"Source video ({self.analyzer.duration:.2f}s) is shorter than target segment ({target_duration:.2f}s)")
            return (0, 0)
            
        # Extract target frames
        self.log(f"Extracting frames from target segment ({target_duration:.2f}s)...")
        target_frames = self.extract_frames(target_video_path, target_start_time, target_duration)
        
        if not target_frames:
            self.log("Failed to extract target frames")
            return (0, 0)
            
        self.log(f"Extracted {len(target_frames)} frames from target segment")
        
        # Search for the best match
        best_start_time = 0
        best_score = -1
        
        self.log(f"Searching for best match in source video (window: {window_duration:.2f}s, step: {step:.2f}s)...")
        
        # Try different starting positions
        current_time = 0
        comparison_count = 0
        
        while current_time <= max_start_time:
            # Extract frames from the source segment
            source_frames = self.extract_frames(self.source_path, current_time, target_duration)
            
            if not source_frames:
                current_time += step
                continue
                
            # Compare segments
            score = self.compare_segments(target_frames, source_frames)
            comparison_count += 1
            
            if comparison_count % 10 == 0:
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
    
    def replace_scene_preserving_audio(self, source_video_path, target_scene, output_path=None, best_match_start=None):
        """
        Replace a scene in the training video with a segment from the source video,
        while preserving the original audio.
        
        Args:
            source_video_path: Path to the source video
            target_scene: Scene number to replace
            output_path: Path to save the result video (optional)
            best_match_start: Start time of the best match in source video (if None, will auto-detect)
            
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
            matcher = VideoMatcher(source_video_path, verbose=self.verbose)
            best_match_start, score = matcher.find_best_match(
                self.training_path,
                target_info['start_time'],
                target_info['duration']
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
                best_match_start=args.start_time
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