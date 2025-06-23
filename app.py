import cv2
import sys
import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from yolo_detector import YOLODetector
from zone_manager import ZoneManager
from detection_analyzer import DetectionAnalyzer
from deep_sort_realtime.deepsort_tracker import DeepSort

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoZoneTracker:
    def __init__(self, model_path: str, zones_file: str, detection_interval_seconds: float = 3.0, 
                 detection_size: tuple = (640, 640)):
        """
        Initialize the VideoZoneTracker
        
        Args:
            model_path: Path to YOLO model file
            zones_file: Path to zones configuration JSON file
            detection_interval_seconds: Run detection every N seconds (default: 3.0)
            detection_size: Size to resize frame for detection (width, height)
        """
        self.detection_interval_seconds = detection_interval_seconds
        self.detection_size = detection_size
        self.frame_count = 0
        self.last_detections = []
        self.last_detection_time = 0
        self.fps = 0
        
        # Initialize components
        try:
            self.detector = YOLODetector(model_path)
            self.zone_manager = ZoneManager(zones_file)
            self.analyzer = DetectionAnalyzer(self.zone_manager)
            self.tracker = DeepSort(max_age=30, n_init=3)
            logger.info("✅ All components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise

    def validate_paths(self, video_path: str, output_path: str) -> tuple[str, str]:
        """Validate and prepare input/output paths"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        return video_path, output_path

    def setup_video_capture(self, video_path: str) -> cv2.VideoCapture:
        """Setup video capture with error handling"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Log video information
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"📹 Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        return cap

    def resize_for_detection(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Resize frame for detection and calculate scale factors
        
        Args:
            frame: Original frame
            
        Returns:
            resized_frame: Frame resized for detection
            scale_x: Scale factor for x coordinates
            scale_y: Scale factor for y coordinates
        """
        original_height, original_width = frame.shape[:2]
        target_width, target_height = self.detection_size
        
        # Calculate scale factors
        scale_x = original_width / target_width
        scale_y = original_height / target_height
        
        # Resize frame
        resized_frame = cv2.resize(frame, (target_width, target_height))
        
        return resized_frame, scale_x, scale_y
    
    def scale_detections(self, detections: List[Dict], scale_x: float, scale_y: float) -> List[Dict]:
        """
        Scale detection coordinates back to original frame size
        
        Args:
            detections: List of detections with coordinates for resized frame
            scale_x: Scale factor for x coordinates
            scale_y: Scale factor for y coordinates
            
        Returns:
            List of detections with coordinates scaled to original frame size
        """
        scaled_detections = []
        
        for det in detections:
            bbox = det['bbox']
            # Scale bounding box coordinates
            scaled_bbox = [
                int(bbox[0] * scale_x),  # x1
                int(bbox[1] * scale_y),  # y1
                int(bbox[2] * scale_x),  # x2
                int(bbox[3] * scale_y)   # y2
            ]
            
            scaled_det = det.copy()
            scaled_det['bbox'] = scaled_bbox
            
            # Recalculate center point
            cx = (scaled_bbox[0] + scaled_bbox[2]) // 2
            cy = (scaled_bbox[1] + scaled_bbox[3]) // 2
            scaled_det['center'] = [cx, cy]
            
            scaled_detections.append(scaled_det)
        
        return scaled_detections
        """Setup video writer with proper codec"""
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Try different codecs for better compatibility
        codecs = ['mp4v', 'XVID', 'MJPG']
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if out.isOpened():
                logger.info(f"✅ Using codec: {codec}")
                return out
        
        raise RuntimeError("Failed to initialize video writer with any codec")

    def setup_video_writer(self, cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
        """Setup video writer with proper codec"""
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Try different codecs for better compatibility
        codecs = ['mp4v', 'XVID', 'MJPG']
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if out.isOpened():
                logger.info(f"✅ Using codec: {codec}")
                return out
        
        raise RuntimeError("Failed to initialize video writer with any codec")

    def process_frame(self, frame: np.ndarray, current_time: float) -> tuple[List[Dict], Dict]:
        """Process a single frame with time-based detection and tracking"""
        
        # Check if it's time to run detection (every N seconds)
        should_detect = (current_time - self.last_detection_time) >= self.detection_interval_seconds
        
        if should_detect:
            try:
                # Resize frame for detection
                resized_frame, scale_x, scale_y = self.resize_for_detection(frame)
                
                logger.info(f"🔍 Running detection at {current_time:.2f}s (resized to {self.detection_size})")
                
                # Run detection on resized frame
                results = self.detector.detect(resized_frame)
                detections = self.detector.get_detections(results)
                
                # Scale detections back to original frame size
                self.last_detections = self.scale_detections(detections, scale_x, scale_y)
                self.last_detection_time = current_time
                
                logger.info(f"✅ Found {len(self.last_detections)} objects")
                
            except Exception as e:
                logger.warning(f"Detection failed at {current_time:.2f}s: {e}")
                # Use previous detections if current detection fails
        
        # Update tracker with latest detections
        if self.last_detections:
            detection_data = [
                [det['bbox'], det['confidence'], det['class_id']] 
                for det in self.last_detections
            ]
            tracks = self.tracker.update_tracks(detection_data, frame=frame)
        else:
            tracks = self.tracker.update_tracks([], frame=frame)
        
        # Convert tracks to detection format
        tracked_detections = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            l, t, r, b = map(int, track.to_ltrb())
            cx = (l + r) // 2
            cy = (t + b) // 2
            
            tracked_detections.append({
                'bbox': [l, t, r, b],
                'center': [cx, cy],
                'class_id': getattr(track, 'det_class', 0),
                'class_name': f"ID-{track.track_id}",
                'confidence': 1.0,
                'track_id': track.track_id
            })
        
        # Analyze detections within zones
        analysis = self.analyzer.analyze_detections(tracked_detections)
        
        return tracked_detections, analysis

    def draw_frame_info(self, frame: np.ndarray, frame_num: int, current_time: float, fps: float):
        """Draw frame information on the video"""
        time_since_last_detection = current_time - self.last_detection_time
        next_detection_in = max(0, self.detection_interval_seconds - time_since_last_detection)
        
        info_lines = [
            f"Frame: {frame_num} | Time: {current_time:.2f}s | FPS: {fps:.1f}",
            f"Detection: Every {self.detection_interval_seconds}s | Next in: {next_detection_in:.1f}s",
            f"Detection size: {self.detection_size[0]}x{self.detection_size[1]} | Objects: {len(self.last_detections)}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + (i * 25)
            cv2.putText(frame, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self, video_path: str, output_path: str = 'output.mp4', 
            show_preview: bool = True, save_video: bool = True) -> Dict[str, Any]:
        """
        Run the video zone tracking pipeline
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            show_preview: Whether to show real-time preview
            save_video: Whether to save output video
            
        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()
        
        # Validate paths
        video_path, output_path = self.validate_paths(video_path, output_path)
        
        # Setup video capture and writer
        cap = self.setup_video_capture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)  # Store FPS for time calculation
        out = self.setup_video_writer(cap, output_path) if save_video else None
        
        # Setup display window
        window_name = "Zone Tracking"
        if show_preview:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1200, 800)
        
        self.frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last_detection_time = 0  # Reset detection timer
        
        logger.info("🚀 Starting video processing...")
        logger.info(f"📐 Detection will run every {self.detection_interval_seconds}s with resize to {self.detection_size}")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                current_time = self.frame_count / self.fps  # Calculate current time in seconds
                
                # Process frame with time-based detection
                tracked_detections, analysis = self.process_frame(frame, current_time)
                
                # Draw visualizations
                self.zone_manager.draw_zones(frame)
                self.analyzer.draw_analysis_results(frame, tracked_detections, analysis)
                
                # Draw frame info
                progress = (self.frame_count / total_frames) * 100
                self.draw_frame_info(frame, self.frame_count, current_time, self.fps)
                
                # Save frame
                if save_video and out:
                    out.write(frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("⏹️ Stopped by user")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_path = f"screenshot_frame_{self.frame_count}.jpg"
                        cv2.imwrite(screenshot_path, frame)
                        logger.info(f"📸 Screenshot saved: {screenshot_path}")
                
                # Progress update
                if self.frame_count % 100 == 0:
                    logger.info(f"📊 Progress: {progress:.1f}% ({self.frame_count}/{total_frames})")
        
        except KeyboardInterrupt:
            logger.info("⏹️ Processing interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error during processing: {e}")
            raise
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Calculate statistics
        processing_time = time.time() - start_time
        avg_fps = self.frame_count / processing_time if processing_time > 0 else 0
        
        stats = {
            'total_frames': self.frame_count,
            'processing_time': processing_time,
            'average_fps': avg_fps,
            'output_path': output_path if save_video else None
        }
        
        logger.info(f"✅ Processing complete!")
        logger.info(f"📊 Stats: {self.frame_count} frames in {processing_time:.2f}s ({avg_fps:.2f} FPS)")
        if save_video:
            logger.info(f"💾 Video saved: {output_path}")
        
        return stats

    def process_batch(self, video_paths: List[str], output_dir: str = "output") -> List[Dict]:
        """Process multiple videos in batch"""
        results = []
        
        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"🎬 Processing video {i}/{len(video_paths)}: {video_path}")
            
            # Generate output path
            video_name = Path(video_path).stem
            output_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
            
            try:
                stats = self.run(video_path, output_path, show_preview=False)
                stats['input_path'] = video_path
                results.append(stats)
            except Exception as e:
                logger.error(f"❌ Failed to process {video_path}: {e}")
                results.append({'input_path': video_path, 'error': str(e)})
        
        return results

def main():
    """Main function with command line argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Zone Tracking System')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--model', default='yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--zones', default='zones/zones.json', help='Path to zones file')
    parser.add_argument('--output', default='output.mp4', help='Output video path')
    parser.add_argument('--no-preview', action='store_true', help='Disable preview window')
    parser.add_argument('--no-save', action='store_true', help='Disable video saving')
    parser.add_argument('--detection-interval', type=float, default=3.0, 
                       help='Run detection every N seconds (default: 3.0)')
    parser.add_argument('--detection-size', type=str, default='640x640',
                       help='Detection frame size as WIDTHxHEIGHT (default: 640x640)')
    
    args = parser.parse_args()
    
    # Parse detection size
    try:
        width, height = map(int, args.detection_size.split('x'))
        detection_size = (width, height)
    except:
        logger.error("❌ Invalid detection size format. Use WIDTHxHEIGHT (e.g., 640x640)")
        sys.exit(1)
    
    try:
        tracker = VideoZoneTracker(
            model_path=args.model,
            zones_file=args.zones,
            detection_interval_seconds=args.detection_interval,
            detection_size=detection_size
        )
        
        tracker.run(
            video_path=args.video_path,
            output_path=args.output,
            show_preview=not args.no_preview,
            save_video=not args.no_save
        )
        
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # For direct execution (backward compatibility)
    if len(sys.argv) == 1:
        tracker = VideoZoneTracker(
            model_path='yolov8n.pt',
            zones_file='zones/zones.json',
            detection_interval_seconds=3.0,  # Detect every 3 seconds
            detection_size=(640, 640)        # Resize to 640x640 for detection
        )
        tracker.run(r'C:\Users\admin\Pictures\222222.mp4')
    else:
        main()