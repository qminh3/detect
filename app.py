import cv2
import sys
import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from yolo_detector import YOLODetector
from zone_manager import ZoneManager
from detection_analyzer import DetectionAnalyzer
from deep_sort_realtime.deepsort_tracker import DeepSort

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoZoneTracker:
    def __init__(self, model_path: str, zones_file: str,
                 detection_interval_seconds: float = 3.0,
                 detection_size: tuple = (640, 640)):
        self.detection_interval_seconds = detection_interval_seconds
        self.detection_size = detection_size
        self.frame_count = 0
        self.last_detections = []
        self.last_detection_time = 0
        self.fps = 0

        try:
            self.detector = YOLODetector(model_path)
            self.zone_manager = ZoneManager(zones_file)
            self.analyzer = DetectionAnalyzer(self.zone_manager)
            self.tracker = DeepSort(max_age=30, n_init=3)
            logger.info("✅ Components initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise

    def validate_paths(self, video_path: str, output_path: str) -> tuple[str, str]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return video_path, output_path

    def setup_video_capture(self, video_path: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        logger.info(f"📹 {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
        return cap

    def setup_video_writer(self, cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        codecs = ['mp4v', 'XVID', 'MJPG']
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if out.isOpened():
                logger.info(f"✅ Using codec: {codec}")
                return out
        raise RuntimeError("Failed to initialize video writer.")

    def resize_for_detection(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        target_w, target_h = self.detection_size
        scale_x = w / target_w
        scale_y = h / target_h
        resized = cv2.resize(frame, (target_w, target_h))
        return resized, scale_x, scale_y

    def scale_detections(self, detections: List[Dict], scale_x: float, scale_y: float):
        scaled = []
        for det in detections:
            bbox = det['bbox']
            scaled_bbox = [
                int(bbox[0] * scale_x),
                int(bbox[1] * scale_y),
                int(bbox[2] * scale_x),
                int(bbox[3] * scale_y)
            ]
            cx = (scaled_bbox[0] + scaled_bbox[2]) // 2
            cy = (scaled_bbox[1] + scaled_bbox[3]) // 2
            det_copy = det.copy()
            det_copy['bbox'] = scaled_bbox
            det_copy['center'] = [cx, cy]
            scaled.append(det_copy)
        return scaled

    def process_frame(self, frame: np.ndarray, current_time: float):
        frame_start = time.time()
        should_detect = (current_time - self.last_detection_time) >= self.detection_interval_seconds
        if should_detect:
            resized, scale_x, scale_y = self.resize_for_detection(frame)
            results = self.detector.detect(resized)
            detections = self.detector.get_detections(results)
            self.last_detections = self.scale_detections(detections, scale_x, scale_y)
            self.last_detection_time = current_time
            logger.info(f"🔍 Detection at {current_time:.2f}s: {len(self.last_detections)} objects")
        if self.last_detections:
            dets = [[d['bbox'], d['confidence'], d['class_id']] for d in self.last_detections]
            tracks = self.tracker.update_tracks(dets, frame=frame)
        else:
            tracks = self.tracker.update_tracks([], frame=frame)

        tracked = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            l, t_, r, b = map(int, t.to_ltrb())
            cx, cy = (l + r) // 2, (t_ + b) // 2
            tracked.append({
                'bbox': [l, t_, r, b],
                'center': [cx, cy],
                'class_id': getattr(t, 'det_class', 0),
                'class_name': f"ID-{t.track_id}",
                'confidence': 1.0,
                'track_id': t.track_id
            })
        analysis = self.analyzer.analyze_detections(tracked)

        # 👉 Log zone status every detect cycle
        if should_detect:
            for zone_id, count in analysis.items():
                logger.info(f"📌 Zone '{zone_id}': {count} people inside")

        frame_time = time.time() - frame_start
        logger.info(f"⏱️ Frame processed in {frame_time:.3f}s")
        return tracked, analysis

    def draw_frame_info(self, frame, frame_num, current_time, fps):
        lines = [
            f"Frame: {frame_num} | Time: {current_time:.2f}s | FPS: {fps:.1f}",
            f"Detect every {self.detection_interval_seconds}s",
            f"Objects: {len(self.last_detections)}"
        ]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self, video_path: str, output_path: str = 'output.mp4',
            show_preview: bool = True, save_video: bool = True):
        start_time = time.time()
        video_path, output_path = self.validate_paths(video_path, output_path)
        cap = self.setup_video_capture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        out = self.setup_video_writer(cap, output_path) if save_video else None

        if show_preview:
            cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tracking", 1200, 800)

        self.frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info("🚀 Processing started...")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame_count += 1
                current_time = self.frame_count / self.fps
                tracked, analysis = self.process_frame(frame, current_time)
                self.zone_manager.draw_zones(frame)
                self.analyzer.draw_analysis_results(frame, tracked, analysis)
                self.draw_frame_info(frame, self.frame_count, current_time, self.fps)
                if save_video and out:
                    out.write(frame)
                if show_preview:
                    cv2.imshow("Tracking", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("⏹️ Stopped by user")
                        break
                if self.frame_count % 100 == 0:
                    logger.info(f"📊 Progress: {self.frame_count}/{total_frames}")
        finally:
            cap.release()
            if out:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
            elapsed = time.time() - start_time
            logger.info(f"✅ Done: {self.frame_count} frames in {elapsed:.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Input video file")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--zones", default="zones/zones.json")
    parser.add_argument("--output", default="output.mp4")
    args = parser.parse_args()

    tracker = VideoZoneTracker(args.model, args.zones)
    tracker.run(args.video_path, args.output)
