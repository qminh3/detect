import cv2
import os
import time
import logging
import numpy as np
from yolo_detector import YOLODetector
from zone_manager import ZoneManager
from detection_analyzer import DetectionAnalyzer
from deep_sort_realtime.deepsort_tracker import DeepSort

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

        self.detector = YOLODetector(model_path)
        self.zone_manager = ZoneManager(zones_file)
        self.analyzer = DetectionAnalyzer(self.zone_manager)
        self.tracker = DeepSort(max_age=30, n_init=3)

        self.last_detections = []
        self.last_analysis = {}
        self.last_detect_frame = None

        logger.info("✅ Components initialized successfully.")

    def validate_paths(self, video_path: str, output_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return video_path, output_path

    def setup_video_capture(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        logger.info(f"📹 {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
        return cap, fps

    def setup_video_writer(self, cap, output_path, fps):
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise RuntimeError("Cannot open video writer.")
        return out

    def resize_for_detection(self, frame):
        h, w = frame.shape[:2]
        target_w, target_h = self.detection_size
        scale_x = w / target_w
        scale_y = h / target_h
        resized = cv2.resize(frame, (target_w, target_h))
        return resized, scale_x, scale_y

    def scale_detections(self, detections, scale_x, scale_y):
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

    def run(self, video_path, output_path='output.mp4',
            show_preview=True, save_video=True):
        video_path, output_path = self.validate_paths(video_path, output_path)
        cap, fps = self.setup_video_capture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = self.setup_video_writer(cap, output_path, fps) if save_video else None

        if show_preview:
            cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tracking", 1200, 800)

        logger.info(f"🚀 Running with detection every {self.detection_interval_seconds}s")

        frame_idx = 0
        detect_frame_interval = int(fps * self.detection_interval_seconds)
        next_detect_frame = 0  # Detect ngay frame 0, tiếp theo frame 90, 180...

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx >= next_detect_frame:
                resized, scale_x, scale_y = self.resize_for_detection(frame)
                results = self.detector.detect(resized)
                detections = self.detector.get_detections(results)
                self.last_detections = self.scale_detections(detections, scale_x, scale_y)

                # Cập nhật tracking
                dets = [[d['bbox'], d['confidence'], d['class_id']] for d in self.last_detections]
                tracks = self.tracker.update_tracks(dets, frame=frame)

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

                self.last_analysis = self.analyzer.analyze_detections(tracked)
                self.last_detect_frame = frame.copy()

                for zone, count in self.last_analysis['zone_detections'].items():
                    logger.info(f"📌 Frame {frame_idx} - Zone '{zone}': {count} objects")

                next_detect_frame = frame_idx + detect_frame_interval

            # Vẽ lại kết quả detect gần nhất
            display_frame = frame.copy()
            if self.last_detect_frame is not None:
                self.zone_manager.draw_zones(display_frame)
                self.analyzer.draw_analysis_results(display_frame, self.last_detections, self.last_analysis)

            cv2.putText(display_frame, f"Frame: {frame_idx} | Time: {frame_idx / fps:.2f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if save_video and out:
                out.write(display_frame)
            if show_preview:
                cv2.imshow("Tracking", display_frame)

            # Giới hạn FPS thực tế
            elapsed = time.time() - start_time
            remaining = max(0, (1.0 / fps) - elapsed)
            if remaining > 0:
                time.sleep(remaining)

            if show_preview:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

        cap.release()
        if out:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()

        logger.info(f"✅ Finished processing {frame_idx} frames.")

    def run_image(self, image_path, output_path='output_image.jpg', show_preview=True):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        # Resize + detect
        resized, scale_x, scale_y = self.resize_for_detection(frame)
        results = self.detector.detect(resized)
        detections = self.detector.get_detections(results)
        scaled_detections = self.scale_detections(detections, scale_x, scale_y)

        # Update tracker (optional, mostly useless for single image)
        dets = [[d['bbox'], d['confidence'], d['class_id']] for d in scaled_detections]
        tracks = self.tracker.update_tracks(dets, frame=frame)

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

        for zone, count in analysis['zone_detections'].items():
            logger.info(f"📌 Zone '{zone}': {count} objects")

        self.zone_manager.draw_zones(frame)
        self.analyzer.draw_analysis_results(frame, tracked, analysis)

        if output_path:
            cv2.imwrite(output_path, frame)
            logger.info(f"✅ Saved output to: {output_path}")

        if show_preview:
            cv2.imshow("Image Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "yolov8n.pt"
    zones_path = "zones/zones.json"
    output_path = "output1.mp4"

    tracker = VideoZoneTracker(model_path, zones_path)
    image_path = r"E:\Code\detect\test\test1.jpg"
    tracker.run_image(image_path, output_path='output_image.jpg')
