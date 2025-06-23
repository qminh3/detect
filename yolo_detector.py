"""
YOLO Detection Module
Chứa các hàm liên quan đến YOLO detection
"""
import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Khởi tạo YOLO detector
        Args:
            model_path: đường dẫn model YOLO (mặc định yolov8n.pt)
        """
        self.model = YOLO(model_path)
        print(f"✅ YOLODetector: Đã load model {model_path}")

    def detect(self, image):
        """
        Thực hiện detection trên ảnh
        Args:
            image: ảnh input (numpy array BGR)
        Returns:
            results: kết quả detection từ YOLO
        """
        results = self.model.predict(image, verbose=False)[0]
        return results

    def get_detections(self, results):
        """
        Trích xuất danh sách detection từ YOLO Results
        Args:
            results: YOLO results object
        Returns:
            list: [{ bbox, center, class_id, class_name, confidence }]
        """
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # (N,4)
            classes = results.boxes.cls.cpu().numpy()  # (N,)
            confs = results.boxes.conf.cpu().numpy()  # (N,)

            for box, class_id, confidence in zip(boxes, classes, confs):
                x1, y1, x2, y2 = map(int, box.tolist())
                class_id = int(class_id)
                confidence = float(confidence)
                class_name = self.model.names.get(class_id, f"Object_{class_id}")

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                })

        return detections

    def draw_detections(self, image, detections, color=(255, 0, 0), thickness=2):
        """
        Vẽ các detection lên ảnh
        Args:
            image: ảnh input
            detections: list detections
            color: màu box
            thickness: độ dày viền
        Returns:
            image: ảnh đã vẽ
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']

            label = f"{class_name}: {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(image, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, max(thickness - 1, 1))

        return image
