"""
Detection Analysis Module
Phân tích detection và zone
"""
import cv2
import numpy as np

class DetectionAnalyzer:
    def __init__(self, zone_manager):
      
        self.zone_manager = zone_manager
        self.danger_count = 0
        self.total_count = 0
    
    def analyze_detections(self, detections):
        
        analysis = {
            'total_detections': len(detections),
            'danger_detections': 0,
            'zone_detections': {},
            'outside_detections': 0,
            'detection_details': []
        }
        
        for detection in detections:
            center = detection['center']
            zones_containing = self.zone_manager.get_zones_for_point(center)
            
            detection_info = {
                'detection': detection,
                'zones': zones_containing,
                'is_danger': False
            }
            
            if zones_containing:
                # Đối tượng nằm trong ít nhất 1 zone
                for zone_idx, zone in zones_containing:
                    zone_name = zone['name']
                    
                    # Đếm detection theo zone
                    if zone_name not in analysis['zone_detections']:
                        analysis['zone_detections'][zone_name] = 0
                    analysis['zone_detections'][zone_name] += 1
                    
                    # Kiểm tra có phải danger zone không
                    if 'DANGER' in zone_name.upper():
                        analysis['danger_detections'] += 1
                        detection_info['is_danger'] = True
                        break
            else:
                # Đối tượng nằm ngoài tất cả zones
                analysis['outside_detections'] += 1
            
            analysis['detection_details'].append(detection_info)
        
        return analysis
    
    def draw_analysis_results(self, image, detections, analysis):
       
        for detail in analysis['detection_details']:
            detection = detail['detection']
            zones = detail['zones']
            is_danger = detail['is_danger']
            
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            if is_danger:
                # Đối tượng trong vùng nguy hiểm
                color = (0, 0, 255)  # Đỏ
                thickness = 3
                cv2.putText(image, f"DANGER! {class_name}", (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness)
                
                # Hiển thị tên zone
                if zones:
                    zone_name = zones[0][1]['name']
                    cv2.putText(image, f"IN {zone_name}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            elif zones:
                # Đối tượng trong zone bình thường
                color = (0, 255, 255)  # Vàng
                thickness = 2
                zone_name = zones[0][1]['name']
                cv2.putText(image, f"IN {zone_name}", (x1, y1 - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
                cv2.putText(image, f"{class_name}: {confidence:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            else:
                # Đối tượng ngoài tất cả zones
                color = (255, 0, 0)  # Xanh dương
                thickness = 2
                cv2.putText(image, f"{class_name}: {confidence:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            
            # Vẽ bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Hiển thị thống kê tổng quan
        self._draw_statistics(image, analysis)
        
        return image
    
    def _draw_statistics(self, image, analysis):
  
        y_offset = 30
        
        # Thông tin tổng quan
        total = analysis['total_detections']
        danger = analysis['danger_detections']
        outside = analysis['outside_detections']
        
        info_text = f"Total: {total} | Danger: {danger} | Outside: {outside}"
        
        # Vẽ background cho text
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.rectangle(image, (5, 5), (text_size[0] + 15, 35), (0, 0, 0), -1)
        
        # Vẽ text
        cv2.putText(image, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Thông tin từng zone
        y_offset = 60
        for zone_name, count in analysis['zone_detections'].items():
            zone_text = f"{zone_name}: {count}"
            cv2.putText(image, zone_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
        
        # Cảnh báo nếu có đối tượng trong vùng nguy hiểm
        if danger > 0:
            warning_text = "⚠️ WARNING: Objects in danger zone!"
            cv2.putText(image, warning_text, (10, image.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    def print_analysis_report(self, analysis):
      
        print("\n" + "="*50)
        print("BÁO CÁO PHÂN TÍCH DETECTION")
        print("="*50)
        
        print(f"Tổng số đối tượng phát hiện: {analysis['total_detections']}")
        print(f"Đối tượng trong vùng nguy hiểm: {analysis['danger_detections']}")
        print(f"Đối tượng ngoài tất cả zones: {analysis['outside_detections']}")
        
        print("\nPhân bố theo zones:")
        for zone_name, count in analysis['zone_detections'].items():
            print(f"  - {zone_name}: {count} đối tượng")
        
        if analysis['danger_detections'] > 0:
            print("\n⚠️  CẢNH BÁO: Có đối tượng trong vùng nguy hiểm!")
        
        print("="*50)