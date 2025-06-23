"""
Zone Management Module
Chứa các hàm quản lý và vẽ zones
"""
import cv2
import json
import numpy as np
import os

class ZoneManager:
    def __init__(self, zones_file='zones/zones.json'):
        """Khởi tạo Zone Manager"""
        self.zones_file = zones_file
        self.zones = []
        self.current_zone = []
        self.drawing = False
        self.zone_type = 'rectangle'  # 'rectangle' hoặc 'polygon'
        self.zone_name = ''
        
        # Load zones nếu file tồn tại
        self.load_zones()
    
    def load_zones(self):
        """Load zones từ file JSON"""
        try:
            with open(self.zones_file, 'r') as f:
                data = json.load(f)
                self.zones = data.get('zones', [])
            print(f"Đã load {len(self.zones)} zones từ {self.zones_file}")
        except FileNotFoundError:
            print(f"File {self.zones_file} không tồn tại. Tạo zones mới.")
            self.zones = []
        except json.JSONDecodeError:
            print(f"Lỗi đọc file {self.zones_file}. Tạo zones mới.")
            self.zones = []
    
    def save_zones(self):
        """Lưu zones vào file JSON"""
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(self.zones_file), exist_ok=True)
        
        data = {'zones': self.zones}
        with open(self.zones_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Đã lưu {len(self.zones)} zones vào {self.zones_file}")
    
    def add_zone(self, name, zone_type, coordinates):
        """
        Thêm zone mới
        Args:
            name: tên zone
            zone_type: 'rectangle' hoặc 'polygon'
            coordinates: tọa độ zone
        """
        zone = {
            'name': name,
            'type': zone_type,
            'coordinates': coordinates
        }
        self.zones.append(zone)
        print(f"Đã thêm zone: {name}")
    
    def remove_zone(self, index):
        """Xóa zone theo index"""
        if 0 <= index < len(self.zones):
            removed = self.zones.pop(index)
            print(f"Đã xóa zone: {removed['name']}")
            return True
        return False
    
    def clear_zones(self):
        """Xóa tất cả zones"""
        self.zones.clear()
        print("Đã xóa tất cả zones")
    
    def draw_zones(self, image, show_names=True):
        """
        Vẽ tất cả zones lên ảnh
        Args:
            image: ảnh để vẽ
            show_names: có hiển thị tên zone không
        """
        for i, zone in enumerate(self.zones):
            if zone['type'] == 'rectangle':
                self._draw_rectangle_zone(image, zone, show_names)
            elif zone['type'] == 'polygon':
                self._draw_polygon_zone(image, zone, show_names)
        return image
    
    def _draw_rectangle_zone(self, image, zone, show_names=True):
        """Vẽ rectangle zone"""
        x1, y1, x2, y2 = zone['coordinates']
        
        # Màu đỏ cho Dangerous Zone, xanh lá cho zone khác
        if 'DANGER' in zone['name'].upper():
            color = (0, 0, 255)  # Đỏ
            thickness = 3
        else:
            color = (0, 255, 0)  # Xanh lá
            thickness = 2
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        if show_names:
            cv2.putText(image, zone['name'], (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _draw_polygon_zone(self, image, zone, show_names=True):
        """Vẽ polygon zone"""
        pts = zone['coordinates']
        pts_array = np.array(pts, np.int32)
        pts_array = pts_array.reshape((-1, 1, 2))
        
        # Màu đỏ cho Dangerous Zone, xanh lá cho zone khác
        if 'DANGER' in zone['name'].upper():
            color = (0, 0, 255)  # Đỏ
            thickness = 3
        else:
            color = (0, 255, 0)  # Xanh lá
            thickness = 2
        
        cv2.polylines(image, [pts_array], isClosed=True, color=color, thickness=thickness)
        
        if show_names and len(pts) > 0:
            cv2.putText(image, zone['name'], tuple(pts[0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def check_point_in_zone(self, point, zone):
        """
        Kiểm tra điểm có nằm trong zone không
        Args:
            point: (x, y)
            zone: zone dict
        Returns:
            bool: True nếu điểm nằm trong zone
        """
        x, y = point
        
        if zone['type'] == 'rectangle':
            x1, y1, x2, y2 = zone['coordinates']
            return x1 <= x <= x2 and y1 <= y <= y2
        
        elif zone['type'] == 'polygon':
            pts = np.array(zone['coordinates'], np.int32)
            result = cv2.pointPolygonTest(pts, (x, y), False)
            return result >= 0
        
        return False
    
    def get_zones_for_point(self, point):
        """Lấy tất cả zones chứa điểm"""
        zones_containing_point = []
        for i, zone in enumerate(self.zones):
            if self.check_point_in_zone(point, zone):
                zones_containing_point.append((i, zone))
        return zones_containing_point
    
    def create_center_danger_zone(self, image_width, image_height, zone_size_ratio=0.3):
        """
        Tạo Dangerous Zone ở chính giữa ảnh
        Args:
            image_width: chiều rộng ảnh
            image_height: chiều cao ảnh
            zone_size_ratio: tỷ lệ kích thước zone so với ảnh
        """
        center_x = image_width // 2
        center_y = image_height // 2
        
        zone_width = int(image_width * zone_size_ratio)
        zone_height = int(image_height * zone_size_ratio)
        
        x1 = center_x - zone_width // 2
        y1 = center_y - zone_height // 2
        x2 = center_x + zone_width // 2
        y2 = center_y + zone_height // 2
        
        # Kiểm tra xem đã có Dangerous Zone chưa
        has_danger_zone = any('DANGER' in zone['name'].upper() for zone in self.zones)
        
        if not has_danger_zone:
            self.add_zone('DANGEROUS ZONE', 'rectangle', [x1, y1, x2, y2])
            print(f"Đã tạo Dangerous Zone ở tâm: ({x1}, {y1}) -> ({x2}, {y2})")
        else:
            print("Dangerous Zone đã tồn tại")