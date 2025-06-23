"""
Interactive Zone Drawing Tool
Công cụ vẽ zone tương tác bằng chuột
"""
import cv2
import numpy as np
from zone_manager import ZoneManager

class ZoneDrawer:
    def __init__(self, image_path, zones_file='zones/zones.json'):
        """
        Khởi tạo Zone Drawer
        Args:
            image_path: đường dẫn đến ảnh
            zones_file: file lưu zones
        """
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        
        if self.original_image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        self.image = self.original_image.copy()
        self.zone_manager = ZoneManager(zones_file)
        
        # Trạng thái vẽ
        self.drawing = False
        self.current_points = []
        self.mode = 'rectangle' 
        self.temp_zone_name = 'New Zone'
        
        print("Zone Drawing Tool:")
        print("- Chuột trái: Vẽ zone")
        print("- 'r': Chế độ rectangle")
        print("- 'p': Chế độ polygon")
        print("- 'c': Xóa tất cả zones")
        print("- 'd': Tạo Dangerous Zone ở giữa")
        print("- 's': Lưu zones")
        print("- 'q' hoặc ESC: Thoát")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback xử lý sự kiện chuột"""
        if self.mode == 'rectangle':
            self._handle_rectangle_drawing(event, x, y, flags, param)
        elif self.mode == 'polygon':
            self._handle_polygon_drawing(event, x, y, flags, param)
    
    def _handle_rectangle_drawing(self, event, x, y, flags, param):
        """Xử lý vẽ hình chữ nhật"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_points = [(x, y)]
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Vẽ rectangle tạm thời
            temp_image = self.image.copy()
            cv2.rectangle(temp_image, self.current_points[0], (x, y), (255, 255, 0), 2)
            cv2.imshow('Zone Drawer', temp_image)
        
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1, y1 = self.current_points[0]
            x2, y2 = x, y
            
            # Đảm bảo x1 < x2, y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Thêm zone mới
            zone_name = input(f"Nhập tên zone (Enter = '{self.temp_zone_name}'): ").strip()
            if not zone_name:
                zone_name = self.temp_zone_name
            
            self.zone_manager.add_zone(zone_name, 'rectangle', [x1, y1, x2, y2])
            self.update_display()
    
    def _handle_polygon_drawing(self, event, x, y, flags, param):
        """Xử lý vẽ polygon"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            
            # Vẽ điểm vừa click
            cv2.circle(self.image, (x, y), 3, (255, 255, 0), -1)
            
            # Nếu có ít nhất 2 điểm, vẽ đường nối
            if len(self.current_points) > 1:
                cv2.line(self.image, self.current_points[-2], self.current_points[-1], (255, 255, 0), 2)
            
            cv2.imshow('Zone Drawer', self.image)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Chuột phải để hoàn thành polygon
            if len(self.current_points) >= 3:
                # Nối điểm cuối với điểm đầu
                cv2.line(self.image, self.current_points[-1], self.current_points[0], (255, 255, 0), 2)
                
                # Thêm zone mới
                zone_name = input(f"Nhập tên zone (Enter = '{self.temp_zone_name}'): ").strip()
                if not zone_name:
                    zone_name = self.temp_zone_name
                
                self.zone_manager.add_zone(zone_name, 'polygon', self.current_points.copy())
                self.current_points.clear()
                self.update_display()
            else:
                print("Polygon cần ít nhất 3 điểm!")
                self.current_points.clear()
                self.update_display()
    
    def update_display(self):
        """Cập nhật hiển thị"""
        self.image = self.original_image.copy()
        self.zone_manager.draw_zones(self.image)
        cv2.imshow('Zone Drawer', self.image)
    
    def run(self):
        """Chạy Zone Drawer"""
        cv2.namedWindow('Zone Drawer', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Zone Drawer', self.mouse_callback)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' hoặc ESC
                break
            
            elif key == ord('r'):  # Rectangle mode
                self.mode = 'rectangle'
                self.current_points.clear()
                print("Chế độ: Rectangle")
            
            elif key == ord('p'):  # Polygon mode
                self.mode = 'polygon'
                self.current_points.clear()
                print("Chế độ: Polygon")
            
            elif key == ord('c'):  # Clear all zones
                self.zone_manager.clear_zones()
                self.update_display()
                print("Đã xóa tất cả zones")
            
            elif key == ord('d'):  # Create danger zone
                h, w = self.image.shape[:2]
                self.zone_manager.create_center_danger_zone(w, h)
                self.update_display()
            
            elif key == ord('s'):  # Save zones
                self.zone_manager.save_zones()
                print("Đã lưu zones")
            
            elif key == ord('h'):  # Help
                print("\nHướng dẫn sử dụng:")
                print("- Chuột trái: Vẽ zone")
                print("- Chuột phải: Hoàn thành polygon")
                print("- 'r': Chế độ rectangle")
                print("- 'p': Chế độ polygon") 
                print("- 'c': Xóa tất cả zones")
                print("- 'd': Tạo Dangerous Zone ở giữa")
                print("- 's': Lưu zones")
                print("- 'h': Hiển thị hướng dẫn")
                print("- 'q' hoặc ESC: Thoát")
        
        cv2.destroyAllWindows()
        
        # Hỏi có lưu không trước khi thoát
        save = input("Lưu zones trước khi thoát? (y/n): ").strip().lower()
        if save == 'y':
            self.zone_manager.save_zones()

if __name__ == "__main__":
    # Sử dụng Zone Drawer
    try:
        drawer = ZoneDrawer(r'C:\Users\admin\Pictures\3.png')
        drawer.run()
    except Exception as e:
        print(f"Lỗi: {e}")