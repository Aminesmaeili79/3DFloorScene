import cv2
import numpy as np
from skimage import morphology
import json
import random


class FloorPlanConverter:
    """Handles conversion of 2D floor plans to 3D models"""
    
    def __init__(self, config=None):
        """
        Initialize converter with configuration
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or self._default_config()
    
    def _default_config(self):
        """Default configuration parameters"""
        return {
            'wall_height': 2.7,
            'floor_thickness': 0.3,
            'scale': 0.01,
            'wall_thickness_min': 3,
            'wall_thickness_max': 25,
            'min_wall_length': 30
        }
    
    def preprocess(self, image):
        """Enhanced preprocessing for architectural floor plans"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Use adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Remove very small noise
        kernel_noise = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise, iterations=1)
        
        return binary, gray, image
    
    def detect_main_walls(self, binary):
        """
        Detect only main structural walls by filtering:
        - Line thickness (walls are thicker)
        - Line length (walls are longer)
        - Connectivity (keep main structure only)
        """
        h, w = binary.shape
        
        # STRATEGY 1: Detect thick lines using morphological operations
        # Create structuring elements for different orientations
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))  # Horizontal
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))  # Vertical
        
        # Detect horizontal walls
        horizontal_walls = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
        
        # Detect vertical walls
        vertical_walls = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
        
        # Combine horizontal and vertical
        main_walls = cv2.bitwise_or(horizontal_walls, vertical_walls)
        
        # STRATEGY 2: Thickness filtering using distance transform
        kernel_connect = np.ones((2, 2), np.uint8)
        connected = cv2.dilate(binary, kernel_connect, iterations=1)
        
        # Calculate distance transform to get thickness
        dist_transform = cv2.distanceTransform(connected, cv2.DIST_L2, 5)
        
        # Keep only lines with thickness between min and max wall thickness
        min_thickness = self.config['wall_thickness_min']
        max_thickness = self.config['wall_thickness_max']
        
        thick_lines = np.zeros_like(binary)
        thick_lines[(dist_transform >= min_thickness/2) & (dist_transform <= max_thickness)] = 255
        
        # Combine both strategies
        walls_combined = cv2.bitwise_or(main_walls, thick_lines)
        
        # STRATEGY 3: Remove small disconnected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            walls_combined, connectivity=8
        )
        
        walls_filtered = np.zeros_like(binary)
        min_component_size = self.config['min_wall_length'] * 2
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Keep if large enough AND has wall-like dimensions
            if area >= min_component_size:
                if width >= self.config['min_wall_length'] or height >= self.config['min_wall_length']:
                    walls_filtered[labels == i] = 255
        
        # STRATEGY 4: Keep only the largest connected structures
        num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
            walls_filtered, connectivity=8
        )
        
        if num_labels2 > 1:
            # Get sizes of all components
            sizes = [(i, stats2[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels2)]
            sizes.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top 3 largest components
            main_structure = np.zeros_like(walls_filtered)
            for i, size in sizes[:3]:
                if size > min_component_size * 5:
                    main_structure[labels2 == i] = 255
            
            walls_filtered = main_structure
        
        # Final cleanup
        kernel_clean = np.ones((2,2), np.uint8)
        walls_filtered = cv2.morphologyEx(walls_filtered, cv2.MORPH_OPEN, kernel_clean)
        
        # Get skeleton for clean line representation
        skeleton = morphology.skeletonize(walls_filtered > 0).astype(np.uint8) * 255
        
        # Thicken skeleton to reasonable wall thickness
        kernel_thicken = np.ones((5,5), np.uint8)
        walls_final = cv2.dilate(skeleton, kernel_thicken, iterations=1)
        
        return walls_final, skeleton
    
    def detect_doors_windows(self, walls, binary):
        """
        Detect doors and windows by finding gaps in walls
        
        Args:
            walls: Binary wall mask
            binary: Original binary image
            
        Returns:
            dict: Detected doors and windows
        """
        # Invert to find gaps
        gaps = cv2.bitwise_not(walls)
        
        # Clean up to isolate actual gaps
        kernel = np.ones((3,3), np.uint8)
        gaps = cv2.morphologyEx(gaps, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours of gaps
        contours, _ = cv2.findContours(gaps, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        doors = []
        windows = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip very small gaps (noise)
            if area < 100:
                continue
            
            # Skip very large gaps (not openings)
            if area > 5000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = max(w, h) / (min(w, h) + 1)
            
            # Skip non-linear gaps (should be rectangular)
            if aspect_ratio < 2:
                continue
            
            # Calculate dimensions
            scale = self.config['scale']
            width_m = w * scale
            height_m = h * scale
            
            # Center point
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w//2, y + h//2
            
            opening = {
                'contour': contour,
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'width_m': round(width_m, 2),
                'height_m': round(height_m, 2),
                'area': area,
                'orientation': 'horizontal' if w > h else 'vertical'
            }
            
            # Classify: doors are typically larger than windows
            # Typical door: 0.8-1.2m wide
            # Typical window: 0.3-0.8m wide
            if area > 800 or (width_m > 0.6 and width_m < 1.5):
                opening['type'] = 'door'
                doors.append(opening)
            elif area > 200:
                opening['type'] = 'window'
                windows.append(opening)
        
        return {
            'doors': doors,
            'windows': windows,
            'total_openings': len(doors) + len(windows)
        }
    
    def place_lights(self, rooms):
        """
        Automatically place lights in rooms based on size and type
        
        Args:
            rooms: List of detected rooms
            
        Returns:
            list: Light positions and properties
        """
        lights = []
        wall_h = self.config['wall_height']
        floor_h = self.config['floor_thickness']
        scale = self.config['scale']
        
        for room in rooms:
            cx, cy = room['center']
            
            # Calculate light intensity based on room size
            # Larger rooms need brighter lights
            base_intensity = min(1.0, room['area_m2'] / 20)
            
            # Determine light type based on room
            if room['type'] == 'Kitchen':
                light_type = 'bright'
                intensity = base_intensity * 1.3
                color = [1.0, 1.0, 0.95]  # Slightly warm white
            elif room['type'] == 'Bathroom':
                light_type = 'bright'
                intensity = base_intensity * 1.2
                color = [1.0, 1.0, 1.0]  # Pure white
            elif room['type'] == 'Bedroom':
                light_type = 'warm'
                intensity = base_intensity * 0.8
                color = [1.0, 0.95, 0.85]  # Warm white
            elif room['type'] == 'Living Room':
                light_type = 'ambient'
                intensity = base_intensity * 1.0
                color = [1.0, 0.98, 0.9]  # Soft warm
            else:
                light_type = 'normal'
                intensity = base_intensity
                color = [1.0, 1.0, 0.95]
            
            light = {
                'position': [cx * scale, cy * scale, floor_h + wall_h - 0.1],
                'intensity': round(intensity, 2),
                'color': color,
                'type': light_type,
                'room_id': room['id'],
                'room_type': room['type']
            }
            
            lights.append(light)
        
        return lights
    
    def detect_rooms(self, walls, binary):
        """
        Detect rooms by finding enclosed spaces between walls
        
        Args:
            walls: Binary wall mask
            binary: Original binary image
            
        Returns:
            dict: Room information including contours, areas, and labels
        """
        # Invert walls to get spaces
        spaces = cv2.bitwise_not(walls)
        
        # Remove small noise
        kernel = np.ones((3,3), np.uint8)
        spaces = cv2.morphologyEx(spaces, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Fill small gaps in walls to close rooms
        kernel_close = np.ones((7,7), np.uint8)
        spaces_closed = cv2.morphologyEx(spaces, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Find contours (each contour is a potential room)
        contours, _ = cv2.findContours(spaces_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rooms = []
        room_colors = [
            [0.8, 0.9, 1.0],   # Light blue (living room)
            [1.0, 0.9, 0.8],   # Light orange (bedroom)
            [0.9, 1.0, 0.9],   # Light green (kitchen)
            [1.0, 0.95, 0.85], # Light yellow (bathroom)
            [0.95, 0.9, 1.0],  # Light purple (office)
            [1.0, 0.93, 0.93], # Light pink (dining)
        ]
        
        # Room type classification based on size
        room_types = ['Living Room', 'Bedroom', 'Kitchen', 'Bathroom', 'Office', 'Dining Room', 'Hallway', 'Closet']
        
        min_room_area = 500  # Minimum pixels for a room
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < min_room_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate room properties
            perimeter = cv2.arcLength(contour, True)
            
            # Approximate room shape
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Simple room type classification based on area and aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Classify room type
            if area < 1000:
                room_type = 'Closet'
            elif area < 2000:
                room_type = 'Bathroom'
            elif w < h * 3 and h < w * 3:  # Relatively square
                if area > 5000:
                    room_type = 'Living Room'
                else:
                    room_type = 'Bedroom'
            elif aspect_ratio > 2 or aspect_ratio < 0.5:  # Long and narrow
                room_type = 'Hallway'
            else:
                room_type = random.choice(['Kitchen', 'Office', 'Dining Room'])
            
            # Calculate center point for label
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w//2, y + h//2
            
            # Calculate dimensions in meters
            scale = self.config['scale']
            width_m = w * scale
            height_m = h * scale
            area_m2 = area * (scale ** 2)
            
            room_info = {
                'id': idx,
                'type': room_type,
                'contour': contour,
                'area_pixels': int(area),
                'area_m2': round(area_m2, 2),
                'width_m': round(width_m, 2),
                'height_m': round(height_m, 2),
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'color': room_colors[idx % len(room_colors)]
            }
            
            rooms.append(room_info)
        
        # Sort rooms by area (largest first)
        rooms.sort(key=lambda r: r['area_pixels'], reverse=True)
        
        # Re-assign IDs after sorting
        for idx, room in enumerate(rooms):
            room['id'] = idx
        
        return rooms
    
    def build_3d_model(self, walls, dims, rooms=None):
        """Build 3D model directly from wall structure with optional room coloring"""
        h, w = dims
        wall_h = self.config['wall_height']
        floor_h = self.config['floor_thickness']
        scale = self.config['scale']
        
        vertices = []
        faces = []
        colors = []
        
        # 1. CREATE FLOOR BASE
        wall_points = np.argwhere(walls > 0)
        if len(wall_points) == 0:
            return None
        
        min_y, min_x = wall_points.min(axis=0)
        max_y, max_x = wall_points.max(axis=0)
        
        # Add margin around walls
        margin = 30
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x = min(w, max_x + margin)
        max_y = min(h, max_y + margin)
        
        # Floor corners
        floor_corners = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])
        
        # Floor bottom vertices
        for pt in floor_corners:
            vertices.append([pt[0]*scale, pt[1]*scale, 0])
        
        # Floor top vertices
        for pt in floor_corners:
            vertices.append([pt[0]*scale, pt[1]*scale, floor_h])
        
        # Floor bottom faces
        faces.extend([[0,1,2], [0,2,3]])
        colors.extend([[0.83,0.65,0.46], [0.83,0.65,0.46]])
        
        # Floor top faces (will be covered by room floors if rooms detected)
        if rooms is None or len(rooms) == 0:
            faces.extend([[4,6,5], [4,7,6]])
            colors.extend([[0.87,0.72,0.53], [0.87,0.72,0.53]])
        
        # Floor side faces
        for i in range(4):
            ni = (i+1) % 4
            faces.extend([[i,ni,i+4], [ni,ni+4,i+4]])
            colors.extend([[0.78,0.60,0.42], [0.78,0.60,0.42]])
        
        # 1.5 CREATE COLORED ROOM FLOORS (if rooms detected)
        if rooms and len(rooms) > 0:
            for room in rooms:
                contour = room['contour']
                room_color = room['color']
                
                # Simplify contour for floor
                epsilon = 1.0
                approx = cv2.approxPolyDP(contour, epsilon, True)
                pts = approx.reshape(-1, 2)
                
                if len(pts) < 3:
                    continue
                
                start_v = len(vertices)
                n = len(pts)
                
                # Add vertices at floor top level
                for pt in pts:
                    vertices.append([pt[0]*scale, pt[1]*scale, floor_h])
                
                # Triangulate the room floor using fan triangulation
                for i in range(1, n - 1):
                    faces.append([start_v, start_v + i, start_v + i + 1])
                    colors.append(room_color)
        
        # 2. CREATE WALLS
        # Find wall contours
        contours, _ = cv2.findContours(walls, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        wall_count = 0
        for contour in contours:
            if len(contour) < 2:
                continue
            
            # Simplify contour
            epsilon = 2.0
            approx = cv2.approxPolyDP(contour, epsilon, False)
            pts = approx.reshape(-1, 2)
            
            if len(pts) < 2:
                continue
            
            start_v = len(vertices)
            n = len(pts)
            
            # Bottom vertices (on floor top)
            for pt in pts:
                vertices.append([pt[0]*scale, pt[1]*scale, floor_h])
            
            # Top vertices
            for pt in pts:
                vertices.append([pt[0]*scale, pt[1]*scale, floor_h + wall_h])
            
            # Create wall faces
            for i in range(n-1):
                v1, v2 = start_v + i, start_v + i + 1
                v3, v4 = start_v + n + i, start_v + n + i + 1
                
                faces.extend([[v1,v3,v2], [v2,v3,v4]])
                colors.extend([[0.9,0.9,0.9], [0.9,0.9,0.9]])
            
            wall_count += 1
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        colors = np.array(colors, dtype=np.float32)
        
        return {
            'vertices': vertices,
            'faces': faces,
            'colors': colors,
            'scale': scale,
            'wall_height': wall_h
        }
    
    def create_preview(self, walls, rgb):
        """
        Create preview image showing detected walls
        
        Args:
            walls: Binary wall mask
            rgb: Original RGB image
            
        Returns:
            numpy.ndarray: Preview image
        """
        preview = rgb.copy()
        if len(preview.shape) == 2:
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
        preview[walls > 0] = [255, 0, 0]  # Red walls
        return cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    
    def create_preview_with_rooms(self, walls, rgb, rooms):
        """
        Create preview image with walls and room labels
        
        Args:
            walls: Binary wall mask
            rgb: Original RGB image
            rooms: List of detected rooms
            
        Returns:
            numpy.ndarray: Preview image with room labels
        """
        preview = rgb.copy()
        if len(preview.shape) == 2:
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
        
        # Draw room fills with transparency effect
        overlay = preview.copy()
        for room in rooms:
            # Convert room color from 0-1 to 0-255
            color_bgr = tuple(int(c * 255) for c in reversed(room['color']))
            cv2.drawContours(overlay, [room['contour']], -1, color_bgr, -1)
        
        # Blend overlay with original
        preview = cv2.addWeighted(preview, 0.6, overlay, 0.4, 0)
        
        # Draw walls on top
        preview[walls > 0] = [255, 0, 0]  # Red walls
        
        # Add room labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        for room in rooms:
            cx, cy = room['center']
            
            # Room type
            text = room['type']
            font_scale = 0.6
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(preview, 
                         (cx - text_width//2 - 5, cy - text_height - 5),
                         (cx + text_width//2 + 5, cy + 5),
                         (255, 255, 255), -1)
            
            # Draw text
            cv2.putText(preview, text, 
                       (cx - text_width//2, cy), 
                       font, font_scale, (0, 0, 0), thickness)
            
            # Draw area below
            area_text = f"{room['area_m2']}m²"
            font_scale_small = 0.4
            thickness_small = 1
            (area_width, area_height), _ = cv2.getTextSize(area_text, font, font_scale_small, thickness_small)
            
            cv2.rectangle(preview,
                         (cx - area_width//2 - 3, cy + 10),
                         (cx + area_width//2 + 3, cy + 10 + area_height + 6),
                         (255, 255, 255), -1)
            
            cv2.putText(preview, area_text,
                       (cx - area_width//2, cy + 10 + area_height),
                       font, font_scale_small, (100, 100, 100), thickness_small)
        
        return cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    
    def convert(self, image):
        """
        Main conversion pipeline with room detection, door/window detection, and lighting
        
        Args:
            image: Input image (BGR format from cv2.imread)
            
        Returns:
            dict: Conversion results including model, rooms, openings, lights, and preview
        """
        # Preprocess
        binary, gray, rgb = self.preprocess(image)
        
        # Detect walls
        walls, skeleton = self.detect_main_walls(binary)
        
        # Detect rooms
        rooms = self.detect_rooms(walls, binary)
        
        # Detect doors and windows
        openings = self.detect_doors_windows(walls, binary)
        
        # Place lights in rooms
        lights = self.place_lights(rooms)
        
        # Build 3D model with room colors
        model = self.build_3d_model(walls, binary.shape, rooms)
        
        # Create simple preview without room labels
        preview = self.create_preview(walls, rgb)
        
        return {
            'model': model,
            'preview': preview,
            'walls': walls,
            'skeleton': skeleton,
            'binary': binary,
            'rooms': rooms,
            'openings': openings,
            'lights': lights
        }
    
    def export_obj(self, model, filename):
        """
        Export 3D model as OBJ file
        
        Args:
            model: 3D model dictionary
            filename: Output filename
        """
        v, f = model['vertices'], model['faces']
        
        with open(filename, 'w') as file:
            file.write("# 3D Floor Plan Model\n")
            file.write(f"# Generated with {len(v)} vertices and {len(f)} faces\n\n")
            
            for vertex in v:
                file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            file.write("\n")
            
            for face in f:
                file.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    def export_metadata(self, model, filename):
        """
        Export model metadata as JSON
        
        Args:
            model: 3D model dictionary
            filename: Output filename
        """
        metadata = {
            'vertices_count': len(model['vertices']),
            'faces_count': len(model['faces']),
            'scale': model['scale'],
            'wall_height': model['wall_height'],
            'config': self.config
        }
        
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)