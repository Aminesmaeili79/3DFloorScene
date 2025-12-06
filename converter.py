import cv2
import numpy as np
from skimage import morphology
import json


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
        kernel_noise = np.ones((2,2), np.uint8)
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
        kernel_connect = np.ones((3,3), np.uint8)
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
    
    def build_3d_model(self, walls, dims):
        """Build 3D model directly from wall structure"""
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
        colors.extend([[0.6,0.5,0.4], [0.6,0.5,0.4]])
        
        # Floor top faces (wood color)
        faces.extend([[4,6,5], [4,7,6]])
        colors.extend([[0.75,0.65,0.55], [0.75,0.65,0.55]])
        
        # Floor side faces
        for i in range(4):
            ni = (i+1) % 4
            faces.extend([[i,ni,i+4], [ni,ni+4,i+4]])
            colors.extend([[0.6,0.5,0.4], [0.6,0.5,0.4]])
        
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