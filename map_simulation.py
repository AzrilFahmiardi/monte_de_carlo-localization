import pygame
import math

class MapSimulation:
    def __init__(self, width, height, grid_size):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.line_thickness = 5
        self.field_lines = [
            # Outer rectangle
            {'start': (100, 100), 'end': (700, 100)},  # Top
            {'start': (700, 100), 'end': (700, 500)},  # Right
            {'start': (700, 500), 'end': (100, 500)},  # Bottom
            {'start': (100, 500), 'end': (100, 100)},  # Left
            
            # Center line
            {'start': (400, 100), 'end': (400, 500)},
            
            # Center circle (approximated with 16 line segments)
            *self._create_circle_segments(400, 300, 50, 16)
        ]

    def _create_circle_segments(self, cx, cy, radius, segments):
        """Create line segments to approximate a circle"""
        circle_lines = []
        for i in range(segments):
            angle1 = 2 * math.pi * i / segments
            angle2 = 2 * math.pi * (i + 1) / segments
            x1 = cx + radius * math.cos(angle1)
            y1 = cy + radius * math.sin(angle1)
            x2 = cx + radius * math.cos(angle2)
            y2 = cy + radius * math.sin(angle2)
            circle_lines.append({'start': (x1, y1), 'end': (x2, y2)})
        return circle_lines

    def draw_map(self, screen):
        screen.fill((0, 128, 0))  # Background green for field
        # Draw white lines
        for line in self.field_lines:
            pygame.draw.line(screen, (255, 255, 255), 
                           line['start'], line['end'], 
                           self.line_thickness)

    def _distance_to_line_segment(self, px, py, start, end):
        """Calculate the distance from point (px,py) to line segment (start-end)"""
        x1, y1 = start
        x2, y2 = end
        
        # Vector from line start to point
        A = px - x1
        B = py - y1
        # Vector from line start to line end
        C = x2 - x1
        D = y2 - y1
        
        # Length of line segment squared
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        # Compute projection parameter
        if len_sq != 0:  # Avoid division by zero
            param = dot / len_sq
        else:
            param = -1
        
        if param < 0:
            # Point is nearest to the start of the line segment
            x = x1
            y = y1
        elif param > 1:
            # Point is nearest to the end of the line segment
            x = x2
            y = y2
        else:
            # Nearest point lies on the line segment
            x = x1 + param * C
            y = y1 + param * D
        
        # Calculate distance to nearest point
        dx = px - x
        dy = py - y
        return math.sqrt(dx * dx + dy * dy)

    def is_white_line(self, x, y):
        """Check if point (x,y) is near any white line"""
        threshold = self.line_thickness / 2 + 1  # Half line thickness plus small margin
        
        for line in self.field_lines:
            distance = self._distance_to_line_segment(x, y, line['start'], line['end'])
            if distance <= threshold:
                return True
        
        return False