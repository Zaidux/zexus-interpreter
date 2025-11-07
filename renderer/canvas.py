# renderer/canvas.py
"""
Advanced canvas system for dynamic graphics and animations
"""

import math
import time
from typing import List, Tuple, Dict, Any, Callable
from .color_system import RGBColor

class Point:
    """2D point with floating-point precision"""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)
    
    def rotate(self, angle: float, center: 'Point' = None) -> 'Point':
        """Rotate point around center (or origin)"""
        if center is None:
            center = Point(0, 0)
        
        # Translate to origin
        translated = self - center
        
        # Rotate
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x_new = translated.x * cos_a - translated.y * sin_a
        y_new = translated.x * sin_a + translated.y * cos_a
        
        # Translate back
        return Point(x_new + center.x, y_new + center.y)
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Canvas:
    """Drawing canvas for dynamic graphics"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.buffer = [[' ' for _ in range(width)] for _ in range(height)]
        self.color_buffer = [[None for _ in range(width)] for _ in range(height)]
        self.bg_buffer = [[None for _ in range(width)] for _ in range(height)]
        self.depth_buffer = [[-float('inf') for _ in range(width)] for _ in range(height)]
    
    def clear(self, char: str = ' ', color: RGBColor = None, bg_color: RGBColor = None):
        """Clear canvas with optional fill"""
        for y in range(self.height):
            for x in range(self.width):
                self.buffer[y][x] = char
                if color:
                    self.color_buffer[y][x] = color
                if bg_color:
                    self.bg_buffer[y][x] = bg_color
                self.depth_buffer[y][x] = -float('inf')
    
    def set_pixel(self, x: int, y: int, char: str = '█', 
                  color: RGBColor = None, bg_color: RGBColor = None, depth: float = 0):
        """Set a pixel with optional depth testing"""
        if 0 <= x < self.width and 0 <= y < self.height:
            if depth >= self.depth_buffer[y][x]:
                self.buffer[y][x] = char
                self.color_buffer[y][x] = color
                self.bg_buffer[y][x] = bg_color
                self.depth_buffer[y][x] = depth
    
    def draw_line(self, x1: int, y1: int, x2: int, y2: int, 
                  char: str = '█', color: RGBColor = None, thickness: int = 1):
        """Draw a line using Bresenham's algorithm"""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            # Draw thick line
            for ty in range(-thickness//2, thickness//2 + 1):
                for tx in range(-thickness//2, thickness//2 + 1):
                    self.set_pixel(x1 + tx, y1 + ty, char, color)
            
            if x1 == x2 and y1 == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
    
    def draw_circle(self, center_x: int, center_y: int, radius: int,
                   char: str = '█', color: RGBColor = None, fill: bool = False):
        """Draw a circle using midpoint algorithm"""
        if fill:
            self._draw_filled_circle(center_x, center_y, radius, char, color)
        else:
            self._draw_circle_outline(center_x, center_y, radius, char, color)
    
    def _draw_circle_outline(self, center_x: int, center_y: int, radius: int,
                            char: str, color: RGBColor):
        """Draw circle outline"""
        x = radius
        y = 0
        err = 0
        
        while x >= y:
            self.set_pixel(center_x + x, center_y + y, char, color)
            self.set_pixel(center_x + y, center_y + x, char, color)
            self.set_pixel(center_x - y, center_y + x, char, color)
            self.set_pixel(center_x - x, center_y + y, char, color)
            self.set_pixel(center_x - x, center_y - y, char, color)
            self.set_pixel(center_x - y, center_y - x, char, color)
            self.set_pixel(center_x + y, center_y - x, char, color)
            self.set_pixel(center_x + x, center_y - y, char, color)
            
            y += 1
            err += 1 + 2*y
            if 2*(err - x) + 1 > 0:
                x -= 1
                err += 1 - 2*x
    
    def _draw_filled_circle(self, center_x: int, center_y: int, radius: int,
                           char: str, color: RGBColor):
        """Draw filled circle"""
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                if x*x + y*y <= radius*radius:
                    self.set_pixel(center_x + x, center_y + y, char, color)
    
    def draw_rectangle(self, x: int, y: int, width: int, height: int,
                      char: str = '█', color: RGBColor = None, fill: bool = False):
        """Draw a rectangle"""
        if fill:
            for dy in range(height):
                for dx in range(width):
                    self.set_pixel(x + dx, y + dy, char, color)
        else:
            # Outline
            for dx in range(width):
                self.set_pixel(x + dx, y, char, color)
                self.set_pixel(x + dx, y + height - 1, char, color)
            for dy in range(height):
                self.set_pixel(x, y + dy, char, color)
                self.set_pixel(x + width - 1, y + dy, char, color)
    
    def draw_polygon(self, points: List[Tuple[int, int]], 
                    char: str = '█', color: RGBColor = None, fill: bool = False):
        """Draw a polygon"""
        if len(points) < 2:
            return
        
        # Draw outline
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            self.draw_line(x1, y1, x2, y2, char, color)
        
        # Fill polygon (simple scanline fill)
        if fill and len(points) >= 3:
            self._fill_polygon(points, char, color)
    
    def _fill_polygon(self, points: List[Tuple[int, int]], char: str, color: RGBColor):
        """Fill polygon using scanline algorithm"""
        min_y = min(y for x, y in points)
        max_y = max(y for x, y in points)
        
        for y in range(min_y, max_y + 1):
            intersections = []
            for i in range(len(points)):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % len(points)]
                
                if (y1 < y and y2 >= y) or (y2 < y and y1 >= y):
                    if y1 != y2:
                        x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                        intersections.append(x_intersect)
            
            intersections.sort()
            
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    x_start = int(intersections[i])
                    x_end = int(intersections[i + 1])
                    for x in range(x_start, x_end + 1):
                        self.set_pixel(x, y, char, color)
    
    def draw_arc(self, center_x: int, center_y: int, radius: int,
                start_angle: float, end_angle: float,
                char: str = '█', color: RGBColor = None, steps: int = 100):
        """Draw an arc"""
        angle_range = end_angle - start_angle
        for i in range(steps + 1):
            angle = start_angle + angle_range * (i / steps)
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            self.set_pixel(x, y, char, color)
    
    def draw_text(self, x: int, y: int, text: str, color: RGBColor = None):
        """Draw text on canvas"""
        for i, char in enumerate(text):
            self.set_pixel(x + i, y, char, color)
    
    def merge_with(self, other: 'Canvas', x: int = 0, y: int = 0):
        """Merge another canvas into this one"""
        for dy in range(other.height):
            for dx in range(other.width):
                if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                    if other.buffer[dy][dx] != ' ':
                        self.buffer[y + dy][x + dx] = other.buffer[dy][dx]
                        self.color_buffer[y + dy][x + dx] = other.color_buffer[dy][dx]
                        self.bg_buffer[y + dy][x + dx] = other.bg_buffer[dy][dx]