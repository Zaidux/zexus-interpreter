# renderer/graphics.py
"""
Advanced graphics components for clocks, animations, and visualizations
"""

import time
import math
from typing import Dict, Any, Callable
from .canvas import Canvas, Point
from .color_system import ColorPalette

class Clock:
    """Animated analog clock component"""
    
    def __init__(self, x: int, y: int, radius: int, palette: ColorPalette):
        self.x = x
        self.y = y
        self.radius = radius
        self.palette = palette
        self.show_seconds = True
        self.show_numbers = True
        self.style = "classic"  # classic, modern, minimal
    
    def draw(self, canvas: Canvas, current_time: time.struct_time = None):
        """Draw clock with current time"""
        if current_time is None:
            current_time = time.localtime()
        
        hours = current_time.tm_hour % 12
        minutes = current_time.tm_min
        seconds = current_time.tm_sec
        
        center_x, center_y = self.x, self.y
        
        # Draw clock face
        if self.style == "classic":
            self._draw_classic_face(canvas, center_x, center_y)
        elif self.style == "modern":
            self._draw_modern_face(canvas, center_x, center_y)
        else:  # minimal
            self._draw_minimal_face(canvas, center_x, center_y)
        
        # Draw hour numbers
        if self.show_numbers:
            self._draw_numbers(canvas, center_x, center_y)
        
        # Calculate hand angles
        hour_angle = math.radians((hours * 30) + (minutes * 0.5) - 90)
        minute_angle = math.radians((minutes * 6) - 90)
        second_angle = math.radians((seconds * 6) - 90)
        
        # Draw hands
        self._draw_hand(canvas, center_x, center_y, hour_angle, 
                       self.radius * 0.5, 3, self.palette.get_color("blue"))
        
        self._draw_hand(canvas, center_x, center_y, minute_angle,
                       self.radius * 0.7, 2, self.palette.get_color("green"))
        
        if self.show_seconds:
            self._draw_hand(canvas, center_x, center_y, second_angle,
                           self.radius * 0.9, 1, self.palette.get_color("red"))
        
        # Draw center dot
        canvas.draw_circle(center_x, center_y, 2, '●', 
                          self.palette.get_color("black"), fill=True)
    
    def _draw_classic_face(self, canvas: Canvas, center_x: int, center_y: int):
        """Draw classic clock face"""
        # Outer circle
        canvas.draw_circle(center_x, center_y, self.radius, '○', 
                          self.palette.get_color("white"))
        
        # Hour markers
        for hour in range(1, 13):
            angle = math.radians(hour * 30 - 90)
            marker_x = center_x + int((self.radius - 5) * math.cos(angle))
            marker_y = center_y + int((self.radius - 5) * math.sin(angle))
            canvas.draw_circle(marker_x, marker_y, 1, '•', 
                              self.palette.get_color("white"), fill=True)
    
    def _draw_modern_face(self, canvas: Canvas, center_x: int, center_y: int):
        """Draw modern clock face"""
        # Gradient background
        for r in range(self.radius, 0, -1):
            color = self.palette.mix("indigo", "purple", r/self.radius)
            canvas.draw_circle(center_x, center_y, r, ' ', bg_color=color)
    
    def _draw_minimal_face(self, canvas: Canvas, center_x: int, center_y: int):
        """Draw minimal clock face"""
        canvas.draw_circle(center_x, center_y, self.radius, '○', 
                          self.palette.get_color("gray"))
    
    def _draw_numbers(self, canvas: Canvas, center_x: int, center_y: int):
        """Draw hour numbers"""
        for hour in range(1, 13):
            angle = math.radians(hour * 30 - 90)
            dist = self.radius - 8
            x = center_x + int(dist * math.cos(angle)) - 1
            y = center_y + int(dist * math.sin(angle))
            canvas.draw_text(x, y, str(hour), self.palette.get_color("white"))
    
    def _draw_hand(self, canvas: Canvas, center_x: int, center_y: int, 
                  angle: float, length: int, thickness: int, color: RGBColor):
        """Draw clock hand"""
        end_x = center_x + int(length * math.cos(angle))
        end_y = center_y + int(length * math.sin(angle))
        canvas.draw_line(center_x, center_y, end_x, end_y, '█', color, thickness)

class ProgressMeter:
    """Animated progress meter for mining simulations"""
    
    def __init__(self, x: int, y: int, width: int, height: int, palette: ColorPalette):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.palette = palette
        self.progress = 0.0
        self.animation_time = 0.0
    
    def draw(self, canvas: Canvas, current_time: float = None):
        """Draw progress meter with animation"""
        if current_time is None:
            current_time = time.time()
        
        # Animated progress bar with pulsing effect
        animated_progress = self.progress + 0.1 * math.sin(current_time * 5)
        animated_progress = max(0, min(1, animated_progress))
        
        # Draw background
        canvas.draw_rectangle(self.x, self.y, self.width, self.height, 
                            ' ', bg_color=self.palette.get_color("gray4"))
        
        # Draw progress bar
        progress_width = int(self.width * animated_progress)
        if progress_width > 0:
            # Gradient from green to blue based on progress
            progress_color = self.palette.mix("green", "blue", animated_progress)
            canvas.draw_rectangle(self.x, self.y, progress_width, self.height,
                                ' ', bg_color=progress_color)
        
        # Draw border
        canvas.draw_rectangle(self.x, self.y, self.width, self.height, '│',
                            self.palette.get_color("white"))
        
        # Draw percentage text
        percent_text = f"{int(self.progress * 100)}%"
        text_x = self.x + (self.width - len(percent_text)) // 2
        text_y = self.y + self.height // 2
        canvas.draw_text(text_x, text_y, percent_text, self.palette.get_color("white"))

class MiningVisualization:
    """Complete mining operation visualization"""
    
    def __init__(self, x: int, y: int, width: int, height: int, palette: ColorPalette):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.palette = palette
        self.clock = Clock(x + width - 50, y + 20, 20, palette)
        self.progress_meter = ProgressMeter(x + 10, y + height - 10, width - 20, 5, palette)
        self.hash_rate = 0.0
        self.blocks_mined = 0
        self.uptime = 0.0
    
    def draw(self, canvas: Canvas, current_time: float = None):
        """Draw complete mining visualization"""
        if current_time is None:
            current_time = time.time()
        
        # Draw main container
        canvas.draw_rectangle(self.x, self.y, self.width, self.height, 
                            ' ', bg_color=self.palette.get_color("black"))
        canvas.draw_rectangle(self.x, self.y, self.width, self.height, '│',
                            self.palette.get_color("blue"))
        
        # Draw title
        title = "⛏️ Mining Operation"
        canvas.draw_text(self.x + 2, self.y + 1, title, self.palette.get_color("yellow"))
        
        # Draw stats
        stats = [
            f"Hash Rate: {self.hash_rate:.1f} MH/s",
            f"Blocks Mined: {self.blocks_mined}",
            f"Uptime: {self.uptime:.1f}s"
        ]
        
        for i, stat in enumerate(stats):
            canvas.draw_text(self.x + 2, self.y + 3 + i, stat, 
                           self.palette.get_color("green"))
        
        # Draw clock
        self.clock.draw(canvas)
        
        # Draw progress meter
        self.progress_meter.progress = (current_time % 10) / 10  # Simulated progress
        self.progress_meter.draw(canvas, current_time)
        
        # Draw mining animation
        self._draw_mining_animation(canvas, current_time)
    
    def _draw_mining_animation(self, canvas: Canvas, current_time: float):
        """Draw animated mining activity"""
        animation_x = self.x + 10
        animation_y = self.y + 10
        
        # Animated pickaxe
        swing_angle = math.sin(current_time * 3) * 0.5
        pickaxe_chars = ['⛏', '⛏', '⛏']
        
        for i, char in enumerate(pickaxe_chars):
            offset_x = int(swing_angle * (i + 1) * 2)
            canvas.draw_text(animation_x + offset_x, animation_y + i, char,
                           self.palette.get_color("orange"))
        
        # Particle effects
        for i in range(5):
            particle_x = animation_x + 15 + int(math.sin(current_time * 2 + i) * 3)
            particle_y = animation_y + 2 + int(math.cos(current_time * 2 + i) * 2)
            canvas.draw_text(particle_x, particle_y, '•', 
                           self.palette.get_color("yellow"))

class AnimationSystem:
    """System for managing and playing animations"""
    
    def __init__(self):
        self.animations = {}
        self.running_animations = {}
    
    def create_animation(self, name: str, duration: float, 
                        update_callback: Callable, 
                        draw_callback: Callable):
        """Create a new animation"""
        self.animations[name] = {
            'duration': duration,
            'update': update_callback,
            'draw': draw_callback,
            'start_time': None,
            'running': False
        }
    
    def start_animation(self, name: str, start_time: float = None):
        """Start an animation"""
        if name in self.animations:
            if start_time is None:
                start_time = time.time()
            self.animations[name]['start_time'] = start_time
            self.animations[name]['running'] = True
    
    def update_animations(self, current_time: float, canvas: Canvas):
        """Update and draw all running animations"""
        for name, anim in self.animations.items():
            if anim['running'] and anim['start_time'] is not None:
                elapsed = current_time - anim['start_time']
                progress = min(1.0, elapsed / anim['duration'])
                
                # Update animation state
                anim['update'](progress)
                
                # Draw animation
                anim['draw'](canvas, progress)
                
                # Check if animation finished
                if progress >= 1.0:
                    anim['running'] = False