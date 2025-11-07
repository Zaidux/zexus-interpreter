# renderer/layout.py
"""
Advanced layout system with screen components and inheritance
"""

from typing import Dict, List, Any, Optional

class ScreenComponent:
    """Base class for all screen components"""
    
    def __init__(self, name: str, **properties):
        self.name = name
        self.properties = properties
        self.children = []
        self.parent = None
    
    def add_child(self, component):
        """Add a child component"""
        component.parent = self
        self.children.append(component)
    
    def get_property(self, key: str, default=None):
        """Get property with inheritance from parent"""
        if key in self.properties:
            return self.properties[key]
        elif self.parent:
            return self.parent.get_property(key, default)
        return default
    
    def render(self, width: int, height: int) -> List[str]:
        """Render component to lines of text"""
        raise NotImplementedError("Subclasses must implement render")

class Screen(ScreenComponent):
    """Main screen container with your desired syntax"""
    
    def __init__(self, name: str, **properties):
        super().__init__(name, **properties)
        self.type = "screen"
        
        # Set defaults
        self.properties.setdefault('height', 24)
        self.properties.setdefault('width', 80)
        self.properties.setdefault('color', 'white')
        self.properties.setdefault('title', '')
        self.properties.setdefault('border', True)
    
    def render(self, width: int, height: int) -> List[str]:
        """Render entire screen with all components"""
        screen_width = self.get_property('width', width)
        screen_height = self.get_property('height', height)
        screen_color = self.get_property('color', 'white')
        border = self.get_property('border', True)
        
        # Create buffer
        buffer = [[' ' for _ in range(screen_width)] for _ in range(screen_height)]
        
        # Draw border if enabled
        if border:
            self._draw_border(buffer, screen_width, screen_height)
        
        # Draw title
        title = self.get_property('title', '')
        if title:
            self._draw_title(buffer, title, screen_width)
        
        # Render children with proper positioning
        for child in self.children:
            child_buffer = child.render(screen_width - 4, screen_height - 4)
            x = child.get_property('x', 2)
            y = child.get_property('y', 2)
            
            for dy, line in enumerate(child_buffer):
                for dx, char in enumerate(line):
                    if 0 <= y+dy < screen_height and 0 <= x+dx < screen_width:
                        buffer[y+dy][x+dx] = char
        
        # Convert buffer to lines
        return [''.join(row) for row in buffer]
    
    def _draw_border(self, buffer, width, height):
        """Draw screen border"""
        border_chars = {
            'top_left': '╭', 'top_right': '╮',
            'bottom_left': '╰', 'bottom_right': '╯',
            'horizontal': '─', 'vertical': '│'
        }
        
        # Corners
        buffer[0][0] = border_chars['top_left']
        buffer[0][width-1] = border_chars['top_right']
        buffer[height-1][0] = border_chars['bottom_left']
        buffer[height-1][width-1] = border_chars['bottom_right']
        
        # Horizontal borders
        for x in range(1, width-1):
            buffer[0][x] = border_chars['horizontal']
            buffer[height-1][x] = border_chars['horizontal']
        
        # Vertical borders
        for y in range(1, height-1):
            buffer[y][0] = border_chars['vertical']
            buffer[y][width-1] = border_chars['vertical']
    
    def _draw_title(self, buffer, title, width):
        """Draw screen title"""
        if len(title) > width - 4:
            title = title[:width-7] + "..."
        title_line = f" {title} "
        start_x = (width - len(title_line)) // 2
        
        for i, char in enumerate(title_line):
            if 0 <= start_x + i < width:
                buffer[0][start_x + i] = char

class Button(ScreenComponent):
    """Button component matching your syntax"""
    
    def __init__(self, name: str, **properties):
        super().__init__(name, **properties)
        self.type = "button"
        
        # Button-specific defaults
        self.properties.setdefault('text', 'Button')
        self.properties.setdefault('width', 12)
        self.properties.setdefault('height', 3)
        self.properties.setdefault('color', 'blue')
        self.properties.setdefault('x', 0)
        self.properties.setdefault('y', 0)
        self.properties.setdefault('enabled', True)
    
    def render(self, width: int, height: int) -> List[str]:
        """Render button"""
        btn_width = self.get_property('width')
        btn_height = self.get_property('height')
        btn_text = self.get_property('text')
        enabled = self.get_property('enabled')
        color = self.get_property('color')
        
        lines = []
        
        # Top border
        if enabled:
            lines.append("┌" + "─" * (btn_width - 2) + "┐")
        else:
            lines.append("┌" + "─" * (btn_width - 2) + "┐")
        
        # Content line with text
        text_line = btn_text.center(btn_width - 2)
        if enabled:
            lines.append("│" + text_line + "│")
        else:
            lines.append("│" + text_line + "│")
        
        # Bottom border
        if enabled:
            lines.append("└" + "─" * (btn_width - 2) + "┘")
        else:
            lines.append("└" + "─" * (btn_width - 2) + "┘")
        
        # Fill remaining height
        while len(lines) < btn_height:
            lines.append(" " * btn_width)
        
        return lines

class TextBox(ScreenComponent):
    """Text input box component"""
    
    def __init__(self, name: str, **properties):
        super().__init__(name, **properties)
        self.type = "textbox"
        
        self.properties.setdefault('text', '')
        self.properties.setdefault('width', 20)
        self.properties.setdefault('height', 3)
        self.properties.setdefault('placeholder', '')
        self.properties.setdefault('x', 0)
        self.properties.setdefault('y', 0)
    
    def render(self, width: int, height: int) -> List[str]:
        """Render text box"""
        box_width = self.get_property('width')
        box_height = self.get_property('height')
        text = self.get_property('text')
        placeholder = self.get_property('placeholder')
        
        lines = []
        
        # Top border
        lines.append("┌" + "─" * (box_width - 2) + "┐")
        
        # Content line
        display_text = text if text else placeholder
        if len(display_text) > box_width - 4:
            display_text = display_text[:box_width-7] + "..."
        padded_text = display_text + " " * (box_width - 4 - len(display_text))
        lines.append("│ " + padded_text + " │")
        
        # Bottom border
        lines.append("└" + "─" * (box_width - 2) + "┘")
        
        # Fill remaining height
        while len(lines) < box_height:
            lines.append(" " * box_width)
        
        return lines

class Label(ScreenComponent):
    """Simple text label"""
    
    def __init__(self, name: str, **properties):
        super().__init__(name, **properties)
        self.type = "label"
        
        self.properties.setdefault('text', '')
        self.properties.setdefault('x', 0)
        self.properties.setdefault('y', 0)
    
    def render(self, width: int, height: int) -> List[str]:
        """Render label"""
        text = self.get_property('text')
        return [text]

class ScreenRegistry:
    """Registry for screen templates and components"""
    
    def __init__(self):
        self.screens = {}
        self.components = {}
    
    def register_screen(self, name: str, screen: Screen):
        """Register a screen template"""
        self.screens[name] = screen
    
    def register_component(self, name: str, component: ScreenComponent):
        """Register a reusable component"""
        self.components[name] = component
    
    def create_screen(self, name: str, **overrides) -> Screen:
        """Create a screen instance from template"""
        if name not in self.screens:
            raise ValueError(f"Screen '{name}' not found")
        
        template = self.screens[name]
        screen = Screen(name, **template.properties)
        
        # Apply overrides
        for key, value in overrides.items():
            screen.properties[key] = value
        
        # Copy children
        for child in template.children:
            screen.add_child(child)
        
        return screen
    
    def get_component(self, name: str) -> ScreenComponent:
        """Get a reusable component"""
        return self.components.get(name)