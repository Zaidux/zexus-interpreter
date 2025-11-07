# renderer/painter.py
"""
Modern terminal painter with gradients, shadows, and advanced styling
"""

from .color_system import ColorPalette, RGBColor, Theme

class ModernPainter:
    """Modern terminal graphics with advanced features"""
    
    def __init__(self):
        self.palette = ColorPalette()
        self.current_theme = None
        self.buffer = []
        self.width = 0
        self.height = 0
        
        # Define some beautiful default themes
        self._setup_default_themes()
    
    def _setup_default_themes(self):
        """Setup beautiful default themes"""
        
        # Modern Dark Theme
        dark_theme = Theme("dark", self.palette)
        dark_theme.set_primary("blue")
        dark_theme.set_accent("mint")
        dark_theme.set_background("black")
        dark_theme.set_text("white")
        self.themes = {'dark': dark_theme}
        
        # Light Theme
        light_theme = Theme("light", self.palette)
        light_theme.set_primary("indigo")
        light_theme.set_accent("pink")
        light_theme.set_background("gray6")
        light_theme.set_text("black")
        self.themes['light'] = light_theme
        
        # Set default theme
        self.current_theme = dark_theme
    
    def init_screen(self, width: int, height: int):
        """Initialize screen buffer"""
        self.width = width
        self.height = height
        self.buffer = [[' ' for _ in range(width)] for _ in range(height)]
        self.color_buffer = [[None for _ in range(width)] for _ in range(height)]
        self.bg_buffer = [[None for _ in range(width)] for _ in range(height)]
    
    def set_theme(self, theme_name: str):
        """Set current theme"""
        if theme_name in self.themes:
            self.current_theme = self.themes[theme_name]
        else:
            raise ValueError(f"Unknown theme: {theme_name}")
    
    def create_theme(self, name: str, primary: str, accent: str, background: str, text: str):
        """Create a new theme"""
        theme = Theme(name, self.palette)
        theme.set_primary(primary)
        theme.set_accent(accent)
        theme.set_background(background)
        theme.set_text(text)
        self.themes[name] = theme
        return theme
    
    def mix_colors(self, color1: str, color2: str, ratio: float = 0.5) -> RGBColor:
        """Mix two colors by name"""
        return self.palette.mix(color1, color2, ratio)
    
    def define_color(self, name: str, r: int, g: int, b: int):
        """Define a custom color"""
        self.palette.define_color(name, r, g, b)
    
    def create_gradient(self, start_color: str, end_color: str, steps: int, name: str):
        """Create a color gradient"""
        return self.palette.create_gradient(start_color, end_color, steps, name)
    
    def apply_style(self, text: str, color: str = None, background: str = None, 
                   style: str = None, use_theme: bool = True) -> str:
        """Apply advanced styling with theme support"""
        
        # Use theme colors if requested
        if use_theme and self.current_theme:
            if color and color.startswith('theme.'):
                theme_color = color[6:]  # Remove 'theme.' prefix
                color_obj = self.current_theme.get(theme_color)
            else:
                color_obj = self.palette.get_color(color) if color else None
            
            if background and background.startswith('theme.'):
                theme_bg = background[6:]
                bg_obj = self.current_theme.get(theme_bg)
            else:
                bg_obj = self.palette.get_color(background) if background else None
        else:
            color_obj = self.palette.get_color(color) if color else None
            bg_obj = self.palette.get_color(background) if background else None
        
        # Build style string
        style_parts = []
        
        if color_obj:
            style_parts.append(color_obj.to_ansi())
        
        if bg_obj:
            style_parts.append(bg_obj.to_ansi(background=True))
        
        if style:
            style_map = {
                'bold': '\033[1m',
                'dim': '\033[2m',
                'italic': '\033[3m',
                'underline': '\033[4m',
                'blink': '\033[5m',
                'reverse': '\033[7m',
            }
            if style in style_map:
                style_parts.append(style_map[style])
        
        if style_parts:
            return ''.join(style_parts) + text + '\033[0m'
        else:
            return text
    
    def draw_rounded_rect(self, x: int, y: int, width: int, height: int, 
                         color: str = None, background: str = None):
        """Draw a rounded rectangle"""
        rounded_chars = {
            'top_left': '╭', 'top_right': '╮',
            'bottom_left': '╰', 'bottom_right': '╯',
            'horizontal': '─', 'vertical': '│'
        }
        
        # Draw corners
        self._set_char(x, y, rounded_chars['top_left'], color, background)
        self._set_char(x + width - 1, y, rounded_chars['top_right'], color, background)
        self._set_char(x, y + height - 1, rounded_chars['bottom_left'], color, background)
        self._set_char(x + width - 1, y + height - 1, rounded_chars['bottom_right'], color, background)
        
        # Draw horizontal borders
        for dx in range(1, width - 1):
            self._set_char(x + dx, y, rounded_chars['horizontal'], color, background)
            self._set_char(x + dx, y + height - 1, rounded_chars['horizontal'], color, background)
        
        # Draw vertical borders
        for dy in range(1, height - 1):
            self._set_char(x, y + dy, rounded_chars['vertical'], color, background)
            self._set_char(x + width - 1, y + dy, rounded_chars['vertical'], color, background)
        
        # Fill interior
        for dy in range(1, height - 1):
            for dx in range(1, width - 1):
                self._set_char(x + dx, y + dy, ' ', color, background)
    
    def draw_gradient_background(self, start_color: str, end_color: str, direction: str = 'vertical'):
        """Draw a gradient background"""
        if direction == 'vertical':
            gradient = self.palette.create_gradient(start_color, end_color, self.height, 'temp_gradient')
            for y in range(self.height):
                color = gradient[y]
                for x in range(self.width):
                    self.bg_buffer[y][x] = color
        else:  # horizontal
            gradient = self.palette.create_gradient(start_color, end_color, self.width, 'temp_gradient')
            for x in range(self.width):
                color = gradient[x]
                for y in range(self.height):
                    self.bg_buffer[y][x] = color
    
    def _set_char(self, x: int, y: int, char: str, color: str = None, background: str = None):
        """Set character with color at position"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.buffer[y][x] = char
            if color:
                self.color_buffer[y][x] = self.palette.get_color(color)
            if background:
                self.bg_buffer[y][x] = self.palette.get_color(background)
    
    def render(self) -> str:
        """Render final screen with colors"""
        screen_output = []
        for y in range(self.height):
            line = []
            for x in range(self.width):
                char = self.buffer[y][x]
                fg_color = self.color_buffer[y][x]
                bg_color = self.bg_buffer[y][x]
                
                if fg_color or bg_color:
                    styled_char = ""
                    if fg_color:
                        styled_char += fg_color.to_ansi()
                    if bg_color:
                        styled_char += bg_color.to_ansi(background=True)
                    styled_char += char + '\033[0m'
                    line.append(styled_char)
                else:
                    line.append(char)
            
            screen_output.append(''.join(line))
        
        return '\n'.join(screen_output)
    
    def clear(self):
        """Clear the screen buffers"""
        self.buffer = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        self.color_buffer = [[None for _ in range(self.width)] for _ in range(self.height)]
        self.bg_buffer = [[None for _ in range(self.width)] for _ in range(self.height)]