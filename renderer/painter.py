# renderer/painter.py
"""
Advanced terminal painter with colors and styling
"""

class ColorPalette:
    """Terminal color definitions"""
    
    COLORS = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'reset': '\033[0m'
    }
    
    BACKGROUNDS = {
        'black': '\033[40m',
        'red': '\033[41m',
        'green': '\033[42m',
        'yellow': '\033[43m',
        'blue': '\033[44m',
        'magenta': '\033[45m',
        'cyan': '\033[46m',
        'white': '\033[47m',
    }
    
    STYLES = {
        'bold': '\033[1m',
        'dim': '\033[2m',
        'italic': '\033[3m',
        'underline': '\033[4m',
    }

class AdvancedPainter:
    """Advanced terminal graphics painter"""
    
    def __init__(self):
        self.colors = ColorPalette()
        self.buffer = []
        self.width = 0
        self.height = 0
    
    def init_screen(self, width: int, height: int):
        """Initialize screen buffer"""
        self.width = width
        self.height = height
        self.buffer = [[' ' for _ in range(width)] for _ in range(height)]
    
    def apply_color(self, text: str, color: str, background: str = None, style: str = None) -> str:
        """Apply colors and styles to text"""
        result = ""
        
        if color in self.colors.COLORS:
            result += self.colors.COLORS[color]
        
        if background in self.colors.BACKGROUNDS:
            result += self.colors.BACKGROUNDS[background]
        
        if style in self.colors.STYLES:
            result += self.colors.STYLES[style]
        
        result += text + self.colors.COLORS['reset']
        return result
    
    def draw_component(self, component, x: int, y: int):
        """Draw a screen component at position"""
        rendered = component.render(self.width, self.height)
        color = component.get_property('color', 'white')
        
        for dy, line in enumerate(rendered):
            for dx, char in enumerate(line):
                if char.strip():  # Only draw non-space characters
                    pos_x, pos_y = x + dx, y + dy
                    if 0 <= pos_x < self.width and 0 <= pos_y < self.height:
                        colored_char = self.apply_color(char, color)
                        self.buffer[pos_y][pos_x] = colored_char
    
    def render(self) -> str:
        """Render final screen output"""
        screen_output = []
        for row in self.buffer:
            screen_output.append(''.join(row))
        return '\n'.join(screen_output)
    
    def clear(self):
        """Clear the screen buffer"""
        self.buffer = [[' ' for _ in range(self.width)] for _ in range(self.height)]