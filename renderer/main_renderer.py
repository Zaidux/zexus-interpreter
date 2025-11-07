# renderer/main_renderer.py
"""
Main screen renderer with Zexus integration
"""

from .layout import Screen, Button, TextBox, Label, ScreenRegistry
from .painter import AdvancedPainter

class ZexusScreenRenderer:
    """Main renderer that supports your desired syntax"""
    
    def __init__(self):
        self.registry = ScreenRegistry()
        self.painter = AdvancedPainter()
        self._setup_builtin_components()
    
    def _setup_builtin_components(self):
        """Setup built-in screen components"""
        
        # Built-in button styles
        login_button = Button("login_button", 
                            text="Login",
                            color="green",
                            width=12,
                            height=3)
        
        cancel_button = Button("cancel_button",
                             text="Cancel",
                             color="red",
                             width=12,
                             height=3)
        
        self.registry.register_component("login_button", login_button)
        self.registry.register_component("cancel_button", cancel_button)
    
    def define_screen(self, name: str, **properties) -> Screen:
        """Define a new screen template - matches your syntax!"""
        screen = Screen(name, **properties)
        self.registry.register_screen(name, screen)
        return screen
    
    def define_component(self, name: str, component_type: str, **properties) -> ScreenComponent:
        """Define a reusable component"""
        component_classes = {
            'button': Button,
            'textbox': TextBox,
            'label': Label,
            'screen': Screen
        }
        
        if component_type not in component_classes:
            raise ValueError(f"Unknown component type: {component_type}")
        
        component = component_classes[component_type](name, **properties)
        self.registry.register_component(name, component)
        return component
    
    def render_screen(self, screen_name: str, **overrides) -> str:
        """Render a screen to terminal output"""
        screen = self.registry.create_screen(screen_name, **overrides)
        
        # Initialize painter with screen dimensions
        width = screen.get_property('width', 80)
        height = screen.get_property('height', 24)
        self.painter.init_screen(width, height)
        
        # Render the screen
        self.painter.draw_component(screen, 0, 0)
        
        return self.painter.render()
    
    def add_to_screen(self, screen_name: str, component_name: str, **properties):
        """Add a component to a screen template"""
        if screen_name not in self.registry.screens:
            raise ValueError(f"Screen '{screen_name}' not found")
        
        component = self.registry.get_component(component_name)
        if not component:
            raise ValueError(f"Component '{component_name}' not found")
        
        # Create a copy of the component with overrides
        new_component = type(component)(component_name, **component.properties)
        for key, value in properties.items():
            new_component.properties[key] = value
        
        self.registry.screens[screen_name].add_child(new_component)

# Global renderer instance
_renderer = ZexusScreenRenderer()

# Zexus-facing API
def define_screen(name: str, **properties):
    """Zexus function to define screens"""
    return _renderer.define_screen(name, **properties)

def define_component(name: str, component_type: str, **properties):
    """Zexus function to define components"""
    return _renderer.define_component(name, component_type, **properties)

def render_screen(screen_name: str, **overrides) -> str:
    """Zexus function to render screens"""
    return _renderer.render_screen(screen_name, **overrides)

def add_to_screen(screen_name: str, component_name: str, **properties):
    """Zexus function to add components to screens"""
    return _renderer.add_to_screen(screen_name, component_name, **properties)