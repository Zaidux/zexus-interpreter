"""
Zexus Stack Trace Formatter

Provides beautiful, informative stack traces for runtime errors.
"""

from typing import List, Optional, Dict, Any


class StackFrame:
    """Represents a single frame in the call stack"""
    
    def __init__(
        self,
        function_name: str,
        filename: Optional[str] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
        source_line: Optional[str] = None,
        node_type: Optional[str] = None
    ):
        self.function_name = function_name
        self.filename = filename or "<unknown>"
        self.line = line
        self.column = column
        self.source_line = source_line
        self.node_type = node_type
    
    def __str__(self):
        location = f"{self.filename}"
        if self.line:
            location += f":{self.line}"
            if self.column:
                location += f":{self.column}"
        
        result = f"  at {self.function_name} ({location})"
        
        if self.source_line and self.line:
            # Show source context
            result += f"\n    {self.line} | {self.source_line.strip()}"
            if self.column:
                # Add pointer
                padding = len(str(self.line)) + 3 + self.column
                result += f"\n    {' ' * padding}^"
        
        return result


class StackTraceFormatter:
    """Formats stack traces for display"""
    
    def __init__(self):
        self.max_frames = 10  # Maximum frames to display
        self.context_lines = 1  # Lines of context around error
    
    def format_stack_trace(
        self,
        frames: List[StackFrame],
        error_message: str,
        show_full: bool = False
    ) -> str:
        """
        Format a complete stack trace.
        
        Args:
            frames: List of stack frames
            error_message: The error message
            show_full: Show all frames or truncate
        
        Returns:
            Formatted stack trace string
        """
        if not frames:
            return error_message
        
        parts = []
        parts.append(f"Traceback (most recent call last):")
        
        # Limit frames if not showing full
        display_frames = frames
        if not show_full and len(frames) > self.max_frames:
            # Show first few and last few
            first_half = self.max_frames // 2
            second_half = self.max_frames - first_half
            display_frames = (
                frames[:first_half] +
                [None] +  # Marker for truncation
                frames[-second_half:]
            )
        
        for i, frame in enumerate(display_frames):
            if frame is None:
                parts.append(f"  ... {len(frames) - self.max_frames} frames omitted ...")
            else:
                parts.append(str(frame))
        
        parts.append(f"\n{error_message}")
        
        return "\n".join(parts)
    
    def format_simple_trace(self, frames: List[StackFrame], limit: int = 3) -> str:
        """
        Format a simple stack trace (just function names and locations).
        
        Args:
            frames: List of stack frames
            limit: Maximum frames to show
        
        Returns:
            Formatted trace string
        """
        if not frames:
            return ""
        
        display_frames = frames[-limit:] if len(frames) > limit else frames
        
        parts = ["Call stack:"]
        for frame in display_frames:
            location = f"{frame.filename}"
            if frame.line:
                location += f":{frame.line}"
            parts.append(f"  {frame.function_name} at {location}")
        
        return "\n".join(parts)
    
    def create_frame_from_node(
        self,
        node,
        function_name: str,
        error_reporter=None
    ) -> StackFrame:
        """
        Create a stack frame from an AST node.
        
        Args:
            node: AST node
            function_name: Name of the function/context
            error_reporter: Error reporter for source lookup
        
        Returns:
            StackFrame instance
        """
        filename = None
        line = None
        column = None
        source_line = None
        node_type = type(node).__name__ if node else None
        
        # Extract location info from node
        if hasattr(node, 'token') and node.token:
            token = node.token
            line = getattr(token, 'line', None)
            column = getattr(token, 'column', None)
        
        # Try to get source line if error reporter available
        if error_reporter and line:
            source_line = error_reporter.get_source_line(filename, line)
        
        return StackFrame(
            function_name=function_name,
            filename=filename,
            line=line,
            column=column,
            source_line=source_line,
            node_type=node_type
        )


# Global formatter instance
_stack_trace_formatter = StackTraceFormatter()


def get_stack_trace_formatter() -> StackTraceFormatter:
    """Get the global stack trace formatter"""
    return _stack_trace_formatter


def format_call_stack(frames: List[str], limit: int = 5) -> str:
    """
    Format a simple call stack from string frames.
    
    Args:
        frames: List of frame strings
        limit: Maximum frames to display
    
    Returns:
        Formatted stack string
    """
    if not frames:
        return ""
    
    display_frames = frames[-limit:] if len(frames) > limit else frames
    
    parts = ["Call stack:"]
    for frame in display_frames:
        parts.append(f"  {frame}")
    
    if len(frames) > limit:
        parts.insert(1, f"  ... {len(frames) - limit} frames omitted ...")
    
    return "\n".join(parts)


def create_frame_info(
    function_name: str,
    filename: str = "<unknown>",
    line: Optional[int] = None,
    node_type: Optional[str] = None
) -> str:
    """
    Create a simple frame info string.
    
    Args:
        function_name: Name of function
        filename: Source filename
        line: Line number
        node_type: Type of AST node
    
    Returns:
        Frame info string
    """
    parts = [function_name]
    
    if filename != "<unknown>" or line:
        location = filename
        if line:
            location += f":{line}"
        parts.append(f"({location})")
    
    if node_type:
        parts.append(f"[{node_type}]")
    
    return " ".join(parts)
