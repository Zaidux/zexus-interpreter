"""JSON module for Zexus standard library."""

import json as json_lib
from typing import Any, Optional


class JsonModule:
    """Provides JSON encoding and decoding operations."""

    @staticmethod
    def parse(text: str) -> Any:
        """Parse JSON string to object."""
        return json_lib.loads(text)

    @staticmethod
    def stringify(obj: Any, indent: Optional[int] = None, sort_keys: bool = False) -> str:
        """Convert object to JSON string."""
        return json_lib.dumps(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=False)

    @staticmethod
    def load(file_path: str) -> Any:
        """Load JSON from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json_lib.load(f)

    @staticmethod
    def save(file_path: str, obj: Any, indent: Optional[int] = 2, sort_keys: bool = False) -> None:
        """Save object to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json_lib.dump(obj, f, indent=indent, sort_keys=sort_keys, ensure_ascii=False)

    @staticmethod
    def validate(text: str) -> bool:
        """Check if string is valid JSON."""
        try:
            json_lib.loads(text)
            return True
        except (json_lib.JSONDecodeError, ValueError):
            return False

    @staticmethod
    def merge(obj1: dict, obj2: dict) -> dict:
        """Merge two JSON objects."""
        result = obj1.copy()
        result.update(obj2)
        return result

    @staticmethod
    def pretty_print(obj: Any, indent: int = 2) -> str:
        """Pretty print JSON with indentation."""
        return json_lib.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False)


# Export functions for easy access
parse = JsonModule.parse
stringify = JsonModule.stringify
load = JsonModule.load
save = JsonModule.save
validate = JsonModule.validate
merge = JsonModule.merge
pretty_print = JsonModule.pretty_print
