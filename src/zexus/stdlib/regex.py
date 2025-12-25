"""Regex module for Zexus standard library."""

import re
from typing import List, Dict, Any, Optional


class RegexModule:
    """Provides regular expression operations."""

    @staticmethod
    def match(pattern: str, text: str, flags: int = 0) -> Optional[Dict[str, Any]]:
        """Match pattern at start of text."""
        m = re.match(pattern, text, flags)
        if m:
            return {
                'matched': True,
                'groups': list(m.groups()),
                'group_dict': m.groupdict(),
                'span': m.span(),
                'start': m.start(),
                'end': m.end()
            }
        return None

    @staticmethod
    def search(pattern: str, text: str, flags: int = 0) -> Optional[Dict[str, Any]]:
        """Search for pattern in text."""
        m = re.search(pattern, text, flags)
        if m:
            return {
                'matched': True,
                'groups': list(m.groups()),
                'group_dict': m.groupdict(),
                'span': m.span(),
                'start': m.start(),
                'end': m.end()
            }
        return None

    @staticmethod
    def findall(pattern: str, text: str, flags: int = 0) -> List[str]:
        """Find all occurrences of pattern."""
        return re.findall(pattern, text, flags)

    @staticmethod
    def finditer(pattern: str, text: str, flags: int = 0) -> List[Dict[str, Any]]:
        """Find all matches as iterator."""
        matches = []
        for m in re.finditer(pattern, text, flags):
            matches.append({
                'matched': True,
                'groups': list(m.groups()),
                'group_dict': m.groupdict(),
                'span': m.span(),
                'start': m.start(),
                'end': m.end()
            })
        return matches

    @staticmethod
    def sub(pattern: str, replacement: str, text: str, count: int = 0, flags: int = 0) -> str:
        """Replace pattern with replacement."""
        return re.sub(pattern, replacement, text, count, flags)

    @staticmethod
    def subn(pattern: str, replacement: str, text: str, count: int = 0, flags: int = 0) -> tuple:
        """Replace pattern and return (new_string, number_of_subs)."""
        return re.subn(pattern, replacement, text, count, flags)

    @staticmethod
    def split(pattern: str, text: str, maxsplit: int = 0, flags: int = 0) -> List[str]:
        """Split text by pattern."""
        return re.split(pattern, text, maxsplit, flags)

    @staticmethod
    def escape(pattern: str) -> str:
        """Escape special characters in pattern."""
        return re.escape(pattern)

    @staticmethod
    def compile(pattern: str, flags: int = 0) -> str:
        """Compile regex pattern (returns pattern string for Zexus)."""
        # In Zexus, we can't return compiled pattern objects
        # Just validate and return the pattern
        re.compile(pattern, flags)  # Validate
        return pattern

    @staticmethod
    def is_match(pattern: str, text: str, flags: int = 0) -> bool:
        """Check if pattern matches text."""
        return re.match(pattern, text, flags) is not None

    @staticmethod
    def is_search(pattern: str, text: str, flags: int = 0) -> bool:
        """Check if pattern exists in text."""
        return re.search(pattern, text, flags) is not None

    @staticmethod
    def count_matches(pattern: str, text: str, flags: int = 0) -> int:
        """Count number of matches."""
        return len(re.findall(pattern, text, flags))

    @staticmethod
    def extract_groups(pattern: str, text: str, flags: int = 0) -> List[str]:
        """Extract all groups from first match."""
        m = re.search(pattern, text, flags)
        if m:
            return list(m.groups())
        return []

    @staticmethod
    def extract_all_groups(pattern: str, text: str, flags: int = 0) -> List[List[str]]:
        """Extract groups from all matches."""
        matches = []
        for m in re.finditer(pattern, text, flags):
            matches.append(list(m.groups()))
        return matches

    # Flag constants
    IGNORECASE = re.IGNORECASE
    MULTILINE = re.MULTILINE
    DOTALL = re.DOTALL
    VERBOSE = re.VERBOSE
    ASCII = re.ASCII
    UNICODE = re.UNICODE


# Export functions for easy access
match = RegexModule.match
search = RegexModule.search
findall = RegexModule.findall
finditer = RegexModule.finditer
sub = RegexModule.sub
subn = RegexModule.subn
split = RegexModule.split
escape = RegexModule.escape
compile = RegexModule.compile
is_match = RegexModule.is_match
is_search = RegexModule.is_search
count_matches = RegexModule.count_matches
extract_groups = RegexModule.extract_groups
extract_all_groups = RegexModule.extract_all_groups
IGNORECASE = RegexModule.IGNORECASE
MULTILINE = RegexModule.MULTILINE
DOTALL = RegexModule.DOTALL
VERBOSE = RegexModule.VERBOSE
ASCII = RegexModule.ASCII
UNICODE = RegexModule.UNICODE
