"""Code formatter for Zexus source files."""

import re
from typing import List


class ZexusFormatter:
    """Formats Zexus source code with consistent style."""

    # Keywords that open a block (followed by body / braces)
    _BLOCK_OPENERS = {
        'if', 'else', 'for', 'while', 'action', 'entity', 'contract',
        'match', 'case', 'default', 'try', 'catch', 'finally',
    }

    # Operators that should be surrounded by single spaces
    _SPACED_OPS = re.compile(
        r'(?<!=)(?<!!)(?<!<)(?<!>)'   # negative look-behind for !=, ==, <=, >=
        r'([=!<>]=|[+\-*/]=?|==|!=)'
        r'(?!=)',                       # negative look-ahead for ==
    )

    def format_document(self, source: str, indent_size: int = 4) -> str:
        """Format a full Zexus source document."""
        lines = source.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        formatted = self._reindent(lines, indent_size)
        formatted = [self._normalize_operators(line) for line in formatted]
        formatted = [line.rstrip() for line in formatted]
        # Ensure single trailing newline
        text = '\n'.join(formatted)
        text = text.rstrip('\n') + '\n'
        return text

    # ------------------------------------------------------------------
    # Indentation
    # ------------------------------------------------------------------

    def _reindent(self, lines: List[str], indent_size: int) -> List[str]:
        """Re-indent lines based on brace depth."""
        result: List[str] = []
        depth = 0
        indent = ' ' * indent_size

        for raw in lines:
            stripped = raw.strip()
            if not stripped:
                result.append('')
                continue

            # Decrease depth *before* this line if it starts with '}'
            if stripped.startswith('}'):
                depth = max(depth - 1, 0)

            result.append(f"{indent * depth}{stripped}")

            # Count unbalanced braces on this line (ignore braces in strings)
            open_b = self._count_outside_strings(stripped, '{')
            close_b = self._count_outside_strings(stripped, '}')
            depth += open_b - close_b

            # If a block-opening keyword appears without a '{', still indent
            first_word = stripped.split('(')[0].split()[0] if stripped else ''
            if (first_word in self._BLOCK_OPENERS
                    and '{' not in stripped
                    and not stripped.endswith('}')):
                # Only bump if the next non-empty line isn't '{'
                pass  # conservative: only indent on braces

            depth = max(depth, 0)

        return result

    @staticmethod
    def _count_outside_strings(line: str, char: str) -> int:
        """Count occurrences of *char* that are not inside string literals."""
        count = 0
        in_str: str | None = None
        escape = False
        for ch in line:
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if in_str:
                if ch == in_str:
                    in_str = None
                continue
            if ch in ('"', "'"):
                in_str = ch
                continue
            if ch == char:
                count += 1
        return count

    # ------------------------------------------------------------------
    # Operator normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_operators(line: str) -> str:
        """Ensure single spaces around common operators.

        We work only outside of string literals to avoid mangling user
        strings.
        """
        result: list[str] = []
        in_str: str | None = None
        escape = False
        i = 0
        chars = line

        while i < len(chars):
            ch = chars[i]

            if escape:
                result.append(ch)
                escape = False
                i += 1
                continue
            if ch == '\\':
                escape = True
                result.append(ch)
                i += 1
                continue

            # Toggle string mode
            if ch in ('"', "'") and not in_str:
                in_str = ch
                result.append(ch)
                i += 1
                continue
            if in_str:
                if ch == in_str:
                    in_str = None
                result.append(ch)
                i += 1
                continue

            # Two-char operators
            two = chars[i:i + 2]
            if two in ('==', '!=', '<=', '>=', '+=', '-=', '*=', '/='):
                # strip trailing space before operator
                while result and result[-1] == ' ':
                    result.pop()
                result.append(' ')
                result.append(two)
                result.append(' ')
                i += 2
                # skip extra spaces after
                while i < len(chars) and chars[i] == ' ':
                    i += 1
                continue

            # Single-char operators (but not unary minus/plus at line start)
            if ch in ('=', '+', '-', '*', '/'):
                # Avoid touching '//' comments
                if ch == '/' and i + 1 < len(chars) and chars[i + 1] == '/':
                    result.append(chars[i:])
                    break
                while result and result[-1] == ' ':
                    result.pop()
                result.append(' ')
                result.append(ch)
                result.append(' ')
                i += 1
                while i < len(chars) and chars[i] == ' ':
                    i += 1
                continue

            result.append(ch)
            i += 1

        return ''.join(result)
