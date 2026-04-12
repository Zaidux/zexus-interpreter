"""Server-side string templating module for Zexus standard library."""

import re
import html
from typing import Any, Dict, List, Optional, Union


class TemplateModule:
    """Provides server-side string templating with variable interpolation,
    conditionals, loops, and filters."""

    # --- Filters -----------------------------------------------------------

    _BUILTIN_FILTERS: Dict[str, Any] = {
        "upper": lambda v: str(v).upper(),
        "lower": lambda v: str(v).lower(),
        "capitalize": lambda v: str(v).capitalize(),
        "escape": lambda v: html.escape(str(v)),
        "trim": lambda v: str(v).strip(),
        "length": lambda v: str(len(v)),
    }

    # Patterns for filters that accept an argument
    _ARG_FILTER_RE = re.compile(r'^(\w+)\((.+)\)$')

    # --- Core helpers ------------------------------------------------------

    @staticmethod
    def _resolve(name: str, context: Dict[str, Any]) -> Any:
        """Resolve a dotted name against *context*.

        Supports dict key access and attribute access for nested lookups.
        Returns ``""`` when the name cannot be resolved.
        """
        parts = name.strip().split(".")
        current: Any = context
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return ""
        return current

    @staticmethod
    def _apply_filter(value: Any, filter_expr: str) -> str:
        """Apply a single filter expression to *value*.

        Handles both simple filters (``upper``) and parameterised filters
        (``default("fallback")``, ``truncate(50)``).
        """
        filter_expr = filter_expr.strip()

        # Check for parameterised filter
        m = TemplateModule._ARG_FILTER_RE.match(filter_expr)
        if m:
            fname, raw_arg = m.group(1), m.group(2).strip()
            # Strip surrounding quotes from string arguments
            if (raw_arg.startswith('"') and raw_arg.endswith('"')) or \
               (raw_arg.startswith("'") and raw_arg.endswith("'")):
                arg: Any = raw_arg[1:-1]
            else:
                try:
                    arg = int(raw_arg)
                except ValueError:
                    try:
                        arg = float(raw_arg)
                    except ValueError:
                        arg = raw_arg

            if fname == "default":
                return str(value) if value != "" else str(arg)
            if fname == "truncate":
                s = str(value)
                limit = int(arg)
                return s[:limit] + "..." if len(s) > limit else s
            return str(value)

        # Simple (no-arg) built-in filter
        fn = TemplateModule._BUILTIN_FILTERS.get(filter_expr)
        if fn is not None:
            return fn(value)

        # Unknown filter – pass through unchanged
        return str(value)

    @staticmethod
    def _apply_filters(value: Any, filters: List[str]) -> str:
        """Chain multiple filters left-to-right."""
        result: Any = value
        for f in filters:
            result = TemplateModule._apply_filter(result, f)
        return str(result)

    # --- Variable interpolation --------------------------------------------

    _VAR_RE = re.compile(r'\{\{\s*(.+?)\s*\}\}')

    @staticmethod
    def _interpolate(template_str: str, context: Dict[str, Any],
                     auto_escape: bool = False) -> str:
        """Replace ``{{expr}}`` tokens with values from *context*."""

        def _replacer(match: re.Match) -> str:
            expr = match.group(1)
            parts = [p.strip() for p in expr.split("|")]
            name = parts[0]
            filters = parts[1:]
            value = TemplateModule._resolve(name, context)
            escape_already = any(f.split(":", 1)[0] == "escape" for f in filters)
            if filters:
                value = TemplateModule._apply_filters(value, filters)
            else:
                value = str(value)
            if auto_escape and not escape_already:
                value = html.escape(value)
            return value

        return TemplateModule._VAR_RE.sub(_replacer, template_str)

    # --- Block processing (for / if) --------------------------------------

    @staticmethod
    def _process_blocks(template_str: str, context: Dict[str, Any],
                        auto_escape: bool = False) -> str:
        """Process ``{% for %}`` and ``{% if %}`` blocks recursively."""
        result = template_str

        # Process for-loops (innermost first to handle nesting)
        result = TemplateModule._process_for_blocks(result, context, auto_escape)

        # Process conditionals (innermost first)
        result = TemplateModule._process_if_blocks(result, context, auto_escape)

        return result

    # -- for blocks ---------------------------------------------------------

    _FOR_RE = re.compile(
        r'\{%\s*for\s+(\w+)\s+in\s+(\w[\w.]*)\s*%\}(.*?)\{%\s*endfor\s*%\}',
        re.DOTALL,
    )

    @staticmethod
    def _process_for_blocks(template_str: str, context: Dict[str, Any],
                            auto_escape: bool = False) -> str:
        """Expand ``{% for item in items %}...{% endfor %}`` blocks."""

        def _for_replacer(match: re.Match) -> str:
            var_name = match.group(1)
            iterable_name = match.group(2)
            body = match.group(3)

            items = TemplateModule._resolve(iterable_name, context)
            if not isinstance(items, (list, tuple)):
                return ""

            total = len(items)
            parts: List[str] = []
            for idx, item in enumerate(items):
                loop_ctx = {
                    "index": idx,
                    "index1": idx + 1,
                    "first": idx == 0,
                    "last": idx == total - 1,
                    "length": total,
                }
                child_context = {**context, var_name: item, "loop": loop_ctx}
                rendered = TemplateModule._process_blocks(body, child_context, auto_escape)
                rendered = TemplateModule._interpolate(rendered, child_context, auto_escape)
                parts.append(rendered)
            return "".join(parts)

        # Repeat until no more for-blocks are found (handles nesting)
        prev = None
        result = template_str
        while prev != result:
            prev = result
            result = TemplateModule._FOR_RE.sub(_for_replacer, result)
        return result

    # -- if blocks ----------------------------------------------------------

    _IF_ELSE_RE = re.compile(
        r'\{%\s*if\s+(.+?)\s*%\}(.*?)\{%\s*else\s*%\}(.*?)\{%\s*endif\s*%\}',
        re.DOTALL,
    )
    _IF_RE = re.compile(
        r'\{%\s*if\s+(.+?)\s*%\}(.*?)\{%\s*endif\s*%\}',
        re.DOTALL,
    )

    @staticmethod
    def _evaluate_condition(expr: str, context: Dict[str, Any]) -> bool:
        """Evaluate a simple condition expression against *context*.

        Supports:
        - truthiness of a variable: ``{% if user %}``
        - ``not`` prefix: ``{% if not logged_in %}``
        - equality / inequality: ``{% if role == "admin" %}``
        - comparison operators: ``==``, ``!=``, ``>``, ``<``, ``>=``, ``<=``
        """
        expr = expr.strip()

        # Handle "not" prefix
        if expr.startswith("not "):
            return not TemplateModule._evaluate_condition(expr[4:], context)

        # Handle comparison operators
        for op in ("==", "!=", ">=", "<=", ">", "<"):
            if op in expr:
                left_str, right_str = expr.split(op, 1)
                left = TemplateModule._resolve(left_str.strip(), context)
                right_raw = right_str.strip()
                # Parse the right-hand side literal or variable
                if (right_raw.startswith('"') and right_raw.endswith('"')) or \
                   (right_raw.startswith("'") and right_raw.endswith("'")):
                    right: Any = right_raw[1:-1]
                else:
                    try:
                        right = int(right_raw)
                    except ValueError:
                        try:
                            right = float(right_raw)
                        except ValueError:
                            right = TemplateModule._resolve(right_raw, context)

                ops = {
                    "==": lambda a, b: a == b,
                    "!=": lambda a, b: a != b,
                    ">":  lambda a, b: a > b,
                    "<":  lambda a, b: a < b,
                    ">=": lambda a, b: a >= b,
                    "<=": lambda a, b: a <= b,
                }
                try:
                    return ops[op](left, right)
                except TypeError:
                    return False

        # Simple truthiness
        value = TemplateModule._resolve(expr, context)
        return bool(value)

    @staticmethod
    def _process_if_blocks(template_str: str, context: Dict[str, Any],
                           auto_escape: bool = False) -> str:
        """Expand ``{% if %}...{% else %}...{% endif %}`` blocks."""

        def _if_else_replacer(match: re.Match) -> str:
            condition = match.group(1)
            true_body = match.group(2)
            false_body = match.group(3)
            if TemplateModule._evaluate_condition(condition, context):
                body = true_body
            else:
                body = false_body
            body = TemplateModule._process_blocks(body, context, auto_escape)
            return TemplateModule._interpolate(body, context, auto_escape)

        def _if_replacer(match: re.Match) -> str:
            condition = match.group(1)
            body = match.group(2)
            if TemplateModule._evaluate_condition(condition, context):
                body = TemplateModule._process_blocks(body, context, auto_escape)
                return TemplateModule._interpolate(body, context, auto_escape)
            return ""

        prev = None
        result = template_str
        while prev != result:
            prev = result
            result = TemplateModule._IF_ELSE_RE.sub(_if_else_replacer, result)
            result = TemplateModule._IF_RE.sub(_if_replacer, result)
        return result

    # --- Public API --------------------------------------------------------

    @staticmethod
    def render(template_str: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Render a template string with the given context.

        Supports variable interpolation (``{{var}}``), filters
        (``{{var|upper}}``), conditionals (``{% if %}``), and loops
        (``{% for %}``).

        Missing variables resolve to an empty string.

        Args:
            template_str: The template string to render.
            context: A dictionary of variables available inside the template.

        Returns:
            The rendered string.
        """
        if context is None:
            context = {}
        result = TemplateModule._process_blocks(template_str, context, auto_escape=False)
        result = TemplateModule._interpolate(result, context, auto_escape=False)
        return result

    @staticmethod
    def render_safe(template_str: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Render a template with automatic HTML entity escaping.

        Behaves identically to :meth:`render` but every interpolated value
        is passed through :func:`html.escape` before insertion.

        Args:
            template_str: The template string to render.
            context: A dictionary of variables available inside the template.

        Returns:
            The rendered string with HTML-escaped values.
        """
        if context is None:
            context = {}
        result = TemplateModule._process_blocks(template_str, context, auto_escape=True)
        result = TemplateModule._interpolate(result, context, auto_escape=True)
        return result

    @staticmethod
    def render_file(filepath: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Load a template from a file and render it.

        Args:
            filepath: Path to the template file.
            context: A dictionary of variables available inside the template.

        Returns:
            The rendered string.

        Raises:
            FileNotFoundError: If *filepath* does not exist.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            template_str = f.read()
        return TemplateModule.render(template_str, context)


# Export functions for easy access
render = TemplateModule.render
render_safe = TemplateModule.render_safe
render_file = TemplateModule.render_file
