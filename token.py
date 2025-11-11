# token.py

# Compatibility shim: provide EXACT_TOKEN_TYPES so Python's tokenize module can import
# from the stdlib 'token' module even if this project has a top-level 'token.py'.
# NOTE: Ideally this file should be removed/renamed; this shim reduces breakage for now.
"""Compatibility shim for the Python standard library `token` module.

This repository previously contained a top-level `token.py` which shadowed
the standard library `token` module, causing imports like `import tokenize`
to fail (they import attributes such as EXACT_TOKEN_TYPES from `token`).

To avoid that breakage while preserving repository layout, this file now
provides a minimal compatible surface that the stdlib expects. The real
language tokens used by the project live in `src/zexus/zexus_token.py`.

If possible, remove or rename this file in future to avoid the name clash.
"""

__all__ = [
    'tok_name', 'NAME', 'NUMBER', 'STRING', 'NEWLINE', 'INDENT', 'DEDENT',
    'NL', 'OP', 'ERRORTOKEN', 'ENDMARKER', 'EXACT_TOKEN_TYPES'
]

# Minimal numeric token values (these values are not used by the project;
# they merely satisfy the interface expected by the stdlib tokenize module).
ENDMARKER = 0
NAME = 1
NUMBER = 2
STRING = 3
NEWLINE = 4
INDENT = 5
DEDENT = 6
NL = 7
OP = 8
ERRORTOKEN = 9

tok_name = {
    ENDMARKER: 'ENDMARKER', NAME: 'NAME', NUMBER: 'NUMBER', STRING: 'STRING',
    NEWLINE: 'NEWLINE', INDENT: 'INDENT', DEDENT: 'DEDENT', NL: 'NL',
    OP: 'OP', ERRORTOKEN: 'ERRORTOKEN'
}

# The tokenize module imports EXACT_TOKEN_TYPES from token; provide an empty
# mapping so that import succeeds.
EXACT_TOKEN_TYPES = {}

# Keep a small legacy lookup to avoid AttributeError when other code expects
# lookup_ident or similar; these are no-ops here.
def lookup_ident(ident):
    return ident