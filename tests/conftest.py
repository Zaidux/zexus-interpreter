"""
Pytest configuration for Zexus tests.
"""
import sys
import os

# Ensure both import styles work:
# - `import zexus...` (src/ on sys.path)
# - `import src.zexus...` (repo root on sys.path + src/ is a package)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SRC_DIR = os.path.join(_ROOT, 'src')

for _p in (_ROOT, _SRC_DIR):
	if _p not in sys.path:
		sys.path.insert(0, _p)
