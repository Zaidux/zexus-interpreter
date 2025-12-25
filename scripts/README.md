# Scripts Directory

Entry points and utilities for running Zexus.

## Main Entry Points

- **main.py** - Primary CLI entry point (imported by zx-dev, zx-run)
- **__main__.py** - Allows running as module: `python -m scripts`

## Utilities

- **run_zexus_tests.py** - Test runner
- **zpm.py** - Zexus Package Manager
- **integration_demo.py** - Integration examples

## Important Note

Scripts should **call** the engine (src/zexus/), not contain parts of it.

- ❌ Don't duplicate lexer/parser/token logic here
- ✅ Import from `src.zexus.*` and use the API
- ✅ Keep scripts thin - just CLI wrappers

For duplicated logic files (token.py, zexus_ast.py), see archive/dev-scripts/
