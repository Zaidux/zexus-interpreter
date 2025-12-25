# Repository Structure

This document explains the organization of the Zexus interpreter repository.

## Core Directories

```
zexus-interpreter/
├── src/zexus/              # Core interpreter engine (DO NOT duplicate this in scripts/)
│   ├── lexer/              # Tokenization
│   ├── parser/             # AST generation
│   ├── evaluator/          # Execution engine
│   ├── object/             # Runtime object system
│   ├── cli/                # Command-line interface
│   └── builtin/            # Built-in functions
│
├── scripts/                # Entry points (thin wrappers - import from src/)
│   ├── main.py             # Primary CLI entry
│   └── zpm.py              # Package manager
│
├── docs/                   # Active documentation
│   ├── INDEX.md            # Documentation index
│   ├── QUICK_START.md      # Getting started guide
│   ├── guides/             # Feature guides
│   └── archive/            # Historical docs (enhancement-package, etc.)
│
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
│
├── examples/               # Example code
│
├── archive/                # Historical artifacts (not for active dev)
│   ├── reports/            # Test reports, status docs
│   ├── summaries/          # Implementation summaries
│   ├── proofs/             # Proof documents
│   ├── dev-notes/          # Development notes
│   └── dev-scripts/        # Old debug/temp scripts
│
└── zx, zx-dev, zx-run      # CLI executables
```

## File Policies

### Never Commit (in .gitignore)
- `chain_data/` - Runtime blockchain state
- `*.sqlite` - Database files
- `__pycache__/` - Python cache
- `generated.*` - Generated files
- `*.log` - Log files
- Temporary test files

### Root Should Contain Only
- Core documentation (README.md, CHANGELOG.md, LICENSE)
- Configuration (pyproject.toml, setup.py, pytest.ini)
- CLI executables (zx, zx-dev, zx-run)
- Install scripts (install.sh, setup_stdlib.sh)

### Avoid in Root
- Implementation reports (→ archive/reports/)
- Status summaries (→ archive/summaries/)
- Temporary scripts (→ archive/temp-files/)
- Generated files (→ .gitignore)
- Debug artifacts (→ archive/dev-scripts/)

## Development Workflow

### Adding Features
1. Implement in `src/zexus/`
2. Write tests in `tests/`
3. Update docs in `docs/`
4. Never duplicate logic in `scripts/`

### Scripts Should
- Import from `src.zexus.*`
- Be thin CLI wrappers
- Call the engine, not contain it

### Scripts Should NOT
- Duplicate lexer/parser/token classes
- Contain business logic
- Import each other (except main.py)

## Documentation Organization

### Active Docs (`/docs`)
- Feature guides
- API reference
- Tutorials
- Architecture docs

### Archive (`/archive` and `/docs/archive`)
- Historical enhancement packages
- Completed implementation summaries
- Proof documents
- Old status reports

**Rule**: If you're linking to it from active docs, it shouldn't be in archive.

## Questions?

- How do I run Zexus? → See README.md
- Where are the docs? → See docs/INDEX.md
- What's in archive? → Historical reference only (safe to ignore)
- Scripts vs src? → Scripts call src, never duplicate it
