# zexus-core

This directory contains the optional Rust execution core for Zexus, exposed to Python as the `zexus_core` module via PyO3.

## Build (from this repo)

Requirements:
- Python 3.8+
- Rust toolchain (`cargo`)

Commands:
- `python -m pip install -U maturin`
- `python -m maturin develop -m rust_core/Cargo.toml --release`

If the extension is not available, Zexus falls back to the pure-Python VM automatically.
