#!/bin/bash
# ~/zexus-interpreter/install.sh

echo "🚀 Installing Zexus Programming Language..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.8+ is required but not installed."
    echo "Please install Python from https://python.org"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.8+ is required. Found Python $PYTHON_VERSION"
    exit 1
fi

# Install Zexus (and optional extras) in development mode
echo "📦 Installing Zexus + dependencies..."
python3 -m pip install --upgrade pip

# "full" pulls in blockchain + networking + security helpers and the tooling
# needed to compile the Rust VM (best-effort).
python3 -m pip install -e ".[full]"

# Best-effort: build/install Rust VM extension if Rust toolchain is present.
if command -v cargo &> /dev/null; then
    if [ -f "rust_core/Cargo.toml" ]; then
        echo "🦀 Building Rust VM extension (zexus_core)..."
        python3 -m pip install --upgrade maturin
        # Install into the current Python environment.
        python3 -m maturin develop -m rust_core/Cargo.toml --release || echo "⚠️  Rust VM build failed; continuing with pure-Python VM."
    fi
else
    echo "ℹ️  Rust toolchain not found (cargo missing); skipping Rust VM build."
fi

# Verify installation
if command -v zx &> /dev/null; then
    echo ""
    echo "✅ [bold green]Zexus installed successfully![/bold green]"
    echo ""
    echo "🎯 Quick start commands:"
    echo "   zx run examples/hello_world.zx"
    echo "   zx repl"
    echo "   zx --help"
    echo ""
    echo "💡 Try: zx run examples/blockchain_demo.zx"
else
    echo "❌ Installation failed. Please check the errors above."
    exit 1
fi
