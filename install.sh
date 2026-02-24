#!/bin/bash
# ~/zexus-interpreter/install.sh

echo "🚀 Installing Zexus Programming Language..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "⚠️  Python 3 not found. Attempting to install..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-pip python3-venv
    elif command -v brew &> /dev/null; then
        brew install python@3
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3 python3-pip
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm python python-pip
    else
        echo "❌ Could not auto-install Python 3."
        echo "Please install Python 3.8+ from https://python.org"
        exit 1
    fi
fi

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

echo "✓ Found Python $PYTHON_VERSION"

# Ensure pip is available
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo "   pip not found — bootstrapping..."
    python3 -m ensurepip --upgrade 2>/dev/null || true
fi

# Install Zexus (and optional extras) in development mode
echo "📦 Installing Zexus + dependencies..."
python3 -m pip install --upgrade pip

# "full" pulls in blockchain + networking + security helpers and the tooling
# needed to compile the Rust VM (best-effort).
python3 -m pip install -e ".[full]"

# Best-effort: build/install Rust VM extension if Rust toolchain is present.
if ! command -v cargo &> /dev/null; then
    echo ""
    echo "🦀 Rust toolchain (cargo) not found."
    echo "   Attempting to install Rust via rustup..."
    if curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; then
        # Source cargo env for this session
        export PATH="$HOME/.cargo/bin:$PATH"
        if [ -f "$HOME/.cargo/env" ]; then
            source "$HOME/.cargo/env"
        fi
        echo "✓ Rust toolchain installed via rustup"
    else
        echo "⚠️  Could not auto-install Rust; continuing with pure-Python VM."
    fi
fi

if command -v cargo &> /dev/null; then
    if [ -f "rust_core/Cargo.toml" ]; then
        # Check if zexus_core is already importable
        if python3 -c "import zexus_core" 2>/dev/null; then
            echo "✓ Rust VM extension (zexus_core) already installed"
        else
            echo "🔨 Building Rust VM extension (zexus_core)..."
            python3 -m pip install --upgrade maturin
            # Install into the current Python environment.
            if python3 -m maturin develop -m rust_core/Cargo.toml --release; then
                echo "✓ Rust VM extension built and installed"
            else
                echo "⚠️  Rust VM build failed; continuing with pure-Python VM."
            fi
        fi
    fi
else
    echo "ℹ️  Rust toolchain not available; skipping Rust VM build."
fi

# Verify installation
if command -v zx &> /dev/null; then
    echo ""
    echo "✅ Zexus installed successfully!"
    echo ""
    echo "🎯 Quick start commands:"
    echo "   zx run examples/hello_world.zx"
    echo "   zx repl"
    echo "   zx --help"
    echo ""
    echo "💡 Try: zx run examples/blockchain_demo.zx"
else
    echo ""
    echo "✅ Zexus core installed. You may need to add the install location to your PATH."
    echo "   Try: python3 -m zexus --help"
fi
