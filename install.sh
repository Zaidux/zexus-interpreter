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

# Install dependencies first
echo "📦 Installing dependencies..."
pip install click rich pygments

# Install Zexus in development mode
echo "🔧 Installing Zexus..."
pip install -e .

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
