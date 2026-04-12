#!/bin/bash

echo "🚀 Setting up Zexus Standard Library..."

# Create required directories 
mkdir -p zpm_modules
sudo mkdir -p /usr/local/lib/zexus/stdlib

# Install zexus-core
echo "📦 Installing zexus-core..."
if [ -d "zpm_modules/zexus-core" ]; then
    sudo cp -r zpm_modules/zexus-core /usr/local/lib/zexus/stdlib/
fi

# Clone and install standard libraries if needed
if [ ! -d "zpm_modules/zexus-math" ]; then
    echo "📦 Installing zexus-math..."
    git clone https://github.com/zexus/zexus-math.git zpm_modules/zexus-math
fi

if [ ! -d "zpm_modules/zexus-network" ]; then
    echo "📦 Installing zexus-network..."
    git clone https://github.com/zexus/zexus-network.git zpm_modules/zexus-network
fi

if [ ! -d "zpm_modules/zexus-blockchain" ]; then
    echo "📦 Installing zexus-blockchain..."
    git clone https://github.com/zexus/zexus-blockchain.git zpm_modules/zexus-blockchain
fi

# Configure stdlib path
mkdir -p ~/.zexus
cat > ~/.zexus/config.json << EOL
{
    "stdlibPath": "/usr/local/lib/zexus/stdlib",
    "defaultImports": [
        "/usr/local/lib/zexus/stdlib/zexus-core/index.zx"
    ],
    "modulePaths": [
        "./zpm_modules",
        "/usr/local/lib/zexus/stdlib"
    ]
}
EOL

# Set permissions
sudo chmod -R 755 /usr/local/lib/zexus
sudo chown -R $USER:$USER ~/.zexus

echo "✅ Standard Library installed successfully!"
echo ""
echo "💡 Usage:"
echo "   use 'zexus-math' as math"
echo "   use 'zexus-network' as net" 
echo "   use 'zexus-blockchain' as blockchain"
