#!/bin/bash

echo "🚀 Setting up Zexus Standard Library..."

# Create zpm_modules directory if it doesn't exist
mkdir -p zpm_modules

# Copy standard libraries from the parent directory
echo "📦 Installing zexus-math..."
cp -r ../Zexus-standard-library-/zexus-math zpm_modules/

echo "📦 Installing zexus-network..."
cp -r ../Zexus-standard-library-/zexus-network zpm_modules/

echo "📦 Installing zexus-blockchain..."
cp -r ../Zexus-standard-library-/zexus-blockchain zpm_modules/

echo "✅ Standard Library installed successfully!"
echo ""
echo "💡 Usage:"
echo "   use 'zexus-math' as math"
echo "   use 'zexus-network' as net" 
echo "   use 'zexus-blockchain' as blockchain"
