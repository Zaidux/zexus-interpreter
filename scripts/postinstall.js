#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('\n🚀 Installing Zexus Programming Language...\n');

// Check if Python is available
try {
  const pythonVersion = execSync('python3 --version', { encoding: 'utf-8' });
  console.log(`✓ Found ${pythonVersion.trim()}`);
} catch (error) {
  console.error('❌ Python 3.8+ is required but not found.');
  console.error('Please install Python 3.8 or higher: https://www.python.org/downloads/');
  process.exit(1);
}

// Check Python version
try {
  const versionCheck = execSync('python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"');
  console.log('✓ Python version is 3.8 or higher');
} catch (error) {
  console.error('❌ Python 3.8 or higher is required.');
  process.exit(1);
}

// Install Zexus Python package
console.log('\n📦 Installing Zexus Python package...');
try {
  execSync('pip3 install zexus', { stdio: 'inherit' });
  console.log('\n✓ Zexus Python package installed successfully');
} catch (error) {
  console.error('\n❌ Failed to install Zexus Python package.');
  console.error('Please run manually: pip3 install zexus');
  process.exit(1);
}

console.log('\n✅ Zexus installed successfully!\n');
console.log('Get started:');
console.log('  zexus --help       # Show help');
console.log('  zx --version       # Check version');
console.log('  zexus examples/    # Explore examples\n');
console.log('Documentation: https://github.com/Zaidux/zexus-interpreter\n');
