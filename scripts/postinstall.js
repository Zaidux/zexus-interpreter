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
  // Install with "full" extras so blockchain/network/security features work out of the box.
  // Use --user to avoid permission issues on global Python installs.
  execSync('pip3 install --user "zexus[full]"', { stdio: 'inherit' });
  console.log('\n✓ Zexus Python package installed successfully');
} catch (error) {
  console.error('\n❌ Failed to install Zexus Python package.');
  console.error('Please run manually: pip3 install --user "zexus[full]"');
  process.exit(1);
}

// Best-effort: build/install Rust VM extension from bundled sources when available.
// This requires a Rust toolchain and may fail on systems without build tooling.
function hasCommand(cmd) {
  try {
    execSync(`${cmd} --version`, { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

const pkgRoot = path.resolve(__dirname, '..');
const cargoToml = path.join(pkgRoot, 'rust_core', 'Cargo.toml');

if (fs.existsSync(cargoToml) && hasCommand('cargo')) {
  console.log('\n🦀 Rust toolchain detected — attempting to build zexus_core...');
  try {
    execSync('python3 -m pip install --user --upgrade maturin', { stdio: 'inherit' });
    execSync(`python3 -m maturin develop -m "${cargoToml}" --release`, { stdio: 'inherit' });
    execSync('python3 -c "import zexus_core; print(\"zexus_core OK\", zexus_core.version())"', { stdio: 'inherit' });
    console.log('✓ Rust VM extension built and installed');
  } catch (error) {
    console.warn('\n⚠️  Rust VM build failed; continuing with pure-Python VM.');
    console.warn('   To retry manually (from source repo):');
    console.warn('     pip install maturin && maturin develop -m rust_core/Cargo.toml --release');
  }
} else {
  console.log('\nℹ️  Skipping Rust VM build (cargo not found or rust_core not bundled).');
}

console.log('\n✅ Zexus installed successfully!\n');
console.log('Get started:');
console.log('  zexus --help       # Show help');
console.log('  zx --version       # Check version');
console.log('  zexus examples/    # Explore examples\n');
console.log('Documentation: https://github.com/Zaidux/zexus-interpreter\n');
