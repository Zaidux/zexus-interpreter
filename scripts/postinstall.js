#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

console.log('\n🚀 Installing Zexus Programming Language...\n');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function hasCommand(cmd) {
  try {
    execSync(`${cmd} --version`, { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

function run(cmd, opts = {}) {
  return execSync(cmd, { encoding: 'utf-8', stdio: 'inherit', ...opts });
}

function runQuiet(cmd) {
  return execSync(cmd, { encoding: 'utf-8', stdio: 'pipe' });
}

const isWin = os.platform() === 'win32';
const pip = isWin ? 'pip' : 'pip3';
const python = isWin ? 'python' : 'python3';

// ---------------------------------------------------------------------------
// 1. Python — check or install
// ---------------------------------------------------------------------------

let pythonAvailable = false;

if (hasCommand(python)) {
  try {
    const ver = runQuiet(`${python} --version`).trim();
    console.log(`✓ Found ${ver}`);
    // Verify >= 3.8
    runQuiet(`${python} -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"`);
    console.log('✓ Python version is 3.8 or higher');
    pythonAvailable = true;
  } catch {
    console.error('❌ Python was found but version is below 3.8.');
    console.error('   Please upgrade to Python 3.8+: https://www.python.org/downloads/');
  }
} else {
  console.warn('⚠️  python3 not found on PATH.');
  // Attempt auto-install on common platforms
  if (os.platform() === 'linux') {
    console.log('   Attempting to install python3 via apt...');
    try {
      run('sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-pip python3-venv');
      pythonAvailable = true;
      console.log('✓ Python 3 installed via apt');
    } catch {
      console.warn('   Could not auto-install Python. Please install manually:');
      console.warn('   https://www.python.org/downloads/');
    }
  } else if (os.platform() === 'darwin') {
    if (hasCommand('brew')) {
      console.log('   Attempting to install python3 via Homebrew...');
      try {
        run('brew install python@3');
        pythonAvailable = true;
        console.log('✓ Python 3 installed via Homebrew');
      } catch {
        console.warn('   Could not auto-install Python. Please install manually:');
        console.warn('   https://www.python.org/downloads/');
      }
    } else {
      console.warn('   Please install Python 3.8+: https://www.python.org/downloads/');
    }
  } else {
    console.warn('   Please install Python 3.8+: https://www.python.org/downloads/');
  }
}

if (!pythonAvailable) {
  console.error('\n❌ Python 3.8+ is required but could not be found or installed.');
  console.error('   Install it from: https://www.python.org/downloads/');
  console.error('   Then re-run: npm rebuild zexus');
  process.exit(1);
}

// ---------------------------------------------------------------------------
// 2. pip — ensure available
// ---------------------------------------------------------------------------

if (!hasCommand(pip) && !hasCommand('pip')) {
  console.log('   pip not found — bootstrapping...');
  try {
    run(`${python} -m ensurepip --upgrade`);
  } catch {
    try {
      run(`${python} -m pip install --upgrade pip`);
    } catch {
      console.warn('⚠️  Could not bootstrap pip. Python package install may fail.');
    }
  }
}

// ---------------------------------------------------------------------------
// 3. Install Zexus Python package
// ---------------------------------------------------------------------------

console.log('\n📦 Installing Zexus Python package...');

// Check if zexus is already installed and up-to-date
let zexusInstalled = false;
try {
  runQuiet(`${python} -c "import zexus"`);
  zexusInstalled = true;
  console.log('✓ Zexus Python package already installed');
} catch {
  // Not installed yet
}

if (!zexusInstalled) {
  try {
    run(`${python} -m pip install --user "zexus[full]"`);
    console.log('✓ Zexus Python package installed successfully');
  } catch {
    // Retry without --user (some environments like venvs don't need it)
    try {
      run(`${python} -m pip install "zexus[full]"`);
      console.log('✓ Zexus Python package installed successfully');
    } catch {
      console.error('❌ Failed to install Zexus Python package.');
      console.error(`   Please run manually: ${python} -m pip install "zexus[full]"`);
      // Don't exit — commands that don't need Python may still work
    }
  }
}

// ---------------------------------------------------------------------------
// 4. Rust toolchain — check or install, then build rust_core
// ---------------------------------------------------------------------------

const pkgRoot = path.resolve(__dirname, '..');
const cargoToml = path.join(pkgRoot, 'rust_core', 'Cargo.toml');

if (fs.existsSync(cargoToml)) {
  let cargoReady = hasCommand('cargo');

  if (!cargoReady) {
    console.log('\n🦀 Rust toolchain (cargo) not found.');
    console.log('   Attempting to install Rust via rustup...');
    try {
      if (isWin) {
        // On Windows, download and run rustup-init
        console.log('   Please install Rust manually: https://rustup.rs');
      } else {
        // Unix: install rustup non-interactively
        run('curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y');
        // Source the cargo env so it's available in this session
        const cargoEnv = path.join(os.homedir(), '.cargo', 'env');
        if (fs.existsSync(cargoEnv)) {
          // Read the cargo bin path and add to PATH for child processes
          const cargoBin = path.join(os.homedir(), '.cargo', 'bin');
          process.env.PATH = `${cargoBin}${path.delimiter}${process.env.PATH}`;
        }
        cargoReady = hasCommand('cargo');
        if (cargoReady) {
          console.log('✓ Rust toolchain installed via rustup');
        }
      }
    } catch (err) {
      console.warn('⚠️  Could not auto-install Rust. Continuing with pure-Python VM.');
      console.warn('   To install manually: https://rustup.rs');
    }
  } else {
    console.log('\n✓ Rust toolchain detected');
  }

  if (cargoReady) {
    // Check if zexus_core is already importable
    let rustVmInstalled = false;
    try {
      runQuiet(`${python} -c "import zexus_core"`);
      rustVmInstalled = true;
      console.log('✓ Rust VM extension (zexus_core) already installed');
    } catch {
      // Need to build
    }

    if (!rustVmInstalled) {
      console.log('🔨 Building Rust VM extension (zexus_core)...');
      try {
        // Ensure maturin is available
        try {
          runQuiet(`${python} -m maturin --version`);
        } catch {
          console.log('   Installing maturin build tool...');
          run(`${python} -m pip install --user --upgrade maturin`);
        }
        run(`${python} -m maturin develop -m "${cargoToml}" --release`);
        runQuiet(`${python} -c "import zexus_core"`);
        console.log('✓ Rust VM extension built and installed');
      } catch (error) {
        console.warn('\n⚠️  Rust VM build failed; continuing with pure-Python VM.');
        console.warn('   To retry manually:');
        console.warn(`     ${python} -m pip install maturin && ${python} -m maturin develop -m rust_core/Cargo.toml --release`);
      }
    }
  }
} else {
  console.log('\nℹ️  rust_core/Cargo.toml not bundled; skipping Rust VM build.');
}

// ---------------------------------------------------------------------------
// Done
// ---------------------------------------------------------------------------

console.log('\n✅ Zexus installed successfully!\n');
console.log('Get started:');
console.log('  zexus --help       # Show help');
console.log('  zx --version       # Check version');
console.log('  zx run file.zx     # Run a Zexus file');
console.log('  zexus examples/    # Explore examples\n');
console.log('Documentation: https://github.com/Zaidux/zexus-interpreter\n');
