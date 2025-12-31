# Publishing Zexus to NPM

## Quick Steps (for you)

### 1. Install/Update npm (one-time setup)
```bash
# Check if npm is installed
npm --version
node --version

# If not installed, install Node.js and npm
# Ubuntu/Debian
sudo apt update
sudo apt install nodejs npm

# Or use nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install --lts
```

### 2. Create NPM Account (if you don't have one)
- Go to https://www.npmjs.com/signup
- Verify your email
- Enable 2FA (recommended for security)

### 3. Login to NPM
```bash
cd /workspaces/zexus-interpreter
npm login

# You'll be prompted for:
# Username: your-npm-username
# Password: your-password
# Email: your-email@example.com
# One-time password (if 2FA enabled): xxxxxx
```

### 4. Prepare the package

#### A. Create wrapper scripts for binaries
The NPM package needs shell wrapper scripts in `bin/` directory:

```bash
mkdir -p bin
```

Create these wrapper files:

**bin/zexus:**
```bash
#!/usr/bin/env bash
python3 -m zexus "$@"
```

**bin/zx:**
```bash
#!/usr/bin/env bash
python3 -m zexus "$@"
```

**bin/zpm:**
```bash
#!/usr/bin/env bash
python3 -m zexus.zpm "$@"
```

**bin/zx-run:**
```bash
#!/usr/bin/env bash
python3 -m zexus.runner "$@"
```

**bin/zx-dev:**
```bash
#!/usr/bin/env bash
python3 -m zexus.dev "$@"
```

**bin/zx-deploy:**
```bash
#!/usr/bin/env bash
python3 -m zexus.deploy "$@"
```

**bin/zpics:**
```bash
#!/usr/bin/env bash
python3 -m zexus.pics "$@"
```

Make them executable:
```bash
chmod +x bin/*
```

#### B. Create post-install script
Create `scripts/postinstall.js`:

```javascript
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
```

Make it executable:
```bash
chmod +x scripts/postinstall.js
```

### 5. Test locally before publishing
```bash
# Check what will be included in the package
npm pack --dry-run

# Or create actual tarball to inspect
npm pack
# This creates zexus-1.6.2.tgz - you can extract and inspect it
```

### 6. Publish to NPM

#### Option A: Test on npm with limited scope first
```bash
# Publish as a scoped package first (safer for testing)
npm publish --access public --tag beta
```

#### Option B: Publish directly to NPM
```bash
npm publish --access public
```

### 7. Test installation
```bash
# In a different directory or machine:
npm install -g zexus

# Test it works:
zexus --version
zx --version
zpm --help
```

## Result
Users can now simply run:
```bash
npm install -g zexus
```

And get Zexus installed with:
- `zexus` command - Main interpreter
- `zx` command - Shorthand for running scripts
- `zpm` command - Zexus Package Manager
- `zx-run` command - Script runner
- `zx-dev` command - Development mode
- `zx-deploy` command - Deployment tool
- `zpics` command - Package info & check system

## For Future Releases
1. Update version in `package.json` and `pyproject.toml`
2. Run `npm publish --access public`
3. Done!

### Updating a Version
```bash
# Increment version automatically
npm version patch   # 1.6.2 -> 1.6.3
npm version minor   # 1.6.2 -> 1.7.0
npm version major   # 1.6.2 -> 2.0.0

# Then publish
npm publish --access public
```

## Troubleshooting

### "You do not have permission to publish"
```bash
# Make sure you're logged in
npm whoami

# If not logged in:
npm login
```

### "Package name already taken"
Someone else already published "zexus" - we need to check NPM first or:
- Use a scoped package: `@zaidux/zexus`
- Choose a different name: `zexus-lang`

To use scoped package, update `package.json`:
```json
{
  "name": "@zaidux/zexus",
  ...
}
```

Then publish:
```bash
npm publish --access public
```

### "ENEEDAUTH" error
Your authentication token expired:
```bash
npm logout
npm login
```

### "Python not found" after installation
Users need Python 3.8+ installed. Add to package README:
```markdown
## Prerequisites
- Node.js 14+ and npm 6+
- Python 3.8 or higher
- pip (Python package manager)
```

### Files missing from package
Check `.npmignore` or add files to the `files` array in `package.json`

### Permission denied when running commands
The bin scripts need to be executable:
```bash
chmod +x bin/*
```

## Alternative: Using GitHub Packages
You can also publish to GitHub Packages Registry:

```bash
# Add to package.json
{
  "publishConfig": {
    "registry": "https://npm.pkg.github.com"
  }
}

# Login to GitHub Packages
npm login --registry=https://npm.pkg.github.com

# Publish
npm publish
```

## Notes
- Version is 1.6.2 (matching PyPI)
- Requires Python 3.8+ and Node.js 14+
- Python package `zexus` will be auto-installed via postinstall script
- All CLI commands will be available globally after installation
- Post-install message guides users to documentation

## Security Considerations
- Enable 2FA on your npm account
- Use `npm audit` to check for vulnerabilities
- Never commit `.npmrc` with auth tokens to git
- Consider using automation tokens for CI/CD

## CI/CD Publishing (Advanced)
You can automate publishing with GitHub Actions:

```yaml
# .github/workflows/publish-npm.yml
name: Publish to NPM

on:
  release:
    types: [created]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          registry-url: 'https://registry.npmjs.org'
      - run: npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

Add your NPM token to GitHub repository secrets as `NPM_TOKEN`.
