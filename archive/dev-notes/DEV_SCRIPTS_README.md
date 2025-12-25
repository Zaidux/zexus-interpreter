# Zexus Development & Deployment Scripts

## Quick Reference

```bash
# Development Mode (instant testing without installation)
./zx-dev run script.zx
./zx-dev check script.zx
./zx-dev --help

# Deploy with version bump
./zx-deploy --patch      # 1.0.0 → 1.0.1
./zx-deploy --minor      # 1.0.0 → 1.1.0
./zx-deploy --major      # 1.0.0 → 2.0.0
```

## zx-dev: Development Mode

**Purpose:** Run the Zexus interpreter directly from source without installation. Changes to the codebase take effect immediately.

**Usage:**
```bash
./zx-dev run myfile.zx          # Run a Zexus file
./zx-dev check myfile.zx        # Check syntax
./zx-dev --debug run myfile.zx  # Run with debug output
```

**Benefits:**
- ✅ Instant feedback - no reinstallation needed
- ✅ Test fixes in real-time
- ✅ Perfect for rapid development
- ✅ All changes take effect immediately

**When to use:**
- During active development
- Testing interpreter fixes
- Debugging issues
- Rapid prototyping

**Example workflow:**
```bash
# 1. Make changes to interpreter code
vim src/zexus/evaluator/statements.py

# 2. Test immediately
./zx-dev run test.zx

# 3. Iterate - no reinstall needed!
```

## zx-deploy: Production Deployment

**Purpose:** Package and install the Zexus interpreter for production use. Updates the system-wide `zx` command.

**Usage:**
```bash
# Deploy current version
./zx-deploy

# Deploy with version bump
./zx-deploy --patch      # Increment patch version (bug fixes)
./zx-deploy --minor      # Increment minor version (new features)
./zx-deploy --major      # Increment major version (breaking changes)

# Skip tests (faster deployment)
./zx-deploy --patch --skip-tests

# Force deployment even if tests fail
./zx-deploy --patch --force
```

**What it does:**
1. 📦 Bumps version in setup.py and pyproject.toml (if requested)
2. 🧪 Runs tests (unless --skip-tests)
3. 🗑️ Uninstalls old version
4. 🧹 Cleans build artifacts
5. 📦 Installs new version with pip
6. 🔍 Verifies installation

**When to use:**
- After completing a feature
- Before committing major changes
- When you want to test with the production `zx` command
- Preparing for release

**Example workflow:**
```bash
# 1. Develop with zx-dev
./zx-dev run test.zx

# 2. When ready, deploy with version bump
./zx-deploy --patch

# 3. Test production command
zx run test.zx
```

## Workflow: Development to Production

### Phase 1: Development (Use zx-dev)
```bash
# Make changes
vim src/zexus/parser/strategy_context.py

# Test immediately - no install
./zx-dev run test_script.zx

# Debug if needed
./zx-dev --debug run test_script.zx

# Iterate quickly
```

### Phase 2: Validation (Still using zx-dev)
```bash
# Test multiple files
./zx-dev run tests/test1.zx
./zx-dev run tests/test2.zx
./zx-dev check examples/*.zx

# Verify blockchain integration
cd /workspaces/Ziver-Chain
/workspaces/zexus-interpreter/zx-dev run src/core/block.zx
```

### Phase 3: Deployment (Switch to zx-deploy)
```bash
# Deploy with appropriate version bump
cd /workspaces/zexus-interpreter

# Bug fix
./zx-deploy --patch

# New feature
./zx-deploy --minor

# Breaking change
./zx-deploy --major
```

### Phase 4: Production Testing
```bash
# Test with system zx command
zx run test_script.zx
zx --version

# Verify blockchain works
cd /workspaces/Ziver-Chain
zx run src/main.zx
```

## Versioning Guide

**Patch (x.y.Z):** Bug fixes, minor improvements
```bash
./zx-deploy --patch
# Example: 1.0.0 → 1.0.1
```

**Minor (x.Y.z):** New features, backward compatible
```bash
./zx-deploy --minor
# Example: 1.0.0 → 1.1.0
```

**Major (X.y.z):** Breaking changes, major features
```bash
./zx-deploy --major
# Example: 1.0.0 → 2.0.0
```

## Troubleshooting

### zx-dev not working
```bash
# Make sure it's executable
chmod +x zx-dev

# Check Python path
which python3

# Run directly
python3 zx-dev run script.zx
```

### zx-deploy fails
```bash
# Check if old version can be uninstalled
pip uninstall zexus -y

# Clean everything
rm -rf build/ dist/ *.egg-info

# Try again
./zx-deploy --patch --skip-tests
```

### Changes not taking effect
```bash
# If using zx-dev: changes are immediate
./zx-dev run script.zx

# If using zx: redeploy
./zx-deploy --skip-tests
```

## Best Practices

1. **During Development:**
   - Always use `zx-dev`
   - Test frequently
   - Keep iterations fast

2. **Before Committing:**
   - Deploy with `zx-deploy`
   - Test production `zx` command
   - Verify version bump is appropriate

3. **Version Control:**
   - Commit version bumps with meaningful messages
   - Tag releases: `git tag v1.0.1`
   - Push tags: `git push --tags`

## Current Version

Run to check installed version:
```bash
zx --version
```

Current development version (in code):
```bash
grep version setup.py
```

## Examples

### Fix a bug in the parser
```bash
# 1. Edit the file
vim src/zexus/parser/strategy_context.py

# 2. Test fix immediately
./zx-dev run broken_script.zx

# 3. Verify it works
./zx-dev run tests/*.zx

# 4. Deploy with patch bump
./zx-deploy --patch

# 5. Verify production
zx run broken_script.zx
```

### Add a new feature
```bash
# 1. Develop with zx-dev
./zx-dev run feature_test.zx

# 2. Test thoroughly
./zx-dev run comprehensive_tests.zx

# 3. Deploy with minor bump
./zx-deploy --minor

# 4. Update changelog
echo "v1.1.0 - Added new feature" >> CHANGELOG.md
```

## Support

For issues or questions:
- Check the logs: deployment script shows full output
- Verify Python environment: `which python3`
- Check pip: `pip list | grep zexus`
- Test import: `python3 -c "from zexus.cli.main import cli"`
