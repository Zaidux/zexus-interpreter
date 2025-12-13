# Publishing Zexus to PyPI

## Quick Steps (for you)

### 1. Install build tools (one-time setup)
```bash
pip install build twine
```

### 2. Build the package
```bash
cd /workspaces/zexus-interpreter
python -m build
```

This creates `dist/` with:
- `zexus-1.0.0-py3-none-any.whl` (wheel package)
- `zexus-1.0.0.tar.gz` (source distribution)

### 3. Create PyPI Account (if you don't have one)
- Go to https://pypi.org/account/register/
- Verify your email
- Enable 2FA (recommended)
- Create an API token at https://pypi.org/manage/account/token/

### 4. Configure PyPI credentials

Option A: Using API token (recommended)
```bash
# When prompted by twine, use:
# Username: __token__
# Password: pypi-AgEI... (your API token)
```

Option B: Save in ~/.pypirc
```ini
[pypi]
username = __token__
password = pypi-AgEI...  # your API token here
```

### 5. Test on TestPyPI first (optional but recommended)
```bash
# Upload to test server
python -m twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ zexus
```

### 6. Upload to real PyPI
```bash
python -m twine upload dist/*
```

### 7. Test installation
```bash
# In a new environment/terminal:
pip install zexus

# Test it works:
zexus --version
zx --version
```

## Result
Users can now simply run:
```bash
pip install zexus
```

And get Zexus installed with:
- `zexus` command - Main interpreter
- `zx` command - Shorthand for running scripts
- `zpm` command - Zexus Package Manager

## For Future Releases
1. Update version in `pyproject.toml` and `setup.py`
2. Run `python -m build`
3. Run `python -m twine upload dist/*`
4. Done!

## Troubleshooting

### "File already exists"
Remove old dist files: `rm -rf dist/ build/ *.egg-info`

### "Invalid credentials"
- Check your API token is correct
- Make sure username is `__token__` (with double underscores)

### "Package name already taken"
Someone else already published "zexus" - we need to check PyPI first or choose different name

## Notes
- Version is now 1.0.0 (stable release)
- Python 3.8+ required
- All dependencies will be auto-installed (click, rich)
- Post-install message will guide users to documentation
