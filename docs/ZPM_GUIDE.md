# ZPM - Zexus Package Manager

The official package manager for Zexus programming language.

## Features

- 📦 **Install packages** from the ZPM registry
- 🔍 **Search packages** across the ecosystem
- 📤 **Publish packages** to share with the community
- 🔒 **Dependency management** with lock files
- 🏗️ **Project initialization** with templates
- 🌐 **Built-in packages** for common tasks

## Installation

ZPM is included with Zexus:

```bash
pip install zexus
```

Verify installation:

```bash
zpm --version
```

## Quick Start

### 1. Initialize a Project

```bash
mkdir my-zexus-app
cd my-zexus-app
zpm init
```

This creates a `zexus.json` file:

```json
{
  "name": "my-zexus-app",
  "version": "0.1.0",
  "description": "",
  "main": "main.zx",
  "dependencies": {},
  "devDependencies": {},
  "scripts": {
    "test": "zx run tests/test_*.zx",
    "build": "zx compile main.zx"
  },
  "author": "",
  "license": "MIT"
}
```

### 2. Install Packages

```bash
# Install all dependencies from zexus.json
zpm install

# Install a specific package
zpm install std
zpm install crypto
zpm install web@0.2.0

# Install as dev dependency
zpm install testing -D
```

### 3. Use Packages in Code

```zexus
// Import from installed packages
use std from "std"
use {encrypt, decrypt} from "crypto"
use {Server, Router} from "web"

// Use the imports
let data = "secret message"
let encrypted = encrypt(data, "password")
print("Encrypted: " + encrypted)
```

### 4. Publish Your Package

```bash
# Make sure your zexus.json is complete
zpm info

# Publish to registry
zpm publish
```

## Commands

### `zpm init`

Initialize a new Zexus project.

```bash
zpm init                    # Interactive init
zpm init -n my-app -v 1.0.0 # With options
```

### `zpm install`

Install packages.

```bash
zpm install                 # Install all from zexus.json
zpm install <package>       # Install specific package
zpm install <package>@1.2.3 # Install specific version
zpm install <package> -D    # Install as devDependency
```

### `zpm uninstall`

Remove a package.

```bash
zpm uninstall <package>
```

### `zpm list`

List installed packages.

```bash
zpm list
```

Output:
```
┏━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Name     ┃ Version ┃ Path                ┃
┡━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ std      │ 0.1.0   │ ./zpm_modules/std   │
│ crypto   │ 0.1.0   │ ./zpm_modules/crypto│
└──────────┴─────────┴─────────────────────┘
```

### `zpm search`

Search for packages.

```bash
zpm search crypto
zpm search web framework
```

### `zpm publish`

Publish package to registry.

```bash
zpm publish
```

### `zpm info`

Show project information.

```bash
zpm info
```

### `zpm clean`

Remove `zpm_modules` directory.

```bash
zpm clean
```

## Package Structure

A Zexus package has this structure:

```
my-package/
├── zexus.json          # Package metadata
├── main.zx             # Main entry point
├── lib/                # Library code
│   ├── module1.zx
│   └── module2.zx
├── tests/              # Tests
│   └── test_main.zx
├── README.md           # Documentation
└── LICENSE             # License file
```

## zexus.json Format

```json
{
  "name": "my-package",
  "version": "1.0.0",
  "description": "My awesome Zexus package",
  "main": "main.zx",
  "keywords": ["utility", "helper"],
  "author": "Your Name <you@example.com>",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/username/my-package"
  },
  "dependencies": {
    "std": "^0.1.0",
    "crypto": "~0.2.0"
  },
  "devDependencies": {
    "testing": "^1.0.0"
  },
  "scripts": {
    "test": "zx run tests/test_*.zx",
    "build": "zx compile main.zx",
    "lint": "zx check *.zx"
  }
}
```

## Built-in Packages

ZPM includes several built-in packages:

### `std` - Standard Library
```zexus
use {map, filter, reduce} from "std"

let numbers = [1, 2, 3, 4, 5]
let doubled = map(numbers, transform: it * 2)
```

### `crypto` - Cryptography
```zexus
use {hash, encrypt, decrypt} from "crypto"

let hashed = hash("password", "sha256")
let encrypted = encrypt("data", "key")
```

### `web` - Web Framework
```zexus
use {Server, Router, Request, Response} from "web"

let app = Server()
let router = Router()

router.get("/", action(req: Request) -> Response {
    return Response("Hello, World!")
})

app.use(router)
app.listen(8080)
```

### `blockchain` - Blockchain Utilities
```zexus
use {Contract, Transaction, Address} from "blockchain"

let token = Contract({
    name: "MyToken",
    symbol: "MTK"
})
```

## Version Ranges

ZPM supports semantic versioning:

- `1.2.3` - Exact version
- `^1.2.3` - Compatible with 1.x.x (>=1.2.3 <2.0.0)
- `~1.2.3` - Approximately 1.2.x (>=1.2.3 <1.3.0)
- `>=1.2.3` - Greater than or equal
- `*` or `latest` - Latest version

## Lock File

`zexus-lock.json` ensures consistent installations:

```json
{
  "lockfileVersion": 1,
  "packages": {
    "std": {
      "version": "0.1.0",
      "path": "./zpm_modules/std"
    },
    "crypto": {
      "version": "0.2.1",
      "path": "./zpm_modules/crypto"
    }
  }
}
```

## Publishing Guidelines

### Before Publishing

1. **Test your package**:
   ```bash
   zpm install
   zx run tests/
   ```

2. **Update version**:
   Follow [semantic versioning](https://semver.org/):
   - Major: Breaking changes (1.0.0 → 2.0.0)
   - Minor: New features (1.0.0 → 1.1.0)
   - Patch: Bug fixes (1.0.0 → 1.0.1)

3. **Write documentation**:
   - Clear README.md
   - Usage examples
   - API documentation

4. **Add LICENSE**:
   - Choose appropriate license
   - Include LICENSE file

### Publishing

```bash
zpm publish
```

The package is:
1. Validated
2. Packaged into a tarball
3. Uploaded to registry
4. Made available for installation

## Registry

The default registry is `https://registry.zexus.dev` (coming soon).

Custom registry:

```bash
export ZPM_REGISTRY=https://my-registry.com
zpm install
```

## Environment Variables

- `ZPM_REGISTRY` - Registry URL (default: https://registry.zexus.dev)
- `ZPM_CACHE` - Cache directory (default: ~/.zpm/cache)

## Best Practices

1. **Use semantic versioning** for your packages
2. **Pin major versions** in dependencies (`^1.0.0`)
3. **Test before publishing**
4. **Keep packages focused** (single responsibility)
5. **Document your API** clearly
6. **Include examples** in README
7. **Use meaningful names** (descriptive, not generic)
8. **Specify license** explicitly

## Troubleshooting

### Package not found
```bash
# Clear cache
rm -rf ~/.zpm/cache

# Try again
zpm install <package>
```

### Installation fails
```bash
# Check zexus.json syntax
cat zexus.json | python -m json.tool

# Reinstall all
zpm clean
zpm install
```

### Publishing fails
```bash
# Verify package info
zpm info

# Check required fields
# - name
# - version
# - main file exists
```

## Roadmap

- [x] Basic package manager
- [x] Local package installation
- [x] Built-in packages
- [ ] Remote registry
- [ ] Package versioning resolution
- [ ] Dependency conflict resolution
- [ ] Package scripts execution
- [ ] Private packages/registries
- [ ] Package signing/verification
- [ ] Monorepo support

## Contributing

Want to add your package to ZPM?

1. Create your package
2. Test thoroughly
3. Publish with `zpm publish`
4. Share with the community!

## License

MIT License - See LICENSE file for details.

---

**Made with ❤️ by the Zexus Team**
