# PyPI releases (Trusted Publishing + cibuildwheel)

This repo is set up to publish **manylinux (x86_64 + aarch64)** + **macOS** + **Windows** wheels to PyPI using:

- **cibuildwheel** for building wheels in a PyPI-compatible environment (manylinux)
- **PyPI Trusted Publishing (OIDC)** so you **don’t need an API token**

The workflow file is: `.github/workflows/wheels.yml`.

## Why this is needed (Termux)

If you build binary wheels on Termux, you commonly get wheels tagged like `linux_aarch64`, and **PyPI rejects those**.

With this setup, you can develop on Termux and publish by pushing a git tag; GitHub Actions builds compliant wheels and uploads them.

## One-time setup

### 1) Create GitHub Environments

In GitHub: **Repo → Settings → Environments**

Create:

- `pypi`

No secrets are required for Trusted Publishing.

### 2) Configure PyPI Trusted Publisher

In PyPI: **Your project → Settings → Publishing → Trusted publishers → Add a new trusted publisher**

Fill these fields:

- **Provider:** GitHub Actions
- **Owner:** `Zaidux`
- **Repository:** `zexus-interpreter`
- **Workflow filename:** `.github/workflows/wheels.yml`
- **Environment name:** `pypi`

Save.

## Publishing a release (tag-based)

Publishing is **tag-triggered**. From anywhere (including Termux), push a version tag:

```bash
# example version
git tag v1.7.3
git push origin v1.7.3
```

That tag triggers GitHub Actions to:

1. build wheels (including manylinux aarch64)
2. build an sdist
3. publish all artifacts to PyPI using OIDC (Trusted Publishing)

## Notes

- Tag format is `v*` (e.g. `v1.7.3`). This matches `on.push.tags: ["v*"]`.
- The build includes native extensions in CI (`ZEXUS_BUILD_EXTENSIONS=1`) so wheels include the C/C++/Cython modules.
- If you want a “dry run”, you can use **Actions → Build wheels (cibuildwheel) → Run workflow** to build artifacts without publishing.

