# Zexus Documentation Index

Welcome to the Zexus Programming Language documentation.

## 🚀 Getting Started

- **[README](../README.md)** — Project overview, installation, quick start
- **[Quick Start](../QUICK_START.md)** — Installation matrix and first program
- **[Zexus Guide](../ZEXUS_GUIDE.md)** — Test-verified tutorial (basics → advanced)
- **[Zexus Rules](../ZEXUS_RULES.md)** — 15 rules to avoid common mistakes
- **[CLI Commands](CLI_COMMANDS.md)** — Command-line reference

## 📚 Core Language Reference

- **[Keywords](KEYWORDS.md)** — Complete keyword reference
- **[Builtins](BUILTINS.md)** — Built-in functions (interpreter + VM)
- **[Type Safety](TYPE_SAFETY.md)** — Type system and checking
- **[Error Reporting](ERROR_REPORTING.md)** — Error messages and diagnostics
- **[Quick Reference](QUICK_REFERENCE.md)** — Syntax cheat sheet

## 🏗️ Architecture & Internals

- **[Architecture](ARCHITECTURE.md)** — System architecture overview
- **[Main Entry Point](MAIN_ENTRY_POINT.md)** — How `zx run` works
- **[Parsing Pipeline](../PARSING_PIPELINE.md)** — Lexer → Parser → AST → Evaluator
- **[Module System](MODULE_SYSTEM.md)** — `use` / `import` / `export`
- **[Performance Features](PERFORMANCE_FEATURES.md)** — VM, JIT, optimizations
- **[Plugin System](PLUGIN_SYSTEM.md)** — Extending Zexus with plugins

## ⛓️ Blockchain

- **[Blockchain Features](BLOCKCHAIN_FEATURES.md)** — 3 consensus engines, MPT, contracts
- **[Contract References](CONTRACT_REFERENCES.md)** — Smart contract patterns
- **[Crypto Functions](CRYPTO_FUNCTIONS.md)** — Hashing, signing, key management

## 🔒 Security

- **[Security](SECURITY.md)** — Capabilities, sanitization, resource limits, overflow protection, taint tracking, access control, audit logging, and new v1.8.4 security modules

## 🔄 Concurrency

- **[Concurrency](CONCURRENCY.md)** — Channels, async/await, atomic operations
- **[WaitGroup & Barrier](WAITGROUP_BARRIER.md)** — Synchronization primitives
- Keyword reference: [Async & Concurrency](keywords/ASYNC_CONCURRENCY.md)

## 📦 Ecosystem

- **[Ecosystem Guide](ECOSYSTEM_GUIDE.md)** — Roadmap and strategy
- **[ZPM Guide](ZPM_GUIDE.md)** — Package manager usage
- **[Package Development](PACKAGE_DEVELOPMENT.md)** — Creating packages
- **[Watch Feature](WATCH_FEATURE.md)** — Reactive state management
- **[Philosophy](PHILOSOPHY.md)** — Language design philosophy

## 📂 Subdirectories

- **[keywords/](keywords/)** — Per-keyword deep-dive documentation
- **[keywords/features/](keywords/features/)** — Feature reference guides (HTTP server, DB drivers, CLI framework, testing, VM, profiler, SSA)
- **[packages/](packages/)** — Package specifications (@zexus/web, @zexus/db, @zexus/ai, @zexus/gui)
- **[stdlib/](stdlib/)** — Standard library module docs (blockchain, crypto, integration)
- **[profiler/](profiler/)** — Profiler usage guide

## 📋 Changelog

See **[CHANGELOG.md](../CHANGELOG.md)** for version history.

---

**Last Updated**: April 12, 2026 | **Version**: 1.8.4
