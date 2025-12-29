# Official Zexus Packages (@zexus/*)

This directory contains specifications for official Zexus packages as part of the **Phase 3: Batteries Included** strategy.

## Package List

### Web Development
- **[@zexus/web](ZEXUS_WEB_PACKAGE.md)** - Full-stack web framework
  - HTTP server and routing
  - Middleware system
  - Template engine
  - WebSocket support
  - Authentication

### Database
- **[@zexus/db](ZEXUS_DB_PACKAGE.md)** - Database ORM and drivers
  - PostgreSQL, MySQL, MongoDB, SQLite drivers
  - ORM with relationships
  - Query builder
  - Migrations

### AI/ML
- **[@zexus/ai](ZEXUS_AI_PACKAGE.md)** - Machine learning framework
  - Neural networks
  - Pre-trained models
  - NLP and computer vision
  - GPU acceleration

### GUI
- **[@zexus/gui](ZEXUS_GUI_PACKAGE.md)** - Cross-platform GUI framework
  - Desktop applications
  - Reactive UI
  - Native widgets
  - Theming

## Additional Packages (Planned)

- **@zexus/cli** - CLI framework
- **@zexus/test** - Testing framework
- **@zexus/crypto** - Extended cryptography
- **@zexus/blockchain** - Blockchain utilities
- **@zexus/mobile** - Mobile app framework
- **@zexus/cloud** - Cloud deployment tools

## Package Status

All packages are currently in **Planning Phase** (Phase 3).

### Development Timeline

1. **Phase 1** (Q1-Q2 2025): Build core libraries WITH Zexus
2. **Phase 2** (Q3-Q4 2025): Integrate native keywords INTO Zexus
3. **Phase 3** (2026+): Release official packages

## Installation

Once released, packages can be installed via ZPM:

```bash
zpm install @zexus/web
zpm install @zexus/db
zpm install @zexus/ai
zpm install @zexus/gui
```

## Contributing

See [Package Development Guide](../PACKAGE_DEVELOPMENT.md) for contribution guidelines.

## Related Documentation

- [Ecosystem Strategy](../ECOSYSTEM_STRATEGY.md)
- [Package Development Guide](../PACKAGE_DEVELOPMENT.md)
- [ZPM Guide](../ZPM_GUIDE.md)

---

**Last Updated**: 2025-12-29
