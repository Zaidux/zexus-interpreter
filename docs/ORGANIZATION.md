# Zexus Repository Organization

This document describes the organization of documentation and tests in the Zexus interpreter repository.

## Documentation Structure

All documentation is organized under the `docs/` directory with the following structure:

### `docs/` - General Documentation
Contains high-level guides and documentation:
- `README.md` - Documentation index
- `INDEX.md` - Quick reference guide
- `ARCHITECTURE.md` - System architecture
- `PHILOSOPHY.md` - Language design philosophy
- `QUICK_START.md` - Getting started guide
- `MODULE_SYSTEM.md` - Module system guide
- `PLUGIN_SYSTEM.md` - Plugin development guide
- `ZPM_GUIDE.md` - Package manager guide
- `ERROR_REPORTING.md` - Error handling guide
- `CONCURRENCY.md` - Concurrency features
- `PERFORMANCE_FEATURES.md` - Performance optimization guide
- `BLOCKCHAIN_FEATURES.md` - Blockchain capabilities
- `SECURITY_FEATURES.md` - Security features overview
- `WATCH_FEATURE.md` - Reactive state management
- `ISSUES.md` - Known issues and troubleshooting
- `STATUS.md` - Project status
- Other guides in `guides/` subdirectory

### `docs/keywords/` - Keyword Documentation
Contains documentation for all language keywords and commands:

#### Core Keywords
- `ACTION_FUNCTION_LAMBDA_RETURN.md` - Function keywords
- `LET.md` - Variable declarations
- `CONST.md` - Constants
- `DATA.md` - Data types
- `IF_ELIF_ELSE.md` - Conditional statements
- `WHILE_FOR_EACH_IN.md` - Loops
- `ASYNC_AWAIT.md` - Async programming
- `ERROR_HANDLING.md` - Error handling keywords
- `MODULE_SYSTEM.md` - Module keywords
- And more...

#### Command Keywords (COMMAND_*.md)
Special command keywords for advanced features:
- `COMMAND_audit.md` - Audit logging
- `COMMAND_buffer.md` - Memory buffers
- `COMMAND_const.md` - Constant declarations
- `COMMAND_defer.md` - Deferred execution
- `COMMAND_elif.md` - Elif conditional
- `COMMAND_enum.md` - Enumerations
- `COMMAND_gc.md` - Garbage collection
- `COMMAND_inline.md` - Function inlining
- `COMMAND_native.md` - Native FFI
- `COMMAND_pattern.md` - Pattern matching
- `COMMAND_restrict.md` - Input validation
- `COMMAND_sandbox.md` - Sandboxed execution
- `COMMAND_simd.md` - SIMD operations
- `COMMAND_stream.md` - Event streaming
- `COMMAND_trail.md` - Event tracking
- `COMMAND_watch.md` - Reactive state

#### Advanced Features
- `ADVANCED_KEYWORDS.md` - Advanced keyword reference
- `SPECIAL_KEYWORDS.md` - Special keywords
- `BLOCKCHAIN_KEYWORDS.md` - Blockchain-specific keywords
- `KEYWORD_TESTING_INDEX.md` - Keyword testing index
- `KEYWORD_TESTING_MASTER_LIST.md` - Complete keyword test list
- `KEYWORD_TESTING_PROJECT_SUMMARY.md` - Testing project overview

### `docs/keywords/features/` - Feature Implementation Documentation
Contains detailed implementation reports and summaries for features:

#### Phase Completion Reports
- `PHASE_5_COMPLETION_REPORT.md` - Phase 5 summary
- `PHASE_6_COMPLETION_REPORT.md` - Phase 6 summary
- `PHASE_10_COMPLETION.md` - Phase 10 summary
- `10-PHASE_COMPLETE.md` - Complete phase summary
- `PHASE_1_2_SUMMARY.md` - Phases 1-2 summary

#### Implementation Summaries
- `ADVANCED_FEATURES_IMPLEMENTATION.md` - Advanced features
- `MODULE_SYSTEM_IMPLEMENTATION_SUMMARY.md` - Module system
- `SECURITY_IMPLEMENTATION.md` - Security features
- `SECURITY_IMPLEMENTATION_SUMMARY.md` - Security summary
- `BLOCKCHAIN_IMPLEMENTATION.md` - Blockchain features
- `BLOCKCHAIN_PARSER_EVALUATOR_INTEGRATION.md` - Blockchain integration
- `AUDIT_COMPLETION_SUMMARY.md` - Audit feature
- `AUDIT_SECURITY_INTEGRATION.md` - Audit security
- `SEAL_IMPLEMENTATION_SUMMARY.md` - Seal feature

#### Feature-Specific Documentation
- `MAIN_ENTRY_POINT_IMPLEMENTATION.md` - Main entry point
- `MAIN_ENTRY_POINT_IMPLEMENTATION_STATUS.md` - Entry point status
- `IF_THEN_ELSE_IMPLEMENTATION.md` - If/else implementation
- `KEYWORD_AFTER_DOT_IMPLEMENTATION.md` - Keywords after dot
- `KEYWORD_AFTER_DOT.md` - Keywords after dot guide
- `REQUIRE_TOLERANCE_IMPLEMENTATION.md` - Require tolerance
- `VERIFY_ENHANCEMENT_GUIDE.md` - Verify enhancements
- `VERIFY_IMPLEMENTATION_SUMMARY.md` - Verify summary

#### Session Summaries
- `SESSION_COMPLETION_SUMMARY.md` - Session completion
- `SESSION_PARSER_FIX_SUMMARY.md` - Parser fixes
- `installation_paths_summary.md` - Installation paths
- `parsing_fixes_summary.md` - Parsing fixes
- `module_cache.md` - Module caching
- `parser_fix_summary.md` - Parser fix details
- `evaluator_export_compiler_summary.md` - Evaluator/compiler

#### Fix Summaries
- `FIXES_SUMMARY.md` - General fixes
- `FIX_ISSUES_5_6_SUMMARY.md` - Issues 5-6 fixes
- `FIX_SUMMARY.md` - Fix summary
- `ADVANCED_PARSER_FIX_SUMMARY.md` - Parser fixes

#### Integration Reports
- `INTEGRATION_COMPLETE.md` - Integration complete
- `INTEGRATION_COMPLETE_SUMMARY.md` - Integration summary
- `IMPLEMENTATION_SUMMARY.md` - Implementation summary
- `COMPREHENSIVE_VERIFICATION_REPORT.md` - Verification report

#### VM Documentation
- `VM_INTEGRATION_SUMMARY.md` - VM integration
- `VM_CONNECTION_VERIFIED.md` - VM connection
- `VM_INTEGRATION_COMPLETE.md` - VM completion
- `VM_CRITICAL_ISSUES_FIXED.md` - VM fixes
- `VM_ENHANCEMENT_MASTER_LIST.md` - VM enhancements
- `VM_OPTIMIZATION_PHASE_8_MASTER_LIST.md` - VM optimizations
- `ASYNC_OPTIMIZER.md` - Async optimization
- `MEMORY_POOL_USAGE_GUIDE.md` - Memory pool
- `PROFILER_USAGE_GUIDE.md` - Profiler
- `SSA_AND_REGISTER_ALLOCATION.md` - SSA/registers
- Phase completion reports (2-7)

#### Testing Documentation
- `TEST_SUITE_DOCUMENTATION.md` - Test suite guide
- `PERFORMANCE_FEATURES_SUMMARY.md` - Performance features
- `ZEXUS_IN_ACTION.md` - Usage examples

## Test Structure

All tests are organized under the `tests/` directory with the following structure:

### `tests/` - Root Level
Main test directory containing:
- Python integration test files (`test_*.py`)
- `conftest.py` - Pytest configuration
- Debug utilities (`debug_*.py`)
- Comprehensive integration tests (`comprehensive_test.zx`, `final_integration_test.zx`)
- `TEST_RESULTS.md` - Test results documentation

### `tests/unit/` - Unit Tests
Python unit tests for individual components (34 files):
- Parser tests
- Evaluator tests
- Lexer tests
- Type system tests
- Feature-specific tests
- Performance tests
- Memory tests
- VM tests

### `tests/integration/` - Integration Tests
Integration tests organized by category:

#### `tests/integration/keywords/` - Keyword Tests
Tests for specific language keywords (17 files):
- Argument handling (`test_args*.zx`)
- Function tests (`test_bare_function.zx`)
- Let/const tests (`test_let_*.zx`)
- If/else tests (`test_if*.zx`, `if_test.zx`)
- Loop tests (`test_loops.zx`, `test_while_simple.zx`)
- Lambda tests (`test_lambda_*.py`, `test_all_lambdas.zx`)
- Action tests (`test_action*.zx`)

#### `tests/integration/features/` - Feature Tests
Tests for advanced features (17 files):
- Production features (`test_production_features.zx`)
- Hot reload (`test_hot_reload*.zx`)
- Scheduling (`test_schedule*.zx`)
- Signal handling (`test_signal_handlers*.zx`)
- Lifecycle hooks (`test_lifecycle_hooks.zx`)
- Module introspection (`test_module_introspection.zx`)
- Daemon features (`test_daemonize.zx`)
- Enhanced runtime (`test_enhanced_run.zx`)
- Arguments (`test_main_args_complete.zx`, `test_run_with_args.zx`)
- Phase tests (`test_phase1_features.zx`)
- Simple features (`test_simple_features.zx`)

#### `tests/integration/core/` - Core Functionality Tests
Tests for core language features (33 files):
- Basic operations (`test_simple.zx`, `test_minimal.zx`, `test_core.zx`)
- Assignments (`test_assignment_simple.zx`, `test_explicit.zx`)
- Expressions (`test_simple_expressions.zx`)
- Booleans (`test_boolean*.zx`)
- Comparisons (`test_comparisons*.zx`, `test_all_comparisons*.zx`)
- Numbers (`test_float.zx`, `test_modulo.zx`)
- Variables (`test_vars.zx`)
- Data structures (`test_map.zx`, `test_data_structures.zx`)
- Control flow (`test_*_control.zx`)
- Print (`test_print*.zx`)
- UI (`test_ui.zx`)
- Syntax (`test_syntax_discovery.zx`, `test_working_syntax.zx`)
- Multi-language (`test_multi_language_fixed.zx`)
- Edge cases (`test_edge_cases.zx`)
- Debug/File I/O (`test_debug_error.zx`, `test_file_io.zx`)
- Date/time (`test_datetime_math.zx`)
- Other core tests

#### `tests/integration/` - General Integration Tests
Existing integration tests (58 files):
- Module tests
- Feature tests
- Issue reproduction tests
- Comprehensive tests

### `tests/vm/` - VM Tests
Virtual machine and compiler tests (27 files):
- JIT compilation tests
- Optimizer tests
- Register allocation tests
- SSA conversion tests
- Memory management tests
- Cache tests
- Profiler tests
- Integration tests

### `tests/keyword_tests/` - Structured Keyword Tests
Organized keyword tests by difficulty:
- `easy/` - 23 basic keyword tests
- `medium/` - 21 intermediate tests
- `complex/` - 21 advanced tests
- `README.md` - Keyword testing guide
- `run_keyword_test.sh` - Test runner script

### `tests/fixtures/` - Test Fixtures
Test data and helper files:
- Module fixtures (`modules/`)
- Test programs (`*.zx`)

### `tests/examples/` - Example Programs
Sample programs demonstrating features (19 files)

### `tests/repro/` - Reproduction Tests
Bug reproduction tests (4 files)

### `tests/builtin_modules/` - Built-in Module Tests
Tests for built-in modules

## Finding Documentation

### By Topic
- **Getting Started**: `docs/QUICK_START.md`, `docs/README.md`
- **Language Reference**: `docs/keywords/` (all keyword docs)
- **Advanced Features**: `docs/keywords/ADVANCED_KEYWORDS.md`, `docs/keywords/features/`
- **Security**: `docs/SECURITY_FEATURES.md`, `docs/keywords/SECURITY.md`
- **Performance**: `docs/PERFORMANCE_FEATURES.md`, `docs/keywords/PERFORMANCE.md`
- **Blockchain**: `docs/BLOCKCHAIN_FEATURES.md`, `docs/keywords/BLOCKCHAIN_KEYWORDS.md`
- **Implementation Details**: `docs/keywords/features/` (all implementation docs)
- **Testing**: `docs/keywords/features/TEST_SUITE_DOCUMENTATION.md`

### By Keyword
All keyword documentation is in `docs/keywords/` with filenames matching the keyword name.

### By Feature
Feature implementation details are in `docs/keywords/features/` with descriptive filenames.

## Recent Changes

This organization was implemented in December 2025 to consolidate scattered documentation and tests:

1. **Documentation Consolidation**:
   - Moved all COMMAND_*.md files to `docs/keywords/`
   - Moved all keyword docs to `docs/keywords/`
   - Moved all feature/implementation docs to `docs/keywords/features/`
   - Removed duplicate files and obsolete `doc/` directory

2. **Test Reorganization**:
   - Created `tests/integration/keywords/` for keyword tests
   - Created `tests/integration/features/` for feature tests
   - Created `tests/integration/core/` for core functionality tests
   - Consolidated `phase1/` and `final_tests/` directories
   - Removed 9 duplicate/obsolete test files

3. **Reference Updates**:
   - Updated README.md to point to new documentation locations
   - Updated docs/INDEX.md to reflect new structure

## Contributing

When adding new documentation:
- General guides → `docs/`
- Keyword documentation → `docs/keywords/`
- Feature implementations → `docs/keywords/features/`

When adding new tests:
- Unit tests → `tests/unit/`
- Keyword tests → `tests/integration/keywords/`
- Feature tests → `tests/integration/features/`
- Core tests → `tests/integration/core/`
- VM tests → `tests/vm/`
