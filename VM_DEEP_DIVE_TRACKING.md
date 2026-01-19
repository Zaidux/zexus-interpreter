# VM Deep Dive Tracking

Status key:
- [ ] Not started
- [~] In progress
- [x] Done

## Phase 1: Correctness + missing opcode coverage
- [x] Remove duplicate opcode handlers (ATOMIC_ADD/ATOMIC_CAS/BARRIER/FOR_ITER)
- [x] Fix fast dispatch IMPORT no-op masking real IMPORT handling
- [x] Implement missing stack opcodes (BUILD_SET, SLICE, WRITE, EXPORT)
- [x] Implement missing call opcodes (CALL_BUILTIN, CALL_FUNC_CONST, SPAWN_CALL)
- [x] Implement missing protocol opcode (DEFINE_PROTOCOL)

## Phase 2: Performance upgrades (planned)
- [x] Expand `fastops` opcode coverage for hot paths
- [x] Inline cache for name/method lookups in stack VM
- [x] Safer JIT + gas metering compatibility
- [x] Broaden register conversion coverage
- [ ] Async optimizer usage audit

## Notes
- Keep changes small and verify behavior with existing tests/benchmarks.