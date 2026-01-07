# Performance Improvements Summary

## Baseline
- Interpreter processing 1,000 transfer actions required ~4.3s (≈232 TPS) prior to Phase 3 work.
- Storage commits executed per action causing excessive disk flushes and JSON rehydration.

## Implemented Enhancements

### Phase 3 – Storage Batching & Serialization
- Added batched SQLite commit logic with configurable thresholds.
- Introduced map entry hashing to persist maps per-key instead of rewriting entire blobs.
- Optimized JSON serializer/deserializer dispatch to reduce isinstance overhead.

### Phase 4 – Map Persistence, Caching & VM Focus
- Wrapped contract maps with `StorageMap` to track dirty keys and deletions.
- Persisted map entries individually via `SET_MAP`/`DELETE_MAP_ENTRY` events.
- Added per-action cache to reuse hydrated storage values during execution.
- Promoted persistent cache for contract storage to avoid redundant scans across actions.
- Raised default SQLite batch size to 512 writes (override with `ZEXUS_STORAGE_BATCH_SIZE`) to cut commit churn.
- Flagged `EvaluationError` instances for O(1) error detection, removing millions of `isinstance` checks from hot loops.
- Introduced evaluator dispatch table for common AST nodes to skip the multi-hundred `isinstance` cascade during execution.
- Specialized stack VM by front-loading a direct opcode dispatch map plus async-aware handlers, eliminating the cascading `if/elif` chain on hot opcodes.
- Added optional `ZEXUS_VM_PROFILE_OPS=1` instrumentation to capture opcode frequencies for future tuning.
- Preallocated stack arrays and reused pooled await buffers to slash transient list allocations inside the VM hot path.

### Profiling & VM Updates
- Extended `profile_performance.py` with `--use-vm` flag for targeted profiling.
- Benchmarked VM path to validate interpreter vs VM split and highlight residual hotspots.

## Benchmark Results

| Scenario | Transactions | Duration | TPS | Notes |
| --- | --- | --- | --- | --- |
| Pre-optimization (baseline) | 1,000 | ~4,300 ms | ~232 | Frequent commits & full-map rewrites |
| Post Phase 3 | 500 | 1,073 ms | 465 | SQLite batching enabled |
| Phase 4 (pre-persistent cache) | 10,000 | 238,166 ms | 41 | Per-key storage without persistent reuse |
| Phase 4 (persistent cache) | 10,000 | 10,249 ms | 975 | Persistent storage cache eliminates rehydration cost |
| Batch tuning + error flag | 1,000 | 1,567 ms | 638 | Persistent cache + larger batch window + fast error checks |
| Batch tuning + error flag | 10,000 | 9,603 ms | 1,041 | Same run with tuned batch size |
| Dispatch table fast-path | 1,000 | 719 ms | 1,390 | Hot AST nodes bypass isinstance chain |
| Dispatch table fast-path | 10,000 | 8,231 ms | 1,214 | VM receives faster evaluator input |
| VM opcode dispatch table | 1,000 | 733 ms | 1,364 | Direct opcode handlers collapse VM branching |
| VM opcode dispatch table | 10,000 | 8,482 ms | 1,178 | Same tuning applied to full 10k transfer run |
| Stack preallocation + pooled await buffers | 1,000 | 670 ms | 1,492 | Batch size 2,048; lowest stack churn observed |
| Stack preallocation + pooled await buffers | 10,000 | 8,064 ms | 1,240 | Batch size 2,048; fewer storage commits |

## Remaining Opportunities
- Further tune SQLite batching thresholds to squeeze commit overhead.
- Investigate VM dispatch hotspots (`eval_node`, `isinstance`) for specialization.
- Explore incremental map hydration to support selective entry loading for very large maps.
- Fix opcode profiling path (current `ZEXUS_VM_PROFILE_OPS=1` runs emit empty datasets).
