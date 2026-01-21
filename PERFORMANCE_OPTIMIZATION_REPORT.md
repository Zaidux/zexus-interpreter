# Performance Optimization Report - Round 2 Complete
## Test: 100 transactions on perf_full_network_10k.zx

### Performance Summary
- **Baseline**: 121.2s (0.83 tx/s)
- **After Round 1**: 113.6s (0.88 tx/s) - **6.3% faster**
- **After Round 2**: 112.7s (0.89 tx/s) - **7.0% faster than baseline**
- **Round 3 (current profiler run)**: VM 34.21s | Interpreter 14.85s | Parse 37.16ms

### Key Findings

**✅ Successful Optimizations:**
1. **Thread locks**: 37.6s → 15.5s (59% improvement) - Lock-free execution path
2. **len() calls**: 13.9s → 9.5s (32% improvement) - Direct stack.sp access
3. **dict.get**: 16.5s → 11.4s (31% improvement) - Better name caching

**⚠️ Attempted But Ineffective:**
- **Sync fast dispatch**: Added ~1% improvement but increased lock contention
- **Bytecode compilation for sync path**: Actions already use bytecode in async path

**❌ Persistent Bottlenecks:**
- **eval_node**: 9.8-10.0s (110k calls) - Interpreter fallback from imports/modules
- **_parse_block_statements**: 6.7-6.8s (606 calls) - Runtime contract parsing
- **isinstance**: 6.7-6.8s (2.7M calls) - Type checking overhead
- **Thread coordination**: 15-26s - Async event loop and child VM creation

### Optimization Details

#### Round 1: Low-Hanging Fruit (121.2s → 113.6s, 6.3% improvement)

1. **Thread Lock Optimization** (`src/zexus/vm/vm.py` line ~1225)
   ```python
   # Before: Always acquired locks
   # After: Skip locks in single-threaded mode
   if self._jit_lock is None:
       # Lock-free path...
   ```
   **Result**: 37.6s → 15.5s

2. **Stack Size Optimization** (`src/zexus/vm/vm.py` line ~2663)
   ```python
   # Before: len(stack) - 2M function calls
   # After: stack.sp - Direct attribute access
   ```
   **Result**: 13.9s → 9.5s

3. **Gas Metering Optimization** (`src/zexus/vm/gas_metering.py`)
   - Added `_cost_cache` for static operation costs
   - Reordered checks to test operation count first
   **Result**: Marginal improvement

#### Round 2: Async Overhead Investigation (113.6s → 112.7s, 0.9% improvement)

4. **Synchronous Fast Dispatch** (`--perf-fast-dispatch` flag)
   - Enabled `_run_stack_bytecode_sync` to bypass asyncio
   - **Unexpected Result**: Minimal improvement, sometimes slower
   - **Analysis**: async overhead is only ~6s, not 33s as initially thought
   - Async calls are necessary for legitimate concurrency (event handlers, futures)

5. **Bytecode Compilation in Sync Path** (`src/zexus/vm/vm.py` line ~1866)
   - Added Action→bytecode compilation to `_invoke_callable_sync`
   - Mirrors async path implementation at line ~3730
   - **Result**: No improvement - sync path rarely used; async path already had it

### Root Cause Analysis

**Why is eval_node still at 10s?**

The 110k `eval_node` calls are NOT from Actions (those use bytecode), but from:
1. **Module imports** - Modules are executed via interpreter (line ~896 in vm.py)
2. **Dynamic code evaluation** - `eval`, `exec`, runtime introspection
3. **Contract method compilation** - Parser called during execution

**Why didn't sync dispatch help?**

The async overhead is mostly from:
- **Necessary async operations**: Event handlers, futures, promises (can't be eliminated)
- **Threading coordination**: Child VM creation, message passing (architectural)
- **Lock contention**: Even with fast path, locks protect shared JIT/memory state

The `_perf_fast_dispatch` flag only affects direct bytecode execution, not:
- High-level ops (always async)
- Builtin calls that spawn tasks
- Module imports
- Child VM creation

### Remaining Optimization Opportunities

**High Impact (estimated 15-20s savings):**

1. **Pre-compile Modules** (6.7s potential)
   - Currently: Modules parsed and evaluated during runtime imports
   - Fix: Parse all modules during initialization, cache ASTs and bytecode
   - Implementation: Module loader pre-compilation phase

2. **Cache Parsed Contracts** (6.7s potential)
   - Currently: `_parse_block_statements` called 606 times during execution
   - Fix: Parse contracts once, store in global registry
   - Implementation: Contract AST cache keyed by source hash

3. **Reduce Module Eval Overhead** (3-5s potential)
   - Currently: Modules use interpreter even with VM available
   - Fix: Compile module bodies to bytecode, execute in VM
   - Implementation: Extend module cache to store bytecode

**Medium Impact (estimated 5-10s savings):**

4. **Optimize isinstance Checks** (3-4s potential)
   - 2.7M isinstance calls for type validation
   - Fix: Type cache, duck typing where safe
   - Implementation: `_type_cache` dict, skip checks for trusted objects

5. **Reduce VM Recreation** (2-3s potential)
   - Child VMs created frequently for function calls
   - Fix: VM pool, reuse VMs for similar scopes
   - Implementation: VM object pool with reset() method

**Low Impact (estimated 2-5s savings):**

6. **Further dict.get Optimization** (2-3s potential)
   - Still 11.4s spent on 2M environment lookups
   - Fix: Scope-local variable caching, closure optimization
   - Implementation: Compiler pass to identify hot variables

### Estimated Final Performance

**Conservative estimate (implementing high-impact items):**
- Current: 112.7s
- After module pre-compilation: ~106s (5.9% improvement)
- After contract caching: ~99s (12.2% improvement)  
- After module bytecode: ~94s (16.6% improvement)
- **Total potential**: ~94s (22% faster than current, 28% faster than baseline)

**Optimistic estimate (all optimizations):**
- With isinstance optimization: ~91s
- With VM pooling: ~89s
- With dict.get optimization: ~86s
- **Total potential**: ~86s (24% faster than current, 29% faster than baseline)

### Code Changes Summary

**Files Modified:**
1. `src/zexus/vm/vm.py` - Lock optimization, stack.sp, bytecode in sync path
2. `src/zexus/vm/gas_metering.py` - Cost caching
3. `scripts/profile_full_network_components.py` - Profiling infrastructure

**Lines Changed**: ~180 lines
**Performance Gain**: 7.0% (121.2s → 112.7s)
**Effort**: Medium (2-3 hours of profiling and optimization)

### Next Steps

1. **Implement Module Pre-compilation**
   - Add `--precompile-modules` flag to profiler
   - Parse all imports before execution
   - Store in global module registry

2. **Contract AST Cache**
   - Hash contract source
   - Cache parsed AST globally
   - Reuse on repeated parses

3. **Module Bytecode Execution**
   - Compile module bodies to bytecode
   - Execute in VM instead of interpreter
   - Store in module cache

### Lessons Learned

1. **Profile First**: Initial assumption about async overhead (33s) was wrong (actually ~6s)
2. **Understand the Path**: Sync dispatch didn't help because most code uses async for valid reasons
3. **Check Both Paths**: Adding bytecode compilation to sync path didn't help because it was already in async path
4. **Measure Everything**: Small optimizations compound, but only if they're on the hot path
5. **Know Your Bottlenecks**: The real issues are module loading and parsing, not async overhead

**Bottom Line**: We achieved 7% improvement with targeted micro-optimizations. The next 20% requires architectural changes (pre-compilation, caching).

---

## Round 3 (Current Work): VM + Interpreter Pair Fixes
**Profiler run**: `scripts/profile_full_network_components.py` on `perf_full_network_10k.zx` with 100 tx, max ops 200k.

### Changes Applied

**VM (2 issues):**
1. **Name resolution fast path** (`_resolve`): reduced dict membership checks by using sentinel + cached getters.
2. **Gas metering locals**: cached `gas_metering` and `consume` references in the hot loop to reduce attribute lookups.

**Interpreter (2 issues):**
1. **Module cache by resolved path**: reuse cached module env for candidate paths and cache under both import spec and resolved path.
2. **Statement starter set reuse**: moved statement-starter tokens to module-level constant to avoid rebuilding on every block parse.

### Results (Round 3)
- **Interpreter**: 14.85s (down from 15.43s in prior run)
- **VM**: 34.21s (up from 33.70s in prior run) → **regression observed**
- **Parse time**: 37.16ms

### Notes
- The interpreter-side changes reduced parsing overhead in `_parse_block_statements` and improved module reuse.
- VM changes did not improve runtime in this pass; further VM optimizations should target lock/wait overhead and opcode dispatch costs.
- The shared event-loop experiment for `_run_coroutine_sync` was **reverted** after showing a slowdown.

---

## Round 4 (Current Work): VM + Interpreter Pair Fixes (500 tx)
**Profiler run**: `scripts/profile_full_network_components.py` on `perf_full_network_10k.zx` with 500 tx, max ops 200k.

### Changes Applied

**VM (2 issues):**
1. **Precomputed dispatch metadata**: prebuilt handler/async/gas-kind tuples for each instruction to reduce per-iteration dict lookups.
2. **Gas kwargs avoidance**: removed per-iteration `gas_kwargs` dict creation by using structured gas-kind handling.

**Interpreter (2 issues):**
1. **Block statement LRU cache**: cached parsed block statements by token signature (bounded to 256 entries).
2. **Meaningful-token check**: moved meaningful token types to a module-level constant and avoided per-call closures.

### Results (Round 4, 500 tx)
- **Interpreter**: 14.36s
- **VM**: 34.35s
- **Parse time**: 36.53ms

### Notes
- Interpreter parsing costs dropped slightly; `_parse_block_statements` is still a hotspot but now with fewer calls (590 vs 606).
- VM core loop cost is still dominated by `_run_stack_bytecode`, with `dict.get` and gas metering as persistent costs.
