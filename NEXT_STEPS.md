# NEXT_STEPS

## What was done
- Added profiling controls and report wiring via stdlib perf: `profile_report`, plus VM config options for sampling and overhead control.
- Improved profiler internals (sampling, bounded timing samples, fixed instruction timing attribution).
- Fixed VM truthiness handling for `JUMP_IF_FALSE` by unwrapping Zexus values and treating `Null` as `None`.
- Preserved contract-like objects (`call_method`/`get_attr`) during VM env sync.
- Added loop diagnostics (compile errors, metadata, opcode previews) and optional bytecode dumps to `/tmp`.
- Added `BreakStatement` compilation to VM bytecode.
- Routed Zexus `Action`/`LambdaFunction` execution through an evaluator inside VM calls so method calls execute correctly.
- Fixed bytecode optimizer dead-code elimination to respect numeric jump targets.
- Added limited CALL_METHOD tracing to confirm VM method execution.

## Current behavior
- VM now executes contract actions from loops; validator registration and transaction submission work when VM is enabled.
- With bytecode optimizer enabled, the benchmark can still hang/time out; with it disabled, runs complete and profiling works.

## What is left
1. **Isolate and fix the remaining bytecode optimizer hang.**
   - Instrument `BytecodeOptimizer` passes to log before/after instruction counts and detect pass that introduces infinite loop or broken control flow.
   - Add guard rails to any unsafe pass (likely jump/threading or instruction combining) when working with loops.
2. **Re-enable optimizer safely.**
   - Once the failing pass is fixed, turn `enable_bytecode_optimizer` back on in the benchmark config.
3. **Optional: keep Action/Lambda execution in VM**
   - If needed, add caching of the evaluator used by VM to avoid repeated instantiation costs.

## How to disable verbose debug logs
- **Environment flags**: stop using these environment variables when running benchmarks:
  - `ZEXUS_VM_PROFILE_VERBOSE=1`
  - `ZEXUS_VM_PROFILE_OPS=1`
- **Benchmark config**: set in `blockchain_test/perf_full_network_10k.zx`:
  - `enable_profiling: false`
  - `profiling_level: "BASIC"` (or leave default when disabled)
  - `vm_dump_bytecode: false`

If you want, I can add a toggle in the benchmark to switch between “profile” and “normal” modes without editing the file.

use ./zx-run to run zexus files
use timeouts to prevent infinite loops(max 60s)
the test file is pref_full_network_10k.zx and the blockchain file is full_network_blockchain.zx
the vm is located in src/zexus/vm
