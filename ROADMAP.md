# Zexus Roadmap

**Source of truth for the unification program.** Phases are sequential;
each phase ships validated (full test suite + live smoke on both engines)
before the next begins. The overall arc: *one grammar → one engine story
→ one native layer → real integration → self-hosting ladder.*

Ground rules for every phase:

1. **Fail loudly.** No silent statement dropping, ever (Phase A's rule is
   permanent). Any parse error is fatal.
2. **Differential harness.** Every language-level change runs the same
   program on the tree-walk evaluator AND the VM; outputs must match.
3. **Zero regressions.** The pre-existing 9 parallel-VM test failures are
   the standing baseline; nothing new may join them.
4. **Docs follow code.** A feature that isn't documented in GRAMMAR.md /
   the guide and pinned by a test doesn't count as shipped.

---

## Completed phases

### Phase A — Unified grammar spec + five priority fixes ✅
*Commit `f01c3fe`.*

- `GRAMMAR.md`: one canonical form per construct, ~35 keywords, full
  legacy→canonical migration table (§9) with a staged rollout
  (warn in 1.9 → error in 2.0). Category power lives in the stdlib, not
  the grammar.
- Parse errors fatal in all four CLI gates + `zx-run` (the "critical"
  substring filter and warn-then-execute are gone).
- Mock-signature hole gated behind `ZEXUS_ALLOW_MOCK_CRYPTO=1`
  (non-PEM keys previously produced forgeable signatures).
- Evaluator double-unescaping removed (the lexer already decodes).
- Parser debug print removed; `persistent storage` is a migration error
  instead of parsed-and-discarded.
- Module registries merged (`use "crypto"` exposes the union of stdlib +
  builtin; the richer builtin module was unreachable before).
- CI: `tests.yml` runs the full suite + a grammar-conformance job.

### Phase B — VM contract execution, string methods, literals ✅
*Commit `e1f8f26`.*

- **ISSUE8 V-001 fixed**: `Counter()` returned null on the VM — the
  callable dispatcher had no SmartContract branch, and actions *named*
  `get`/`set` were swallowed by the map/dict fast path in both CALL_METHOD
  handlers. Canonical contracts now produce identical results on
  `--use-vm` and `--no-vm` (hybrid execution: VM runs the program,
  contract actions keep evaluator semantics).
- **R-031 fixed**: full canonical string-method set on the object model
  (len/trim/contains/upper/lower/slice/split/replace/join/index_of/
  starts_with/ends_with/reverse/to_int/to_hex/from_hex).
- Hex integer literals (`0xFF`) and `\xNN`/`\uNNNN` string escapes with
  hard errors on incomplete sequences.

### Phase C — Capability unification + Bytes type + VM parity ✅
*Commit `9e5971b`.*

- **Capability store unified** (the sandbox story became real). Root
  causes fixed: integration built its own CapabilityManager separate
  from the global one grants wrote to; grants landed on entity `self`
  while checks read context `default`; plugin names never expanded
  (`network` ≠ `network.tcp`); the traditional parser's grant/revoke
  constructors passed `capability=` to an AST expecting `capabilities`
  (TypeError → every grant silently dropped on that engine). Verified
  live: denied → grant → works → revoke → denied, exit 1.
- **Bytes type** (GRAMMAR §4): `b"..."` literals with raw `\xNN` escapes,
  canonical method set, `+`/`==` infix, `bytes_from_hex()`. One
  implementation on the object model shared by both engines; VM binary
  ops preserve the wrapper (was: unwrap-to-raw broke method chains).
- **VM builtin parity**: `get_builtin_module` demanded an evaluator the
  VM never passes → registry never initialized → `sha256` unreachable on
  the VM. Registry lazy-inits regardless; identical output both engines.
- Parser debug prints (`👤 Entity` / `🔑 Capabilities`) silenced.
- `examples/crypto.zx` rewritten (was the same toy functions pasted
  twice); GRAMMAR.md §6 + guide document grants and bytes.

---

## Planned phases

### Phase D — Native layer alignment: Rust-first, retire C/C++
**Goal:** exactly ONE native layer (the Rust core), with pure-Python
fallbacks, so a missing wheel never bricks the interpreter.

Current state (verified): the repo ships prebuilt C/C++ extensions —
`src/zexus/vm/cabi.so`, `fastops.so`, `native_runtime.so` (sources:
`cabi.c`, `fastops.c`/`fastops.pyx`, `native_runtime.cpp`) — consumed at
exactly three guarded sites (`vm.py:165` fastops fast-path,
`native_jit_backend.py:36/42` JIT symbol registration) — *plus* an
unbuilt Rust core (`rust_core/`, PyO3, k256/keccak/merkle/VM) that
everything else wraps in try/except with Python fallback.

Plan:
1. Build `zexus_core` from `rust_core` (maturin; wheels already exist for
   C extensions in CI, extend to Rust) and route the crypto hot paths
   (secp256k1 verify, keccak, merkle) through it.
2. Delete `cabi.c`/`cabi.h`, `fastops.c`/`fastops.pyx`,
   `native_runtime.cpp` and their prebuilt `.so`s; replace the three
   import sites with Rust-backed equivalents behind the same guards
   (fast-path off → pure Python; never fatal).
3. `native_jit_backend.py` (llvmlite JIT) evaluation: keep only if the
   differential harness + benchmarks justify it; otherwise delete.
4. `wheels.yml`: build Rust wheels in CI; pure-Python fallback verified
   by a job with the extension absent.

### Phase E — Riciplay integration (bug hunting + security testing)
**Goal:** Zexus as Riciplay's agent scripting layer.

Phase C delivered the prerequisite: capability-gated, fail-loud scripts
(network/fs only via explicit grants) are exactly what an agent needs to
run generated code safely.

Plan:
1. Riciplay CLI tool `zx_run(program, grants)` — write a `.zx` file to
   the workspace, invoke the interpreter headless, return structured
   results (JSON, not scraped stdout).
2. Grant broker: the agent's tool wrapper maps declared grants
   (network.http for a recon script, fs for a static-analysis script);
   anything undeclared is denied by the language, not by the prompt.
3. Contract-mode hunting: target scan results as Zexus data; exploit
   PoCs as `bytes` payloads; findings written back to the Riciplay
   notebook.
4. Bench harness (riciplay-bench): A/B agent-with-bash vs
   agent-with-zexus on the hardened targets.

### Phase F — Compiler/interpreter alignment
**Goal:** the two engines cannot drift again.

1. **Differential harness as CI** (promoted from a rule to a job): a
   corpus of `.zx` programs executed on both engines with
   output-equality assertions; any divergence fails the build.
2. Port the remaining V-series fixes (V-002..V-010 from ISSUE8) the same
   way V-001 was fixed — with paired regression tests on both engines.
3. Delete `evaluator_original.py` (legacy copy); single statement of
   truth per construct.
4. `fn` keyword arrives with the migration table's warn phase
   (`function`/`action` outside contracts warn, then error in 2.0).

### Phase G — Hybrid execution (compiler + interpreter as one system)
**Goal:** tiered execution decided by measurement, not architecture.

1. Benchmarks per construct: tree-walk vs VM vs (if it survives D) JIT.
2. Policy layer: hot loops → VM; cold/admin code → tree-walk; contract
   actions → evaluator semantics (already hybrid from Phase B).
3. The VM's existing pool/threading infra becomes the tier-2 target.
4. Every tier decision is a benchmark table in the repo, revisited per
   release.

### Phase H — Self-hosting ladder
**Goal:** North Star, sequenced so each rung is independently useful.

1. Stdlib modules written in Zexus (the zexus-stdlib repo, migrated to
   the canonical grammar — it does not parse today).
2. Dev tools in Zexus (zpm, zpics-style snapshot runner).
3. *Compiler* self-hosted in Zexus (the CPython/LLVM approach — the
   runtime stays Rust/Python).
4. Interpreter self-host last, if ever. Nothing before F/G is
   rock-solid ships a rung.

### Phase I — Cleanup and documentation reset ✅ → (this phase)
**Goal:** a hygienic project: no dead directories, no unused features,
no inaccurate documents. Code corrected where it imports removed
things; docs either deleted or completely rewritten to be accurate.

1. **Directory purge** (candidates, each verified unused first):
   `evaluator_original.py`, `zpm_modules/` (duplicated stdlib content),
   legacy launchers (7 entry points → ~2), `.zpics_runtime/`,
   `vscode-extension/` vs `zexus-syntax/` (one survives), dead test
   suites superseded by `tests/grammar/`.
2. **Feature purge**: keyword scaffolding that doesn't parse or execute
   its documented form (remaining `emit {}` object form, `sanitize ...
   as ...` keyword form, `restrict name` form) — remove or implement,
   never leave half-wired.
3. **Document reset**: every root-level `.md` and `docs/` file is either
   deleted or **completely rewritten** against the current code. The
   guide's "✅ verified" table regenerates from executable tests, not
   stdout-lookalikes. QUICK_START.md loses dead links and the VM-flag
   lie; README's "Rust-first 102,000 TPS" claim becomes the honest
   benchmark table from G; the 0-byte LICENSE is filled.
4. **Exit criteria**: a newcomer reads README → QUICK_START → GRAMMAR
   and every command they run works exactly as documented.

---

## Standing decisions

- Legacy syntax maps: warn in v1.9, error in v2.0 (GRAMMAR.md §9).
- The 9 pre-existing parallel-VM test failures are the regression
  baseline; fixing them belongs to Phase F (port of V-series), not
  ad-hoc patches.
- Mock crypto stays env-gated for tests; production paths raise.
- One implementation per construct, shared by both engines (Bytes'
  object-model pattern is the template).
