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

## Proof of capability — the v2.0.0 field test (2026-09-08)

Before Phase E, Zexus was tested by *building real things with it* — the
friction was the test. Two programs, both shipped in `examples/`:

- **`recon_demo.zx`** — a first-pass bug-bounty recon tool run against
  the live riciplay.xyz (authorized): DNS, TLS cert + protocol, security
  header audit, cookie audit, web fingerprint, content-verified endpoint
  probing (correctly identifies SPA-fallback 200s — no false positives),
  subdomain discovery, structured findings with severity stats. Real
  findings produced (missing headers, TXT email enumeration).
- **`api_server_demo.zx`** — a JSON API microservice in pure Zexus on
  raw sockets: HTTP request parsing, routing, query-string stripping,
  correct response headers, 404 handling. All endpoints verified live.

The test surfaced and fixed **seven real language bugs** (this is the
point of dogfooding):

1. netsec module: only 2 of 10 functions registered → 6 more wired
2. http module: cookies never surfaced → added with security flags
3. List `.len()`/`.join()` missing (GRAMMAR §4 conformance)
4. pentest module: `discover_subdomains`/`severity_stats`/`add_finding`
   unregistered → wired, with the report Map keeping its live native
   dict (round-tripping nested findings through Map pairs loses them)
5. **grant/revoke parser over-advance** — ANY statement directly after
   a grant was mis-parsed (one-token over-consumption)
6. **`function`/`action` lexer boundary** — IDENT missing from the
   keyword boundary set, so a function declaration after an
   identifier-ending statement lexed as a plain identifier
7. socket server docs/return shape (auto-start; wrapper is stop/is_running)

v2.0.0 published: PyPI (zexus 2.0.0) + npm (zexus 2.0.0, files
whitelisted to bin/postinstall/README/GRAMMAR after the first publish
attempt failed on cargo's hardlinked target/ artifacts — 51.6 MB → 13
files).

Verdict for Phase E: **capable.** The language can express recon logic,
binary-ish payload work, and backend services today; the remaining
friction (module registration completeness, docs accuracy) is exactly
what Phase I addresses.

## Phase F — Compiler/interpreter alignment ✅
*(this session)*

- **Differential harness as a CI job** (`tests/grammar/test_differential.py`
  + `tests.yml:differential`): 16 canonical constructs run on BOTH engines
  with output-equality assertions; any divergence fails the build. The
  two remaining divergences are documented xfails (error-model design
  item; anonymous closures V-004/R-029).
- **match works for the first time (R-027 — Critical)**: canonical
  `pattern => block|expr` arms with `_` wildcard. Root causes fixed:
  the LBRACE infix (Entity{...} constructor) swallowed the arm block
  while parsing the match value; the LAMBDA infix (arrow lambdas)
  swallowed the arm separator while parsing the pattern; the tree-walk
  had NO evaluator handler at all; the VM compiled `_` as a variable
  lookup so the fallback arm never matched. Block and expression bodies,
  statement and value positions — all at parity on both engines.
- **Range loops (R-024)**: `for i in 0..N` — `..` previously lexed as a
  FLOAT (`0.`) + property access. New RANGE token + RangeExpression
  (exclusive end, Python semantics, 1M-element guard) on both engines.
- **`not` keyword (R-021)**: lexer maps it to BANG but preserves the
  literal; the evaluator only checked `!`. Both spellings accepted.
- **`fn` keyword (GRAMMAR.md §3 canonical)**: both `fn` and legacy
  `function` accepted (warn phase of the migration table) on both engines.
- **Numeric parity**: `/` is true division on both engines (the
  tree-walk truncated; VM already produced floats).
- **VM parity fixes**: canonical method names (`len`) in the VM's
  string/list tables + `to_hex`/`from_hex`/`to_int`/`reverse`; a bare
  `VM()` now defaults to the shared builtin registry (identical surface
  to the CLI path — `bytes_from_hex` etc. were null standalone).
- **`evaluator_original.py` deleted** (2040 lines, zero importers).

Open (documented xfail, not blocking): error-model unification
(errors-as-values vs exceptions — Phase G design item), anonymous
closures (V-004/R-029).

### Phase I — part 2 ✅

- **Half-wired feature survey** (every keyword probed on both engines).
  Classification: remove / leave-as-legacy-error / **report-valuable**.
  Found + fixed 1 more lexer bug en route (`data` at file start).
  Report of valuable unwired features delivered with this commit.
- **ZEXUS_GUIDE.md completely regenerated** from executed snippets —
  every code block in the new guide was run before publication; a
  dedicated "What is NOT wired" section (15) lists the scaffolding by
  name so users can't mistake it for guarantees.
- **PARSING_PIPELINE.md** reviewed: architecture description matches
  current code; currency banner added.

Sequencing per owner decision: F → **I** → G → E.

## Phase I — Cleanup and documentation reset (in progress)

**Done this pass:**
- Directory purge: `zpm_modules/` (duplicated stdlib content; the zpm
  target is created on demand), `.zpics_runtime/` + `.zpics_snapshots/`
  (runtime artifacts — gitignored), `blockchain_test/` (1 MB, zero
  references), `vscode-extension/` (duplicate of `zexus-syntax/`, which
  survives as the canonical npm-published extension)
- LICENSE filled (MIT — was a 0-byte file; Cargo.toml already claimed MIT)
- README completely rewritten: honest status table, real install paths,
  the safety model, engine parity explanation — the "102,000+ TPS /
  Rust-first / zero Python fallbacks" claims and the 2,745-line v1.8.4
  feature catalog are gone
- QUICK_START completely rewritten: every command executed as written
  (doc verification caught a real bug doing this — see below)
- ZEXUS_RULES.md deleted (v1.8.3-era rules superseded by GRAMMAR.md)
- ZEXUS_GUIDE.md carries an explicit regeneration banner: the stale
  "✅ verified" tables are marked untrustworthy; the differential
  corpus is named as the source of truth
- Doc verification found and fixed TWO real bugs: the ADVANCED parser
  (CLI path) parsed match block-arms as MapLiterals (arm bodies never
  executed via `zx run`), and the evaluator lacked handlers for the
  pattern node types the advanced parser produces (LiteralPattern,
  WildcardPattern, VariablePattern). Match now works identically
  through the CLI on both engines.

**Mid-Phase I (v2 example sweep — every .zx in examples/ run under v2):**
- Launchers unified: 7 bin/ entries → 2 (`zx` + `zxpm`); the other five
  invoked Python modules that mostly never existed (runner/dev/deploy)
- 3 LANGUAGE BUGS the sweep found and fixed:
  1. `data` keyword didn't lex after `RPAREN` (any statement following a
     call) — context set gap
  2. **module-level state mutation from actions/functions silently lost
     on the tree-walk** (R-018's sibling, root-caused): closure captures
     cloned the env's dict-store, so `assign` mutated the copy. Actions
     and functions now capture the LIVE env; demo_simple_working chain
     length 0-vs-3 divergence eliminated (both engines now agree)
  3. RANGE token unsupported in the ADVANCED (strategy) parser — `let r
     = 0..3` parsed as plain IntegerLiteral(0) via `zx run`
- Example migrations to v2 canonical: `#` comments → `//` (2 files),
  token_contract (v1 `state x;` decls → `state {}` block, `limit`/`pure`
  modifiers dropped, `this.`-qualification), contract `let` fields →
  state (demo_backend_simple), `replace()` free-function → method,
  entity-field handler call extracted (R-029 workaround), ziver_chain
  fully migrated (colon bodies, `push(list,x)` → `list.push(x)`, `str`),
  three debug scripts rewritten canonically
- Result: **all 30 examples pass on both engines** (3 DB examples need
  live servers by design; 3 long-running servers verified by timeout+rc)

**Remaining (next pass):** full guide regeneration from the executable
corpus; launcher consolidation (7 bin/ entries → ~2); feature purge of
the remaining half-wired keyword forms (`emit {}` object form,
`sanitize ... as ...`, `restrict name`); PARSING_PIPELINE.md accuracy
review.

Note: one timing-flake test (memory_pool int-pool 0.75s threshold)
failed under this session's own CPU load — verified failing identically
on the stashed pre-change tree; environmental, not a regression.

## Planned phases

### Phase D — Native layer alignment: Rust-first, retire C/C++ ✅
*See commit for this change.*

- **The Rust core builds.** `rust_core/` had never compiled: an unclosed
  match-arm brace in `rust_vm.rs` (the file was committed corrupted —
  `_ =>` wildcard inside the `(Str, Int)` arm body). Fixed; `cargo build
  --release` succeeds; `zexus_core.so` loads and serves RustHasher
  (sha256/keccak), RustMerkle, RustSignature, RustVMExecutor.
- **Hash hot paths route through Rust**: `CryptoPlugin.hash_data` uses
  the Rust core for SHA-256/Keccak-256 when built, hashlib/pycryptodome
  otherwise — results cross-checked identical. This also removes the
  hard pycryptodome dependency for Keccak-256 (Rust provides it).
- **The C/C++ layer is deleted**: `cabi.c`/`cabi.h`, `fastops.c`/
  `fastops.pyx`, `native_runtime.cpp`, their prebuilt `.so`s, the
  setup.py extension build block, and the three import sites. The VM's
  execution tiers are now exactly: Rust VM (when built) → pure Python.
  `native_jit_backend`'s C symbol registration is a no-op hook (LLVM
  resolves symbols at link time); the JIT itself survives pending
  Phase G benchmarking.
- **wheels.yml** builds `zexus_core` with maturin alongside the
  pure-Python wheel; missing extensions never brick anything (all
  consumers guarded — verified by a test).
- Found+fixed en route: the VM print formatter now handles the Bytes
  *wrapper* (fell through to `str(bytes)` raw repr after fastops
  removal shifted the binary-op path).
- Validation: 11 new phase-D tests; full suite 2642 passed / 9 failed
  (the standing baseline). The 233 extra passing tests are the Rust
  bridge suite activating now that the extension is importable.

### Phase D — Native layer alignment: Rust-first, retire C/C++ (original plan)
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
