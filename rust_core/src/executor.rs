// ─────────────────────────────────────────────────────────────────────
// Batch Transaction Executor — Rayon-parallel
// ─────────────────────────────────────────────────────────────────────
//
// Executes a block's worth of transactions, grouping by target contract
// so that non-overlapping groups run in parallel on all CPU cores.
//
// Three execution modes:
//
//   **GIL-free native** (Phase 5, `execute_batch_native`):
//       Pure-Rust execution — transactions carry pre-compiled .zxc
//       bytecode.  Each is executed by a fresh RustVM instance on a
//       Rayon thread.  Zero GIL acquisitions during execution.
//       Near-linear CPU scaling.
//
//   **Batched-GIL** (Phase 0 legacy, `execute_batch`):
//       Acquires the Python GIL *once per contract group* instead of
//       once per transaction.  Kept for contracts that need Python
//       fallback (use CALL_NAME, etc.).
//
//   **Per-tx GIL** (legacy, `execute_batch_pertx`):
//       Acquires the GIL per transaction.  Kept for diagnostic use.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pyo3::ToPyObject;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::binary_bytecode;
use crate::rust_vm::{RustVM, ZxValue, VmError};

// ── Result types ──────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TxReceipt {
    pub success: bool,
    pub gas_used: u64,
    pub error: String,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct TxBatchResult {
    #[pyo3(get)]
    pub total: usize,
    #[pyo3(get)]
    pub succeeded: usize,
    #[pyo3(get)]
    pub failed: usize,
    #[pyo3(get)]
    pub gas_used: u64,
    #[pyo3(get)]
    pub elapsed_secs: f64,
    #[pyo3(get)]
    pub receipts: Vec<String>, // JSON-encoded receipts
}

#[pymethods]
impl TxBatchResult {
    #[getter]
    fn throughput(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.total as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("total".into(), self.total.to_string());
        m.insert("succeeded".into(), self.succeeded.to_string());
        m.insert("failed".into(), self.failed.to_string());
        m.insert("gas_used".into(), self.gas_used.to_string());
        m.insert("elapsed".into(), format!("{:.4}", self.elapsed_secs));
        m.insert("throughput".into(), format!("{:.2}", self.throughput()));
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "TxBatchResult(total={}, ok={}, fail={}, gas={}, {:.1} tx/s)",
            self.total, self.succeeded, self.failed, self.gas_used, self.throughput()
        )
    }
}

// ── Transaction representation ────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TxInput {
    contract: String,
    action: String,
    args: serde_json::Value,
    caller: String,
    gas_limit: Option<u64>,
}

// ── Native transaction (GIL-free, Phase 5) ────────────────────────────

/// A transaction prepared for pure-Rust execution.
/// Contains pre-compiled .zxc bytecode so no Python interaction is needed.
#[derive(Clone, Debug)]
struct NativeTxInput {
    contract_address: String,
    caller: String,
    bytecode: Vec<u8>,
    state: HashMap<String, ZxValue>,
    gas_limit: u64,
    gas_discount: f64,
    /// Position in the original transaction list (for ordering results)
    index: usize,
}

/// Result of a single native transaction execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct NativeTxReceipt {
    success: bool,
    gas_used: u64,
    gas_saved: u64,
    instructions: u64,
    error: String,
    #[serde(skip)]
    state_changes: HashMap<String, ZxValue>,
    #[serde(skip)]
    new_state: HashMap<String, ZxValue>,
    /// Events emitted during execution (Phase 6).
    #[serde(skip)]
    events: Vec<(String, ZxValue)>,
    needs_fallback: bool,
    index: usize,
}

/// Execute a single contract in pure Rust — no GIL, no Python.
fn execute_native_tx(input: &NativeTxInput) -> NativeTxReceipt {
    // Deserialize bytecode
    let module = match binary_bytecode::deserialize_zxc(&input.bytecode, true) {
        Ok(m) => m,
        Err(e) => {
            return NativeTxReceipt {
                success: false,
                gas_used: 0,
                gas_saved: 0,
                instructions: 0,
                error: format!("Deserialization: {}", e),
                state_changes: HashMap::new(),
                new_state: HashMap::new(),
                events: Vec::new(),
                needs_fallback: true,
                index: input.index,
            };
        }
    };

    // Create VM
    let mut vm = RustVM::from_module(&module);
    vm.set_gas_limit(input.gas_limit);
    vm.set_gas_discount(input.gas_discount);
    vm.set_blockchain_state(input.state.clone());
    vm.env_set("_caller", ZxValue::Str(input.caller.clone()));
    vm.env_set(
        "_contract_address",
        ZxValue::Str(input.contract_address.clone()),
    );

    // Execute
    let result = vm.execute();
    let (instr_count, gas_used, _) = vm.get_stats();
    let gas_full = ((gas_used as f64) / input.gas_discount).round() as u64;
    let gas_saved = gas_full.saturating_sub(gas_used);
    let events = vm.get_events().to_vec();

    match result {
        Ok(_) => {
            let new_state = vm.get_blockchain_state().clone();
            // Compute state diff
            let mut changes = HashMap::new();
            for (k, v) in &new_state {
                let changed = match input.state.get(k) {
                    Some(old) => !zx_eq(old, v),
                    None => true,
                };
                if changed {
                    changes.insert(k.clone(), v.clone());
                }
            }
            for k in input.state.keys() {
                if !new_state.contains_key(k) {
                    changes.insert(k.clone(), ZxValue::Null);
                }
            }

            NativeTxReceipt {
                success: true,
                gas_used,
                gas_saved,
                instructions: instr_count,
                error: String::new(),
                state_changes: changes,
                new_state,
                events,
                needs_fallback: false,
                index: input.index,
            }
        }
        Err(VmError::NeedsPythonFallback) => NativeTxReceipt {
            success: false,
            gas_used,
            gas_saved: 0,
            instructions: instr_count,
            error: "NeedsPythonFallback".into(),
            state_changes: HashMap::new(),
            new_state: HashMap::new(),
            events,
            needs_fallback: true,
            index: input.index,
        },
        Err(VmError::OutOfGas { used, limit, opcode }) => NativeTxReceipt {
            success: false,
            gas_used: limit,
            gas_saved: 0,
            instructions: instr_count,
            error: format!("OutOfGas: used={}, limit={}, op={}", used, limit, opcode),
            state_changes: HashMap::new(),
            new_state: HashMap::new(),
            events,
            needs_fallback: false,
            index: input.index,
        },
        Err(e) => NativeTxReceipt {
            success: false,
            gas_used,
            gas_saved: 0,
            instructions: instr_count,
            error: format!("{}", e),
            state_changes: HashMap::new(),
            new_state: HashMap::new(),
            events,
            needs_fallback: false,
            index: input.index,
        },
    }
}

/// Quick ZxValue equality check for state diffs (GIL-free).
fn zx_eq(a: &ZxValue, b: &ZxValue) -> bool {
    match (a, b) {
        (ZxValue::Null, ZxValue::Null) => true,
        (ZxValue::Bool(x), ZxValue::Bool(y)) => x == y,
        (ZxValue::Int(x), ZxValue::Int(y)) => x == y,
        (ZxValue::Float(x), ZxValue::Float(y)) => (x - y).abs() < f64::EPSILON,
        (ZxValue::Str(x), ZxValue::Str(y)) => x == y,
        (ZxValue::Int(x), ZxValue::Float(y)) => (*x as f64 - y).abs() < f64::EPSILON,
        (ZxValue::Float(x), ZxValue::Int(y)) => (x - *y as f64).abs() < f64::EPSILON,
        _ => false,
    }
}

// ── Batch executor ────────────────────────────────────────────────────

#[pyclass]
pub struct RustBatchExecutor {
    max_workers: usize,
    gas_discount: f64,
    default_gas_limit: u64,
    // Accumulated stats across native batches
    native_total: u64,
    native_succeeded: u64,
    native_failed: u64,
    native_fallbacks: u64,
    native_total_gas: u64,
    native_total_saved: u64,
}

#[pymethods]
impl RustBatchExecutor {
    #[new]
    #[pyo3(signature = (max_workers = 0, gas_discount = 0.6, default_gas_limit = 10_000_000))]
    fn new(max_workers: usize, gas_discount: f64, default_gas_limit: u64) -> Self {
        let workers = if max_workers == 0 {
            rayon::current_num_threads()
        } else {
            max_workers
        };
        // Configure Rayon's global thread pool
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global();
        RustBatchExecutor {
            max_workers: workers,
            gas_discount: gas_discount.clamp(0.01, 1.0),
            default_gas_limit,
            native_total: 0,
            native_succeeded: 0,
            native_failed: 0,
            native_fallbacks: 0,
            native_total_gas: 0,
            native_total_saved: 0,
        }
    }

    /// Execute a batch of transactions — **batched-GIL** mode.
    ///
    /// `transactions` — list of dicts with keys: contract, action, args, caller, gas_limit
    /// `vm_callback`  — a Python callable `fn(contract, action, args_json, caller, gas_limit) -> dict`
    ///                   that executes one transaction via the Zexus ContractVM and returns
    ///                   `{"success": bool, "gas_used": int, "error": str}`.
    ///
    /// Non-conflicting contract groups are dispatched in parallel via Rayon.
    /// The GIL is acquired **once per group** — all txs in a group execute
    /// inside a single GIL hold, eliminating per-tx GIL contention.
    fn execute_batch(
        &self,
        py: Python<'_>,
        transactions: Vec<HashMap<String, String>>,
        vm_callback: PyObject,
    ) -> PyResult<TxBatchResult> {
        let start = std::time::Instant::now();
        let total = transactions.len();

        // Parse into TxInput structs
        let txs: Vec<TxInput> = transactions
            .iter()
            .map(|m| TxInput {
                contract: m.get("contract").cloned().unwrap_or_default(),
                action: m.get("action").cloned().unwrap_or_default(),
                args: m
                    .get("args")
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                caller: m.get("caller").cloned().unwrap_or_default(),
                gas_limit: m.get("gas_limit").and_then(|s| s.parse().ok()),
            })
            .collect();

        // Group by contract
        let mut groups: HashMap<String, Vec<TxInput>> = HashMap::new();
        for tx in txs {
            groups
                .entry(tx.contract.clone())
                .or_default()
                .push(tx);
        }

        // Execute — parallel across contract groups.
        // BATCHED-GIL: acquire the GIL once per group, execute all txs
        // in that group inside the single hold, then release.
        let groups_vec: Vec<(String, Vec<TxInput>)> = groups.into_iter().collect();

        let all_receipts: Vec<TxReceipt> = py.allow_threads(|| {
            groups_vec
                .par_iter()
                .flat_map(|(contract_addr, txs)| {
                    // ONE GIL acquisition for the entire group
                    Python::with_gil(|py| {
                        let callback = vm_callback.bind(py);
                        let mut group_receipts = Vec::with_capacity(txs.len());

                        for tx in txs {
                            let args_json =
                                serde_json::to_string(&tx.args).unwrap_or_default();
                            let gas = tx.gas_limit.unwrap_or(0);

                            let call_result = callback.call1((
                                contract_addr.as_str(),
                                tx.action.as_str(),
                                args_json.as_str(),
                                tx.caller.as_str(),
                                gas,
                            ));

                            let receipt = match call_result {
                                Ok(result) => {
                                    let success: bool = result
                                        .get_item("success")
                                        .and_then(|v| v.extract())
                                        .unwrap_or(false);
                                    let gas_used: u64 = result
                                        .get_item("gas_used")
                                        .and_then(|v| v.extract())
                                        .unwrap_or(0);
                                    let error: String = result
                                        .get_item("error")
                                        .and_then(|v| v.extract())
                                        .unwrap_or_default();
                                    TxReceipt { success, gas_used, error }
                                }
                                Err(e) => TxReceipt {
                                    success: false,
                                    gas_used: 0,
                                    error: format!("Rust->Python callback error: {}", e),
                                },
                            };
                            group_receipts.push(receipt);
                        }
                        group_receipts
                    })
                })
                .collect()
        });

        // Aggregate
        let mut succeeded = 0usize;
        let mut failed = 0usize;
        let mut gas_total = 0u64;
        let mut receipt_jsons = Vec::with_capacity(all_receipts.len());

        for r in &all_receipts {
            if r.success {
                succeeded += 1;
            } else {
                failed += 1;
            }
            gas_total += r.gas_used;
            receipt_jsons.push(serde_json::to_string(r).unwrap_or_default());
        }

        Ok(TxBatchResult {
            total,
            succeeded,
            failed,
            gas_used: gas_total,
            elapsed_secs: start.elapsed().as_secs_f64(),
            receipts: receipt_jsons,
        })
    }

    /// Legacy per-tx GIL mode — acquires GIL for every single transaction.
    /// Kept for diagnostic/comparison benchmarks.
    fn execute_batch_pertx(
        &self,
        py: Python<'_>,
        transactions: Vec<HashMap<String, String>>,
        vm_callback: PyObject,
    ) -> PyResult<TxBatchResult> {
        let start = std::time::Instant::now();
        let total = transactions.len();

        let txs: Vec<TxInput> = transactions
            .iter()
            .map(|m| TxInput {
                contract: m.get("contract").cloned().unwrap_or_default(),
                action: m.get("action").cloned().unwrap_or_default(),
                args: m
                    .get("args")
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                caller: m.get("caller").cloned().unwrap_or_default(),
                gas_limit: m.get("gas_limit").and_then(|s| s.parse().ok()),
            })
            .collect();

        let mut groups: HashMap<String, Vec<TxInput>> = HashMap::new();
        for tx in txs {
            groups.entry(tx.contract.clone()).or_default().push(tx);
        }

        let groups_vec: Vec<(String, Vec<TxInput>)> = groups.into_iter().collect();

        let all_receipts: Vec<TxReceipt> = py.allow_threads(|| {
            groups_vec
                .par_iter()
                .flat_map(|(contract_addr, txs)| {
                    let mut group_receipts = Vec::with_capacity(txs.len());
                    for tx in txs {
                        // Per-tx GIL acquisition (legacy)
                        let receipt = Python::with_gil(|py| {
                            let args_json =
                                serde_json::to_string(&tx.args).unwrap_or_default();
                            let gas = tx.gas_limit.unwrap_or(0);
                            let callback = vm_callback.bind(py);
                            let call_result = callback.call1((
                                contract_addr.as_str(),
                                tx.action.as_str(),
                                args_json.as_str(),
                                tx.caller.as_str(),
                                gas,
                            ));
                            match call_result {
                                Ok(result) => {
                                    let success: bool = result
                                        .get_item("success")
                                        .and_then(|v| v.extract())
                                        .unwrap_or(false);
                                    let gas_used: u64 = result
                                        .get_item("gas_used")
                                        .and_then(|v| v.extract())
                                        .unwrap_or(0);
                                    let error: String = result
                                        .get_item("error")
                                        .and_then(|v| v.extract())
                                        .unwrap_or_default();
                                    TxReceipt { success, gas_used, error }
                                }
                                Err(e) => TxReceipt {
                                    success: false,
                                    gas_used: 0,
                                    error: format!("Rust->Python callback error: {}", e),
                                },
                            }
                        });
                        group_receipts.push(receipt);
                    }
                    group_receipts
                })
                .collect()
        });

        let mut succeeded = 0usize;
        let mut failed = 0usize;
        let mut gas_total = 0u64;
        let mut receipt_jsons = Vec::with_capacity(all_receipts.len());
        for r in &all_receipts {
            if r.success { succeeded += 1; } else { failed += 1; }
            gas_total += r.gas_used;
            receipt_jsons.push(serde_json::to_string(r).unwrap_or_default());
        }

        Ok(TxBatchResult {
            total, succeeded, failed,
            gas_used: gas_total,
            elapsed_secs: start.elapsed().as_secs_f64(),
            receipts: receipt_jsons,
        })
    }

    /// Pure-Rust parallel hash batch — no Python callback needed.
    /// Useful for computing tx hashes or block hashes in bulk.
    fn hash_batch(&self, py: Python<'_>, data: Vec<Vec<u8>>) -> Vec<String> {
        use sha2::{Digest, Sha256};
        py.allow_threads(|| {
            data.par_iter()
                .map(|d| {
                    let hash = Sha256::digest(d);
                    hex::encode(hash)
                })
                .collect()
        })
    }

    // ── Phase 5: GIL-free native batch execution ──────────────────

    /// Execute a batch of pre-compiled transactions entirely in Rust.
    ///
    /// **Zero GIL acquisitions during execution.**
    ///
    /// Each transaction dict must contain:
    ///   - `bytecode`: bytes — .zxc serialized bytecode
    ///   - `contract_address`: str
    ///   - `caller`: str
    ///   - `gas_limit`: int (optional, defaults to executor's default)
    ///
    /// Optional per-transaction overrides:
    ///   - `state`: dict — contract state snapshot
    ///   - `gas_discount`: float — per-tx discount override
    ///
    /// Non-conflicting contract groups run in parallel via Rayon.
    /// Returns a `TxBatchResult` with JSON receipts plus a `native_stats` dict.
    fn execute_batch_native(
        &mut self,
        py: Python<'_>,
        transactions: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        let start = std::time::Instant::now();
        let total = transactions.len();

        // ── Parse into NativeTxInputs (requires GIL — one-time) ──
        let mut inputs: Vec<NativeTxInput> = Vec::with_capacity(total);
        for (i, item) in transactions.iter().enumerate() {
            let d: &Bound<'_, PyDict> = item.downcast::<PyDict>()?;

            let bc_obj = d.get_item("bytecode")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                    format!("Transaction {} missing 'bytecode'", i),
                ))?;
            let bc_bytes: &Bound<'_, PyBytes> = bc_obj.downcast::<PyBytes>()?;
            let bytecode = bc_bytes.as_bytes().to_vec();

            let contract_address: String = d.get_item("contract_address")?
                .map(|v| v.extract::<String>().unwrap_or_default())
                .unwrap_or_default();

            let caller: String = d.get_item("caller")?
                .map(|v| v.extract::<String>().unwrap_or_default())
                .unwrap_or_default();

            let gas_limit: u64 = d.get_item("gas_limit")?
                .map(|v| v.extract::<u64>().unwrap_or(self.default_gas_limit))
                .unwrap_or(self.default_gas_limit);

            let gas_discount: f64 = d.get_item("gas_discount")?
                .map(|v| v.extract::<f64>().unwrap_or(self.gas_discount))
                .unwrap_or(self.gas_discount);

            // Per-transaction state (optional)
            let state: HashMap<String, ZxValue> = match d.get_item("state")? {
                Some(v) => {
                    if let Ok(sd) = v.downcast::<PyDict>() {
                        let mut m: HashMap<String, ZxValue> = HashMap::new();
                        for (k, val) in sd.iter() {
                            let key: String = k.extract::<String>().unwrap_or_default();
                            let obj: PyObject = val.to_object(py);
                            let zv = crate::rust_vm::py_to_zx(py, &obj);
                            m.insert(key, zv);
                        }
                        m
                    } else {
                        HashMap::new()
                    }
                }
                None => HashMap::new(),
            };

            inputs.push(NativeTxInput {
                contract_address,
                caller,
                bytecode,
                state,
                gas_limit,
                gas_discount,
                index: i,
            });
        }

        // ── Group by contract address for conflict-free parallelism ──
        let mut groups: HashMap<String, Vec<NativeTxInput>> = HashMap::new();
        for tx in inputs {
            groups.entry(tx.contract_address.clone()).or_default().push(tx);
        }
        let groups_vec: Vec<(String, Vec<NativeTxInput>)> = groups.into_iter().collect();

        // ── Execute in parallel — ZERO GIL ──
        let gas_counter = AtomicU64::new(0);
        let saved_counter = AtomicU64::new(0);

        let mut all_receipts: Vec<NativeTxReceipt> = py.allow_threads(|| {
            let results: Vec<NativeTxReceipt> = groups_vec
                .par_iter()
                .flat_map(|(_addr, txs)| {
                    // Within a contract group, execute sequentially
                    // (state ordering matters for same-contract txs)
                    let mut group_results = Vec::with_capacity(txs.len());
                    let mut running_state: Option<HashMap<String, ZxValue>> = None;

                    for tx in txs {
                        // Chain state: if a prior tx in this group modified state,
                        // use the updated state for subsequent txs
                        let mut input = tx.clone();
                        if let Some(ref s) = running_state {
                            // Merge running state into tx state
                            for (k, v) in s {
                                input.state.insert(k.clone(), v.clone());
                            }
                        }

                        let receipt = execute_native_tx(&input);

                        if receipt.success {
                            // Update running state for next tx in group
                            let mut new_state = input.state.clone();
                            for (k, v) in &receipt.state_changes {
                                new_state.insert(k.clone(), v.clone());
                            }
                            running_state = Some(new_state);
                        }

                        gas_counter.fetch_add(receipt.gas_used, Ordering::Relaxed);
                        saved_counter.fetch_add(receipt.gas_saved, Ordering::Relaxed);
                        group_results.push(receipt);
                    }
                    group_results
                })
                .collect();
            results
        });

        // ── Sort by original index to preserve ordering ──
        all_receipts.sort_by_key(|r| r.index);

        // ── Aggregate results ──
        let elapsed = start.elapsed().as_secs_f64();
        let mut succeeded = 0usize;
        let mut failed = 0usize;
        let mut fallbacks = 0usize;
        let gas_total = gas_counter.load(Ordering::Relaxed);
        let gas_saved_total = saved_counter.load(Ordering::Relaxed);
        let mut receipt_jsons = Vec::with_capacity(all_receipts.len());

        for r in &all_receipts {
            if r.success {
                succeeded += 1;
            } else {
                failed += 1;
                if r.needs_fallback {
                    fallbacks += 1;
                }
            }
            // JSON receipt (lightweight, for compatibility with TxBatchResult)
            let json_str = serde_json::to_string(&serde_json::json!({
                "success": r.success,
                "gas_used": r.gas_used,
                "gas_saved": r.gas_saved,
                "instructions": r.instructions,
                "error": r.error,
                "needs_fallback": r.needs_fallback,
            }))
            .unwrap_or_default();
            receipt_jsons.push(json_str);
        }

        // Update cumulative stats
        self.native_total += total as u64;
        self.native_succeeded += succeeded as u64;
        self.native_failed += failed as u64;
        self.native_fallbacks += fallbacks as u64;
        self.native_total_gas += gas_total;
        self.native_total_saved += gas_saved_total;

        // Build result dict with extra native stats
        let result = TxBatchResult {
            total,
            succeeded,
            failed,
            gas_used: gas_total,
            elapsed_secs: elapsed,
            receipts: receipt_jsons,
        };

        let d = PyDict::new_bound(py);
        let _ = d.set_item("total", result.total);
        let _ = d.set_item("succeeded", result.succeeded);
        let _ = d.set_item("failed", result.failed);
        let _ = d.set_item("gas_used", result.gas_used);
        let _ = d.set_item("elapsed_secs", result.elapsed_secs);
        let _ = d.set_item("throughput", result.throughput());
        let _ = d.set_item("gas_saved", gas_saved_total);
        let _ = d.set_item("fallbacks", fallbacks);
        let _ = d.set_item("gil_acquisitions", 0u64);
        let _ = d.set_item("mode", "native_gil_free");
        let _ = d.set_item("workers", self.max_workers);
        let _ = d.set_item("gas_discount", self.gas_discount);

        // Receipts as Python list
        let receipt_list = PyList::empty_bound(py);
        for r_json in &result.receipts {
            let _ = receipt_list.append(r_json);
        }
        let _ = d.set_item("receipts", receipt_list);

        // State changes per contract (for chain commit)
        let changes_dict = PyDict::new_bound(py);
        for r in &all_receipts {
            if r.success && !r.state_changes.is_empty() {
                // Find contract address from the receipt's index
                let addr = &inputs_addrs(total, &groups_vec, r.index);
                let sc = PyDict::new_bound(py);
                for (k, v) in &r.state_changes {
                    let _ = sc.set_item(k, crate::rust_vm::zx_to_py(py, v));
                }
                let _ = changes_dict.set_item(addr.as_str(), sc);
            }
        }
        let _ = d.set_item("state_changes", changes_dict);

        // Events collected across all receipts (Phase 6)
        let events_list = PyList::empty_bound(py);
        for r in &all_receipts {
            for (event_name, event_data) in &r.events {
                let ev = PyDict::new_bound(py);
                let _ = ev.set_item("event", event_name.as_str());
                let _ = ev.set_item("data", crate::rust_vm::zx_to_py(py, event_data));
                let _ = ev.set_item("index", r.index);
                let _ = events_list.append(ev);
            }
        }
        let _ = d.set_item("events", events_list);

        Ok(d.to_object(py))
    }

    /// Get/set gas discount for native batch execution.
    #[getter]
    fn gas_discount(&self) -> f64 {
        self.gas_discount
    }

    #[setter]
    fn set_gas_discount(&mut self, discount: f64) {
        self.gas_discount = discount.clamp(0.01, 1.0);
    }

    /// Get cumulative native execution stats.
    fn get_native_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        let _ = d.set_item("total", self.native_total);
        let _ = d.set_item("succeeded", self.native_succeeded);
        let _ = d.set_item("failed", self.native_failed);
        let _ = d.set_item("fallbacks", self.native_fallbacks);
        let _ = d.set_item("total_gas", self.native_total_gas);
        let _ = d.set_item("total_gas_saved", self.native_total_saved);
        let tps = if self.native_total > 0 {
            self.native_total as f64  // Not meaningful here, tracked per-batch
        } else {
            0.0
        };
        let _ = d.set_item("gil_acquisitions", 0u64);
        let _ = d.set_item("mode", "native_gil_free");
        Ok(d.to_object(py))
    }

    /// Reset native stats.
    fn reset_native_stats(&mut self) {
        self.native_total = 0;
        self.native_succeeded = 0;
        self.native_failed = 0;
        self.native_fallbacks = 0;
        self.native_total_gas = 0;
        self.native_total_saved = 0;
    }

    fn __repr__(&self) -> String {
        format!(
            "RustBatchExecutor(workers={}, discount={:.2}, native_txs={})",
            self.max_workers, self.gas_discount, self.native_total,
        )
    }
}

// ── Helper: recover contract address from flattened groups by index ──

fn inputs_addrs(
    _total: usize,
    groups: &[(String, Vec<NativeTxInput>)],
    index: usize,
) -> String {
    for (addr, txs) in groups {
        for tx in txs {
            if tx.index == index {
                return addr.clone();
            }
        }
    }
    String::new()
}
