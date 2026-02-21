// ─────────────────────────────────────────────────────────────────────
// Batch Transaction Executor — Rayon-parallel
// ─────────────────────────────────────────────────────────────────────
//
// Executes a block's worth of transactions, grouping by target contract
// so that non-overlapping groups run in parallel on all CPU cores.
//
// Two execution modes:
//
//   **Batched-GIL** (default, `execute_batch`):
//       Acquires the Python GIL *once per contract group* instead of
//       once per transaction.  Inside a single GIL hold, all
//       transactions for that contract execute sequentially via the
//       Python callback.  Rayon dispatches groups in parallel.
//       This reduces GIL contention from O(n) to O(groups).
//
//   **Per-tx GIL** (legacy, `execute_batch_pertx`):
//       Acquires the GIL per transaction.  Kept for diagnostic use.
//
// The actual *contract logic* still runs in Python (the Zexus VM) —
// what Rust does is orchestration, grouping, and parallel dispatch.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

// ── Batch executor ────────────────────────────────────────────────────

#[pyclass]
pub struct RustBatchExecutor {
    max_workers: usize,
}

#[pymethods]
impl RustBatchExecutor {
    #[new]
    #[pyo3(signature = (max_workers = 0))]
    fn new(max_workers: usize) -> Self {
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

    fn __repr__(&self) -> String {
        format!("RustBatchExecutor(workers={})", self.max_workers)
    }
}
