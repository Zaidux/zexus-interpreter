// ─────────────────────────────────────────────────────────────────────
// Zexus Blockchain — Rust ContractVM Orchestration  (Phase 4)
// ─────────────────────────────────────────────────────────────────────
//
// Moves the contract execution orchestration layer to Rust:
//   • Reentrancy detection
//   • Call-depth enforcement
//   • Gas metering + Rust discount
//   • State snapshot / commit / rollback
//   • Receipt generation
//   • Cross-contract call scaffolding
//
// Python passes pre-serialised .zxc bytecode, a state dict, an env
// dict, and configuration.  Rust performs the full execution lifecycle
// and returns a receipt dict.  The only Python involvement after this
// is final persistence of state to Chain.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pyo3::ToPyObject;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::binary_bytecode;
use crate::rust_vm::{RustVM, RustVMExecutor, ZxValue};

// helpers to convert between Python dicts and Rust HashMaps
fn py_dict_to_hashmap(py: Python<'_>, d: &Bound<'_, PyDict>) -> HashMap<String, ZxValue> {
    let mut m = HashMap::new();
    for (k, v) in d.iter() {
        let key = k.extract::<String>().unwrap_or_default();
        let val = py_to_zx(py, &v.to_object(py));
        m.insert(key, val);
    }
    m
}

fn hashmap_to_py_dict(py: Python<'_>, m: &HashMap<String, ZxValue>) -> Py<PyDict> {
    let d = PyDict::new_bound(py);
    for (k, v) in m {
        let _ = d.set_item(k, zx_to_py(py, v));
    }
    d.into()
}

fn py_to_zx(py: Python<'_>, obj: &PyObject) -> ZxValue {
    crate::rust_vm::py_to_zx(py, obj)
}

fn zx_to_py(py: Python<'_>, val: &ZxValue) -> PyObject {
    crate::rust_vm::zx_to_py(py, val)
}

// ── Execution Receipt (Rust-side) ─────────────────────────────────────

#[derive(Debug, Clone)]
struct ContractReceipt {
    success: bool,
    result: ZxValue,
    gas_used: u64,
    gas_limit: u64,
    error: String,
    revert_reason: String,
    state_changes: HashMap<String, ZxValue>,
    instructions_executed: u64,
    output: Vec<String>,
}

impl ContractReceipt {
    fn new_success(
        result: ZxValue,
        gas_used: u64,
        gas_limit: u64,
        state_changes: HashMap<String, ZxValue>,
        instructions_executed: u64,
        output: Vec<String>,
    ) -> Self {
        ContractReceipt {
            success: true,
            result,
            gas_used,
            gas_limit,
            error: String::new(),
            revert_reason: String::new(),
            state_changes,
            instructions_executed,
            output,
        }
    }

    fn new_error(gas_limit: u64, gas_used: u64, error: &str, revert_reason: &str) -> Self {
        ContractReceipt {
            success: false,
            result: ZxValue::Null,
            gas_used,
            gas_limit,
            error: error.to_string(),
            revert_reason: revert_reason.to_string(),
            state_changes: HashMap::new(),
            instructions_executed: 0,
            output: Vec::new(),
        }
    }

    fn to_py_dict(&self, py: Python<'_>) -> PyObject {
        let d = PyDict::new_bound(py);
        let _ = d.set_item("success", self.success);
        let _ = d.set_item("result", zx_to_py(py, &self.result));
        let _ = d.set_item("gas_used", self.gas_used);
        let _ = d.set_item("gas_limit", self.gas_limit);
        let _ = d.set_item("error", &self.error);
        let _ = d.set_item("revert_reason", &self.revert_reason);

        let changes = PyDict::new_bound(py);
        for (k, v) in &self.state_changes {
            let _ = changes.set_item(k, zx_to_py(py, v));
        }
        let _ = d.set_item("state_changes", changes);
        let _ = d.set_item("instructions_executed", self.instructions_executed);

        let output_list = PyList::empty_bound(py);
        for line in &self.output {
            let _ = output_list.append(line);
        }
        let _ = d.set_item("output", output_list);

        d.to_object(py)
    }
}

// ── RustContractVM ────────────────────────────────────────────────────

/// Rust-side contract VM orchestrator.
///
/// Handles the full contract execution lifecycle in Rust:
///   1. Reentrancy detection
///   2. Call-depth enforcement (max 10)
///   3. State snapshot + rollback on failure
///   4. Rust bytecode VM execution with gas discount
///   5. Receipt generation
///   6. State diff computation
///
/// Python wrapper calls this with pre-serialized data and merges
/// results back to chain state.
#[pyclass]
pub struct RustContractVM {
    /// Gas discount factor (default 0.6 = 40% cheaper in Rust)
    gas_discount: f64,
    /// Default gas limit
    default_gas_limit: u64,
    /// Max call depth for cross-contract calls
    max_call_depth: u32,
    /// Currently executing contracts (reentrancy guard)
    executing: HashSet<String>,
    /// Current call depth
    call_depth: u32,
    /// Execution statistics
    stats: ContractVMStats,
}

#[derive(Debug, Clone, Default)]
struct ContractVMStats {
    total_executions: u64,
    successful: u64,
    failed: u64,
    total_gas_used: u64,
    total_gas_saved: u64,
    total_instructions: u64,
    reentrancy_blocked: u64,
    depth_exceeded: u64,
    out_of_gas: u64,
    fallback_to_python: u64,
}

#[pymethods]
impl RustContractVM {
    #[new]
    #[pyo3(signature = (gas_discount=0.6, default_gas_limit=10_000_000, max_call_depth=10))]
    fn new(gas_discount: f64, default_gas_limit: u64, max_call_depth: u32) -> Self {
        RustContractVM {
            gas_discount: gas_discount.clamp(0.01, 1.0),
            default_gas_limit,
            max_call_depth,
            executing: HashSet::new(),
            call_depth: 0,
            stats: ContractVMStats::default(),
        }
    }

    /// Get/set gas discount
    #[getter]
    fn gas_discount(&self) -> f64 {
        self.gas_discount
    }

    #[setter]
    fn set_gas_discount(&mut self, discount: f64) {
        self.gas_discount = discount.clamp(0.01, 1.0);
    }

    /// Execute a contract action in Rust.
    ///
    /// Args:
    ///     contract_address: str — address of the contract
    ///     action_bytecode: bytes — .zxc serialized bytecode for the action
    ///     state: dict — current contract state (snapshot from chain)
    ///     env: dict — environment variables (TX, _blockchain_state, etc.)
    ///     args: dict — action arguments
    ///     gas_limit: int — gas budget (0 = use default)
    ///     caller: str — caller address
    ///
    /// Returns:
    ///     dict with keys: success, result, gas_used, gas_limit, error,
    ///                     revert_reason, state_changes, instructions_executed,
    ///                     output, needs_fallback, gas_discount,
    ///                     gas_saved (amount saved by Rust discount)
    #[pyo3(signature = (contract_address, action_bytecode, state=None, env=None, args=None, gas_limit=0, caller=""))]
    fn execute_contract(
        &mut self,
        py: Python<'_>,
        contract_address: &str,
        action_bytecode: &Bound<'_, PyBytes>,
        state: Option<&Bound<'_, PyDict>>,
        env: Option<&Bound<'_, PyDict>>,
        args: Option<&Bound<'_, PyDict>>,
        gas_limit: u64,
        caller: &str,
    ) -> PyResult<PyObject> {
        let gas_limit = if gas_limit > 0 {
            gas_limit
        } else {
            self.default_gas_limit
        };

        self.stats.total_executions += 1;

        // ── 1. Reentrancy guard ──
        if self.executing.contains(contract_address) {
            self.stats.reentrancy_blocked += 1;
            self.stats.failed += 1;
            let receipt = ContractReceipt::new_error(
                gas_limit,
                0,
                "ReentrancyGuard",
                &format!("Reentrant call to contract {}", contract_address),
            );
            return Ok(receipt.to_py_dict(py));
        }

        // ── 2. Call-depth guard ──
        if self.call_depth >= self.max_call_depth {
            self.stats.depth_exceeded += 1;
            self.stats.failed += 1;
            let receipt = ContractReceipt::new_error(
                gas_limit,
                0,
                "CallDepthExceeded",
                &format!(
                    "Call depth {} exceeds max {}",
                    self.call_depth, self.max_call_depth
                ),
            );
            return Ok(receipt.to_py_dict(py));
        }

        // ── 3. State snapshot ──
        let state_snapshot: HashMap<String, ZxValue> = if let Some(py_state) = state {
            py_dict_to_hashmap(py, py_state)
        } else {
            HashMap::new()
        };

        // Mark as executing
        self.executing.insert(contract_address.to_string());
        self.call_depth += 1;

        // ── 4. Deserialize bytecode ──
        let raw = action_bytecode.as_bytes();
        let module = match binary_bytecode::deserialize_zxc(raw, true) {
            Ok(m) => m,
            Err(e) => {
                self.executing.remove(contract_address);
                self.call_depth -= 1;
                self.stats.fallback_to_python += 1;
                // Return needs_fallback so Python can handle it
                let d = PyDict::new_bound(py);
                let _ = d.set_item("needs_fallback", true);
                let _ = d.set_item("error", format!("Deserialization error: {}", e));
                let _ = d.set_item("success", false);
                let _ = d.set_item("gas_used", 0u64);
                let _ = d.set_item("gas_limit", gas_limit);
                return Ok(d.to_object(py));
            }
        };

        // ── 5. Create and configure VM ──
        let mut vm = RustVM::from_module(&module);

        if gas_limit > 0 {
            vm.set_gas_limit(gas_limit);
        }
        vm.set_gas_discount(self.gas_discount);

        // Set environment
        if let Some(py_env) = env {
            let rust_env = py_dict_to_hashmap(py, py_env);
            vm.set_env(rust_env);
        }

        // Set args into environment
        if let Some(py_args) = args {
            for (k, v) in py_args.iter() {
                let key = k.extract::<String>().unwrap_or_default();
                let val = py_to_zx(py, &v.to_object(py));
                vm.env_set(&key, val);
            }
        }

        // Set blockchain state
        vm.set_blockchain_state(state_snapshot.clone());

        // Add caller and timestamp to environment
        vm.env_set("_caller", ZxValue::Str(caller.to_string()));
        vm.env_set(
            "_contract_address",
            ZxValue::Str(contract_address.to_string()),
        );
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        vm.env_set("_timestamp", ZxValue::Float(ts));

        // ── 6. Execute ──
        let exec_result = vm.execute();

        // ── 7. Process result ──
        let (instr_count, gas_used, _) = vm.get_stats();
        let gas_full_price = ((gas_used as f64) / self.gas_discount).round() as u64;
        let gas_saved = gas_full_price.saturating_sub(gas_used);

        // Cleanup guards
        self.executing.remove(contract_address);
        self.call_depth -= 1;

        match exec_result {
            Ok(result_val) => {
                // Success — compute state diff
                let new_state = vm.get_blockchain_state().clone();
                let state_changes = diff_state(&state_snapshot, &new_state);

                self.stats.successful += 1;
                self.stats.total_gas_used += gas_used;
                self.stats.total_gas_saved += gas_saved;
                self.stats.total_instructions += instr_count;

                let receipt = ContractReceipt::new_success(
                    result_val,
                    gas_used,
                    gas_limit,
                    state_changes,
                    instr_count,
                    vm.get_output().to_vec(),
                );

                let d_py = receipt.to_py_dict(py);
                // Add extra fields
                let d = d_py.downcast_bound::<PyDict>(py).unwrap();
                let _ = d.set_item("needs_fallback", false);
                let _ = d.set_item("gas_discount", self.gas_discount);
                let _ = d.set_item("gas_saved", gas_saved);

                // Return new state for Python to commit
                let new_state_py = PyDict::new_bound(py);
                for (k, v) in vm.get_blockchain_state() {
                    let _ = new_state_py.set_item(k, zx_to_py(py, v));
                }
                let _ = d.set_item("new_state", new_state_py);

                // Return env for syncing back
                let new_env_py = PyDict::new_bound(py);
                for (k, v) in vm.get_env() {
                    let _ = new_env_py.set_item(k, zx_to_py(py, v));
                }
                let _ = d.set_item("env", new_env_py);

                // Phase 6: Return events emitted by Rust builtins
                let events_list = PyList::empty_bound(py);
                for (event_name, event_data) in vm.get_events() {
                    let ev = PyDict::new_bound(py);
                    let _ = ev.set_item("event", event_name.as_str());
                    let _ = ev.set_item("data", zx_to_py(py, event_data));
                    let _ = events_list.append(ev);
                }
                let _ = d.set_item("events", events_list);

                Ok(d_py)
            }
            Err(crate::rust_vm::VmError::OutOfGas {
                used,
                limit,
                opcode,
            }) => {
                // Out of gas — rollback (state_snapshot is not applied)
                self.stats.failed += 1;
                self.stats.out_of_gas += 1;
                self.stats.total_gas_used += gas_limit;

                let receipt = ContractReceipt::new_error(
                    gas_limit,
                    gas_limit,
                    "OutOfGas",
                    &format!("Out of gas: used={}, limit={}, op={}", used, limit, opcode),
                );
                let d_py = receipt.to_py_dict(py);
                let d = d_py.downcast_bound::<PyDict>(py).unwrap();
                let _ = d.set_item("needs_fallback", false);
                let _ = d.set_item("gas_discount", self.gas_discount);
                let _ = d.set_item("gas_saved", 0u64);

                Ok(d_py)
            }
            Err(crate::rust_vm::VmError::RequireFailed(msg)) => {
                self.stats.failed += 1;
                self.stats.total_gas_used += gas_used;

                let receipt = ContractReceipt::new_error(
                    gas_limit,
                    gas_used,
                    "RequireFailed",
                    &msg,
                );
                let d_py = receipt.to_py_dict(py);
                let d = d_py.downcast_bound::<PyDict>(py).unwrap();
                let _ = d.set_item("needs_fallback", false);
                let _ = d.set_item("gas_discount", self.gas_discount);
                let _ = d.set_item("gas_saved", gas_saved);

                Ok(d_py)
            }
            Err(crate::rust_vm::VmError::NeedsPythonFallback) => {
                // A Python-only opcode was hit — tell Python to handle it
                self.stats.fallback_to_python += 1;

                let d = PyDict::new_bound(py);
                let _ = d.set_item("success", false);
                let _ = d.set_item("needs_fallback", true);
                let _ = d.set_item("gas_used", gas_used);
                let _ = d.set_item("gas_limit", gas_limit);
                let _ = d.set_item("error", "NeedsPythonFallback");
                let _ = d.set_item("revert_reason", "");
                let _ = d.set_item("gas_discount", self.gas_discount);
                let _ = d.set_item("gas_saved", 0u64);

                Ok(d.to_object(py))
            }
            Err(e) => {
                self.stats.failed += 1;
                self.stats.total_gas_used += gas_used;

                let receipt = ContractReceipt::new_error(
                    gas_limit,
                    gas_used,
                    "RuntimeError",
                    &format!("{}", e),
                );
                let d_py = receipt.to_py_dict(py);
                let d = d_py.downcast_bound::<PyDict>(py).unwrap();
                let _ = d.set_item("needs_fallback", false);
                let _ = d.set_item("gas_discount", self.gas_discount);
                let _ = d.set_item("gas_saved", gas_saved);

                Ok(d_py)
            }
        }
    }

    /// Execute a batch of contract actions in sequence.
    /// Returns a list of receipt dicts.
    #[pyo3(signature = (calls))]
    fn execute_batch(
        &mut self,
        py: Python<'_>,
        calls: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        let results = PyList::empty_bound(py);

        for item in calls.iter() {
            let call_dict = item.downcast::<PyDict>()?;

            let addr = call_dict
                .get_item("contract_address")?
                .map(|v| v.extract::<String>().unwrap_or_default())
                .unwrap_or_default();

            let bytecode_obj = call_dict
                .get_item("bytecode")?
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'bytecode' in call dict")
                })?;
            let bytecode = bytecode_obj.downcast::<PyBytes>()?;

            // Extract optional dicts — downcast to PyDict directly
            let state_opt: Option<Bound<'_, PyDict>> = call_dict
                .get_item("state")?
                .and_then(|v| v.downcast::<PyDict>().ok().cloned());

            let env_opt: Option<Bound<'_, PyDict>> = call_dict
                .get_item("env")?
                .and_then(|v| v.downcast::<PyDict>().ok().cloned());

            let args_opt: Option<Bound<'_, PyDict>> = call_dict
                .get_item("args")?
                .and_then(|v| v.downcast::<PyDict>().ok().cloned());

            let gas_limit = call_dict
                .get_item("gas_limit")?
                .map(|v| v.extract::<u64>().unwrap_or(0))
                .unwrap_or(0);

            let caller = call_dict
                .get_item("caller")?
                .map(|v| v.extract::<String>().unwrap_or_default())
                .unwrap_or_default();

            let result = self.execute_contract(
                py,
                &addr,
                bytecode,
                state_opt.as_ref(),
                env_opt.as_ref(),
                args_opt.as_ref(),
                gas_limit,
                &caller,
            )?;
            results.append(result)?;
        }

        Ok(results.to_object(py))
    }

    /// Get execution statistics.
    fn get_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        let _ = d.set_item("total_executions", self.stats.total_executions);
        let _ = d.set_item("successful", self.stats.successful);
        let _ = d.set_item("failed", self.stats.failed);
        let _ = d.set_item("total_gas_used", self.stats.total_gas_used);
        let _ = d.set_item("total_gas_saved", self.stats.total_gas_saved);
        let _ = d.set_item("total_instructions", self.stats.total_instructions);
        let _ = d.set_item("reentrancy_blocked", self.stats.reentrancy_blocked);
        let _ = d.set_item("depth_exceeded", self.stats.depth_exceeded);
        let _ = d.set_item("out_of_gas", self.stats.out_of_gas);
        let _ = d.set_item("fallback_to_python", self.stats.fallback_to_python);
        let _ = d.set_item("gas_discount", self.gas_discount);
        let _ = d.set_item("default_gas_limit", self.default_gas_limit);
        let _ = d.set_item("max_call_depth", self.max_call_depth);
        let _ = d.set_item("current_call_depth", self.call_depth);
        let success_rate = if self.stats.total_executions > 0 {
            self.stats.successful as f64 / self.stats.total_executions as f64 * 100.0
        } else {
            0.0
        };
        let _ = d.set_item("success_rate", success_rate);
        let avg_gas = if self.stats.successful > 0 {
            self.stats.total_gas_used / self.stats.successful
        } else {
            0
        };
        let _ = d.set_item("avg_gas_per_execution", avg_gas);

        Ok(d.to_object(py))
    }

    /// Reset execution statistics.
    fn reset_stats(&mut self) {
        self.stats = ContractVMStats::default();
    }

    /// Check if a contract is currently executing (reentrancy check).
    fn is_executing(&self, contract_address: &str) -> bool {
        self.executing.contains(contract_address)
    }

    /// Get current call depth.
    #[getter]
    fn call_depth(&self) -> u32 {
        self.call_depth
    }

    /// Get/set max call depth.
    #[getter]
    fn max_call_depth(&self) -> u32 {
        self.max_call_depth
    }

    #[setter]
    fn set_max_call_depth(&mut self, depth: u32) {
        self.max_call_depth = depth.max(1);
    }
}

// ── State diff computation ────────────────────────────────────────────

fn diff_state(
    before: &HashMap<String, ZxValue>,
    after: &HashMap<String, ZxValue>,
) -> HashMap<String, ZxValue> {
    let mut changes = HashMap::new();
    // Check modified and new keys
    for (k, v) in after {
        match before.get(k) {
            Some(old) => {
                if !zx_values_equal(old, v) {
                    changes.insert(k.clone(), v.clone());
                }
            }
            None => {
                changes.insert(k.clone(), v.clone());
            }
        }
    }
    // Check deleted keys
    for k in before.keys() {
        if !after.contains_key(k) {
            changes.insert(k.clone(), ZxValue::Null);
        }
    }
    changes
}

fn zx_values_equal(a: &ZxValue, b: &ZxValue) -> bool {
    match (a, b) {
        (ZxValue::Null, ZxValue::Null) => true,
        (ZxValue::Bool(x), ZxValue::Bool(y)) => x == y,
        (ZxValue::Int(x), ZxValue::Int(y)) => x == y,
        (ZxValue::Float(x), ZxValue::Float(y)) => (x - y).abs() < f64::EPSILON,
        (ZxValue::Str(x), ZxValue::Str(y)) => x == y,
        (ZxValue::Int(x), ZxValue::Float(y)) => (*x as f64 - y).abs() < f64::EPSILON,
        (ZxValue::Float(x), ZxValue::Int(y)) => (x - *y as f64).abs() < f64::EPSILON,
        _ => false, // Lists/Maps/PyObj: treat as changed for safety
    }
}
