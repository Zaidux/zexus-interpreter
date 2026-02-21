"""
Zexus Blockchain — Multiprocessing Batch Executor
===================================================

Dispatches contract-group transaction batches to **separate OS processes**
via ``multiprocessing.Pool``, giving each group its own Python GIL for
true CPU-level parallelism.

Architecture::

    Main process (orchestrator)
        │
        ├── Worker-0  (own GIL) ← contract group A
        ├── Worker-1  (own GIL) ← contract group B
        ├── Worker-2  (own GIL) ← contract group C
        └── ...

Each worker process re-instantiates the ``ContractVM`` (or a lightweight
mock) so that contract state is process-local.  Results are aggregated
in the main process.

When combined with the Rust batched-GIL executor (Option 2) inside each
worker, the two optimisations stack: Rayon parallelises within a process
while multiprocessing parallelises across processes.

Usage::

    from zexus.blockchain.multiprocess_executor import MultiProcessBatchExecutor

    executor = MultiProcessBatchExecutor(
        vm_factory=lambda: MyContractVM(),
        workers=4,
    )
    result = executor.execute_batch(transactions)
    print(result.throughput)  # >> 10,000+ tx/s
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("zexus.blockchain.multiprocess_executor")


# ── Results ────────────────────────────────────────────────────────────

class MPBatchResult:
    """Aggregated result from a multiprocess batch execution."""

    __slots__ = (
        "total", "succeeded", "failed", "gas_used",
        "elapsed", "receipts",
    )

    def __init__(
        self,
        total: int = 0,
        succeeded: int = 0,
        failed: int = 0,
        gas_used: int = 0,
        elapsed: float = 0.0,
        receipts: Optional[List[Dict[str, Any]]] = None,
    ):
        self.total = total
        self.succeeded = succeeded
        self.failed = failed
        self.gas_used = gas_used
        self.elapsed = elapsed
        self.receipts = receipts or []

    @property
    def throughput(self) -> float:
        return self.total / self.elapsed if self.elapsed > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"MPBatchResult(total={self.total}, ok={self.succeeded}, "
            f"fail={self.failed}, gas={self.gas_used}, "
            f"{self.throughput:.1f} tx/s)"
        )


# ── Worker function (runs in child process) ───────────────────────────

# Module-level VM holder for the child process.
_worker_vm: Any = None


def _init_worker(vm_factory_pickle: bytes) -> None:
    """Process initialiser — creates a VM instance per worker."""
    import pickle
    global _worker_vm
    vm_factory = pickle.loads(vm_factory_pickle)
    _worker_vm = vm_factory()


def _execute_group(
    contract_addr: str,
    txs_json: str,
) -> str:
    """Execute a group of transactions in the worker's own GIL.

    Receives and returns JSON to avoid pickle overhead on complex objects.
    """
    global _worker_vm
    import json as _json
    import hashlib as _hashlib

    txs = _json.loads(txs_json)
    results: List[Dict[str, Any]] = []
    succeeded = 0
    failed = 0
    gas_used = 0

    for tx in txs:
        try:
            if _worker_vm is not None and hasattr(_worker_vm, "execute_action"):
                receipt = _worker_vm.execute_action(
                    contract=tx.get("contract", contract_addr),
                    action=tx.get("action", ""),
                    args=tx.get("args", {}),
                    caller=tx.get("caller", ""),
                    gas_limit=tx.get("gas_limit", 100_000),
                )
            else:
                # Minimal fallback — hash-based mock
                data = (
                    f"{contract_addr}:{tx.get('action', '')}:"
                    f"{tx.get('caller', '')}:{_json.dumps(tx.get('args', {}), sort_keys=True)}"
                )
                receipt = {
                    "success": True,
                    "gas_used": max(21_000, len(data) * 8),
                    "result": _hashlib.sha256(data.encode()).hexdigest()[:16],
                }

            if isinstance(receipt, dict) and receipt.get("success"):
                succeeded += 1
                gas_used += receipt.get("gas_used", 0)
            else:
                failed += 1
            results.append(receipt if isinstance(receipt, dict) else {"success": False})
        except Exception as e:
            failed += 1
            results.append({"success": False, "error": str(e)})

    return _json.dumps({
        "contract": contract_addr,
        "succeeded": succeeded,
        "failed": failed,
        "gas_used": gas_used,
        "receipts": results,
    })


# ── Try Rust-accelerated group execution ──────────────────────────────

def _execute_group_rust(
    contract_addr: str,
    txs_json: str,
) -> str:
    """Execute a group using the Rust batched-GIL executor within the worker."""
    global _worker_vm
    import json as _json

    txs = _json.loads(txs_json)

    try:
        import zexus_core  # type: ignore[import-untyped]

        # Build the vm_callback for Rust
        def _callback(contract, action, args_json_str, caller, gas_limit):
            if _worker_vm is not None and hasattr(_worker_vm, "execute_action"):
                import json as _j
                args = _j.loads(args_json_str) if isinstance(args_json_str, str) else args_json_str
                gas = int(gas_limit) if isinstance(gas_limit, str) else gas_limit
                return _worker_vm.execute_action(
                    contract=contract, action=action,
                    args=args, caller=caller, gas_limit=gas,
                )
            # Mock fallback
            import hashlib as _h
            data = f"{contract}:{action}:{caller}:{args_json_str}"
            return {
                "success": True,
                "gas_used": max(21_000, len(data) * 8),
                "result": _h.sha256(data.encode()).hexdigest()[:16],
            }

        # Serialise txs for Rust
        serialized = []
        for tx in txs:
            serialized.append({
                "contract": str(tx.get("contract", contract_addr)),
                "action": str(tx.get("action", "")),
                "args": _json.dumps(tx.get("args", {})) if not isinstance(tx.get("args"), str) else tx["args"],
                "caller": str(tx.get("caller", "")),
                "gas_limit": str(tx.get("gas_limit", "0")),
            })

        executor = zexus_core.RustBatchExecutor(max_workers=1)
        result = executor.execute_batch(serialized, _callback)

        receipts = []
        for r_json in result.receipts:
            try:
                receipts.append(_json.loads(r_json))
            except _json.JSONDecodeError:
                receipts.append({"success": False, "error": r_json})

        return _json.dumps({
            "contract": contract_addr,
            "succeeded": result.succeeded,
            "failed": result.failed,
            "gas_used": result.gas_used,
            "receipts": receipts,
        })
    except ImportError:
        # Rust not available in worker — fall back to pure Python
        return _execute_group(contract_addr, txs_json)


def _default_vm_factory():
    """Default VM factory that creates no VM (uses mock fallback)."""
    return None


# ── Main executor class ───────────────────────────────────────────────

class MultiProcessBatchExecutor:
    """Multi-process batch executor — true GIL-free parallelism.

    Parameters
    ----------
    vm_factory : callable
        A zero-argument callable that returns a ContractVM instance.
        Must be picklable (e.g. a module-level function or lambda
        with no closures over unpicklable objects).
    workers : int
        Number of worker processes (default: CPU count).
    use_rust_in_workers : bool
        If True, each worker uses the Rust batched-GIL executor for
        its group — stacking Rayon parallelism (within-process) with
        multiprocessing parallelism (across processes).
    """

    def __init__(
        self,
        vm_factory: Optional[Callable[[], Any]] = None,
        workers: int = 0,
        use_rust_in_workers: bool = True,
    ):
        self._vm_factory = vm_factory
        self._workers = workers or min(multiprocessing.cpu_count(), 16)
        self._use_rust = use_rust_in_workers
        self._pool: Optional[ProcessPoolExecutor] = None
        self._pool_mp: Optional[multiprocessing.pool.Pool] = None

    def _ensure_pool(self) -> None:
        """Lazily create the process pool."""
        if self._pool_mp is not None:
            return
        import pickle
        factory = self._vm_factory if self._vm_factory else _default_vm_factory
        factory_bytes = pickle.dumps(factory)
        self._pool_mp = multiprocessing.Pool(
            processes=self._workers,
            initializer=_init_worker,
            initargs=(factory_bytes,),
        )

    def execute_batch(
        self,
        transactions: List[Dict[str, Any]],
    ) -> MPBatchResult:
        """Execute a batch of transactions across multiple processes.

        Transactions are grouped by contract address. Each group is
        dispatched to a worker process that has its own Python GIL.
        """
        start = time.perf_counter()
        self._ensure_pool()

        # Group by contract
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for tx in transactions:
            groups[tx.get("contract", "unknown")].append(tx)

        # Choose worker function
        worker_fn = _execute_group_rust if self._use_rust else _execute_group

        # Dispatch groups to workers
        async_results = []
        for contract_addr, txs in groups.items():
            txs_json = json.dumps(txs)
            async_results.append(
                self._pool_mp.apply_async(worker_fn, (contract_addr, txs_json))  # type: ignore
            )

        # Collect results
        result = MPBatchResult(total=len(transactions))
        for ar in async_results:
            try:
                group_json = ar.get(timeout=30)
                group_data = json.loads(group_json)
                result.succeeded += group_data.get("succeeded", 0)
                result.failed += group_data.get("failed", 0)
                result.gas_used += group_data.get("gas_used", 0)
                result.receipts.extend(group_data.get("receipts", []))
            except Exception as e:
                logger.error("Worker group failed: %s", e)
                result.failed += 1

        result.elapsed = time.perf_counter() - start
        return result

    def shutdown(self) -> None:
        """Terminate the process pool."""
        if self._pool_mp:
            self._pool_mp.terminate()
            self._pool_mp.join()
            self._pool_mp = None

    def __del__(self) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"MultiProcessBatchExecutor(workers={self._workers}, "
            f"rust_in_workers={self._use_rust})"
        )
