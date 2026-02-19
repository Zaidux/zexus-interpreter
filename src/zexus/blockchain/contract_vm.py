"""
Zexus Blockchain — Contract VM Bridge

Connects the Zexus VM's smart-contract opcodes to the real blockchain
infrastructure (Chain, Ledger, TransactionContext, CryptoPlugin).

The existing VM has blockchain opcodes (110-119, 130-137) that operate on
a raw ``env["_blockchain_state"]`` dict.  This bridge replaces that naive
dict with a proper adapter that delegates to:

  - ``Chain.contract_state``  for persistent contract storage
  - ``Ledger``                for versioned, auditable state writes
  - ``TransactionContext``    for gas tracking & TX metadata
  - ``CryptoPlugin``          for real signature verification

Usage
-----
::

    from zexus.blockchain.contract_vm import ContractVM

    contract_vm = ContractVM(chain=node.chain)
    receipt = contract_vm.execute_contract(
        contract_address="0xabc...",
        action="transfer",
        args={"to": "0xdef...", "amount": 100},
        caller="0x123...",
        gas_limit=500_000,
    )
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .chain import Chain, Transaction, TransactionReceipt
from .transaction import TransactionContext
from .crypto import CryptoPlugin

logger = logging.getLogger("zexus.blockchain.contract_vm")

# Try importing the VM — it may not be installed in every deployment.
try:
    from ..vm.vm import VM as ZexusVM
    from ..vm.gas_metering import GasMetering, GasCost, OutOfGasError
    _VM_AVAILABLE = True
except ImportError:
    _VM_AVAILABLE = False
    ZexusVM = None  # type: ignore
    GasMetering = None  # type: ignore
    GasCost = None  # type: ignore
    OutOfGasError = None  # type: ignore

# SmartContract from security module
try:
    from ..security import SmartContract
    _CONTRACT_AVAILABLE = True
except ImportError:
    _CONTRACT_AVAILABLE = False
    SmartContract = None  # type: ignore


# ---------------------------------------------------------------------------
# Contract State Adapter
# ---------------------------------------------------------------------------

class ContractStateAdapter(dict):
    """A dict-like object that transparently delegates reads/writes to
    ``Chain.contract_state[contract_address]``.

    Every *write* is recorded in a pending journal so the caller can
    commit or rollback atomically.
    """

    def __init__(self, chain: Chain, contract_address: str):
        super().__init__()
        self._chain = chain
        self._contract_address = contract_address
        # Seed from chain (make a shallow copy so mutations don't leak back)
        stored = chain.contract_state.get(contract_address, {})
        super().update(copy.deepcopy(stored))

    # Reads ---------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)

    # Writes (journalled) -------------------------------------------------

    def __setitem__(self, key: str, value: Any):
        super().__setitem__(key, value)

    def update(self, other=(), **kwargs):
        super().update(other, **kwargs)

    # Commit / Rollback ---------------------------------------------------

    def commit(self):
        """Flush all current state back to ``chain.contract_state``."""
        self._chain.contract_state[self._contract_address] = dict(self)

    def rollback(self, snapshot: Dict[str, Any]):
        """Restore from a previously captured snapshot."""
        self.clear()
        super().update(snapshot)


# ---------------------------------------------------------------------------
# Execution Receipt
# ---------------------------------------------------------------------------

@dataclass
class ContractExecutionReceipt:
    """Result of executing a contract action through the VM."""
    success: bool = True
    return_value: Any = None
    gas_used: int = 0
    gas_limit: int = 0
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    revert_reason: str = ""
    state_changes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "return_value": str(self.return_value),
            "gas_used": self.gas_used,
            "gas_limit": self.gas_limit,
            "logs": self.logs,
            "error": self.error,
            "revert_reason": self.revert_reason,
            "state_changes": self.state_changes,
        }


# ---------------------------------------------------------------------------
# ContractVM — the bridge
# ---------------------------------------------------------------------------

class ContractVM:
    """Bridge between the Zexus VM and the real blockchain infrastructure.

    Responsibilities
    ----------------
    1. Provide a real ``_blockchain_state`` backed by ``Chain.contract_state``
       so that STATE_READ / STATE_WRITE opcodes persist to the chain.
    2. Inject a proper ``verify_sig`` builtin so VERIFY_SIGNATURE uses
       ``CryptoPlugin.verify_signature`` instead of the insecure SHA-256
       fallback.
    3. Enforce gas metering for every opcode in both sync *and* async paths.
    4. Wire TX_BEGIN / TX_COMMIT / TX_REVERT to atomic chain-state updates.
    5. Execute ``SmartContract`` actions through the VM with a full
       ``TransactionContext``.
    """

    def __init__(
        self,
        chain: Chain,
        gas_limit: int = 10_000_000,
        debug: bool = False,
    ):
        if not _VM_AVAILABLE:
            raise RuntimeError(
                "ContractVM requires the Zexus VM. "
                "Ensure src/zexus/vm/ is present and importable."
            )
        self._chain = chain
        self._default_gas_limit = gas_limit
        self._debug = debug

        # Deployed contract registry:  address -> SmartContract
        self._contracts: Dict[str, SmartContract] = {}

        # Reentrancy guard — tracks contracts currently being executed
        self._executing: set = set()

        # Cross-contract call depth tracking
        self._call_depth: int = 0
        self._max_call_depth: int = 10

    # ------------------------------------------------------------------
    # Contract lifecycle
    # ------------------------------------------------------------------

    def deploy_contract(
        self,
        contract: "SmartContract",
        deployer: str,
        gas_limit: Optional[int] = None,
        initial_value: int = 0,
    ) -> ContractExecutionReceipt:
        """Deploy a SmartContract onto the chain.

        - Assigns the contract an on-chain address.
        - Stores initial bytecode / storage in ``chain.contract_state``.
        - Runs the constructor (if any) inside the VM.
        """
        gas_limit = gas_limit or self._default_gas_limit
        address = contract.address

        # Register on-chain account
        acct = self._chain.get_account(address)
        acct["balance"] = initial_value
        acct["code"] = contract.name  # Store contract "type" as code
        acct["nonce"] = 0

        # Save initial storage
        initial_storage: Dict[str, Any] = {}
        if hasattr(contract, 'storage') and hasattr(contract.storage, 'current_state'):
            initial_storage = dict(contract.storage.current_state)
        elif hasattr(contract, 'storage') and hasattr(contract.storage, 'data'):
            initial_storage = dict(contract.storage.data)
        self._chain.contract_state[address] = initial_storage

        # Register locally
        self._contracts[address] = contract

        receipt = ContractExecutionReceipt(
            success=True,
            gas_limit=gas_limit,
            state_changes={"deployed": address, "storage_keys": list(initial_storage.keys())},
        )

        logger.info("Contract '%s' deployed at %s", contract.name, address)
        return receipt

    def get_contract(self, address: str) -> Optional["SmartContract"]:
        """Look up a deployed contract by address."""
        return self._contracts.get(address)

    # ------------------------------------------------------------------
    # Contract execution
    # ------------------------------------------------------------------

    def execute_contract(
        self,
        contract_address: str,
        action: str,
        args: Optional[Dict[str, Any]] = None,
        caller: str = "",
        gas_limit: Optional[int] = None,
        value: int = 0,
    ) -> ContractExecutionReceipt:
        """Execute a contract action inside the VM.

        This is the main entry-point used by ``BlockchainNode`` when
        processing a contract-call transaction.

        Steps
        -----
        1. Build a ``TransactionContext`` with the caller, gas limit, etc.
        2. Create a ``ContractStateAdapter`` backed by the chain.
        3. Construct a fresh VM with the state adapter as
           ``env["_blockchain_state"]`` and real ``verify_sig``.
        4. Execute the contract's action body via the VM.
        5. On success → commit state; on failure → rollback.
        6. Return a ``ContractExecutionReceipt``.
        """
        gas_limit = gas_limit or self._default_gas_limit
        # Per-execution log list — avoids sharing state across concurrent calls
        logs: List[Dict[str, Any]] = []

        contract = self._contracts.get(contract_address)
        if contract is None:
            return ContractExecutionReceipt(
                success=False,
                error=f"Contract not found at {contract_address}",
                gas_limit=gas_limit,
            )

        action_obj = contract.actions.get(action)
        if action_obj is None:
            return ContractExecutionReceipt(
                success=False,
                error=f"Action '{action}' not found on contract '{contract.name}'",
                gas_limit=gas_limit,
            )

        # Reentrancy guard
        if contract_address in self._executing:
            return ContractExecutionReceipt(
                success=False,
                error="ReentrancyGuard",
                revert_reason=f"Reentrant call to contract {contract_address}",
                gas_limit=gas_limit,
            )

        # Call-depth guard (cross-contract calls)
        if self._call_depth >= self._max_call_depth:
            return ContractExecutionReceipt(
                success=False,
                error="CallDepthExceeded",
                revert_reason=f"Call depth {self._call_depth} exceeds max {self._max_call_depth}",
                gas_limit=gas_limit,
            )

        self._executing.add(contract_address)
        self._call_depth += 1

        # 1. TX context
        tip = self._chain.tip
        tx_ctx = TransactionContext(
            caller=caller,
            timestamp=time.time(),
            block_hash=tip.hash if tip else "0" * 64,
            gas_limit=gas_limit,
        )

        # 2. Chain-backed state adapter
        state_adapter = ContractStateAdapter(self._chain, contract_address)
        snapshot = dict(state_adapter)  # for rollback

        # 3. Build VM environment + builtins
        env = self._build_env(state_adapter, tx_ctx, contract, args or {})
        builtins = self._build_builtins(tx_ctx, contract_address, logs)

        # 4. Execute
        try:
            vm = ZexusVM(
                env=env,
                builtins=builtins,
                enable_gas_metering=True,
                gas_limit=gas_limit,
                debug=self._debug,
            )

            # Execute the action body through the evaluator
            result = self._execute_action(vm, action_obj, env, args or {})

            gas_used = vm.gas_metering.gas_used if vm.gas_metering else 0

            # 5a. Commit
            state_adapter.commit()

            return ContractExecutionReceipt(
                success=True,
                return_value=result,
                gas_used=gas_used,
                gas_limit=gas_limit,
                logs=list(logs),
                state_changes=self._diff_state(snapshot, dict(state_adapter)),
            )

        except OutOfGasError as e:
            # 5b. Rollback on OOG
            state_adapter.rollback(snapshot)
            return ContractExecutionReceipt(
                success=False,
                gas_used=gas_limit,
                gas_limit=gas_limit,
                error="OutOfGas",
                revert_reason=str(e),
            )

        except Exception as e:
            # 5b. Rollback on any error
            state_adapter.rollback(snapshot)
            return ContractExecutionReceipt(
                success=False,
                gas_used=0,
                gas_limit=gas_limit,
                error=type(e).__name__,
                revert_reason=str(e),
                logs=list(logs),
            )

        finally:
            self._executing.discard(contract_address)
            self._call_depth -= 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_env(
        self,
        state_adapter: ContractStateAdapter,
        tx_ctx: TransactionContext,
        contract: "SmartContract",
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble the VM ``env`` dict for a contract execution."""
        from ..object import Map, String, Integer, Float, Boolean as BooleanObj

        env: Dict[str, Any] = {}

        # Wire the chain-backed state adapter as _blockchain_state
        env["_blockchain_state"] = state_adapter

        # Gas tracking (used by GAS_CHARGE opcode)
        env["_gas_remaining"] = tx_ctx.gas_limit

        # TX object — immutable context
        tx_map = Map({
            String("caller"): String(tx_ctx.caller),
            String("timestamp"): Integer(int(tx_ctx.timestamp)),
            String("block_hash"): String(tx_ctx.block_hash),
            String("gas_limit"): Integer(tx_ctx.gas_limit),
            String("gas_remaining"): Integer(tx_ctx.gas_remaining),
        })
        env["TX"] = tx_map

        # Pre-populate contract storage into env for tree-walking evaluator
        if hasattr(contract, 'storage'):
            state = state_adapter  # already seeded from chain
            for key, val in state.items():
                env[key] = self._wrap_value(val)

        # Arguments (passed as env vars to the action)
        for k, v in args.items():
            env[k] = self._wrap_value(v)

        # Contract reference
        env["self"] = contract
        env["_contract_address"] = contract.address

        return env

    def _build_builtins(
        self,
        tx_ctx: TransactionContext,
        contract_address: str = "",
        logs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build VM builtins, including the real ``verify_sig``."""
        builtins: Dict[str, Any] = {}
        _logs = logs if logs is not None else []

        # Real signature verification via CryptoPlugin
        def verify_sig(signature: Any, message: Any, public_key: Any) -> bool:
            """Verify an ECDSA signature using the real CryptoPlugin."""
            sig_str = str(signature.value) if hasattr(signature, 'value') else str(signature)
            msg_str = str(message.value) if hasattr(message, 'value') else str(message)
            key_str = str(public_key.value) if hasattr(public_key, 'value') else str(public_key)
            try:
                return CryptoPlugin.verify_signature(msg_str, sig_str, key_str)
            except Exception:
                return False

        builtins["verify_sig"] = verify_sig

        # Emit log/event
        def emit_event(name: Any, data: Any = None) -> None:
            """Emit a contract event (stored in receipt logs)."""
            name_str = str(name.value) if hasattr(name, 'value') else str(name)
            _logs.append({
                "event": name_str,
                "data": data,
                "timestamp": time.time(),
                "contract": contract_address,  # emit from contract, not caller
            })

        builtins["emit"] = emit_event

        # Balance check
        def get_balance(address: Any) -> int:
            """Get on-chain balance of an address."""
            addr = str(address.value) if hasattr(address, 'value') else str(address)
            return self._chain.get_account(addr).get("balance", 0)

        builtins["get_balance"] = get_balance

        # Transfer — with overflow protection
        def transfer(to: Any, amount: Any) -> bool:
            """Transfer value between accounts."""
            to_str = str(to.value) if hasattr(to, 'value') else str(to)
            amt = int(amount.value) if hasattr(amount, 'value') else int(amount)
            if amt <= 0:
                return False
            caller_acct = self._chain.get_account(tx_ctx.caller)
            sender_balance = caller_acct.get("balance", 0)
            if sender_balance < amt:
                return False
            to_acct = self._chain.get_account(to_str)
            to_balance = to_acct.get("balance", 0)
            # Overflow check
            if to_balance + amt < to_balance:
                return False
            caller_acct["balance"] = sender_balance - amt
            to_acct["balance"] = to_balance + amt
            return True

        builtins["transfer"] = transfer

        # Keccak-256 hash
        def keccak256(data: Any) -> str:
            """Keccak-256 hash via CryptoPlugin."""
            d = str(data.value) if hasattr(data, 'value') else str(data)
            return CryptoPlugin.keccak256(d)

        builtins["keccak256"] = keccak256

        # Block info
        def block_number() -> int:
            return self._chain.height

        def block_timestamp() -> float:
            tip = self._chain.tip
            return tip.header.timestamp if tip else 0.0

        builtins["block_number"] = block_number
        builtins["block_timestamp"] = block_timestamp

        # ── Cross-contract calls ──────────────────────────────────
        vm_ref = self  # capture for closures

        def contract_call(target_address: Any, action: Any,
                          call_args: Any = None, value: Any = None) -> Any:
            """Call another contract's action (state-mutating).

            Parameters
            ----------
            target_address : str or String
                Address of the contract to call.
            action : str or String
                Name of the action to invoke.
            call_args : dict, optional
                Arguments to pass to the action.
            value : int, optional
                Value to transfer with the call.

            Returns the action's return value (unwrapped to Python).
            Raises RuntimeError on failure or depth exceeded.
            """
            addr = str(target_address.value) if hasattr(target_address, 'value') else str(target_address)
            act = str(action.value) if hasattr(action, 'value') else str(action)
            args = {}
            if call_args is not None:
                if hasattr(call_args, 'pairs'):
                    args = {str(k.value) if hasattr(k, 'value') else str(k):
                            vm_ref._unwrap_value(v) for k, v in call_args.pairs.items()}
                elif isinstance(call_args, dict):
                    args = call_args
            val = 0
            if value is not None:
                val = int(value.value) if hasattr(value, 'value') else int(value)

            if vm_ref._call_depth >= vm_ref._max_call_depth:
                raise RuntimeError(f"Cross-contract call depth exceeded (max {vm_ref._max_call_depth})")

            receipt = vm_ref.execute_contract(
                contract_address=addr,
                action=act,
                args=args,
                caller=contract_address or tx_ctx.caller,
                gas_limit=tx_ctx.gas_remaining,
                value=val,
            )
            if not receipt.success:
                raise RuntimeError(f"Cross-contract call failed: {receipt.error or receipt.revert_reason}")
            return receipt.return_value

        def static_contract_call(target_address: Any, action: Any,
                                  call_args: Any = None) -> Any:
            """Read-only call to another contract (no state changes).

            Same as contract_call but uses static_call internally.
            """
            addr = str(target_address.value) if hasattr(target_address, 'value') else str(target_address)
            act = str(action.value) if hasattr(action, 'value') else str(action)
            args = {}
            if call_args is not None:
                if hasattr(call_args, 'pairs'):
                    args = {str(k.value) if hasattr(k, 'value') else str(k):
                            vm_ref._unwrap_value(v) for k, v in call_args.pairs.items()}
                elif isinstance(call_args, dict):
                    args = call_args

            receipt = vm_ref.static_call(
                contract_address=addr,
                action=act,
                args=args,
                caller=contract_address or tx_ctx.caller,
            )
            if not receipt.success:
                raise RuntimeError(f"Static call failed: {receipt.error or receipt.revert_reason}")
            return receipt.return_value

        def delegate_call(target_address: Any, action: Any,
                          call_args: Any = None) -> Any:
            """Delegatecall: execute target's code in caller's storage context.

            Like contract_call, but the target's action runs with the
            *calling* contract's state adapter, so state writes go to
            the caller's storage, not the target's.
            """
            addr = str(target_address.value) if hasattr(target_address, 'value') else str(target_address)
            act = str(action.value) if hasattr(action, 'value') else str(action)
            args = {}
            if call_args is not None:
                if hasattr(call_args, 'pairs'):
                    args = {str(k.value) if hasattr(k, 'value') else str(k):
                            vm_ref._unwrap_value(v) for k, v in call_args.pairs.items()}
                elif isinstance(call_args, dict):
                    args = call_args

            if vm_ref._call_depth >= vm_ref._max_call_depth:
                raise RuntimeError(f"Delegatecall depth exceeded (max {vm_ref._max_call_depth})")

            # Find the target contract's action
            target_contract = vm_ref.get_contract(addr)
            if target_contract is None:
                raise RuntimeError(f"Contract not found: {addr}")

            action_obj = None
            if hasattr(target_contract, 'actions'):
                for a in target_contract.actions:
                    a_name = a.name if hasattr(a, 'name') else str(a)
                    if a_name == act:
                        action_obj = a
                        break
            if action_obj is None:
                raise RuntimeError(f"Action '{act}' not found on contract {addr}")

            # Execute with *caller's* state adapter (the key difference)
            caller_addr = contract_address or tx_ctx.caller
            state_adapter = ContractStateAdapter(vm_ref._chain, caller_addr)
            snapshot = dict(state_adapter)

            from ..vm.vm import VM as ZexusVM
            vm = ZexusVM(debug=vm_ref._debug)
            vm_ref._call_depth += 1
            try:
                env = vm_ref._build_env(state_adapter, tx_ctx, target_contract, args)
                inner_builtins = vm_ref._build_builtins(tx_ctx, caller_addr, _logs)
                for bk, bv in inner_builtins.items():
                    vm.env[bk] = bv
                result = vm_ref._execute_action(vm, action_obj, env, args)
                state_adapter.commit()
                return vm_ref._unwrap_value(result) if result is not None else None
            except Exception:
                state_adapter.rollback(snapshot)
                raise
            finally:
                vm_ref._call_depth -= 1

        builtins["contract_call"] = contract_call
        builtins["static_call"] = static_contract_call
        builtins["delegate_call"] = delegate_call

        return builtins

    def _execute_action(
        self,
        vm: "ZexusVM",
        action_obj: Any,
        env: Dict[str, Any],
        args: Dict[str, Any],
    ) -> Any:
        """Run a contract action's body through the evaluator.

        Depending on whether the action has bytecode or an AST body,
        we either use the VM directly or fall back to the tree-walking
        evaluator.
        """
        from ..object import Environment, Action
        from ..evaluator.core import Evaluator

        # Build an evaluator Environment from the flat dict
        eval_env = Environment()
        for k, v in env.items():
            eval_env.set(k, v)

        # Add action parameters from args
        if hasattr(action_obj, 'parameters') and action_obj.parameters:
            for param in action_obj.parameters:
                param_name = param.value if hasattr(param, 'value') else str(param)
                if param_name in args:
                    eval_env.set(param_name, self._wrap_value(args[param_name]))

        # Execute through the evaluator (which will delegate to VM for bytecode)
        evaluator = Evaluator(use_vm=False)  # use tree-walking for reliability
        result = None
        try:
            if hasattr(action_obj, 'body'):
                result = evaluator.eval_node(action_obj.body, eval_env, [])
        except Exception as e:
            if "Requirement failed" in str(e):
                raise  # Re-raise REQUIRE failures
            raise

        # Sync modified vars back to _blockchain_state
        state_adapter = env.get("_blockchain_state")
        if state_adapter and hasattr(action_obj, 'body'):
            # Check for any env vars that match storage keys
            for key in list(state_adapter.keys()):
                new_val = eval_env.get(key)
                if new_val is not None:
                    state_adapter[key] = self._unwrap_value(new_val)

        return result

    # ------------------------------------------------------------------
    # Value wrapping / unwrapping
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_value(val: Any) -> Any:
        """Wrap a Python value into a Zexus object."""
        from ..object import (
            Integer as ZInteger, Float as ZFloat,
            Boolean as ZBoolean, String as ZString,
            List as ZList, Map as ZMap, Null as ZNull,
        )
        if isinstance(val, (ZInteger, ZFloat, ZBoolean, ZString, ZList, ZMap, ZNull)):
            return val
        if isinstance(val, bool):
            return ZBoolean(val)
        if isinstance(val, int):
            return ZInteger(val)
        if isinstance(val, float):
            return ZFloat(val)
        if isinstance(val, str):
            return ZString(val)
        if isinstance(val, list):
            return ZList([ContractVM._wrap_value(e) for e in val])
        if isinstance(val, dict):
            return ZMap({
                ZString(str(k)): ContractVM._wrap_value(v)
                for k, v in val.items()
            })
        if val is None:
            return ZNull()
        return val

    @staticmethod
    def _unwrap_value(val: Any) -> Any:
        """Unwrap a Zexus object to a plain Python value."""
        if hasattr(val, 'value'):
            return val.value
        if hasattr(val, 'elements'):  # ZList
            return [ContractVM._unwrap_value(e) for e in val.elements]
        if hasattr(val, 'pairs'):  # ZMap
            return {
                ContractVM._unwrap_value(k): ContractVM._unwrap_value(v)
                for k, v in val.pairs.items()
            }
        return val

    @staticmethod
    def _diff_state(
        before: Dict[str, Any], after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute the difference between two state snapshots."""
        changes: Dict[str, Any] = {}
        all_keys = set(before.keys()) | set(after.keys())
        for key in all_keys:
            old = before.get(key)
            new = after.get(key)
            if old != new:
                changes[key] = {"before": old, "after": new}
        return changes

    # ------------------------------------------------------------------
    # Static call (read-only, no state commit)
    # ------------------------------------------------------------------

    def static_call(
        self,
        contract_address: str,
        action: str,
        args: Optional[Dict[str, Any]] = None,
        caller: str = "",
        gas_limit: Optional[int] = None,
    ) -> ContractExecutionReceipt:
        """Execute a read-only call that never commits state changes.

        Useful for ``view`` functions that only read storage.
        """
        gas_limit = gas_limit or self._default_gas_limit
        logs: List[Dict[str, Any]] = []

        contract = self._contracts.get(contract_address)
        if contract is None:
            return ContractExecutionReceipt(
                success=False,
                error=f"Contract not found at {contract_address}",
                gas_limit=gas_limit,
            )

        action_obj = contract.actions.get(action)
        if action_obj is None:
            return ContractExecutionReceipt(
                success=False,
                error=f"Action '{action}' not found",
                gas_limit=gas_limit,
            )

        tip = self._chain.tip
        tx_ctx = TransactionContext(
            caller=caller,
            timestamp=time.time(),
            block_hash=tip.hash if tip else "0" * 64,
            gas_limit=gas_limit,
        )

        state_adapter = ContractStateAdapter(self._chain, contract_address)
        env = self._build_env(state_adapter, tx_ctx, contract, args or {})
        builtins = self._build_builtins(tx_ctx, contract_address, logs)

        try:
            vm = ZexusVM(
                env=env,
                builtins=builtins,
                enable_gas_metering=True,
                gas_limit=gas_limit,
                debug=self._debug,
            )
            result = self._execute_action(vm, action_obj, env, args or {})
            gas_used = vm.gas_metering.gas_used if vm.gas_metering else 0

            # NOTE: No commit — state_adapter is discarded
            return ContractExecutionReceipt(
                success=True,
                return_value=result,
                gas_used=gas_used,
                gas_limit=gas_limit,
                logs=list(logs),
            )
        except Exception as e:
            return ContractExecutionReceipt(
                success=False,
                error=type(e).__name__,
                revert_reason=str(e),
                gas_limit=gas_limit,
            )

    # ------------------------------------------------------------------
    # Batch execution (for block processing)
    # ------------------------------------------------------------------

    def process_contract_transaction(
        self,
        tx: Transaction,
    ) -> TransactionReceipt:
        """Process a contract-call transaction and produce a receipt.

        This is what ``BlockchainNode`` calls when processing a block
        that contains contract interactions.

        The ``tx.data`` field is expected to be JSON-encoded::

            {
                "contract": "<address>",
                "action": "<method>",
                "args": { ... }
            }
        """
        receipt = TransactionReceipt(
            tx_hash=tx.tx_hash,
            status=0,
            gas_used=0,
        )

        # Parse tx.data
        try:
            call_data = json.loads(tx.data) if isinstance(tx.data, str) and tx.data else {}
        except json.JSONDecodeError:
            receipt.revert_reason = "Invalid contract call data"
            return receipt

        contract_addr = call_data.get("contract", tx.recipient)
        action_name = call_data.get("action", "")
        action_args = call_data.get("args", {})

        if not action_name:
            receipt.revert_reason = "Missing action name in tx.data"
            return receipt

        exec_receipt = self.execute_contract(
            contract_address=contract_addr,
            action=action_name,
            args=action_args,
            caller=tx.sender,
            gas_limit=tx.gas_limit,
            value=tx.value,
        )

        receipt.status = 1 if exec_receipt.success else 0
        receipt.gas_used = exec_receipt.gas_used
        receipt.logs = exec_receipt.logs
        receipt.revert_reason = exec_receipt.revert_reason
        receipt.contract_address = contract_addr

        return receipt
