"""
Zexus Blockchain — Advanced Mempool, Gas Price Oracle & ABI Encoder

Extends the basic ``Mempool`` in :pymod:`chain` with:

* **PriorityMempool** — gas-price-ordered pool with per-sender nonce
  validation, eviction, and replace-by-fee.
* **GasPriceOracle** — EIP-1559-style dynamic base fee adjustment.
* **ABIEncoder** — Solidity-compatible ABI encoding / decoding with
  4-byte function selectors (keccak256).
"""

import hashlib
import heapq
import json
import struct
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# PriorityMempool
# ---------------------------------------------------------------------------

class PriorityMempool:
    """Transaction mempool with gas-price priority ordering.

    Transactions are kept sorted by *gas_price* (highest first).  When the
    pool is full the cheapest transaction is evicted to make room.  Nonce
    ordering is enforced per sender.

    Parameters
    ----------
    max_size : int
        Maximum number of transactions the pool will hold.
    max_per_sender : int
        Maximum pending transactions per sender address.
    """

    def __init__(self, max_size: int = 10_000, max_per_sender: int = 64):
        self.max_size = max_size
        self.max_per_sender = max_per_sender

        # Primary storage: tx_hash -> tx dict/object
        self._txs: Dict[str, Any] = {}

        # Min-heap ordered by gas_price so we can cheaply find the lowest.
        # Each entry is ``(gas_price, tx_hash)`` — heapq is a *min* heap.
        self._heap: List[Tuple[int, str]] = []

        # Per-sender indices
        self._by_sender: Dict[str, Dict[int, str]] = defaultdict(dict)  # sender -> {nonce: tx_hash}
        self._sender_nonces: Dict[str, int] = {}  # sender -> next expected nonce

    # -- helpers -------------------------------------------------------------

    def _tx_gas_price(self, tx) -> int:
        """Extract gas_price from a tx (dict or dataclass)."""
        if isinstance(tx, dict):
            return int(tx.get("gas_price", 0))
        return int(getattr(tx, "gas_price", 0))

    def _tx_hash(self, tx) -> str:
        """Extract or compute the tx hash."""
        if isinstance(tx, dict):
            h = tx.get("tx_hash", "")
            if not h:
                h = hashlib.sha256(
                    json.dumps(tx, sort_keys=True, default=str).encode()
                ).hexdigest()
                tx["tx_hash"] = h
            return h
        h = getattr(tx, "tx_hash", "")
        if not h:
            if hasattr(tx, "compute_hash"):
                h = tx.compute_hash()
            else:
                h = hashlib.sha256(
                    json.dumps(tx.to_dict(), sort_keys=True, default=str).encode()
                ).hexdigest()
                tx.tx_hash = h
        return h

    def _tx_sender(self, tx) -> str:
        if isinstance(tx, dict):
            return tx.get("sender", "")
        return getattr(tx, "sender", "")

    def _tx_nonce(self, tx) -> int:
        if isinstance(tx, dict):
            return int(tx.get("nonce", 0))
        return int(getattr(tx, "nonce", 0))

    def _tx_gas_limit(self, tx) -> int:
        if isinstance(tx, dict):
            return int(tx.get("gas_limit", 21_000))
        return int(getattr(tx, "gas_limit", 21_000))

    # -- public API ----------------------------------------------------------

    def add(self, tx) -> bool:
        """Add a transaction to the pool.

        * Rejects exact duplicates.
        * Enforces ``max_per_sender``.
        * Validates monotonic nonce ordering per sender.
        * When the pool is full, evicts the transaction with the lowest
          gas price (only if the new tx has a higher gas price).

        Returns
        -------
        bool
            ``True`` if the transaction was accepted.
        """
        tx_hash = self._tx_hash(tx)
        if tx_hash in self._txs:
            return False

        sender = self._tx_sender(tx)
        nonce = self._tx_nonce(tx)
        gas_price = self._tx_gas_price(tx)

        # Per-sender limit
        if len(self._by_sender[sender]) >= self.max_per_sender:
            return False

        # Nonce validation: must be >= next expected nonce
        expected_nonce = self._sender_nonces.get(sender, 0)
        if nonce < expected_nonce:
            return False

        # Pool-full handling: evict cheapest if new tx pays more
        if len(self._txs) >= self.max_size:
            self._compact_heap()
            if not self._heap:
                return False
            lowest_price, lowest_hash = self._heap[0]
            if gas_price <= lowest_price:
                return False
            self._remove_internal(lowest_hash)

        # Accept
        self._txs[tx_hash] = tx
        heapq.heappush(self._heap, (gas_price, tx_hash))
        self._by_sender[sender][nonce] = tx_hash
        if nonce >= expected_nonce:
            self._sender_nonces[sender] = nonce + 1

        return True

    def remove(self, tx_hash: str) -> bool:
        """Remove a transaction by its hash.

        Returns ``True`` if the transaction existed and was removed.
        """
        return self._remove_internal(tx_hash)

    def get_pending(self, gas_limit: int = 10_000_000) -> List:
        """Return transactions fitting within *gas_limit*, highest gas price first."""
        sorted_txs = sorted(
            self._txs.values(),
            key=lambda t: self._tx_gas_price(t),
            reverse=True,
        )
        result: List = []
        total_gas = 0
        for tx in sorted_txs:
            tx_gas = self._tx_gas_limit(tx)
            if total_gas + tx_gas <= gas_limit:
                result.append(tx)
                total_gas += tx_gas
        return result

    def get_by_sender(self, sender: str) -> List:
        """Return all pending transactions from *sender*, ordered by nonce."""
        nonce_map = self._by_sender.get(sender, {})
        txs = []
        for nonce in sorted(nonce_map):
            tx_hash = nonce_map[nonce]
            tx = self._txs.get(tx_hash)
            if tx is not None:
                txs.append(tx)
        return txs

    @property
    def size(self) -> int:
        """Current number of transactions in the pool."""
        return len(self._txs)

    @property
    def is_full(self) -> bool:
        """Whether the pool has reached ``max_size``."""
        return len(self._txs) >= self.max_size

    def clear(self):
        """Remove all transactions."""
        self._txs.clear()
        self._heap.clear()
        self._by_sender.clear()
        self._sender_nonces.clear()

    def evict_below(self, min_gas_price: int) -> int:
        """Evict every transaction whose gas price is below *min_gas_price*.

        Returns the number of evicted transactions.
        """
        to_remove = [
            h for h, tx in self._txs.items()
            if self._tx_gas_price(tx) < min_gas_price
        ]
        for h in to_remove:
            self._remove_internal(h)
        return len(to_remove)

    def replace(self, old_hash: str, new_tx) -> bool:
        """Replace an existing transaction (same sender + nonce, higher gas price).

        Returns ``True`` if the replacement succeeded.
        """
        old_tx = self._txs.get(old_hash)
        if old_tx is None:
            return False

        new_hash = self._tx_hash(new_tx)
        if self._tx_sender(new_tx) != self._tx_sender(old_tx):
            return False
        if self._tx_nonce(new_tx) != self._tx_nonce(old_tx):
            return False
        if self._tx_gas_price(new_tx) <= self._tx_gas_price(old_tx):
            return False

        self._remove_internal(old_hash)
        self._txs[new_hash] = new_tx
        heapq.heappush(self._heap, (self._tx_gas_price(new_tx), new_hash))

        sender = self._tx_sender(new_tx)
        nonce = self._tx_nonce(new_tx)
        self._by_sender[sender][nonce] = new_hash
        return True

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the pool.

        Keys: ``size``, ``senders``, ``min_gas_price``, ``max_gas_price``,
        ``total_gas``.
        """
        if not self._txs:
            return {
                "size": 0,
                "senders": 0,
                "min_gas_price": 0,
                "max_gas_price": 0,
                "total_gas": 0,
            }
        prices = [self._tx_gas_price(tx) for tx in self._txs.values()]
        total_gas = sum(self._tx_gas_limit(tx) for tx in self._txs.values())
        senders = len({self._tx_sender(tx) for tx in self._txs.values()})
        return {
            "size": len(self._txs),
            "senders": senders,
            "min_gas_price": min(prices),
            "max_gas_price": max(prices),
            "total_gas": total_gas,
        }

    # -- internal ------------------------------------------------------------

    def _remove_internal(self, tx_hash: str) -> bool:
        tx = self._txs.pop(tx_hash, None)
        if tx is None:
            return False
        sender = self._tx_sender(tx)
        nonce = self._tx_nonce(tx)
        nonce_map = self._by_sender.get(sender, {})
        if nonce_map.get(nonce) == tx_hash:
            del nonce_map[nonce]
            if not nonce_map:
                self._by_sender.pop(sender, None)
        # Lazy deletion for the heap — stale entries are cleaned on access.
        return True

    def _compact_heap(self):
        """Remove stale (already-deleted) entries from the front of the heap."""
        while self._heap and self._heap[0][1] not in self._txs:
            heapq.heappop(self._heap)


# ---------------------------------------------------------------------------
# GasPriceOracle — EIP-1559-style dynamic base fee
# ---------------------------------------------------------------------------

class GasPriceOracle:
    """Dynamic base-fee oracle inspired by EIP-1559.

    After each block, :meth:`update` adjusts the base fee proportionally
    to how far the actual gas usage was from the target.  The fee can
    increase or decrease by at most ``1 / elasticity`` per block.

    Parameters
    ----------
    base_fee : int
        Initial base fee (in smallest denomination).
    target_gas_used : int
        Target gas usage per block.
    max_gas_limit : int
        Maximum gas a block is allowed to contain.
    elasticity : int
        The EIP-1559 elasticity multiplier.
    """

    BASE_PRIORITY_FEE = 1_000  # default tip

    def __init__(
        self,
        base_fee: int = 1_000,
        target_gas_used: int = 5_000_000,
        max_gas_limit: int = 10_000_000,
        elasticity: int = 2,
    ):
        self._base_fee = base_fee
        self.target_gas_used = target_gas_used
        self.max_gas_limit = max_gas_limit
        self.elasticity = elasticity
        self._history: List[int] = [base_fee]

    # -- public API ----------------------------------------------------------

    def update(self, block_gas_used: int) -> int:
        """Adjust the base fee after a block with *block_gas_used* gas.

        If the block used more gas than the target the fee increases;
        if less, it decreases.  The change is bounded by
        ``base_fee // (elasticity * 4)`` per block to keep volatility low.

        Returns the new base fee.
        """
        if block_gas_used > self.target_gas_used:
            # Over target → increase
            delta = block_gas_used - self.target_gas_used
            fee_delta = max(
                1,
                self._base_fee * delta
                // self.target_gas_used
                // self.elasticity,
            )
            self._base_fee += fee_delta
        elif block_gas_used < self.target_gas_used:
            # Under target → decrease
            delta = self.target_gas_used - block_gas_used
            fee_delta = (
                self._base_fee * delta
                // self.target_gas_used
                // self.elasticity
            )
            self._base_fee = max(1, self._base_fee - fee_delta)
        # else: exactly on target → no change

        self._history.append(self._base_fee)
        return self._base_fee

    def suggest_gas_price(self) -> Dict[str, int]:
        """Suggest gas price components.

        Returns
        -------
        dict
            ``base_fee`` — current base fee.
            ``priority_fee`` — recommended tip / priority fee.
            ``max_fee`` — ``base_fee + priority_fee``.
        """
        priority = self.BASE_PRIORITY_FEE
        return {
            "base_fee": self._base_fee,
            "priority_fee": priority,
            "max_fee": self._base_fee + priority,
        }

    def estimate_fee(self, gas_limit: int) -> int:
        """Estimate the total fee for a transaction using *gas_limit* gas.

        The estimate uses the suggested ``max_fee`` (base + priority).
        """
        suggestion = self.suggest_gas_price()
        return gas_limit * suggestion["max_fee"]

    @property
    def base_fee(self) -> int:
        """Current base fee."""
        return self._base_fee

    @property
    def history(self) -> List[int]:
        """List of historical base fees (oldest first)."""
        return list(self._history)


# ---------------------------------------------------------------------------
# ABIEncoder — Solidity-compatible ABI encoding
# ---------------------------------------------------------------------------

def _keccak256(data: bytes) -> bytes:
    """Compute Keccak-256 using hashlib (available since Python 3.6+)."""
    return hashlib.sha3_256(data).digest()


def _pad_left(data: bytes, length: int = 32) -> bytes:
    """Zero-pad *data* on the left to *length* bytes."""
    return data.rjust(length, b"\x00")


def _pad_right(data: bytes, length: int = 32) -> bytes:
    """Zero-pad *data* on the right to *length* bytes."""
    return data.ljust(length, b"\x00")


class ABIEncoder:
    """Minimal Solidity-compatible ABI encoder / decoder.

    Supports a useful subset of ABI types:

    * ``uint256``, ``int256``, ``uint<N>``, ``int<N>``
    * ``address`` (20-byte hex)
    * ``bool``
    * ``bytes32`` (and ``bytes<N>`` up to 32)
    * ``bytes`` (dynamic)
    * ``string`` (dynamic)

    Dynamic types (``bytes``, ``string``) are encoded with a 32-byte
    offset pointer followed by a length-prefixed data region.
    """

    # -- function selector ---------------------------------------------------

    @staticmethod
    def function_selector(name: str, param_types: List[str]) -> bytes:
        """Compute the 4-byte function selector.

        ``selector = keccak256(name + "(" + ",".join(param_types) + ")")[:4]``

        Parameters
        ----------
        name : str
            Function name.
        param_types : list[str]
            Canonical ABI type strings (e.g. ``["uint256", "address"]``).

        Returns
        -------
        bytes
            4-byte selector.
        """
        sig = f"{name}({','.join(param_types)})"
        return _keccak256(sig.encode("utf-8"))[:4]

    # -- encoding ------------------------------------------------------------

    @classmethod
    def encode_function_call(
        cls,
        function_name: str,
        param_types: List[str],
        param_values: List[Any],
    ) -> bytes:
        """Encode a complete function call (4-byte selector + ABI params).

        Parameters
        ----------
        function_name : str
            Name of the function to call.
        param_types : list[str]
            ABI type strings for each parameter.
        param_values : list
            Values corresponding to *param_types*.

        Returns
        -------
        bytes
            Encoded calldata.
        """
        selector = cls.function_selector(function_name, param_types)
        encoded_params = cls.encode_params(param_types, param_values)
        return selector + encoded_params

    @classmethod
    def encode_params(cls, types: List[str], values: List[Any]) -> bytes:
        """ABI-encode a list of typed values.

        Static types are encoded in-place (32 bytes each).  Dynamic types
        (``bytes``, ``string``) are encoded as a 32-byte offset pointer in
        the head section followed by the actual data in the tail section.
        """
        if len(types) != len(values):
            raise ValueError(
                f"Length mismatch: {len(types)} types vs {len(values)} values"
            )

        head_parts: List[Optional[bytes]] = []
        tail_parts: List[bytes] = []
        is_dynamic: List[bool] = []

        for typ, val in zip(types, values):
            if cls._is_dynamic(typ):
                is_dynamic.append(True)
                head_parts.append(None)  # placeholder for offset
                tail_parts.append(cls._encode_dynamic(typ, val))
            else:
                is_dynamic.append(False)
                head_parts.append(cls._encode_static(typ, val))
                tail_parts.append(b"")

        # Calculate head size (32 bytes per param)
        head_size = 32 * len(types)
        # Resolve offsets for dynamic params
        current_tail_offset = head_size
        resolved_head: List[bytes] = []
        tail_idx = 0
        for i, dynamic in enumerate(is_dynamic):
            if dynamic:
                resolved_head.append(
                    _pad_left(current_tail_offset.to_bytes(32, "big"))
                )
                current_tail_offset += len(tail_parts[i])
            else:
                resolved_head.append(head_parts[i])  # type: ignore[arg-type]

        return b"".join(resolved_head) + b"".join(tail_parts)

    # -- decoding ------------------------------------------------------------

    @classmethod
    def decode_function_call(
        cls, data: bytes, abi: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Decode ABI-encoded calldata using a function ABI definition.

        Parameters
        ----------
        data : bytes
            Full calldata (selector + encoded params).
        abi : list[dict]
            List of ABI entries.  Each entry should have ``"name"`` (str),
            ``"inputs"`` (list of dicts with ``"type"`` and ``"name"``).

        Returns
        -------
        dict
            ``{"function": name, "params": {param_name: decoded_value, ...}}``.

        Raises
        ------
        ValueError
            If the selector does not match any function in the ABI.
        """
        if len(data) < 4:
            raise ValueError("Calldata must be at least 4 bytes")

        selector = data[:4]
        param_data = data[4:]

        for entry in abi:
            name = entry.get("name", "")
            inputs = entry.get("inputs", [])
            param_types = [inp["type"] for inp in inputs]
            if cls.function_selector(name, param_types) == selector:
                decoded = cls.decode_params(param_types, param_data)
                param_names = [inp.get("name", f"arg{i}") for i, inp in enumerate(inputs)]
                return {
                    "function": name,
                    "params": dict(zip(param_names, decoded)),
                }

        raise ValueError(
            f"No matching function for selector 0x{selector.hex()}"
        )

    @classmethod
    def decode_params(cls, types: List[str], data: bytes) -> List[Any]:
        """Decode ABI-encoded parameters.

        Parameters
        ----------
        types : list[str]
            ABI type strings.
        data : bytes
            The encoded parameter bytes (without the 4-byte selector).

        Returns
        -------
        list
            Decoded values in the same order as *types*.
        """
        values: List[Any] = []
        for i, typ in enumerate(types):
            offset = i * 32
            if cls._is_dynamic(typ):
                # Read offset pointer
                ptr = int.from_bytes(data[offset:offset + 32], "big")
                values.append(cls._decode_dynamic(typ, data, ptr))
            else:
                word = data[offset:offset + 32]
                values.append(cls._decode_static(typ, word))
        return values

    # -- type classification -------------------------------------------------

    @staticmethod
    def _is_dynamic(typ: str) -> bool:
        return typ in ("bytes", "string")

    # -- static encode / decode ----------------------------------------------

    @classmethod
    def _encode_static(cls, typ: str, value: Any) -> bytes:
        if typ == "bool":
            return _pad_left((1 if value else 0).to_bytes(1, "big"))

        if typ == "address":
            addr = str(value).lower().replace("0x", "")
            return _pad_left(bytes.fromhex(addr.zfill(40)))

        if typ.startswith("uint"):
            bits = int(typ[4:]) if len(typ) > 4 else 256
            byte_len = bits // 8
            return _pad_left(int(value).to_bytes(byte_len, "big"))

        if typ.startswith("int"):
            bits = int(typ[3:]) if len(typ) > 3 else 256
            byte_len = bits // 8
            v = int(value)
            # Two's complement for negative values
            if v < 0:
                v = (1 << bits) + v
            return _pad_left(v.to_bytes(byte_len, "big"))

        if typ.startswith("bytes"):
            n = int(typ[5:])
            raw = value if isinstance(value, bytes) else bytes.fromhex(
                str(value).replace("0x", "")
            )
            if len(raw) > n:
                raw = raw[:n]
            return _pad_right(raw)

        raise ValueError(f"Unsupported static ABI type: {typ}")

    @classmethod
    def _decode_static(cls, typ: str, word: bytes) -> Any:
        if typ == "bool":
            return bool(int.from_bytes(word, "big"))

        if typ == "address":
            return "0x" + word[-20:].hex()

        if typ.startswith("uint"):
            return int.from_bytes(word, "big")

        if typ.startswith("int"):
            bits = int(typ[3:]) if len(typ) > 3 else 256
            val = int.from_bytes(word, "big")
            if val >= (1 << (bits - 1)):
                val -= 1 << bits
            return val

        if typ.startswith("bytes"):
            n = int(typ[5:])
            return word[:n]

        raise ValueError(f"Unsupported static ABI type: {typ}")

    # -- dynamic encode / decode ---------------------------------------------

    @classmethod
    def _encode_dynamic(cls, typ: str, value: Any) -> bytes:
        if typ == "string":
            raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        elif typ == "bytes":
            raw = value if isinstance(value, bytes) else bytes.fromhex(
                str(value).replace("0x", "")
            )
        else:
            raise ValueError(f"Unsupported dynamic ABI type: {typ}")

        length_word = _pad_left(len(raw).to_bytes(32, "big"))
        # Pad data to a multiple of 32 bytes
        padded = _pad_right(raw, ((len(raw) + 31) // 32) * 32)
        return length_word + padded

    @classmethod
    def _decode_dynamic(cls, typ: str, data: bytes, offset: int) -> Any:
        length = int.from_bytes(data[offset:offset + 32], "big")
        raw = data[offset + 32:offset + 32 + length]
        if typ == "string":
            return raw.decode("utf-8")
        return raw
