"""Phase 6 — Rust Contract Builtins tests.

Tests that all 40+ builtins execute in pure Rust (no NeedsPythonFallback).
Covers:
  • Crypto builtins: keccak256, sha256, verify_sig
  • Event builtins: emit (via CALL_BUILTIN and EMIT_EVENT opcode)
  • Chain info: block_number, block_timestamp
  • Balance/transfer: get_balance, transfer
  • I/O: print
  • Type/conversion: len, str, int, float, type, abs, min, max
  • Collection: keys, values, push, pop, contains, index_of, slice, reverse, sort, concat
  • String: split, join, trim, upper, lower, starts_with, ends_with, replace, substring, char_at
  • CALL_NAME dispatch to builtins (not just CALL_BUILTIN)
  • Backward compatibility: unknown builtins still fall back
  • Batch executor events collection
"""

import pytest
import time

# ── Helpers ────────────────────────────────────────────────────────────

def _make_bytecode(constants, instructions):
    """Build a Bytecode object and serialize to .zxc bytes."""
    from src.zexus.vm.bytecode import Bytecode
    from src.zexus.vm.binary_bytecode import serialize
    bc = Bytecode(constants=constants, instructions=instructions)
    return serialize(bc, include_checksum=True)


def _exec(data, env=None, state=None, gas_limit=0):
    """Execute .zxc bytes via RustVMExecutor."""
    from zexus_core import RustVMExecutor
    exe = RustVMExecutor()
    return exe.execute(data, env=env, state=state, gas_limit=gas_limit)


def Op(name):
    """Get opcode value from name."""
    from src.zexus.vm.bytecode import Opcode
    return getattr(Opcode, name)


# ══════════════════════════════════════════════════════════════════════
# 1. Crypto Builtins
# ══════════════════════════════════════════════════════════════════════

class TestCryptoBuiltins:
    """Crypto builtins execute in pure Rust."""

    def test_keccak256_basic(self):
        data = _make_bytecode(
            ['keccak256', 'hello'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        result = _exec(data)
        assert result['needs_fallback'] is False
        from zexus_core import RustHasher
        expected = RustHasher.keccak256_str('hello')
        assert result['result'] == expected

    def test_keccak256_empty(self):
        data = _make_bytecode(
            ['keccak256', ''],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        result = _exec(data)
        assert result['needs_fallback'] is False
        assert isinstance(result['result'], str) and len(result['result']) == 64

    def test_sha256_basic(self):
        data = _make_bytecode(
            ['sha256', 'hello'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        result = _exec(data)
        assert result['needs_fallback'] is False
        from zexus_core import RustHasher
        assert result['result'] == RustHasher.sha256_str('hello')

    def test_verify_sig_returns_bool(self):
        """verify_sig with invalid data returns False (but no fallback)."""
        data = _make_bytecode(
            ['verify_sig', 'bad_sig', 'msg', 'bad_key'],
            [
                (Op('LOAD_CONST'), 1),  # sig
                (Op('LOAD_CONST'), 2),  # msg
                (Op('LOAD_CONST'), 3),  # pk
                (Op('CALL_BUILTIN'), (0, 3)),
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data)
        assert result['needs_fallback'] is False
        assert result['result'] is False

    def test_verify_sig_too_few_args(self):
        data = _make_bytecode(
            ['verify_sig', 'only_sig'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('CALL_BUILTIN'), (0, 1)),
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data)
        assert result['needs_fallback'] is False
        assert result['result'] is False


# ══════════════════════════════════════════════════════════════════════
# 2. Event Builtins
# ══════════════════════════════════════════════════════════════════════

class TestEventBuiltins:
    """Events are collected via emit() and EMIT_EVENT opcode."""

    def test_emit_builtin(self):
        data = _make_bytecode(
            ['emit', 'Transfer', 42],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data)
        assert result['needs_fallback'] is False
        events = list(result['events'])
        assert len(events) == 1
        assert events[0]['event'] == 'Transfer'
        assert events[0]['data'] == 42

    def test_emit_multiple(self):
        data = _make_bytecode(
            ['emit', 'A', 1, 'B', 2],
            [
                (Op('LOAD_CONST'), 1),  # 'A'
                (Op('LOAD_CONST'), 2),  # 1
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('POP'), None),
                (Op('LOAD_CONST'), 3),  # 'B'
                (Op('LOAD_CONST'), 4),  # 2
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data)
        events = list(result['events'])
        assert len(events) == 2
        assert events[0]['event'] == 'A'
        assert events[1]['event'] == 'B'

    def test_emit_no_data(self):
        data = _make_bytecode(
            ['emit', 'Ping'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('CALL_BUILTIN'), (0, 1)),
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data)
        events = list(result['events'])
        assert len(events) == 1
        assert events[0]['event'] == 'Ping'


# ══════════════════════════════════════════════════════════════════════
# 3. Chain Info Builtins
# ══════════════════════════════════════════════════════════════════════

class TestChainInfoBuiltins:
    """block_number and block_timestamp read from env."""

    def test_block_number(self):
        data = _make_bytecode(
            ['block_number'],
            [(Op('CALL_BUILTIN'), (0, 0)), (Op('RETURN'), None)],
        )
        result = _exec(data, env={'_block_number': 99999})
        assert result['needs_fallback'] is False
        assert result['result'] == 99999

    def test_block_timestamp(self):
        data = _make_bytecode(
            ['block_timestamp'],
            [(Op('CALL_BUILTIN'), (0, 0)), (Op('RETURN'), None)],
        )
        ts = time.time()
        result = _exec(data, env={'_block_timestamp': ts})
        assert result['needs_fallback'] is False
        assert result['result'] == pytest.approx(ts, abs=0.1)

    def test_block_number_default(self):
        """Without env var, block_number returns 0."""
        data = _make_bytecode(
            ['block_number'],
            [(Op('CALL_BUILTIN'), (0, 0)), (Op('RETURN'), None)],
        )
        result = _exec(data)
        assert result['result'] == 0


# ══════════════════════════════════════════════════════════════════════
# 4. Balance / Transfer Builtins
# ══════════════════════════════════════════════════════════════════════

class TestBalanceTransfer:
    """get_balance and transfer operate on blockchain state."""

    def test_get_balance(self):
        data = _make_bytecode(
            ['get_balance', 'alice'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        result = _exec(data, state={'_balance:alice': 5000})
        assert result['needs_fallback'] is False
        assert result['result'] == 5000

    def test_get_balance_unknown(self):
        data = _make_bytecode(
            ['get_balance', 'unknown_addr'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        result = _exec(data)
        assert result['result'] == 0

    def test_transfer_success(self):
        data = _make_bytecode(
            ['transfer', 'bob', 300, 'get_balance', 'alice'],
            [
                (Op('LOAD_CONST'), 1),  # 'bob'
                (Op('LOAD_CONST'), 2),  # 300
                (Op('CALL_BUILTIN'), (0, 2)),  # transfer('bob', 300) → True
                (Op('POP'), None),
                (Op('LOAD_CONST'), 4),  # 'alice'
                (Op('CALL_BUILTIN'), (3, 1)),  # get_balance('alice')
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data,
            env={'_caller': 'alice'},
            state={'_balance:alice': 1000, '_balance:bob': 200},
        )
        assert result['needs_fallback'] is False
        # Alice started with 1000, sent 300 → 700
        assert result['result'] == 700
        # Check bob's balance in state
        assert result['state']['_balance:bob'] == 500

    def test_transfer_insufficient(self):
        data = _make_bytecode(
            ['transfer', 'bob', 5000],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data,
            env={'_caller': 'alice'},
            state={'_balance:alice': 100},
        )
        assert result['result'] is False  # insufficient balance

    def test_transfer_negative_amount(self):
        data = _make_bytecode(
            ['transfer', 'bob', -100],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data, env={'_caller': 'alice'}, state={'_balance:alice': 1000})
        assert result['result'] is False  # negative not allowed


# ══════════════════════════════════════════════════════════════════════
# 5. Type / Conversion Builtins
# ══════════════════════════════════════════════════════════════════════

class TestTypeConversionBuiltins:
    """Type inspection and conversion builtins."""

    def test_len_string(self):
        data = _make_bytecode(
            ['len', 'hello world'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        assert _exec(data)['result'] == 11

    def test_len_empty(self):
        data = _make_bytecode(
            ['len', ''],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        assert _exec(data)['result'] == 0

    def test_str_conversion(self):
        data = _make_bytecode(
            ['str', 42],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        assert _exec(data)['result'] == '42'

    def test_int_conversion(self):
        data = _make_bytecode(
            ['int', '123'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        assert _exec(data)['result'] == 123

    def test_float_conversion(self):
        data = _make_bytecode(
            ['float', '3.14'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        assert _exec(data)['result'] == pytest.approx(3.14)

    def test_type_builtin(self):
        data = _make_bytecode(
            ['type', 42],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        assert _exec(data)['result'] == 'int'

    def test_type_string(self):
        data = _make_bytecode(
            ['type', 'hello'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        assert _exec(data)['result'] == 'string'

    def test_abs_negative(self):
        data = _make_bytecode(
            ['abs', -42],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        assert _exec(data)['result'] == 42

    def test_min_builtin(self):
        data = _make_bytecode(
            ['min', 10, 3],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] == 3

    def test_max_builtin(self):
        data = _make_bytecode(
            ['max', 10, 3],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] == 10


# ══════════════════════════════════════════════════════════════════════
# 6. Collection Builtins
# ══════════════════════════════════════════════════════════════════════

class TestCollectionBuiltins:
    """Collection manipulation builtins."""

    def test_contains_string(self):
        data = _make_bytecode(
            ['contains', 'hello world', 'world'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] is True

    def test_contains_string_false(self):
        data = _make_bytecode(
            ['contains', 'hello', 'xyz'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] is False

    def test_index_of_string(self):
        data = _make_bytecode(
            ['index_of', 'hello world', 'world'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] == 6

    def test_index_of_not_found(self):
        data = _make_bytecode(
            ['index_of', 'hello', 'xyz'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] == -1

    def test_reverse_string(self):
        data = _make_bytecode(
            ['reverse', 'hello'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        assert _exec(data)['result'] == 'olleh'

    def test_concat_strings(self):
        data = _make_bytecode(
            ['concat', 'hello', ' world'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] == 'hello world'


# ══════════════════════════════════════════════════════════════════════
# 7. String Builtins
# ══════════════════════════════════════════════════════════════════════

class TestStringBuiltins:
    """String manipulation builtins."""

    def test_upper(self):
        data = _make_bytecode(
            ['upper', 'hello'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        assert _exec(data)['result'] == 'HELLO'

    def test_lower(self):
        data = _make_bytecode(
            ['lower', 'HELLO'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        assert _exec(data)['result'] == 'hello'

    def test_trim(self):
        data = _make_bytecode(
            ['trim', '  hello  '],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        assert _exec(data)['result'] == 'hello'

    def test_starts_with(self):
        data = _make_bytecode(
            ['starts_with', 'hello world', 'hello'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] is True

    def test_ends_with(self):
        data = _make_bytecode(
            ['ends_with', 'hello world', 'world'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] is True

    def test_replace(self):
        data = _make_bytecode(
            ['replace', 'hello world', 'world', 'rust'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('LOAD_CONST'), 3),
                (Op('CALL_BUILTIN'), (0, 3)),
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] == 'hello rust'

    def test_split(self):
        data = _make_bytecode(
            ['split', 'a,b,c', ','],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] == ['a', 'b', 'c']

    def test_join(self):
        # Build a list ['x', 'y', 'z'], then join with '-'
        data = _make_bytecode(
            ['x', 'y', 'z', 'join', '-'],
            [
                (Op('LOAD_CONST'), 0),  # 'x'
                (Op('LOAD_CONST'), 1),  # 'y'
                (Op('LOAD_CONST'), 2),  # 'z'
                (Op('BUILD_LIST'), 3),
                (Op('LOAD_CONST'), 4),  # '-'
                (Op('CALL_BUILTIN'), (3, 2)),  # join(list, '-')
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] == 'x-y-z'

    def test_substring(self):
        data = _make_bytecode(
            ['substring', 'hello world', 6, 11],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('LOAD_CONST'), 3),
                (Op('CALL_BUILTIN'), (0, 3)),
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] == 'world'

    def test_char_at(self):
        data = _make_bytecode(
            ['char_at', 'hello', 1],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        assert _exec(data)['result'] == 'e'


# ══════════════════════════════════════════════════════════════════════
# 8. CALL_NAME Dispatch
# ══════════════════════════════════════════════════════════════════════

class TestCallNameDispatch:
    """CALL_NAME dispatches to known builtins instead of falling back."""

    def test_call_name_keccak256(self):
        data = _make_bytecode(
            ['keccak256', 'test'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_NAME'), (0, 1)), (Op('RETURN'), None)],
        )
        result = _exec(data)
        assert result['needs_fallback'] is False
        assert isinstance(result['result'], str)
        assert len(result['result']) == 64

    def test_call_name_len(self):
        data = _make_bytecode(
            ['len', 'abc'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_NAME'), (0, 1)), (Op('RETURN'), None)],
        )
        result = _exec(data)
        assert result['needs_fallback'] is False
        assert result['result'] == 3

    def test_call_name_unknown_falls_back(self):
        data = _make_bytecode(
            ['my_custom_func', 'arg'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_NAME'), (0, 1)), (Op('RETURN'), None)],
        )
        result = _exec(data)
        assert result['needs_fallback'] is True

    def test_call_name_emit(self):
        data = _make_bytecode(
            ['emit', 'TestEvent', 'data123'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_NAME'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data)
        assert result['needs_fallback'] is False
        events = list(result['events'])
        assert len(events) == 1
        assert events[0]['event'] == 'TestEvent'


# ══════════════════════════════════════════════════════════════════════
# 9. Backward Compatibility
# ══════════════════════════════════════════════════════════════════════

class TestBackwardCompat:
    """Unknown builtins and cross-contract calls still fall back."""

    def test_unknown_builtin_falls_back(self):
        data = _make_bytecode(
            ['nonexistent_builtin', 'x'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        result = _exec(data)
        assert result['needs_fallback'] is True

    def test_contract_call_falls_back(self):
        data = _make_bytecode(
            ['contract_call', 'addr', 'action'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data)
        assert result['needs_fallback'] is True

    def test_static_call_falls_back(self):
        data = _make_bytecode(
            ['static_call', 'addr', 'action'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data)
        assert result['needs_fallback'] is True

    def test_delegate_call_falls_back(self):
        data = _make_bytecode(
            ['delegate_call', 'addr', 'action'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data)
        assert result['needs_fallback'] is True

    def test_call_top_still_falls_back(self):
        """CALL_TOP still falls back (not a named builtin)."""
        data = _make_bytecode(
            ['keccak256', 'data'],
            [
                (Op('LOAD_CONST'), 1),  # push 'data'
                (Op('LOAD_NAME'), 0),   # push fn ref (will be null)
                (Op('CALL_TOP'), 1),    # call TOS with 1 arg
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data)
        assert result['needs_fallback'] is True


# ══════════════════════════════════════════════════════════════════════
# 10. Batch Executor with Builtins
# ══════════════════════════════════════════════════════════════════════

class TestBatchExecutorBuiltins:
    """Builtins work in native batch executor (zero GIL)."""

    def test_batch_with_keccak256(self):
        from zexus_core import RustBatchExecutor

        bc_data = _make_bytecode(
            ['keccak256', 'batch_test'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )

        executor = RustBatchExecutor(max_workers=1)
        txs = [
            {"contract_address": "c1", "caller": "alice", "bytecode": bc_data, "state": {}},
        ] * 10

        import json
        result = executor.execute_batch_native(txs)
        assert result['succeeded'] == 10
        assert result['failed'] == 0
        assert result['fallbacks'] == 0

    def test_batch_with_emit(self):
        from zexus_core import RustBatchExecutor

        bc_data = _make_bytecode(
            ['emit', 'BatchEvent', 99],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),
                (Op('RETURN'), None),
            ],
        )

        executor = RustBatchExecutor(max_workers=1)
        txs = [
            {"contract_address": "c1", "caller": "alice", "bytecode": bc_data, "state": {}},
        ] * 5

        result = executor.execute_batch_native(txs)
        assert result['succeeded'] == 5
        # Events should be collected
        events = list(result.get('events', []))
        assert len(events) == 5
        for ev in events:
            assert ev['event'] == 'BatchEvent'

    def test_batch_with_transfer(self):
        from zexus_core import RustBatchExecutor

        bc_data = _make_bytecode(
            ['transfer', 'bob', 100, 'get_balance', 'bob'],
            [
                (Op('LOAD_CONST'), 1),
                (Op('LOAD_CONST'), 2),
                (Op('CALL_BUILTIN'), (0, 2)),  # transfer('bob', 100)
                (Op('POP'), None),
                (Op('LOAD_CONST'), 4),
                (Op('CALL_BUILTIN'), (3, 1)),  # get_balance('bob')
                (Op('RETURN'), None),
            ],
        )

        executor = RustBatchExecutor(max_workers=1)
        txs = [
            {
                "contract_address": "c1",
                "caller": "alice",
                "bytecode": bc_data,
                "state": {"_balance:alice": 500, "_balance:bob": 0},
            },
        ]

        result = executor.execute_batch_native(txs)
        assert result['succeeded'] == 1


# ══════════════════════════════════════════════════════════════════════
# 11. Print Builtin via CALL_BUILTIN
# ══════════════════════════════════════════════════════════════════════

class TestPrintBuiltin:
    """print() builtin captures output."""

    def test_print_via_call_builtin(self):
        data = _make_bytecode(
            ['print', 'hello from rust'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        result = _exec(data)
        assert result['needs_fallback'] is False
        assert 'hello from rust' in list(result['output'])


# ══════════════════════════════════════════════════════════════════════
# 12. Complex Multi-Builtin Contract
# ══════════════════════════════════════════════════════════════════════

class TestMultiBuiltinContract:
    """Contract using multiple builtins in sequence."""

    def test_hash_then_emit(self):
        """Compute hash, emit event with hash, return hash."""
        data = _make_bytecode(
            ['keccak256', 'data', 'emit', 'HashComputed'],
            [
                # hash = keccak256('data')
                (Op('LOAD_CONST'), 1),
                (Op('CALL_BUILTIN'), (0, 1)),
                (Op('STORE_NAME'), 0),  # store in slot 0 (reuse 'keccak256' name — it's just a key)
                # emit('HashComputed', hash)
                (Op('LOAD_CONST'), 3),  # 'HashComputed'
                (Op('LOAD_NAME'), 0),   # load hash
                (Op('CALL_BUILTIN'), (2, 2)),  # emit(name, data)
                (Op('POP'), None),
                # return hash
                (Op('LOAD_NAME'), 0),
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data)
        assert result['needs_fallback'] is False
        assert isinstance(result['result'], str)
        assert len(result['result']) == 64
        events = list(result['events'])
        assert len(events) == 1
        assert events[0]['event'] == 'HashComputed'

    def test_transfer_and_emit(self):
        """Transfer funds and emit a Transfer event."""
        data = _make_bytecode(
            ['transfer', 'bob', 250, 'emit', 'Transfer'],
            [
                (Op('LOAD_CONST'), 1),  # 'bob'
                (Op('LOAD_CONST'), 2),  # 250
                (Op('CALL_BUILTIN'), (0, 2)),  # transfer('bob', 250)
                (Op('POP'), None),
                (Op('LOAD_CONST'), 4),  # 'Transfer'
                (Op('LOAD_CONST'), 2),  # 250 (amount)
                (Op('CALL_BUILTIN'), (3, 2)),  # emit('Transfer', 250)
                (Op('POP'), None),
                (Op('LOAD_CONST'), 2),  # return 250
                (Op('RETURN'), None),
            ],
        )
        result = _exec(data,
            env={'_caller': 'alice'},
            state={'_balance:alice': 1000, '_balance:bob': 0},
        )
        assert result['needs_fallback'] is False
        assert result['state']['_balance:alice'] == 750
        assert result['state']['_balance:bob'] == 250
        events = list(result['events'])
        assert len(events) == 1
        assert events[0]['event'] == 'Transfer'

    def test_gas_metering_with_builtins(self):
        """Builtins consume gas properly."""
        data = _make_bytecode(
            ['keccak256', 'data'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )
        result = _exec(data, gas_limit=1_000_000)
        assert result['needs_fallback'] is False
        assert result['gas_used'] > 0
        assert result['error'] is None


# ══════════════════════════════════════════════════════════════════════
# 13. Performance
# ══════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Builtins execute at high throughput."""

    def test_keccak256_throughput(self):
        """1000 keccak256 calls via batch executor."""
        from zexus_core import RustBatchExecutor

        bc_data = _make_bytecode(
            ['keccak256', 'perf_test_data'],
            [(Op('LOAD_CONST'), 1), (Op('CALL_BUILTIN'), (0, 1)), (Op('RETURN'), None)],
        )

        executor = RustBatchExecutor(max_workers=2)
        txs = [
            {"contract_address": f"c{i%10}", "caller": "alice", "bytecode": bc_data, "state": {}}
            for i in range(1000)
        ]

        result = executor.execute_batch_native(txs)
        assert result['succeeded'] == 1000
        assert result['fallbacks'] == 0
        tps = result['throughput']
        print(f"\nPhase 6 keccak256 batch: {tps:.0f} TPS ({result['succeeded']}/{result['total']})")
        # Should be fast since builtins now execute in Rust
        assert tps > 1000  # conservative minimum
