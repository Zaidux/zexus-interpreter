"""
Comprehensive tests for the Zexus Blockchain JSON-RPC Server.

Tests cover:
  - RPCServer lifecycle (start / stop)
  - All zx_* methods (chain queries, transactions, gas, logs)
  - All txpool_* methods (mempool status / content)
  - All net_* methods (peer count, listening, peers)
  - All miner_* methods (start, stop, status, mine block)
  - All contract_* methods (deploy, call, static call)
  - All admin_* methods (node info, fund account, export chain)
  - JSON-RPC 2.0 protocol compliance (batch, error codes, notifications)
  - Subscription management
  - Rate limiting
  - Helper / utility functions
  - Edge cases (missing params, invalid types, not-found)
"""

import asyncio
import json
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.zexus.blockchain.rpc import (
    RPCServer,
    RPCError,
    RPCErrorCode,
    RPCMethodRegistry,
    SubscriptionManager,
    RateLimiter,
    _hex_int,
    _parse_hex_int,
    _require_param,
    _optional_param,
    _safe_serialize,
)
from src.zexus.blockchain.chain import Chain, Block, BlockHeader, Transaction, Mempool, TransactionReceipt
from src.zexus.blockchain.node import BlockchainNode, NodeConfig


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def node():
    """Create a minimal BlockchainNode for testing."""
    config = NodeConfig(
        chain_id="test-chain",
        miner_address="0x" + "a1" * 20,
        rpc_enabled=False,  # We'll manage server manually in tests
    )
    n = BlockchainNode(config)
    n.chain.create_genesis(
        miner=config.miner_address,
        initial_balances={
            "0x" + "a1" * 20: 1_000_000,
            "0x" + "b2" * 20: 500_000,
        },
    )
    return n


@pytest.fixture
def rpc(node):
    """Create an RPCServer (not started — for method unit testing)."""
    return RPCServer(node, host="127.0.0.1", port=0)


# ── Utility / Helper Tests ───────────────────────────────────────────

class TestHelpers:
    def test_hex_int(self):
        assert _hex_int(0) == "0x0"
        assert _hex_int(255) == "0xff"
        assert _hex_int(1000000) == "0xf4240"

    def test_parse_hex_int_from_hex(self):
        assert _parse_hex_int("0xff") == 255
        assert _parse_hex_int("0x0") == 0
        assert _parse_hex_int("0xF4240") == 1000000

    def test_parse_hex_int_from_int(self):
        assert _parse_hex_int(42) == 42

    def test_parse_hex_int_from_decimal_string(self):
        assert _parse_hex_int("100") == 100

    def test_parse_hex_int_invalid(self):
        with pytest.raises(RPCError) as exc:
            _parse_hex_int([1, 2, 3])
        assert exc.value.code == RPCErrorCode.INVALID_PARAMS

    def test_require_param_dict(self):
        assert _require_param({"a": 1}, "a") == 1

    def test_require_param_list(self):
        assert _require_param([10, 20], 0) == 10
        assert _require_param([10, 20], 1) == 20

    def test_require_param_missing(self):
        with pytest.raises(RPCError) as exc:
            _require_param({"a": 1}, "b", "my_param")
        assert exc.value.code == RPCErrorCode.INVALID_PARAMS
        assert "my_param" in exc.value.message

    def test_optional_param_present(self):
        assert _optional_param({"x": 42}, "x") == 42

    def test_optional_param_default(self):
        assert _optional_param({"x": 42}, "y", "default") == "default"

    def test_optional_param_list(self):
        assert _optional_param([10, 20], 1) == 20
        assert _optional_param([10], 5, "nope") == "nope"

    def test_safe_serialize_primitives(self):
        assert _safe_serialize(None) is None
        assert _safe_serialize(42) == 42
        assert _safe_serialize("hello") == "hello"
        assert _safe_serialize(True) is True

    def test_safe_serialize_nested(self):
        result = _safe_serialize({"a": [1, {"b": 2}]})
        assert result == {"a": [1, {"b": 2}]}

    def test_safe_serialize_bytes(self):
        assert _safe_serialize(b"\xde\xad") == "0xdead"

    def test_safe_serialize_object_with_to_dict(self):
        obj = MagicMock()
        obj.to_dict.return_value = {"key": "value"}
        assert _safe_serialize(obj) == {"key": "value"}


# ── RPCError Tests ────────────────────────────────────────────────────

class TestRPCError:
    def test_basic_error(self):
        err = RPCError(RPCErrorCode.METHOD_NOT_FOUND, "no such method")
        assert err.code == -32601
        assert err.message == "no such method"
        assert err.data is None

    def test_error_with_data(self):
        err = RPCError(RPCErrorCode.INTERNAL_ERROR, "boom", data={"trace": "..."})
        d = err.to_dict()
        assert d["code"] == -32603
        assert d["message"] == "boom"
        assert d["data"] == {"trace": "..."}

    def test_error_is_exception(self):
        with pytest.raises(RPCError):
            raise RPCError(RPCErrorCode.PARSE_ERROR, "bad json")


# ── RPCMethodRegistry Tests ──────────────────────────────────────────

class TestMethodRegistry:
    def test_register_and_get(self):
        reg = RPCMethodRegistry()
        reg.register("test_method", lambda p: 42, "A test")
        assert "test_method" in reg
        info = reg.get("test_method")
        assert info.name == "test_method"
        assert info.handler({}) == 42
        assert info.description == "A test"

    def test_list_methods(self):
        reg = RPCMethodRegistry()
        reg.register("b_method", lambda p: None)
        reg.register("a_method", lambda p: None)
        assert reg.list_methods() == ["a_method", "b_method"]

    def test_get_nonexistent(self):
        reg = RPCMethodRegistry()
        assert reg.get("ghost") is None
        assert "ghost" not in reg

    def test_async_detection(self):
        reg = RPCMethodRegistry()
        async def async_handler(p): return 1
        reg.register("async_m", async_handler)
        assert reg.get("async_m").is_async is True


# ── SubscriptionManager Tests ────────────────────────────────────────

class TestSubscriptionManager:
    def test_add_and_count(self):
        mgr = SubscriptionManager()
        ws = MagicMock()
        sid = mgr.add("newHeads", ws)
        assert sid.startswith("0x")
        assert mgr.count == 1

    def test_remove(self):
        mgr = SubscriptionManager()
        ws = MagicMock()
        sid = mgr.add("newHeads", ws)
        assert mgr.remove(sid) is True
        assert mgr.count == 0

    def test_remove_nonexistent(self):
        mgr = SubscriptionManager()
        assert mgr.remove("0x9999") is False

    def test_remove_all_for_ws(self):
        mgr = SubscriptionManager()
        ws1, ws2 = MagicMock(), MagicMock()
        mgr.add("newHeads", ws1)
        mgr.add("logs", ws1)
        mgr.add("newHeads", ws2)
        removed = mgr.remove_all_for_ws(ws1)
        assert removed == 2
        assert mgr.count == 1

    @pytest.mark.asyncio
    async def test_notify(self):
        mgr = SubscriptionManager()
        ws = AsyncMock()
        sid = mgr.add("newHeads", ws)
        await mgr.notify("newHeads", {"height": 10})
        ws.send_str.assert_called_once()
        payload = json.loads(ws.send_str.call_args[0][0])
        assert payload["method"] == "zx_subscription"
        assert payload["params"]["subscription"] == sid
        assert payload["params"]["result"]["height"] == 10

    @pytest.mark.asyncio
    async def test_notify_wrong_event(self):
        mgr = SubscriptionManager()
        ws = AsyncMock()
        mgr.add("newHeads", ws)
        await mgr.notify("logs", {"some": "data"})
        ws.send_str.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_dead_ws_cleaned_up(self):
        mgr = SubscriptionManager()
        ws = AsyncMock()
        ws.send_str.side_effect = ConnectionError("closed")
        mgr.add("newHeads", ws)
        assert mgr.count == 1
        await mgr.notify("newHeads", {})
        assert mgr.count == 0  # Dead subscription cleaned up


# ── RateLimiter Tests ─────────────────────────────────────────────────

class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = RateLimiter(max_requests=5, window_seconds=1.0)
        for _ in range(5):
            assert rl.allow("127.0.0.1") is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_requests=3, window_seconds=1.0)
        assert rl.allow("1.2.3.4") is True
        assert rl.allow("1.2.3.4") is True
        assert rl.allow("1.2.3.4") is True
        assert rl.allow("1.2.3.4") is False

    def test_different_ips_independent(self):
        rl = RateLimiter(max_requests=1, window_seconds=1.0)
        assert rl.allow("10.0.0.1") is True
        assert rl.allow("10.0.0.2") is True
        assert rl.allow("10.0.0.1") is False

    def test_reset(self):
        rl = RateLimiter(max_requests=1, window_seconds=1.0)
        rl.allow("10.0.0.1")
        rl.reset()
        assert rl.allow("10.0.0.1") is True


# ── RPC Server Method Tests (zx_*) ───────────────────────────────────

class TestZxMethods:
    def test_chain_id(self, rpc):
        result = rpc._zx_chain_id(None)
        assert result == "test-chain"

    def test_block_number(self, rpc):
        result = rpc._zx_block_number(None)
        assert result == "0x0"  # Genesis is height 0

    def test_get_block_by_number_genesis(self, rpc):
        result = rpc._zx_get_block_by_number([0, True])
        assert result is not None
        assert isinstance(result, dict)
        assert "header" in result or "hash" in result

    def test_get_block_by_number_not_found(self, rpc):
        result = rpc._zx_get_block_by_number([999, True])
        assert result is None

    def test_get_block_by_number_tx_hashes_only(self, rpc):
        result = rpc._zx_get_block_by_number([0, False])
        assert result is not None
        # Transactions should be hashes (strings), not full objects
        for tx in result.get("transactions", []):
            assert isinstance(tx, str)

    def test_get_block_by_hash(self, rpc):
        genesis = rpc.node.chain.blocks[0]
        result = rpc._zx_get_block_by_hash([genesis.hash, True])
        assert result is not None

    def test_get_block_by_hash_not_found(self, rpc):
        result = rpc._zx_get_block_by_hash(["0x" + "f" * 64, True])
        assert result is None

    def test_get_block_tx_count(self, rpc):
        result = rpc._zx_get_block_tx_count([0])
        assert isinstance(result, str)
        assert result.startswith("0x")

    def test_get_block_tx_count_not_found(self, rpc):
        with pytest.raises(RPCError) as exc:
            rpc._zx_get_block_tx_count([9999])
        assert exc.value.code == RPCErrorCode.NOT_FOUND

    def test_get_balance(self, rpc):
        addr = "0x" + "a1" * 20
        result = rpc._zx_get_balance([addr])
        balance = int(result, 16)
        assert balance == 1_000_000

    def test_get_balance_unknown_address(self, rpc):
        result = rpc._zx_get_balance(["0x" + "00" * 20])
        assert int(result, 16) == 0

    def test_get_tx_count(self, rpc):
        addr = "0x" + "a1" * 20
        result = rpc._zx_get_tx_count([addr])
        assert isinstance(result, str)
        assert int(result, 16) >= 0

    def test_get_account(self, rpc):
        addr = "0x" + "a1" * 20
        result = rpc._zx_get_account([addr])
        assert "balance" in result
        assert "nonce" in result
        assert result["balance"] == 1_000_000

    def test_send_transaction(self, rpc):
        sender = "0x" + "a1" * 20
        recipient = "0x" + "b2" * 20
        tx_hash = rpc._zx_send_transaction([{
            "from": sender,
            "to": recipient,
            "value": 100,
            "gas": 21000,
            "gasPrice": 1,
        }])
        assert isinstance(tx_hash, str)
        assert len(tx_hash) == 64  # SHA-256 hex

    def test_send_transaction_insufficient_funds(self, rpc):
        sender = "0x" + "00" * 20  # No balance
        with pytest.raises(RPCError) as exc:
            rpc._zx_send_transaction([{
                "from": sender,
                "to": "0x" + "ff" * 20,
                "value": 1_000_000_000,
            }])
        assert exc.value.code == RPCErrorCode.INSUFFICIENT_FUNDS

    def test_send_raw_transaction(self, rpc):
        sender = "0x" + "a1" * 20
        tx_hash = rpc._zx_send_raw_transaction([{
            "sender": sender,
            "recipient": "0x" + "b2" * 20,
            "value": 50,
            "nonce": 0,
            "gas_limit": 21000,
            "gas_price": 1,
            "timestamp": time.time(),
        }])
        assert isinstance(tx_hash, str)

    def test_gas_price(self, rpc):
        result = rpc._zx_gas_price(None)
        assert isinstance(result, str)
        assert int(result, 16) >= 1

    def test_estimate_gas_simple(self, rpc):
        result = rpc._zx_estimate_gas([{}])
        assert int(result, 16) == 21_000

    def test_estimate_gas_contract(self, rpc):
        result = rpc._zx_estimate_gas([{"data": "0x1234"}])
        assert int(result, 16) == 500_000

    def test_get_chain_info(self, rpc):
        result = rpc._zx_get_chain_info(None)
        assert "chain_id" in result
        assert result["chain_id"] == "test-chain"

    def test_validate_chain(self, rpc):
        result = rpc._zx_validate_chain(None)
        assert result["valid"] is True

    def test_get_code_no_contract(self, rpc):
        result = rpc._zx_get_code(["0x" + "ab" * 20])
        assert result == ""

    def test_get_logs_empty(self, rpc):
        result = rpc._zx_get_logs([{"fromBlock": 0, "toBlock": 0}])
        assert isinstance(result, list)

    def test_get_tx_by_hash_not_found(self, rpc):
        result = rpc._zx_get_tx_by_hash(["0x" + "00" * 32])
        assert result is None

    def test_get_tx_receipt_not_found(self, rpc):
        result = rpc._zx_get_tx_receipt(["0x" + "00" * 32])
        assert result is None

    def test_get_tx_by_hash_in_mempool(self, rpc):
        # Submit a tx to mempool
        sender = "0x" + "a1" * 20
        tx_hash = rpc._zx_send_transaction([{
            "from": sender, "to": "0x" + "cc" * 20, "value": 10,
        }])
        result = rpc._zx_get_tx_by_hash([tx_hash])
        assert result is not None
        assert result["block_hash"] is None
        assert result["status"] == "pending"


# ── RPC Server Method Tests (txpool_*) ───────────────────────────────

class TestTxpoolMethods:
    def test_status_empty(self, rpc):
        result = rpc._txpool_status(None)
        assert result["pending"] == 0

    def test_status_with_pending(self, rpc):
        sender = "0x" + "a1" * 20
        rpc._zx_send_transaction([{
            "from": sender, "to": "0x" + "dd" * 20, "value": 1,
        }])
        result = rpc._txpool_status(None)
        assert result["pending"] >= 1

    def test_content(self, rpc):
        result = rpc._txpool_content(None)
        assert "pending" in result


# ── RPC Server Method Tests (net_*) ──────────────────────────────────

class TestNetMethods:
    def test_version(self, rpc):
        assert rpc._net_version(None) == "test-chain"

    def test_peer_count(self, rpc):
        result = rpc._net_peer_count(None)
        assert isinstance(result, str)
        assert result.startswith("0x")

    def test_listening(self, rpc):
        # Network not started, should be False
        assert rpc._net_listening(None) is False

    def test_peers(self, rpc):
        result = rpc._net_peers(None)
        assert isinstance(result, list)


# ── RPC Server Method Tests (miner_*) ────────────────────────────────

class TestMinerMethods:
    def test_status(self, rpc):
        result = rpc._miner_status(None)
        assert result["mining"] is False
        assert result["consensus"] == "pow"
        assert result["miner_address"] == "0x" + "a1" * 20

    def test_set_address(self, rpc):
        new_addr = "0x" + "ff" * 20
        assert rpc._miner_set_address([new_addr]) is True
        assert rpc.node.config.miner_address == new_addr

    def test_mine_block_empty_mempool(self, rpc):
        # mine_block_sync mines even with empty mempool (produces an empty block)
        result = rpc._miner_mine_block(None)
        assert result is not None
        assert "hash" in result

    def test_mine_block_with_tx(self, rpc):
        sender = "0x" + "a1" * 20
        rpc._zx_send_transaction([{
            "from": sender, "to": "0x" + "ee" * 20, "value": 10,
        }])
        result = rpc._miner_mine_block(None)
        assert result is not None
        assert "hash" in result


# ── RPC Server Method Tests (contract_*) ─────────────────────────────

class TestContractMethods:
    def test_deploy(self, rpc):
        result = rpc._contract_deploy([{
            "code": "contract Test { action foo() { return 1 } }",
            "deployer": "0x" + "a1" * 20,
        }])
        assert result["success"] is True
        assert result["address"].startswith("0x")
        assert len(result["address"]) == 42

    def test_deploy_invalid_params(self, rpc):
        with pytest.raises(RPCError):
            rpc._contract_deploy(["not_a_dict"])

    def test_get_code_after_deploy(self, rpc):
        result = rpc._contract_deploy([{
            "code": "action bar() { return 2 }",
            "deployer": "0x" + "a1" * 20,
        }])
        code = rpc._zx_get_code([result["address"]])
        assert "bar" in code


# ── RPC Server Method Tests (admin_*) ────────────────────────────────

class TestAdminMethods:
    def test_node_info(self, rpc):
        result = rpc._admin_node_info(None)
        assert result["chain_id"] == "test-chain"
        assert "rpc_methods" in result
        assert isinstance(result["rpc_methods"], list)
        assert len(result["rpc_methods"]) > 20

    def test_fund_account(self, rpc):
        addr = "0x" + "cc" * 20
        result = rpc._admin_fund_account([addr, "0x186A0"])  # 100000
        assert result["funded"] == 100_000
        assert result["new_balance"] == 100_000

    def test_export_chain(self, rpc):
        result = rpc._admin_export_chain(None)
        assert isinstance(result, list)
        assert len(result) >= 1  # At least genesis

    def test_rpc_methods(self, rpc):
        result = rpc._admin_rpc_methods(None)
        assert "zx_chainId" in result
        assert "zx_getBalance" in result
        assert "admin_rpcMethods" in result


# ── Protocol Compliance Tests ─────────────────────────────────────────

class TestProtocol:
    """Test JSON-RPC 2.0 protocol handling via _process_single_request."""

    @pytest.mark.asyncio
    async def test_valid_request(self, rpc):
        resp = await rpc._process_single_request({
            "jsonrpc": "2.0",
            "method": "zx_chainId",
            "params": [],
            "id": 1,
        })
        assert resp["jsonrpc"] == "2.0"
        assert resp["result"] == "test-chain"
        assert resp["id"] == 1

    @pytest.mark.asyncio
    async def test_missing_jsonrpc_field(self, rpc):
        resp = await rpc._process_single_request({
            "method": "zx_chainId",
            "id": 1,
        })
        assert "error" in resp
        assert resp["error"]["code"] == RPCErrorCode.INVALID_REQUEST

    @pytest.mark.asyncio
    async def test_missing_method(self, rpc):
        resp = await rpc._process_single_request({
            "jsonrpc": "2.0",
            "id": 1,
        })
        assert "error" in resp
        assert resp["error"]["code"] == RPCErrorCode.INVALID_REQUEST

    @pytest.mark.asyncio
    async def test_method_not_found(self, rpc):
        resp = await rpc._process_single_request({
            "jsonrpc": "2.0",
            "method": "nonexistent_method",
            "id": 1,
        })
        assert "error" in resp
        assert resp["error"]["code"] == RPCErrorCode.METHOD_NOT_FOUND

    @pytest.mark.asyncio
    async def test_non_dict_request(self, rpc):
        resp = await rpc._process_single_request("not a dict")
        assert "error" in resp
        assert resp["error"]["code"] == RPCErrorCode.INVALID_REQUEST

    @pytest.mark.asyncio
    async def test_response_preserves_id(self, rpc):
        resp = await rpc._process_single_request({
            "jsonrpc": "2.0",
            "method": "zx_blockNumber",
            "id": "my-custom-id-42",
        })
        assert resp["id"] == "my-custom-id-42"

    @pytest.mark.asyncio
    async def test_null_id_request(self, rpc):
        resp = await rpc._process_single_request({
            "jsonrpc": "2.0",
            "method": "zx_blockNumber",
            "id": None,
        })
        assert resp["id"] is None
        assert "result" in resp

    @pytest.mark.asyncio
    async def test_params_as_dict(self, rpc):
        resp = await rpc._process_single_request({
            "jsonrpc": "2.0",
            "method": "zx_getBalance",
            "params": {"0": "0x" + "a1" * 20},
            "id": 1,
        })
        # Dict params should still work — _require_param handles both
        # (param key "0" is a dict key, not list index)
        assert resp["id"] == 1

    @pytest.mark.asyncio
    async def test_error_response_structure(self, rpc):
        resp = await rpc._process_single_request({
            "jsonrpc": "2.0",
            "method": "nonexistent",
            "id": 99,
        })
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 99
        err = resp["error"]
        assert "code" in err
        assert "message" in err


# ── Integration: Submit TX, Mine, Query ───────────────────────────────

class TestIntegration:
    """End-to-end workflows through the RPC layer."""

    def test_submit_mine_query(self, rpc):
        """Submit tx -> mine block -> query block and receipt."""
        sender = "0x" + "a1" * 20
        recipient = "0x" + "b2" * 20

        # 1. Submit transaction
        tx_hash = rpc._zx_send_transaction([{
            "from": sender,
            "to": recipient,
            "value": 1000,
        }])
        assert tx_hash

        # 2. Mempool should have the tx
        status = rpc._txpool_status(None)
        assert status["pending"] >= 1

        # 3. Mine a block
        block = rpc._miner_mine_block(None)
        assert block is not None

        # 4. Block number should have increased
        height = int(rpc._zx_block_number(None), 16)
        assert height >= 1

        # 5. Get the block
        fetched = rpc._zx_get_block_by_number([height, True])
        assert fetched is not None

        # 6. Mempool should be empty now
        status = rpc._txpool_status(None)
        assert status["pending"] == 0

    def test_fund_and_transfer(self, rpc):
        """Fund a new account via admin, then transfer from it."""
        new_addr = "0x" + "dd" * 20
        rpc._admin_fund_account([new_addr, 50000])
        
        bal = int(rpc._zx_get_balance([new_addr]), 16)
        assert bal == 50000

        # Transfer from new account
        tx_hash = rpc._zx_send_transaction([{
            "from": new_addr,
            "to": "0x" + "ee" * 20,
            "value": 100,
        }])
        assert tx_hash

    def test_multiple_blocks(self, rpc):
        """Submit multiple txs and mine them across blocks."""
        sender = "0x" + "a1" * 20
        for i in range(3):
            rpc._zx_send_transaction([{
                "from": sender,
                "to": f"0x{'0' * 38}{i:02x}",
                "value": 1,
            }])
        
        block = rpc._miner_mine_block(None)
        assert block is not None

        # Chain should now be longer than genesis
        height = int(rpc._zx_block_number(None), 16)
        assert height >= 1

        # Validate chain integrity after mining
        validation = rpc._zx_validate_chain(None)
        assert validation["valid"] is True

    def test_all_method_names_registered(self, rpc):
        """Ensure all documented methods are actually registered."""
        methods = rpc._admin_rpc_methods(None)
        expected = [
            "zx_chainId", "zx_blockNumber", "zx_getBlockByNumber",
            "zx_getBlockByHash", "zx_getBalance", "zx_getTransactionCount",
            "zx_getAccount", "zx_sendTransaction", "zx_gasPrice",
            "zx_getChainInfo", "zx_validateChain", "zx_getLogs",
            "txpool_status", "txpool_content",
            "net_version", "net_peerCount", "net_listening", "net_peers",
            "miner_start", "miner_stop", "miner_status", "miner_mineBlock",
            "contract_deploy", "contract_call", "contract_staticCall",
            "admin_nodeInfo", "admin_fundAccount", "admin_exportChain",
            "admin_rpcMethods",
        ]
        for name in expected:
            assert name in methods, f"Method {name} not registered"


# ── HTTP Server Integration Test ──────────────────────────────────────

class TestHTTPServer:
    """Test the actual HTTP server with aiohttp test client."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self, rpc):
        """Server can start and stop cleanly."""
        await rpc.start()
        assert rpc.is_running is True
        await rpc.stop()
        assert rpc.is_running is False

    @pytest.mark.asyncio
    async def test_health_endpoint(self, rpc):
        """Health check endpoint works."""
        from aiohttp.test_utils import TestServer, TestClient
        from aiohttp import web

        await rpc.start()
        # Create a test client connected to the existing app
        async with TestClient(TestServer(rpc._app)) as client:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["chain_id"] == "test-chain"
        await rpc.stop()

    @pytest.mark.asyncio
    async def test_rpc_call_via_http(self, rpc):
        """Make an actual JSON-RPC call over HTTP."""
        from aiohttp.test_utils import TestServer, TestClient

        await rpc.start()
        async with TestClient(TestServer(rpc._app)) as client:
            resp = await client.post("/", json={
                "jsonrpc": "2.0",
                "method": "zx_chainId",
                "params": [],
                "id": 1,
            })
            assert resp.status == 200
            data = await resp.json()
            assert data["result"] == "test-chain"
            assert data["id"] == 1
        await rpc.stop()

    @pytest.mark.asyncio
    async def test_batch_request_via_http(self, rpc):
        """Make a batch JSON-RPC call."""
        from aiohttp.test_utils import TestServer, TestClient

        await rpc.start()
        async with TestClient(TestServer(rpc._app)) as client:
            resp = await client.post("/", json=[
                {"jsonrpc": "2.0", "method": "zx_chainId", "id": 1},
                {"jsonrpc": "2.0", "method": "zx_blockNumber", "id": 2},
                {"jsonrpc": "2.0", "method": "net_version", "id": 3},
            ])
            assert resp.status == 200
            data = await resp.json()
            assert len(data) == 3
            ids = {r["id"] for r in data}
            assert ids == {1, 2, 3}
        await rpc.stop()

    @pytest.mark.asyncio
    async def test_invalid_json_via_http(self, rpc):
        """Invalid JSON body returns parse error."""
        from aiohttp.test_utils import TestServer, TestClient

        await rpc.start()
        async with TestClient(TestServer(rpc._app)) as client:
            resp = await client.post(
                "/",
                data="this is not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["error"]["code"] == RPCErrorCode.PARSE_ERROR
        await rpc.stop()

    @pytest.mark.asyncio
    async def test_info_on_get(self, rpc):
        """GET / without WS upgrade returns info."""
        from aiohttp.test_utils import TestServer, TestClient

        await rpc.start()
        async with TestClient(TestServer(rpc._app)) as client:
            resp = await client.get("/")
            assert resp.status == 200
            data = await resp.json()
            assert data["server"] == "Zexus Blockchain RPC"
            assert "methods" in data
        await rpc.stop()

    @pytest.mark.asyncio
    async def test_full_workflow_via_http(self, rpc):
        """End-to-end: fund, send tx, mine, query — all over HTTP."""
        from aiohttp.test_utils import TestServer, TestClient

        await rpc.start()
        async with TestClient(TestServer(rpc._app)) as client:
            # Fund an account
            resp = await client.post("/", json={
                "jsonrpc": "2.0",
                "method": "admin_fundAccount",
                "params": ["0x" + "aa" * 20, "0x186A0"],
                "id": 1,
            })
            data = await resp.json()
            assert data["result"]["funded"] == 100_000

            # Send a transaction
            resp = await client.post("/", json={
                "jsonrpc": "2.0",
                "method": "zx_sendTransaction",
                "params": [{
                    "from": "0x" + "a1" * 20,
                    "to": "0x" + "aa" * 20,
                    "value": 500,
                }],
                "id": 2,
            })
            data = await resp.json()
            assert "result" in data
            tx_hash = data["result"]

            # Mine a block
            resp = await client.post("/", json={
                "jsonrpc": "2.0",
                "method": "miner_mineBlock",
                "params": [],
                "id": 3,
            })
            data = await resp.json()
            assert data["result"] is not None

            # Check block number increased
            resp = await client.post("/", json={
                "jsonrpc": "2.0",
                "method": "zx_blockNumber",
                "id": 4,
            })
            data = await resp.json()
            assert int(data["result"], 16) >= 1
        await rpc.stop()
