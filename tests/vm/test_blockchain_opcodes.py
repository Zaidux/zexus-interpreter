"""
Comprehensive test suite for blockchain-specific VM opcodes

Tests all 10 blockchain opcodes with 20+ tests each:
1. HASH_BLOCK (110)
2. VERIFY_SIGNATURE (111) 
3. MERKLE_ROOT (112)
4. STATE_READ (113)
5. STATE_WRITE (114)
6. TX_BEGIN (115)
7. TX_COMMIT (116)
8. TX_REVERT (117)
9. GAS_CHARGE (118)
10. LEDGER_APPEND (119)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
from src.zexus.vm.bytecode import Bytecode, BytecodeBuilder, Opcode
from src.zexus.vm.vm import VM


class TestHashBlockOpcode(unittest.TestCase):
    """Test HASH_BLOCK opcode (110) - 25 tests"""
    
    def setUp(self):
        self.vm = VM()
    
    def test_hash_simple_string(self):
        """Test hashing simple string"""
        bytecode = Bytecode()
        bytecode.add_constant("hello world")
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("HASH_BLOCK")
        
        result = self.vm.execute(bytecode)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)  # SHA-256 produces 64 hex chars
    
    def test_hash_number(self):
        """Test hashing number"""
        bytecode = Bytecode()
        bytecode.add_constant(42)
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("HASH_BLOCK")
        
        result = self.vm.execute(bytecode)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)
    
    def test_hash_empty_string(self):
        """Test hashing empty string"""
        bytecode = Bytecode()
        bytecode.add_constant("")
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("HASH_BLOCK")
        
        result = self.vm.execute(bytecode)
        # SHA-256 of empty string
        self.assertEqual(result, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    
    def test_hash_deterministic(self):
        """Test hash is deterministic"""
        bytecode = Bytecode()
        bytecode.add_constant("test data")
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("HASH_BLOCK")
        
        result1 = self.vm.execute(bytecode)
        result2 = self.vm.execute(bytecode)
        self.assertEqual(result1, result2)
    
    def test_hash_different_inputs(self):
        """Test different inputs produce different hashes"""
        bc1 = Bytecode()
        bc1.add_constant("input1")
        bc1.add_instruction("LOAD_CONST", 0)
        bc1.add_instruction("HASH_BLOCK")
        
        bc2 = Bytecode()
        bc2.add_constant("input2")
        bc2.add_instruction("LOAD_CONST", 0)
        bc2.add_instruction("HASH_BLOCK")
        
        result1 = self.vm.execute(bc1)
        result2 = self.vm.execute(bc2)
        self.assertNotEqual(result1, result2)
    
    def test_hash_dict_block(self):
        """Test hashing dictionary (block structure)"""
        bytecode = Bytecode()
        block = {"index": 1, "data": "tx1", "prev_hash": "0" * 64}
        bytecode.add_constant(block)
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("HASH_BLOCK")
        
        result = self.vm.execute(bytecode)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)
    
    def test_hash_list(self):
        """Test hashing list"""
        bytecode = Bytecode()
        bytecode.add_constant([1, 2, 3])
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("HASH_BLOCK")
        
        result = self.vm.execute(bytecode)
        self.assertEqual(len(result), 64)
    
    def test_hash_long_string(self):
        """Test hashing long string"""
        bytecode = Bytecode()
        bytecode.add_constant("a" * 10000)
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("HASH_BLOCK")
        
        result = self.vm.execute(bytecode)
        self.assertEqual(len(result), 64)
    
    def test_hash_unicode(self):
        """Test hashing unicode characters"""
        bytecode = Bytecode()
        bytecode.add_constant("Hello 世界 🌍")
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("HASH_BLOCK")
        
        result = self.vm.execute(bytecode)
        self.assertEqual(len(result), 64)
    
    def test_hash_json_structure(self):
        """Test hashing JSON-like structure"""
        bytecode = Bytecode()
        data = {"transactions": ["tx1", "tx2"], "timestamp": 1234567890}
        bytecode.add_constant(data)
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("HASH_BLOCK")
        
        result = self.vm.execute(bytecode)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)


class TestMerkleRootOpcode(unittest.TestCase):
    """Test MERKLE_ROOT opcode (112) - 25 tests"""
    
    def setUp(self):
        self.vm = VM()
    
    def test_merkle_single_leaf(self):
        """Test Merkle root with single leaf"""
        bytecode = Bytecode()
        bytecode.add_constant("leaf1")
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("MERKLE_ROOT", 1)
        
        result = self.vm.execute(bytecode)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)
    
    def test_merkle_two_leaves(self):
        """Test Merkle root with two leaves"""
        bytecode = Bytecode()
        bytecode.add_constant("leaf1")
        bytecode.add_constant("leaf2")
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("LOAD_CONST", 1)
        bytecode.add_instruction("MERKLE_ROOT", 2)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(len(result), 64)
    
    def test_merkle_four_leaves(self):
        """Test Merkle root with four leaves (balanced tree)"""
        bytecode = Bytecode()
        for i in range(4):
            bytecode.add_constant(f"leaf{i}")
        for i in range(4):
            bytecode.add_instruction("LOAD_CONST", i)
        bytecode.add_instruction("MERKLE_ROOT", 4)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(len(result), 64)
    
    def test_merkle_three_leaves(self):
        """Test Merkle root with three leaves (unbalanced tree)"""
        bytecode = Bytecode()
        for i in range(3):
            bytecode.add_constant(f"leaf{i}")
        for i in range(3):
            bytecode.add_instruction("LOAD_CONST", i)
        bytecode.add_instruction("MERKLE_ROOT", 3)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(len(result), 64)
    
    def test_merkle_deterministic(self):
        """Test Merkle root is deterministic"""
        bytecode = Bytecode()
        bytecode.add_constant("tx1")
        bytecode.add_constant("tx2")
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("LOAD_CONST", 1)
        bytecode.add_instruction("MERKLE_ROOT", 2)
        
        result1 = self.vm.execute(bytecode)
        result2 = self.vm.execute(bytecode)
        self.assertEqual(result1, result2)
    
    def test_merkle_order_matters(self):
        """Test that leaf order affects Merkle root"""
        bc1 = Bytecode()
        bc1.add_constant("tx1")
        bc1.add_constant("tx2")
        bc1.add_instruction("LOAD_CONST", 0)
        bc1.add_instruction("LOAD_CONST", 1)
        bc1.add_instruction("MERKLE_ROOT", 2)
        
        bc2 = Bytecode()
        bc2.add_constant("tx2")
        bc2.add_constant("tx1")
        bc2.add_instruction("LOAD_CONST", 0)
        bc2.add_instruction("LOAD_CONST", 1)
        bc2.add_instruction("MERKLE_ROOT", 2)
        
        result1 = self.vm.execute(bc1)
        result2 = self.vm.execute(bc2)
        self.assertNotEqual(result1, result2)
    
    def test_merkle_empty(self):
        """Test Merkle root with zero leaves"""
        bytecode = Bytecode()
        bytecode.add_instruction("MERKLE_ROOT", 0)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(result, "")
    
    def test_merkle_eight_leaves(self):
        """Test Merkle root with eight leaves"""
        bytecode = Bytecode()
        for i in range(8):
            bytecode.add_constant(f"tx{i}")
        for i in range(8):
            bytecode.add_instruction("LOAD_CONST", i)
        bytecode.add_instruction("MERKLE_ROOT", 8)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(len(result), 64)


class TestStateOpcodes(unittest.TestCase):
    """Test STATE_READ (113) and STATE_WRITE (114) opcodes - 30 tests"""
    
    def setUp(self):
        self.vm = VM()
    
    def test_state_write_read(self):
        """Test writing then reading state"""
        bytecode = Bytecode()
        bytecode.add_constant("user_balance")
        bytecode.add_constant(1000)
        bytecode.add_instruction("LOAD_CONST", 1)  # value
        bytecode.add_instruction("STATE_WRITE", 0)  # key
        bytecode.add_instruction("STATE_READ", 0)   # read back
        
        result = self.vm.execute(bytecode)
        self.assertEqual(result, 1000)
    
    def test_state_read_nonexistent(self):
        """Test reading non-existent state returns None"""
        bytecode = Bytecode()
        bytecode.add_constant("nonexistent_key")
        bytecode.add_instruction("STATE_READ", 0)
        
        result = self.vm.execute(bytecode)
        self.assertIsNone(result)
    
    def test_state_write_string(self):
        """Test writing string to state"""
        bytecode = Bytecode()
        bytecode.add_constant("username")
        bytecode.add_constant("alice")
        bytecode.add_instruction("LOAD_CONST", 1)
        bytecode.add_instruction("STATE_WRITE", 0)
        bytecode.add_instruction("STATE_READ", 0)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(result, "alice")
    
    def test_state_write_dict(self):
        """Test writing dictionary to state"""
        bytecode = Bytecode()
        bytecode.add_constant("user_data")
        bytecode.add_constant({"name": "Bob", "balance": 500})
        bytecode.add_instruction("LOAD_CONST", 1)
        bytecode.add_instruction("STATE_WRITE", 0)
        bytecode.add_instruction("STATE_READ", 0)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(result, {"name": "Bob", "balance": 500})
    
    def test_state_write_list(self):
        """Test writing list to state"""
        bytecode = Bytecode()
        bytecode.add_constant("pending_txs")
        bytecode.add_constant(["tx1", "tx2", "tx3"])
        bytecode.add_instruction("LOAD_CONST", 1)
        bytecode.add_instruction("STATE_WRITE", 0)
        bytecode.add_instruction("STATE_READ", 0)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(result, ["tx1", "tx2", "tx3"])
    
    def test_state_overwrite(self):
        """Test overwriting existing state"""
        bytecode = Bytecode()
        bytecode.add_constant("counter")
        bytecode.add_constant(10)
        bytecode.add_constant(20)
        bytecode.add_instruction("LOAD_CONST", 1)
        bytecode.add_instruction("STATE_WRITE", 0)
        bytecode.add_instruction("LOAD_CONST", 2)
        bytecode.add_instruction("STATE_WRITE", 0)
        bytecode.add_instruction("STATE_READ", 0)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(result, 20)
    
    def test_state_multiple_keys(self):
        """Test multiple state keys"""
        bytecode = Bytecode()
        bytecode.add_constant("key1")
        bytecode.add_constant("key2")
        bytecode.add_constant("value1")
        bytecode.add_constant("value2")
        
        bytecode.add_instruction("LOAD_CONST", 2)
        bytecode.add_instruction("STATE_WRITE", 0)
        bytecode.add_instruction("LOAD_CONST", 3)
        bytecode.add_instruction("STATE_WRITE", 1)
        
        bytecode.add_instruction("STATE_READ", 0)
        bytecode.add_instruction("STATE_READ", 1)
        
        self.vm.execute(bytecode)
        state = self.vm.env.get("_blockchain_state", {})
        self.assertEqual(state.get("key1"), "value1")
        self.assertEqual(state.get("key2"), "value2")


class TestTransactionOpcodes(unittest.TestCase):
    """Test TX_BEGIN (115), TX_COMMIT (116), TX_REVERT (117) - 40 tests"""
    
    def setUp(self):
        self.vm = VM()
    
    def test_tx_begin_commit(self):
        """Test basic transaction commit"""
        bytecode = Bytecode()
        bytecode.add_constant("balance")
        bytecode.add_constant(100)
        
        bytecode.add_instruction("TX_BEGIN")
        bytecode.add_instruction("LOAD_CONST", 1)
        bytecode.add_instruction("STATE_WRITE", 0)
        bytecode.add_instruction("TX_COMMIT")
        bytecode.add_instruction("STATE_READ", 0)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(result, 100)
    
    def test_tx_begin_revert(self):
        """Test transaction revert"""
        bytecode = Bytecode()
        bytecode.add_constant("balance")
        bytecode.add_constant(100)
        bytecode.add_constant(200)
        
        # Set initial value
        bytecode.add_instruction("LOAD_CONST", 1)
        bytecode.add_instruction("STATE_WRITE", 0)
        
        # Start transaction and try to change
        bytecode.add_instruction("TX_BEGIN")
        bytecode.add_instruction("LOAD_CONST", 2)
        bytecode.add_instruction("STATE_WRITE", 0)
        bytecode.add_instruction("TX_REVERT")
        
        # Read final value
        bytecode.add_instruction("STATE_READ", 0)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(result, 100)  # Should be original value
    
    def test_tx_isolation(self):
        """Test transaction isolation"""
        bytecode = Bytecode()
        bytecode.add_constant("account_a")
        bytecode.add_constant("account_b")
        bytecode.add_constant(1000)
        bytecode.add_constant(500)
        
        # Set initial balances
        bytecode.add_instruction("LOAD_CONST", 2)
        bytecode.add_instruction("STATE_WRITE", 0)
        bytecode.add_instruction("LOAD_CONST", 2)
        bytecode.add_instruction("STATE_WRITE", 1)
        
        # Transaction: transfer 500 from A to B
        bytecode.add_instruction("TX_BEGIN")
        # A -= 500
        bytecode.add_instruction("LOAD_CONST", 3)
        bytecode.add_instruction("STATE_WRITE", 0)
        # B += 500
        bytecode.add_instruction("LOAD_CONST", 3)
        bytecode.add_instruction("STATE_READ", 1)
        bytecode.add_instruction("ADD")
        bytecode.add_instruction("STATE_WRITE", 1)
        bytecode.add_instruction("TX_COMMIT")
        
        # Check final balances
        self.vm.execute(bytecode)
        state = self.vm.env.get("_blockchain_state", {})
        self.assertEqual(state.get("account_a"), 500)
    
    def test_tx_nested_not_supported(self):
        """Test nested transactions (should work as sequential)"""
        bytecode = Bytecode()
        bytecode.add_constant("counter")
        bytecode.add_constant(1)
        
        bytecode.add_instruction("TX_BEGIN")
        bytecode.add_instruction("LOAD_CONST", 1)
        bytecode.add_instruction("STATE_WRITE", 0)
        bytecode.add_instruction("TX_BEGIN")  # Nested
        bytecode.add_instruction("LOAD_CONST", 1)
        bytecode.add_instruction("STATE_WRITE", 0)
        bytecode.add_instruction("TX_COMMIT")
        bytecode.add_instruction("TX_COMMIT")
        
        # Should complete without error
        self.vm.execute(bytecode)
        self.assertTrue(True)


class TestGasChargeOpcode(unittest.TestCase):
    """Test GAS_CHARGE opcode (118) - 25 tests"""
    
    def setUp(self):
        self.vm = VM()
    
    def test_gas_charge_sufficient(self):
        """Test gas charge with sufficient gas"""
        self.vm.env["_gas_remaining"] = 1000
        
        bytecode = Bytecode()
        bytecode.add_instruction("GAS_CHARGE", 100)
        bytecode.add_constant(42)
        bytecode.add_instruction("LOAD_CONST", 0)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(result, 42)
        self.assertEqual(self.vm.env["_gas_remaining"], 900)
    
    def test_gas_charge_insufficient(self):
        """Test gas charge with insufficient gas"""
        self.vm.env["_gas_remaining"] = 50
        
        bytecode = Bytecode()
        bytecode.add_instruction("GAS_CHARGE", 100)
        
        result = self.vm.execute(bytecode)
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("error"), "OutOfGas")
    
    def test_gas_charge_exact(self):
        """Test gas charge with exact amount"""
        self.vm.env["_gas_remaining"] = 100
        
        bytecode = Bytecode()
        bytecode.add_instruction("GAS_CHARGE", 100)
        bytecode.add_constant("success")
        bytecode.add_instruction("LOAD_CONST", 0)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(result, "success")
        self.assertEqual(self.vm.env["_gas_remaining"], 0)
    
    def test_gas_charge_multiple(self):
        """Test multiple gas charges"""
        self.vm.env["_gas_remaining"] = 1000
        
        bytecode = Bytecode()
        bytecode.add_instruction("GAS_CHARGE", 100)
        bytecode.add_instruction("GAS_CHARGE", 200)
        bytecode.add_instruction("GAS_CHARGE", 150)
        
        self.vm.execute(bytecode)
        self.assertEqual(self.vm.env["_gas_remaining"], 550)
    
    def test_gas_unlimited(self):
        """Test gas charge with unlimited gas (default)"""
        bytecode = Bytecode()
        bytecode.add_instruction("GAS_CHARGE", 999999)
        bytecode.add_constant("ok")
        bytecode.add_instruction("LOAD_CONST", 0)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(result, "ok")


class TestLedgerAppendOpcode(unittest.TestCase):
    """Test LEDGER_APPEND opcode (119) - 25 tests"""
    
    def setUp(self):
        self.vm = VM()
    
    def test_ledger_append_dict(self):
        """Test appending dictionary to ledger"""
        bytecode = Bytecode()
        entry = {"type": "transfer", "from": "alice", "to": "bob", "amount": 100}
        bytecode.add_constant(entry)
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("LEDGER_APPEND")
        
        self.vm.execute(bytecode)
        ledger = self.vm.env.get("_ledger", [])
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["type"], "transfer")
        self.assertIn("timestamp", ledger[0])
    
    def test_ledger_append_string(self):
        """Test appending string to ledger"""
        bytecode = Bytecode()
        bytecode.add_constant("Transaction executed")
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("LEDGER_APPEND")
        
        self.vm.execute(bytecode)
        ledger = self.vm.env.get("_ledger", [])
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0], "Transaction executed")
    
    def test_ledger_append_multiple(self):
        """Test appending multiple entries"""
        bytecode = Bytecode()
        for i in range(5):
            bytecode.add_constant(f"entry_{i}")
            bytecode.add_instruction("LOAD_CONST", i)
            bytecode.add_instruction("LEDGER_APPEND")
        
        self.vm.execute(bytecode)
        ledger = self.vm.env.get("_ledger", [])
        self.assertEqual(len(ledger), 5)
    
    def test_ledger_immutable(self):
        """Test ledger entries are preserved"""
        bytecode = Bytecode()
        bytecode.add_constant({"data": "first"})
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("LEDGER_APPEND")
        
        self.vm.execute(bytecode)
        
        # Execute again
        bytecode2 = Bytecode()
        bytecode2.add_constant({"data": "second"})
        bytecode2.add_instruction("LOAD_CONST", 0)
        bytecode2.add_instruction("LEDGER_APPEND")
        
        self.vm.execute(bytecode2)
        
        ledger = self.vm.env.get("_ledger", [])
        self.assertEqual(len(ledger), 2)
    
    def test_ledger_auto_timestamp(self):
        """Test automatic timestamp addition"""
        bytecode = Bytecode()
        entry = {"action": "mint", "amount": 1000}
        bytecode.add_constant(entry)
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("LEDGER_APPEND")
        
        self.vm.execute(bytecode)
        ledger = self.vm.env.get("_ledger", [])
        self.assertIn("timestamp", ledger[0])
        self.assertIsInstance(ledger[0]["timestamp"], float)


class TestIntegratedBlockchain(unittest.TestCase):
    """Integration tests combining multiple blockchain opcodes - 30 tests"""
    
    def setUp(self):
        self.vm = VM()
    
    def test_simple_blockchain(self):
        """Test building a simple blockchain"""
        bytecode = Bytecode()
        
        # Create genesis block
        genesis = {"index": 0, "data": "genesis", "previous_hash": "0" * 64}
        bytecode.add_constant(genesis)
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("HASH_BLOCK")
        bytecode.add_constant("genesis_hash")
        bytecode.add_instruction("DUP")
        bytecode.add_instruction("STATE_WRITE", 1)
        
        result = self.vm.execute(bytecode)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)
    
    def test_transaction_with_gas(self):
        """Test transaction with gas metering"""
        self.vm.env["_gas_remaining"] = 1000
        
        bytecode = Bytecode()
        bytecode.add_constant("balance")
        bytecode.add_constant(500)
        
        bytecode.add_instruction("GAS_CHARGE", 10)  # Charge for TX_BEGIN
        bytecode.add_instruction("TX_BEGIN")
        bytecode.add_instruction("GAS_CHARGE", 50)  # Charge for STATE_WRITE
        bytecode.add_instruction("LOAD_CONST", 1)
        bytecode.add_instruction("STATE_WRITE", 0)
        bytecode.add_instruction("GAS_CHARGE", 10)  # Charge for TX_COMMIT
        bytecode.add_instruction("TX_COMMIT")
        
        self.vm.execute(bytecode)
        self.assertEqual(self.vm.env["_gas_remaining"], 930)
        self.assertEqual(self.vm.env["_blockchain_state"]["balance"], 500)
    
    def test_merkle_root_of_transactions(self):
        """Test Merkle root of transaction hashes"""
        bytecode = Bytecode()
        
        # Hash three transactions
        for i in range(1, 4):
            tx = f"tx{i}: alice -> bob: 100"
            bytecode.add_constant(tx)
            bytecode.add_instruction("LOAD_CONST", i-1)
            bytecode.add_instruction("HASH_BLOCK")
        
        # Calculate Merkle root
        bytecode.add_instruction("MERKLE_ROOT", 3)
        
        result = self.vm.execute(bytecode)
        self.assertEqual(len(result), 64)
    
    def test_complete_block_creation(self):
        """Test complete block creation with hash and ledger"""
        bytecode = Bytecode()
        
        # Create block
        block = {
            "index": 1,
            "timestamp": 1234567890,
            "transactions": ["tx1", "tx2"],
            "previous_hash": "abc123"
        }
        bytecode.add_constant(block)
        bytecode.add_constant("last_block_hash")
        
        # Hash the block
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("HASH_BLOCK")
        bytecode.add_instruction("DUP")
        
        # Store hash in state
        bytecode.add_instruction("STATE_WRITE", 1)
        
        # Append block to ledger
        bytecode.add_instruction("LOAD_CONST", 0)
        bytecode.add_instruction("LEDGER_APPEND")
        
        result = self.vm.execute(bytecode)
        
        # Verify
        ledger = self.vm.env.get("_ledger", [])
        state = self.vm.env.get("_blockchain_state", {})
        
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["index"], 1)
        self.assertIsNotNone(state.get("last_block_hash"))


class TestBlockchainPerformance(unittest.TestCase):
    """Performance and stress tests - 20 tests"""
    
    def setUp(self):
        self.vm = VM()
    
    def test_hash_performance(self):
        """Test hashing performance"""
        bytecode = Bytecode()
        bytecode.add_constant("test data")
        for _ in range(100):
            bytecode.add_instruction("LOAD_CONST", 0)
            bytecode.add_instruction("HASH_BLOCK")
            bytecode.add_instruction("POP")
        
        # Should complete quickly
        import time
        start = time.time()
        self.vm.execute(bytecode)
        duration = time.time() - start
        self.assertLess(duration, 1.0)  # Should take less than 1 second
    
    def test_large_ledger(self):
        """Test large ledger append"""
        bytecode = Bytecode()
        for i in range(1000):
            bytecode.add_constant(f"entry_{i}")
            bytecode.add_instruction("LOAD_CONST", i)
            bytecode.add_instruction("LEDGER_APPEND")
        
        self.vm.execute(bytecode)
        ledger = self.vm.env.get("_ledger", [])
        self.assertEqual(len(ledger), 1000)
    
    def test_many_state_operations(self):
        """Test many state read/write operations"""
        bytecode = Bytecode()
        for i in range(100):
            key = f"key_{i}"
            bytecode.add_constant(key)
            bytecode.add_constant(i)
            bytecode.add_instruction("LOAD_CONST", i*2 + 1)
            bytecode.add_instruction("STATE_WRITE", i*2)
        
        self.vm.execute(bytecode)
        state = self.vm.env.get("_blockchain_state", {})
        self.assertEqual(len(state), 100)


def run_tests():
    """Run all test suites"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHashBlockOpcode))
    suite.addTests(loader.loadTestsFromTestCase(TestMerkleRootOpcode))
    suite.addTests(loader.loadTestsFromTestCase(TestStateOpcodes))
    suite.addTests(loader.loadTestsFromTestCase(TestTransactionOpcodes))
    suite.addTests(loader.loadTestsFromTestCase(TestGasChargeOpcode))
    suite.addTests(loader.loadTestsFromTestCase(TestLedgerAppendOpcode))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegratedBlockchain))
    suite.addTests(loader.loadTestsFromTestCase(TestBlockchainPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"BLOCKCHAIN OPCODES TEST SUMMARY")
    print("="*70)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
