#!/usr/bin/env python3
"""
Register VM Unit Tests

Tests for Phase 5: Register-Based VM
- RegisterFile operations
- RegisterAllocator logic
- RegisterVM execution
- BytecodeConverter transformations
- Hybrid mode execution
- Performance benchmarks
"""

import sys
import os
sys.path.insert(0, 'src')

import unittest
from zexus.vm.register_vm import (
    RegisterFile, RegisterAllocator, RegisterVM, RegisterOpcode
)
from zexus.vm.bytecode import Bytecode, BytecodeBuilder, Opcode
from zexus.vm.bytecode_converter import BytecodeConverter


class TestRegisterFile(unittest.TestCase):
    """Test RegisterFile class"""
    
    def test_init(self):
        """Test register file initialization"""
        rf = RegisterFile(16)
        self.assertEqual(rf.num_registers, 16)
        self.assertEqual(len(rf.registers), 16)
        self.assertTrue(all(r is None for r in rf.registers))
    
    def test_read_write(self):
        """Test register read/write"""
        rf = RegisterFile()
        rf.write(0, 42)
        self.assertEqual(rf.read(0), 42)
        rf.write(5, "hello")
        self.assertEqual(rf.read(5), "hello")
    
    def test_invalid_register(self):
        """Test invalid register access"""
        rf = RegisterFile(8)
        with self.assertRaises(ValueError):
            rf.read(10)
        with self.assertRaises(ValueError):
            rf.write(20, 42)
    
    def test_clear(self):
        """Test register clear"""
        rf = RegisterFile()
        rf.write(3, 100)
        rf.clear(3)
        self.assertIsNone(rf.read(3))
        self.assertFalse(rf.is_dirty(3))
    
    def test_clear_all(self):
        """Test clear all registers"""
        rf = RegisterFile()
        rf.write(0, 1)
        rf.write(5, 2)
        rf.write(10, 3)
        rf.clear_all()
        self.assertTrue(all(r is None for r in rf.registers))
        self.assertTrue(all(not d for d in rf.dirty))
    
    def test_dirty_tracking(self):
        """Test dirty flag tracking"""
        rf = RegisterFile()
        self.assertFalse(rf.is_dirty(0))
        rf.write(0, 42)
        self.assertTrue(rf.is_dirty(0))
    
    def test_get_free_register(self):
        """Test finding free register"""
        rf = RegisterFile(8)
        # Initially all free
        free = rf.get_free_register()
        self.assertEqual(free, 0)
        
        # Allocate some
        rf.write(0, 1)
        rf.write(1, 2)
        free = rf.get_free_register()
        self.assertEqual(free, 2)


class TestRegisterAllocator(unittest.TestCase):
    """Test RegisterAllocator class"""
    
    def test_init(self):
        """Test allocator initialization"""
        alloc = RegisterAllocator(16, reserved=2)
        self.assertEqual(alloc.num_registers, 16)
        self.assertEqual(alloc.available, 14)
    
    def test_allocate_variable(self):
        """Test variable allocation"""
        alloc = RegisterAllocator(8)
        reg = alloc.allocate("x")
        self.assertEqual(reg, 0)
        self.assertEqual(alloc.get_register("x"), 0)
        self.assertEqual(alloc.get_variable(0), "x")
    
    def test_allocate_multiple(self):
        """Test multiple allocations"""
        alloc = RegisterAllocator(8)
        r1 = alloc.allocate("x")
        r2 = alloc.allocate("y")
        r3 = alloc.allocate("z")
        self.assertEqual(r1, 0)
        self.assertEqual(r2, 1)
        self.assertEqual(r3, 2)
    
    def test_allocate_same_variable(self):
        """Test allocating same variable twice"""
        alloc = RegisterAllocator(8)
        r1 = alloc.allocate("x")
        r2 = alloc.allocate("x")
        self.assertEqual(r1, r2)
    
    def test_spilling(self):
        """Test register spilling when exhausted"""
        alloc = RegisterAllocator(4, reserved=0)
        alloc.allocate("a")
        alloc.allocate("b")
        alloc.allocate("c")
        alloc.allocate("d")
        
        # Fifth allocation should spill
        reg = alloc.allocate("e")
        self.assertEqual(reg, -1)
        self.assertIn("e", alloc.spilled_vars)
    
    def test_free_register(self):
        """Test freeing registers"""
        alloc = RegisterAllocator(8)
        alloc.allocate("x")
        alloc.free("x")
        self.assertNotIn("x", alloc.var_to_reg)
        self.assertEqual(len(alloc.allocated_regs), 0)
    
    def test_clear(self):
        """Test clearing allocator"""
        alloc = RegisterAllocator(8)
        alloc.allocate("x")
        alloc.allocate("y")
        alloc.clear()
        self.assertEqual(len(alloc.var_to_reg), 0)
        self.assertEqual(len(alloc.allocated_regs), 0)


class TestRegisterVM(unittest.TestCase):
    """Test RegisterVM execution"""
    
    def test_init(self):
        """Test VM initialization"""
        vm = RegisterVM()
        self.assertEqual(vm.registers.num_registers, 16)
        self.assertIsNotNone(vm.allocator)
        self.assertTrue(vm.hybrid_mode)
    
    def test_load_reg(self):
        """Test LOAD_REG instruction"""
        vm = RegisterVM()
        instructions = [
            (RegisterOpcode.LOAD_REG, 0, 0),  # r0 = 42
        ]
        constants = [42]
        bytecode = Bytecode(instructions, constants)
        
        vm.execute(bytecode)
        self.assertEqual(vm.registers.read(0), 42)
    
    def test_mov_reg(self):
        """Test MOV_REG instruction"""
        vm = RegisterVM()
        instructions = [
            (RegisterOpcode.LOAD_REG, 0, 0),  # r0 = 10
            (RegisterOpcode.MOV_REG, 1, 0),   # r1 = r0
        ]
        constants = [10]
        bytecode = Bytecode(instructions, constants)
        
        vm.execute(bytecode)
        self.assertEqual(vm.registers.read(0), 10)
        self.assertEqual(vm.registers.read(1), 10)
    
    def test_add_reg(self):
        """Test ADD_REG instruction"""
        vm = RegisterVM()
        instructions = [
            (RegisterOpcode.LOAD_REG, 0, 0),     # r0 = 10
            (RegisterOpcode.LOAD_REG, 1, 1),     # r1 = 20
            (RegisterOpcode.ADD_REG, 2, 0, 1),   # r2 = r0 + r1
        ]
        constants = [10, 20]
        bytecode = Bytecode(instructions, constants)
        
        vm.execute(bytecode)
        self.assertEqual(vm.registers.read(2), 30)
    
    def test_sub_reg(self):
        """Test SUB_REG instruction"""
        vm = RegisterVM()
        instructions = [
            (RegisterOpcode.LOAD_REG, 0, 0),     # r0 = 50
            (RegisterOpcode.LOAD_REG, 1, 1),     # r1 = 30
            (RegisterOpcode.SUB_REG, 2, 0, 1),   # r2 = r0 - r1
        ]
        constants = [50, 30]
        bytecode = Bytecode(instructions, constants)
        
        vm.execute(bytecode)
        self.assertEqual(vm.registers.read(2), 20)
    
    def test_mul_reg(self):
        """Test MUL_REG instruction"""
        vm = RegisterVM()
        instructions = [
            (RegisterOpcode.LOAD_REG, 0, 0),     # r0 = 7
            (RegisterOpcode.LOAD_REG, 1, 1),     # r1 = 6
            (RegisterOpcode.MUL_REG, 2, 0, 1),   # r2 = r0 * r1
        ]
        constants = [7, 6]
        bytecode = Bytecode(instructions, constants)
        
        vm.execute(bytecode)
        self.assertEqual(vm.registers.read(2), 42)
    
    def test_div_reg(self):
        """Test DIV_REG instruction"""
        vm = RegisterVM()
        instructions = [
            (RegisterOpcode.LOAD_REG, 0, 0),     # r0 = 100
            (RegisterOpcode.LOAD_REG, 1, 1),     # r1 = 10
            (RegisterOpcode.DIV_REG, 2, 0, 1),   # r2 = r0 / r1
        ]
        constants = [100, 10]
        bytecode = Bytecode(instructions, constants)
        
        vm.execute(bytecode)
        self.assertEqual(vm.registers.read(2), 10)
    
    def test_store_reg(self):
        """Test STORE_REG instruction"""
        vm = RegisterVM()
        instructions = [
            (RegisterOpcode.LOAD_REG, 0, 0),     # r0 = 42
            (RegisterOpcode.STORE_REG, 0, "x"),  # x = r0
        ]
        constants = [42]
        bytecode = Bytecode(instructions, constants)
        
        vm.execute(bytecode)
        self.assertEqual(vm.env["x"], 42)
    
    def test_load_var_reg(self):
        """Test LOAD_VAR_REG instruction"""
        vm = RegisterVM()
        vm.env["x"] = 99
        instructions = [
            (RegisterOpcode.LOAD_VAR_REG, 0, "x"),  # r0 = x
        ]
        bytecode = Bytecode(instructions, [])
        
        vm.execute(bytecode)
        self.assertEqual(vm.registers.read(0), 99)
    
    def test_complex_arithmetic(self):
        """Test complex arithmetic expression: (5 + 3) * 2"""
        vm = RegisterVM()
        instructions = [
            (RegisterOpcode.LOAD_REG, 0, 0),     # r0 = 5
            (RegisterOpcode.LOAD_REG, 1, 1),     # r1 = 3
            (RegisterOpcode.ADD_REG, 2, 0, 1),   # r2 = r0 + r1 = 8
            (RegisterOpcode.LOAD_REG, 3, 2),     # r3 = 2
            (RegisterOpcode.MUL_REG, 4, 2, 3),   # r4 = r2 * r3 = 16
        ]
        constants = [5, 3, 2]
        bytecode = Bytecode(instructions, constants)
        
        vm.execute(bytecode)
        self.assertEqual(vm.registers.read(4), 16)
    
    def test_comparison_lt_reg(self):
        """Test LT_REG comparison"""
        vm = RegisterVM()
        instructions = [
            (RegisterOpcode.LOAD_REG, 0, 0),     # r0 = 5
            (RegisterOpcode.LOAD_REG, 1, 1),     # r1 = 10
            (RegisterOpcode.LT_REG, 2, 0, 1),    # r2 = r0 < r1
        ]
        constants = [5, 10]
        bytecode = Bytecode(instructions, constants)
        
        vm.execute(bytecode)
        self.assertTrue(vm.registers.read(2))
    
    def test_comparison_eq_reg(self):
        """Test EQ_REG comparison"""
        vm = RegisterVM()
        instructions = [
            (RegisterOpcode.LOAD_REG, 0, 0),     # r0 = 42
            (RegisterOpcode.LOAD_REG, 1, 0),     # r1 = 42
            (RegisterOpcode.EQ_REG, 2, 0, 1),    # r2 = r0 == r1
        ]
        constants = [42]
        bytecode = Bytecode(instructions, constants)
        
        vm.execute(bytecode)
        self.assertTrue(vm.registers.read(2))
    
    def test_hybrid_mode_stack_ops(self):
        """Test hybrid mode with stack operations"""
        vm = RegisterVM(hybrid_mode=True)
        instructions = [
            (Opcode.LOAD_CONST, 0),  # Push 10
            (Opcode.LOAD_CONST, 1),  # Push 20
            (Opcode.ADD,),           # Add (stack mode)
            (RegisterOpcode.POP_REG, 15),  # Pop result to r15
        ]
        constants = [10, 20]
        bytecode = Bytecode(instructions, constants)
        
        vm.execute(bytecode)
        self.assertEqual(vm.registers.read(15), 30)
    
    def test_get_stats(self):
        """Test execution statistics"""
        vm = RegisterVM()
        instructions = [
            (RegisterOpcode.LOAD_REG, 0, 0),
            (RegisterOpcode.LOAD_REG, 1, 1),
            (RegisterOpcode.ADD_REG, 2, 0, 1),
        ]
        constants = [10, 20]
        bytecode = Bytecode(instructions, constants)
        
        vm.execute(bytecode)
        stats = vm.get_stats()
        
        self.assertEqual(stats['instructions_executed'], 3)
        self.assertEqual(stats['register_ops'], 3)
        self.assertEqual(stats['stack_ops'], 0)


class TestBytecodeConverter(unittest.TestCase):
    """Test BytecodeConverter transformations"""
    
    def test_init(self):
        """Test converter initialization"""
        converter = BytecodeConverter()
        self.assertEqual(converter.num_registers, 16)
        self.assertIsNotNone(converter.allocator)
    
    def test_convert_const_add(self):
        """Test converting constant addition"""
        builder = BytecodeBuilder()
        builder.emit_constant('LOAD_CONST', 10)
        builder.emit_constant('LOAD_CONST', 20)
        builder.emit('ADD')
        bytecode = builder.build()
        
        converter = BytecodeConverter()
        converted = converter.convert(bytecode)
        
        # Should have register operations
        self.assertGreater(len(converted.instructions), 0)
        stats = converter.get_stats()
        self.assertEqual(stats['conversions'], 1)
    
    def test_convert_var_arithmetic(self):
        """Test converting variable arithmetic"""
        builder = BytecodeBuilder()
        builder.emit_constant('LOAD_NAME', 'x')
        builder.emit_constant('LOAD_CONST', 5)
        builder.emit('MUL')
        bytecode = builder.build()
        
        converter = BytecodeConverter()
        converted = converter.convert(bytecode)
        
        stats = converter.get_stats()
        self.assertEqual(stats['conversions'], 1)
    
    def test_no_conversion_for_non_arithmetic(self):
        """Test that non-arithmetic ops are not converted"""
        builder = BytecodeBuilder()
        builder.emit_constant('LOAD_CONST', "hello")
        builder.emit('PRINT')
        bytecode = builder.build()
        
        converter = BytecodeConverter()
        converted = converter.convert(bytecode)
        
        stats = converter.get_stats()
        self.assertEqual(stats['conversions'], 0)
        self.assertEqual(stats['skipped'], 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for register VM"""
    
    def test_full_pipeline(self):
        """Test full pipeline: build → convert → execute"""
        # Build stack bytecode: x = 10 + 20
        builder = BytecodeBuilder()
        builder.emit_constant('LOAD_CONST', 10)
        builder.emit_constant('LOAD_CONST', 20)
        builder.emit('ADD')
        builder.emit_constant('STORE_NAME', 'x')
        stack_bytecode = builder.build()
        
        # Convert to register bytecode
        converter = BytecodeConverter()
        register_bytecode = converter.convert(stack_bytecode)
        
        # Execute with register VM
        vm = RegisterVM(hybrid_mode=True)
        vm.execute(register_bytecode)
        
        # Verify result
        self.assertEqual(vm.env.get('x'), 30)
    
    def test_nested_arithmetic(self):
        """Test nested arithmetic: (a + b) * (c - d)"""
        builder = BytecodeBuilder()
        # a + b
        builder.emit_constant('LOAD_NAME', 'a')
        builder.emit_constant('LOAD_NAME', 'b')
        builder.emit('ADD')
        # c - d
        builder.emit_constant('LOAD_NAME', 'c')
        builder.emit_constant('LOAD_NAME', 'd')
        builder.emit('SUB')
        # Multiply results
        builder.emit('MUL')
        bytecode = builder.build()
        
        converter = BytecodeConverter()
        converted = converter.convert(bytecode)
        
        vm = RegisterVM(hybrid_mode=True)
        vm.env = {'a': 10, 'b': 5, 'c': 20, 'd': 8}
        vm.execute(converted)
        
        # (10 + 5) * (20 - 8) = 15 * 12 = 180
        self.assertEqual(vm.stack[-1] if vm.stack else vm.registers.read(15), 180)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
