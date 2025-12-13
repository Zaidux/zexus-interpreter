"""
Tests for optimization system.
"""

import unittest
from src.zexus.optimization import (
    BytecodeOp, CompiledFunction, OptimizationType,
    ConstantFoldingPass, DeadCodeEliminationPass, InliningPass,
    LoopOptimizationPass, CommonSubexprEliminationPass,
    OptimizationPipeline, BytecodeCompiler, ExecutionProfile,
    OptimizationFramework, get_optimizer, compile_function
)


class TestBytecodeOp(unittest.TestCase):
    """Test bytecode operations."""
    
    def test_bytecode_op_creation(self):
        """Test creating bytecode ops."""
        op = BytecodeOp("LOAD_CONST", [42])
        self.assertEqual(op.opcode, "LOAD_CONST")
        self.assertEqual(op.operands[0], 42)
    
    def test_bytecode_op_repr(self):
        """Test bytecode op representation."""
        op = BytecodeOp("ADD")
        self.assertEqual(str(op), "ADD")
        
        op2 = BytecodeOp("LOAD_CONST", [42])
        self.assertIn("42", str(op2))


class TestCompiledFunction(unittest.TestCase):
    """Test compiled function."""
    
    def test_compiled_function_creation(self):
        """Test creating compiled function."""
        bytecode = [BytecodeOp("LOAD_CONST", [1])]
        func = CompiledFunction("test", bytecode)
        
        self.assertEqual(func.name, "test")
        self.assertEqual(func.get_bytecode_size(), 1)
    
    def test_add_optimization_pass(self):
        """Test recording optimization passes."""
        bytecode = [BytecodeOp("LOAD_CONST", [1])]
        func = CompiledFunction("test", bytecode)
        
        func.add_pass("ConstantFolding")
        self.assertIn("ConstantFolding", func.optimization_passes)
    
    def test_no_duplicate_passes(self):
        """Test passes not recorded twice."""
        bytecode = [BytecodeOp("LOAD_CONST", [1])]
        func = CompiledFunction("test", bytecode)
        
        func.add_pass("Pass1")
        func.add_pass("Pass1")
        
        self.assertEqual(func.optimization_passes.count("Pass1"), 1)


class TestConstantFoldingPass(unittest.TestCase):
    """Test constant folding optimization."""
    
    def test_fold_addition(self):
        """Test folding constant addition."""
        pass_obj = ConstantFoldingPass()
        bytecode = [
            BytecodeOp("LOAD_CONST", [5]),
            BytecodeOp("LOAD_CONST", [3]),
            BytecodeOp("ADD")
        ]
        
        result = pass_obj.optimize(bytecode)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].opcode, "LOAD_CONST")
        self.assertEqual(result[0].operands[0], 8)
    
    def test_fold_multiplication(self):
        """Test folding constant multiplication."""
        pass_obj = ConstantFoldingPass()
        bytecode = [
            BytecodeOp("LOAD_CONST", [4]),
            BytecodeOp("LOAD_CONST", [7]),
            BytecodeOp("MUL")
        ]
        
        result = pass_obj.optimize(bytecode)
        self.assertEqual(result[0].operands[0], 28)
    
    def test_no_fold_non_const(self):
        """Test doesn't fold non-constant expressions."""
        pass_obj = ConstantFoldingPass()
        bytecode = [
            BytecodeOp("LOAD_VAR", ["x"]),
            BytecodeOp("LOAD_CONST", [1]),
            BytecodeOp("ADD")
        ]
        
        result = pass_obj.optimize(bytecode)
        self.assertEqual(len(result), 3)
    
    def test_pass_name(self):
        """Test pass name."""
        pass_obj = ConstantFoldingPass()
        self.assertEqual(pass_obj.get_name(), "ConstantFolding")


class TestDeadCodeEliminationPass(unittest.TestCase):
    """Test dead code elimination."""
    
    def test_remove_after_return(self):
        """Test removing code after return."""
        pass_obj = DeadCodeEliminationPass()
        bytecode = [
            BytecodeOp("LOAD_CONST", [1]),
            BytecodeOp("RETURN"),
            BytecodeOp("LOAD_CONST", [2]),
            BytecodeOp("LOAD_CONST", [3])
        ]
        
        result = pass_obj.optimize(bytecode)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1].opcode, "RETURN")
    
    def test_keep_before_return(self):
        """Test keeping code before return."""
        pass_obj = DeadCodeEliminationPass()
        bytecode = [
            BytecodeOp("LOAD_CONST", [1]),
            BytecodeOp("LOAD_CONST", [2]),
            BytecodeOp("RETURN")
        ]
        
        result = pass_obj.optimize(bytecode)
        self.assertEqual(len(result), 3)
    
    def test_pass_name(self):
        """Test pass name."""
        pass_obj = DeadCodeEliminationPass()
        self.assertEqual(pass_obj.get_name(), "DeadCodeElimination")


class TestInliningPass(unittest.TestCase):
    """Test function inlining."""
    
    def test_inline_small_function(self):
        """Test inlining small function."""
        pass_obj = InliningPass(size_threshold=5)
        
        func_body = [
            BytecodeOp("LOAD_CONST", [1]),
            BytecodeOp("LOAD_CONST", [2]),
            BytecodeOp("ADD")
        ]
        pass_obj.register_function("add", func_body)
        
        bytecode = [
            BytecodeOp("CALL", ["add"]),
            BytecodeOp("RETURN")
        ]
        
        result = pass_obj.optimize(bytecode)
        self.assertGreater(len(result), len(bytecode))
        self.assertEqual(result[0].opcode, "LOAD_CONST")
    
    def test_no_inline_large_function(self):
        """Test doesn't inline large functions."""
        pass_obj = InliningPass(size_threshold=3)
        
        func_body = [
            BytecodeOp("LOAD_CONST", [1]),
            BytecodeOp("LOAD_CONST", [2]),
            BytecodeOp("LOAD_CONST", [3]),
            BytecodeOp("ADD")
        ]
        pass_obj.register_function("big", func_body)
        
        bytecode = [BytecodeOp("CALL", ["big"])]
        
        result = pass_obj.optimize(bytecode)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].opcode, "CALL")
    
    def test_pass_name(self):
        """Test pass name."""
        pass_obj = InliningPass()
        self.assertEqual(pass_obj.get_name(), "Inlining")


class TestLoopOptimizationPass(unittest.TestCase):
    """Test loop optimization."""
    
    def test_loop_metadata(self):
        """Test loop optimization adds metadata."""
        pass_obj = LoopOptimizationPass()
        bytecode = [
            BytecodeOp("LOOP", [10]),
            BytecodeOp("LOAD_CONST", [1]),
            BytecodeOp("LOOP_END")
        ]
        
        result = pass_obj.optimize(bytecode)
        self.assertTrue(result[0].metadata.get("optimized"))
        self.assertEqual(result[0].metadata.get("size"), 2)
    
    def test_pass_name(self):
        """Test pass name."""
        pass_obj = LoopOptimizationPass()
        self.assertEqual(pass_obj.get_name(), "LoopOptimization")


class TestCommonSubexprEliminationPass(unittest.TestCase):
    """Test common subexpression elimination."""
    
    def test_eliminate_duplicate_expr(self):
        """Test eliminating duplicate expressions."""
        pass_obj = CommonSubexprEliminationPass()
        bytecode = [
            BytecodeOp("LOAD_VAR", ["a"]),
            BytecodeOp("LOAD_VAR", ["b"]),
            BytecodeOp("ADD"),
            BytecodeOp("LOAD_VAR", ["a"]),
            BytecodeOp("LOAD_VAR", ["b"]),
            BytecodeOp("ADD")
        ]
        
        result = pass_obj.optimize(bytecode)
        # Should have CSE operations
        self.assertGreater(len(result), 0)
    
    def test_pass_name(self):
        """Test pass name."""
        pass_obj = CommonSubexprEliminationPass()
        self.assertEqual(pass_obj.get_name(), "CommonSubexprElimination")


class TestOptimizationPipeline(unittest.TestCase):
    """Test optimization pipeline."""
    
    def test_pipeline_creation(self):
        """Test creating pipeline."""
        pipeline = OptimizationPipeline()
        self.assertEqual(len(pipeline.passes), 0)
    
    def test_add_pass(self):
        """Test adding pass to pipeline."""
        pipeline = OptimizationPipeline()
        pass_obj = ConstantFoldingPass()
        
        pipeline.add_pass(pass_obj)
        self.assertEqual(len(pipeline.passes), 1)
    
    def test_enable_disable_pass(self):
        """Test enabling/disabling passes."""
        pipeline = OptimizationPipeline()
        
        initial = len(pipeline.enabled_passes)
        pipeline.disable_pass(OptimizationType.CONSTANT_FOLDING)
        
        self.assertEqual(len(pipeline.enabled_passes), initial - 1)
        
        pipeline.enable_pass(OptimizationType.CONSTANT_FOLDING)
        self.assertEqual(len(pipeline.enabled_passes), initial)
    
    def test_optimize_bytecode(self):
        """Test optimizing bytecode."""
        pipeline = OptimizationPipeline()
        pipeline.add_pass(ConstantFoldingPass())
        
        bytecode = [
            BytecodeOp("LOAD_CONST", [2]),
            BytecodeOp("LOAD_CONST", [3]),
            BytecodeOp("ADD")
        ]
        
        result = pipeline.optimize(bytecode)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].operands[0], 5)


class TestBytecodeCompiler(unittest.TestCase):
    """Test bytecode compiler."""
    
    def test_compiler_creation(self):
        """Test creating compiler."""
        compiler = BytecodeCompiler()
        self.assertEqual(len(compiler.compiled_functions), 0)
    
    def test_compile_without_optimization(self):
        """Test compiling without optimization."""
        compiler = BytecodeCompiler()
        bytecode = [
            BytecodeOp("LOAD_CONST", [1]),
            BytecodeOp("RETURN")
        ]
        
        func = compiler.compile_function("test", bytecode, optimize=False)
        self.assertEqual(func.get_bytecode_size(), 2)
    
    def test_compile_with_optimization(self):
        """Test compiling with optimization."""
        compiler = BytecodeCompiler()
        bytecode = [
            BytecodeOp("LOAD_CONST", [1]),
            BytecodeOp("LOAD_CONST", [2]),
            BytecodeOp("ADD"),
            BytecodeOp("RETURN")
        ]
        
        func = compiler.compile_function("test", bytecode, optimize=True)
        self.assertLessEqual(func.get_bytecode_size(), 4)
    
    def test_get_compiled_function(self):
        """Test retrieving compiled function."""
        compiler = BytecodeCompiler()
        bytecode = [BytecodeOp("LOAD_CONST", [1])]
        
        compiler.compile_function("test", bytecode)
        retrieved = compiler.get_compiled_function("test")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "test")
    
    def test_optimization_stats(self):
        """Test optimization statistics."""
        compiler = BytecodeCompiler()
        bytecode = [
            BytecodeOp("LOAD_CONST", [1]),
            BytecodeOp("LOAD_CONST", [2]),
            BytecodeOp("ADD"),
            BytecodeOp("RETURN"),
            BytecodeOp("LOAD_CONST", [3])
        ]
        
        compiler.compile_function("test", bytecode, optimize=True)
        stats = compiler.get_optimization_stats()
        
        self.assertIn("total_functions", stats)
        self.assertIn("reduction_percentage", stats)
        self.assertEqual(stats["total_functions"], 1)


class TestExecutionProfile(unittest.TestCase):
    """Test execution profile."""
    
    def test_profile_creation(self):
        """Test creating profile."""
        profile = ExecutionProfile("test")
        self.assertEqual(profile.name, "test")
        self.assertEqual(profile.call_count, 0)
    
    def test_record_call(self):
        """Test recording function call."""
        profile = ExecutionProfile("test")
        profile.record_call(0.001)
        
        self.assertEqual(profile.call_count, 1)
        self.assertAlmostEqual(profile.total_time, 0.001)
    
    def test_average_time(self):
        """Test calculating average time."""
        profile = ExecutionProfile("test")
        profile.record_call(0.002)
        profile.record_call(0.004)
        
        avg = profile.get_avg_time()
        self.assertAlmostEqual(avg, 0.003)
    
    def test_should_optimize_hot(self):
        """Test optimization suggestion for hot functions."""
        profile = ExecutionProfile("test")
        for _ in range(11):
            profile.record_call(0.001)
        
        self.assertTrue(profile.should_optimize())
    
    def test_should_optimize_slow(self):
        """Test optimization suggestion for slow functions."""
        profile = ExecutionProfile("test")
        profile.record_call(1.5)
        
        self.assertTrue(profile.should_optimize())


class TestOptimizationFramework(unittest.TestCase):
    """Test optimization framework."""
    
    def test_framework_creation(self):
        """Test creating framework."""
        framework = OptimizationFramework()
        self.assertIsNotNone(framework.compiler)
    
    def test_create_profile(self):
        """Test creating execution profile."""
        framework = OptimizationFramework()
        profile = framework.create_profile("test")
        
        self.assertEqual(profile.name, "test")
        self.assertIn("test", framework.profiles)
    
    def test_get_profile(self):
        """Test retrieving profile."""
        framework = OptimizationFramework()
        framework.create_profile("test")
        
        retrieved = framework.get_profile("test")
        self.assertIsNotNone(retrieved)
    
    def test_get_hot_functions(self):
        """Test finding hot functions."""
        framework = OptimizationFramework()
        profile = framework.create_profile("hot")
        
        for _ in range(15):
            profile.record_call(0.001)
        
        hot = framework.get_hot_functions(threshold=10)
        self.assertIn("hot", hot)
    
    def test_get_slow_functions(self):
        """Test finding slow functions."""
        framework = OptimizationFramework()
        profile = framework.create_profile("slow")
        profile.record_call(2.0)
        
        slow = framework.get_slow_functions(threshold=1.5)
        self.assertIn("slow", slow)


class TestGlobalOptimizer(unittest.TestCase):
    """Test global optimizer functions."""
    
    def test_get_optimizer(self):
        """Test getting global optimizer."""
        optimizer = get_optimizer()
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, OptimizationFramework)
    
    def test_compile_function_global(self):
        """Test compiling with global function."""
        bytecode = [
            BytecodeOp("LOAD_CONST", [1]),
            BytecodeOp("LOAD_CONST", [2]),
            BytecodeOp("ADD")
        ]
        
        func = compile_function("test", bytecode)
        self.assertEqual(func.name, "test")


if __name__ == "__main__":
    unittest.main()
