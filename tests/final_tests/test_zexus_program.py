from src.zexus.lexer import Lexer
from src.zexus.parser.parser import Parser
from src.zexus.evaluator.core import evaluate
from src.zexus.object import Environment

class TestRealZexusIntegration(unittest.TestCase):
    """Test full pipeline with real Zexus code"""
    
    def test_real_zexus_program(self):
        """Test a real Zexus program through the full pipeline"""
        # Real Zexus code
        zexus_code = """
        // Fibonacci function
        action fib(n) {
            if (n <= 1) {
                return n;
            }
            return fib(n - 1) + fib(n - 2);
        }
        
        // Test it
        let result = fib(10);
        result
        """
        
        # Execute through interpreter pipeline
        lexer = Lexer(zexus_code)
        parser = Parser(lexer)
        program = parser.parse_program()
        
        env = Environment()
        
        # Execute with VM support
        result = evaluate(program, env, use_vm=True)
        
        # Expected: fib(10) = 55
        self.assertEqual(result, 55, f"Fibonacci(10) should be 55, got {result}")
        
        print(f"\n🎯 REAL ZEXUS PROGRAM EXECUTION:")
        print(f"   Program: Fibonacci(10)")
        print(f"   Result: {result} (expected: 55)")
    
    def test_blockchain_zexus_program(self):
        """Test Zexus blockchain program with VM"""
        zexus_code = """
        // Simple token transfer
        let sender_balance = 1000;
        let receiver_balance = 0;
        let transfer_amount = 100;
        
        // Simulate transaction
        tx {
            sender_balance = sender_balance - transfer_amount;
            receiver_balance = receiver_balance + transfer_amount;
        }
        
        // Return new balances
        [sender_balance, receiver_balance]
        """
        
        lexer = Lexer(zexus_code)
        parser = Parser(lexer)
        program = parser.parse_program()
        
        env = Environment()
        result = evaluate(program, env, use_vm=True)
        
        # Expected: [900, 100]
        expected = [900, 100]
        self.assertEqual(result, expected, 
                         f"Token transfer should result in {expected}, got {result}")
        
        print(f"\n🔗 BLOCKCHAIN ZEXUS PROGRAM:")
        print(f"   Transfer: 100 tokens")
        print(f"   Result: {result} (expected: {expected})")
    
    def test_performance_comparison_zexus(self):
        """Compare VM vs interpreter performance on real Zexus code"""
        zexus_code = """
        // Compute sum of squares
        let sum = 0;
        let i = 0;
        
        while (i < 1000) {
            sum = sum + (i * i);
            i = i + 1;
        }
        
        sum
        """
        
        lexer = Lexer(zexus_code)
        parser = Parser(lexer)
        program = parser.parse_program()
        
        # Time interpreter (no VM)
        env1 = Environment()
        start = time.perf_counter()
        result_interp = evaluate(program, env1, use_vm=False)
        time_interp = time.perf_counter() - start
        
        # Time with VM
        env2 = Environment()
        start = time.perf_counter()
        result_vm = evaluate(program, env2, use_vm=True)
        time_vm = time.perf_counter() - start
        
        # Results should match
        self.assertEqual(result_interp, result_vm, 
                         f"Results should match: interp={result_interp}, vm={result_vm}")
        
        speedup = time_interp / time_vm if time_vm > 0 else 1
        
        print(f"\n⚡ ZEXUS PERFORMANCE COMPARISON:")
        print(f"   Interpreter: {time_interp*1000:.2f}ms")
        print(f"   VM:          {time_vm*1000:.2f}ms")
        print(f"   Speedup:     {speedup:.2f}x")
        print(f"   Result:      {result_vm}")
        
        # VM should be at least as fast (might be slower for trivial code due to overhead)
        self.assertGreater(speedup, 0.5, 
                          f"VM should not be >2x slower. Speedup: {speedup:.2f}x")
