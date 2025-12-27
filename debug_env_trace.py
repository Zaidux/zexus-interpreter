#!/usr/bin/env python3
"""
Debug script to trace environment variable access and modification in async coroutines.
This helps us understand why shared variables aren't being updated.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def patch_environment():
    """Patch Environment class to trace all get/set/assign operations"""
    from zexus.object import Environment  # Use the correct Environment class!
    
    original_get = Environment.get
    original_set = Environment.set
    original_assign = Environment.assign
    
    def traced_get(self, name):
        result = original_get(self, name)
        env_id = id(self)
        outer_id = f"{id(self.outer):x}" if self.outer else "None"
        print(f"[ENV GET] env={env_id:x} outer={outer_id} name='{name}' found={'YES' if name in self.store else 'NO(outer)'} value={type(result).__name__}", file=sys.stderr)
        return result
    
    def traced_set(self, name, value):
        env_id = id(self)
        outer_id = f"{id(self.outer):x}" if self.outer else "None"
        print(f"[ENV SET] env={env_id:x} outer={outer_id} name='{name}' value={type(value).__name__}", file=sys.stderr)
        return original_set(self, name, value)
    
    def traced_assign(self, name, value):
        env_id = id(self)
        outer_id = f"{id(self.outer):x}" if self.outer else "None"
        in_current = name in self.store
        has_in_outer = self._has_variable(name) if self.outer else False
        print(f"[ENV ASSIGN] env={env_id:x} outer={outer_id} name='{name}' in_current={in_current} in_outer={has_in_outer} value={type(value).__name__}", file=sys.stderr)
        result = original_assign(self, name, value)
        # Check where it ended up
        after_current = name in self.store
        print(f"[ENV ASSIGN RESULT] name='{name}' stored_in_current={after_current}", file=sys.stderr)
        return result
    
    Environment.get = traced_get
    Environment.set = traced_set
    Environment.assign = traced_assign
    print("[DEBUG] Environment patched for tracing", file=sys.stderr)
    
    # Test the patch
    test_env = Environment()
    test_env.set("test", "value")
    print(f"[DEBUG] Test patch: set 'test' in environment", file=sys.stderr)

def patch_async_execution():
    """Patch async execution to trace thread creation and env handling"""
    from zexus.evaluator import expressions
    
    original_eval_async = expressions.ExpressionEvaluatorMixin.eval_async_expression
    
    def traced_eval_async(self, node, env, stack_trace):
        print(f"\n[ASYNC EXEC START] env={id(env):x} has_outer={env.outer is not None}", file=sys.stderr)
        print(f"[ASYNC EXEC] env.store keys: {list(env.store.keys())}", file=sys.stderr)
        result = original_eval_async(self, node, env, stack_trace)
        print(f"[ASYNC EXEC END] result={type(result).__name__}\n", file=sys.stderr)
        return result
    
    expressions.ExpressionEvaluatorMixin.eval_async_expression = traced_eval_async
    print("[DEBUG] Async expression evaluator patched for tracing", file=sys.stderr)

def run_test_file(filename):
    """Run a test file with environment tracing enabled"""
    print(f"[DEBUG] Running {filename} with environment tracing...\n", file=sys.stderr)
    
    # Import and patch BEFORE running anything
    patch_environment()
    patch_async_execution()
    
    # Now import the modules
    from zexus.lexer import Lexer
    from zexus.parser.parser import UltimateParser
    from zexus.evaluator.core import Evaluator
    from zexus.object import Environment
    
    with open(filename, 'r') as f:
        code = f.read()
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    
    if parser.errors:
        print("Parse errors:", parser.errors, file=sys.stderr)
        return
    
    print(f"[DEBUG] Parsed {len(program.statements)} statements\n", file=sys.stderr)
    
    env = Environment()
    evaluator = Evaluator()
    
    print("\n[DEBUG] Starting evaluation...\n", file=sys.stderr)
    result = evaluator.ceval_program(program.statements, env)
    print(f"\n[DEBUG] Evaluation complete, result: {result}\n", file=sys.stderr)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_env_trace.py <zexus_file>")
        sys.exit(1)
    
    run_test_file(sys.argv[1])
