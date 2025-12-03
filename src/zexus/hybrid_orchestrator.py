# src/zexus/hybrid_orchestrator.py
"""
Hybrid Orchestrator - Intelligently chooses between interpreter and compiler
"""

import os
import time
from .lexer import Lexer
from .parser import UltimateParser
# UPDATED: Import from new structure
from .evaluator import evaluate
from .object import Environment
from .config import config

# Try to import compiler components
try:
    from .compiler import ZexusCompiler
    from .vm import ZexusVM
    COMPILER_AVAILABLE = True
except ImportError:
    COMPILER_AVAILABLE = False

class HybridOrchestrator:
    def __init__(self):
        self.interpreter_used = 0
        self.compiler_used = 0
        self.fallbacks = 0

    def should_use_compiler(self, code, syntax_style="auto"):
        """
        Smart rules for when to use compiler vs interpreter
        """
        if not config.use_hybrid_compiler or not COMPILER_AVAILABLE:
            return False

        # Rule 1: Large files (> 100 lines) benefit from compilation
        line_count = len(code.split('\n'))
        if line_count > config.compiler_line_threshold:
            return True

        # Rule 2: Code with complex loops (for, while) 
        complex_constructs = ['for', 'while', 'each', 'function', 'action']
        if any(construct in code for construct in complex_constructs):
            return True

        # Rule 3: Mathematical/computational intensive code
        math_keywords = ['*', '/', '%', 'math.', 'calculate']
        if any(keyword in code for keyword in math_keywords):
            return True

        # Rule 4: User explicitly wants compilation
        if "// compile" in code or "# compile" in code:
            return True

        # Rule 5: Universal syntax is more compiler-friendly
        if syntax_style == "universal":
            return True

        # Default: Use interpreter for simple scripts
        return False

    def compile_and_execute(self, code, environment=None, syntax_style="auto"):
        """
        Execute code using the compiler/VM path
        """
        try:
            if not COMPILER_AVAILABLE:
                raise Exception("Compiler not available")

            print("🔧 Compiling code...")

            # Use the ZexusCompiler
            compiler = ZexusCompiler(code)
            bytecode = compiler.compile()

            if compiler.errors:
                raise Exception(f"Compilation errors: {compiler.errors}")

            # Execute in VM
            vm = ZexusVM(bytecode)
            result = vm.execute()

            self.compiler_used += 1
            return result

        except Exception as e:
            print(f"❌ Compilation failed: {e}")
            if config.fallback_to_interpreter:
                print("🔄 Falling back to interpreter...")
                self.fallbacks += 1
                return self.interpret(code, environment, syntax_style)
            else:
                raise

    def interpret(self, code, environment=None, syntax_style="auto"):
        """
        Execute code using the interpreter path
        """
        lexer = Lexer(code)
        parser = UltimateParser(lexer, syntax_style)
        program = parser.parse_program()

        if len(parser.errors) > 0:
            raise Exception(f"Parse errors: {parser.errors}")

        if environment is None:
            environment = Environment()

        # UPDATED: Use evaluate instead of eval_node
        result = evaluate(program, environment)

        self.interpreter_used += 1
        return result

    def execute(self, code, environment=None, mode="auto", syntax_style="auto"):
        """
        Main entry point - decides execution strategy
        """
        start_time = time.time()

        if mode == "interpreter":
            result = self.interpret(code, environment, syntax_style)
        elif mode == "compiler":
            result = self.compile_and_execute(code, environment, syntax_style)
        else:  # auto mode
            if self.should_use_compiler(code, syntax_style):
                result = self.compile_and_execute(code, environment, syntax_style)
            else:
                result = self.interpret(code, environment, syntax_style)

        execution_time = time.time() - start_time

        if config.enable_debug_logs and config.enable_execution_stats:
            self._print_execution_stats(execution_time, mode)

        return result

    def _print_execution_stats(self, execution_time, mode):
        """Print execution statistics"""
        print(f"\n📊 Execution Statistics:")
        print(f"   Mode: {mode}")
        print(f"   Time: {execution_time:.4f}s")
        print(f"   Interpreter uses: {self.interpreter_used}")
        print(f"   Compiler uses: {self.compiler_used}")
        print(f"   Fallbacks: {self.fallbacks}")
        total = self.interpreter_used + self.compiler_used
        if total > 0:
            compiler_percent = (self.compiler_used / total) * 100
            print(f"   Compiler usage: {compiler_percent:.1f}%")

# Global orchestrator instance
orchestrator = HybridOrchestrator()