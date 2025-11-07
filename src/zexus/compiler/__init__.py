# src/zexus/compiler/__init__.py

"""
Zexus Compiler Phase - Frontend compilation with semantic analysis
"""

# Minimal top-level exports to avoid import-time cycles/errors.
# Do not import .parser, .semantic, .bytecode at module import time unconditionally.
# They will be imported lazily inside ZexusCompiler.compile().

Parser = None  # will be set if parser import succeeds below (best-effort)
ZexusCompiler = None  # defined below

# Expose interpreter builtins to the compiler package for semantic passes.
# Fall back to an empty dict if the evaluator cannot be imported (avoids import cycles).
try:
	from ..evaluator import builtins as BUILTINS
except Exception:
	BUILTINS = {}

# Try to import ProductionParser now but don't fail if it errors (best-effort).
try:
	from .parser import ProductionParser as _ProductionParser
	Parser = _ProductionParser
except Exception:
	# Leave Parser as None; consumers should handle None and provide helpful messages.
	Parser = None

# --- Compiler class (lazy imports inside compile) --------------------------------
class ZexusCompiler:
	def __init__(self, source, enable_optimizations=True):
		self.source = source
		self.enable_optimizations = enable_optimizations
		self.ast = None
		self.bytecode = None
		self.errors = []
		
	def compile(self):
		"""Full compilation pipeline with enhanced error reporting (lazy module imports)"""
		# Import frontend components lazily to avoid import-time circular issues.
		try:
			from .lexer import Lexer
		except Exception as e:
			self.errors.append(f"Compilation import error (lexer): {e}")
			return None

		try:
			from .parser import ProductionParser
		# if parser import fails, record the error and provide a helpful fallback
		except Exception as e:
			self.errors.append(f"Compilation import error (parser): {e}")
			return None

		try:
			from .semantic import SemanticAnalyzer
		except Exception as e:
			self.errors.append(f"Compilation import error (semantic): {e}")
			return None

		try:
			from .bytecode import BytecodeGenerator
		except Exception as e:
			self.errors.append(f"Compilation import error (bytecode): {e}")
			return None

		try:
			# Phase 1: Lexical Analysis
			lexer = Lexer(self.source)
			
			# Phase 2: Syntax Analysis
			parser = ProductionParser(lexer)
			self.ast = parser.parse_program()
			# propagate parser errors
			if getattr(parser, "errors", None):
				self.errors.extend(parser.errors)
			
			if self.errors:
				return None
				
			# Phase 3: Semantic Analysis
			analyzer = SemanticAnalyzer()

			# Best-effort: inject BUILTINS into analyzer environment so compiler can resolve builtins like 'string'
			try:
				# If analyzer exposes an `environment` attribute which supports `set(name, value)`, use it.
				if BUILTINS:
					if hasattr(analyzer, "register_builtins") and callable(getattr(analyzer, "register_builtins")):
						analyzer.register_builtins(BUILTINS)
					elif hasattr(analyzer, "environment"):
						env = getattr(analyzer, "environment")
						# env could be Environment object with set(), or a plain dict
						if hasattr(env, "set") and callable(getattr(env, "set")):
							for k, v in BUILTINS.items():
								try:
									env.set(k, v)
								except Exception:
									# best-effort injection — non-fatal
									pass
						elif isinstance(env, dict):
							for k, v in BUILTINS.items():
								env.setdefault(k, v)
					# otherwise try class-level hook
			except Exception:
				# Do not fail compilation for builtin injection issues; semantic analyzer may handle builtins itself.
				pass

			semantic_errors = analyzer.analyze(self.ast)
			if semantic_errors:
				self.errors.extend(semantic_errors)
			
			if self.errors:
				return None
				
			# Phase 4: Bytecode Generation
			generator = BytecodeGenerator()
			self.bytecode = generator.generate(self.ast)
			
			return self.bytecode
			
		except Exception as e:
			self.errors.append(f"Compilation error: {str(e)}")
			return None
	
	def get_errors(self):
		"""Get formatted error messages with line numbers"""
		return self.errors

# Provide Parser alias for external code expecting it (best-effort)
try:
	from .parser import ProductionParser as _ParserAlias
	Parser = _ParserAlias
except Exception:
	# keep existing Parser (possibly None) and avoid raising on import
	pass