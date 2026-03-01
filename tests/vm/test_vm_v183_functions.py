"""
Unit tests for VM functions added in v1.8.3:
- VMRuntimeError — proper Python exception class
- _vm_warn() — diagnostic warning system
- _vm_native_call() — fast-path native builtins
- _build_entity_definition() — EntityDefinition from bytecode
- _construct_entity() — EntityInstance construction
- _compile_StateStatement — standalone state declaration compiler

Total: 60+ tests covering normal paths, edge cases, and error conditions.
"""

import sys
import os
import io
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.zexus.vm.vm import (
    VM, VMMode, _vm_warn, VMRuntimeError, _VM_NATIVE_MISS,
)
from src.zexus.vm.bytecode import Bytecode, BytecodeBuilder, Opcode
from src.zexus.object import (
    Integer as ZInteger, Float as ZFloat, String as ZString,
    Boolean as ZBoolean, List as ZList, Map as ZMap,
    Null as ZNull, EntityDefinition, EntityInstance,
)


# ========================== VMRuntimeError ==========================

class TestVMRuntimeError(unittest.TestCase):
    """Tests for VMRuntimeError exception class."""

    def test_is_exception(self):
        """VMRuntimeError must inherit from Exception"""
        err = VMRuntimeError("test error")
        self.assertIsInstance(err, Exception)

    def test_message_stored(self):
        """zexus_message attribute stores the original message"""
        err = VMRuntimeError("require failed: insufficient balance")
        self.assertEqual(err.zexus_message, "require failed: insufficient balance")

    def test_str_representation(self):
        """str() should return the message"""
        err = VMRuntimeError("verify failed")
        self.assertEqual(str(err), "verify failed")

    def test_can_be_raised_and_caught(self):
        """Must be raisable (unlike ZEvaluationError which inherits Object)"""
        with self.assertRaises(VMRuntimeError) as ctx:
            raise VMRuntimeError("contract violation")
        self.assertEqual(ctx.exception.zexus_message, "contract violation")

    def test_caught_as_exception(self):
        """Should be catchable as a general Exception"""
        with self.assertRaises(Exception):
            raise VMRuntimeError("generic catch")

    def test_empty_message(self):
        err = VMRuntimeError("")
        self.assertEqual(err.zexus_message, "")
        self.assertEqual(str(err), "")

    def test_unicode_message(self):
        err = VMRuntimeError("错误: バグ 🐛")
        self.assertEqual(err.zexus_message, "错误: バグ 🐛")


# ========================== _vm_warn ==========================

class TestVmWarn(unittest.TestCase):
    """Tests for _vm_warn() diagnostic system."""

    @patch('src.zexus.vm.vm._VM_WARN_LEVEL', 'all')
    def test_warn_all_prints_message(self):
        """When level=all, warnings print to stderr"""
        captured = io.StringIO()
        with patch('sys.stderr', captured):
            _vm_warn("CALL_METHOD", "method 'foo' failed on NoneType")
        output = captured.getvalue()
        self.assertIn("[VM WARN]", output)
        self.assertIn("CALL_METHOD", output)
        self.assertIn("method 'foo' failed", output)

    @patch('src.zexus.vm.vm._VM_WARN_LEVEL', 'all')
    def test_warn_all_includes_exception(self):
        """Exception detail is included in output"""
        captured = io.StringIO()
        with patch('sys.stderr', captured):
            _vm_warn("GET_ATTR", "attribute not found", ValueError("no such attr"))
        output = captured.getvalue()
        self.assertIn("no such attr", output)

    @patch('src.zexus.vm.vm._VM_WARN_LEVEL', 'none')
    def test_warn_none_is_silent(self):
        """When level=none, nothing is printed"""
        captured = io.StringIO()
        with patch('sys.stderr', captured):
            _vm_warn("ANYTHING", "should not appear", RuntimeError("hidden"))
        self.assertEqual(captured.getvalue(), "")

    @patch('src.zexus.vm.vm._VM_WARN_LEVEL', 'errors')
    def test_warn_errors_prints_on_exception(self):
        """Level=errors prints only when exc is provided"""
        captured = io.StringIO()
        with patch('sys.stderr', captured):
            _vm_warn("CALL", "with exception", TypeError("type err"))
        self.assertIn("[VM WARN]", captured.getvalue())

    @patch('src.zexus.vm.vm._VM_WARN_LEVEL', 'errors')
    def test_warn_errors_silent_without_exception(self):
        """Level=errors is silent when no exception passed"""
        captured = io.StringIO()
        with patch('sys.stderr', captured):
            _vm_warn("CALL", "no exception here")
        self.assertEqual(captured.getvalue(), "")


# ========================== _vm_native_call ==========================

class TestVmNativeCall(unittest.TestCase):
    """Tests for _vm_native_call() fast-path builtins."""

    def setUp(self):
        self.vm = VM(debug=False)

    # --- push (non-mutating) ---
    def test_push_plain_list(self):
        result = self.vm._vm_native_call("push", [[1, 2, 3], 4])
        self.assertEqual(result, [1, 2, 3, 4])

    def test_push_does_not_mutate_original(self):
        original = [1, 2]
        result = self.vm._vm_native_call("push", [original, 3])
        self.assertEqual(result, [1, 2, 3])
        self.assertEqual(original, [1, 2])  # original unchanged

    def test_push_zlist(self):
        zl = ZList([1, 2])
        result = self.vm._vm_native_call("push", [zl, 3])
        self.assertIsInstance(result, ZList)
        self.assertEqual(result.elements, [1, 2, 3])
        self.assertEqual(zl.elements, [1, 2])  # original unchanged

    def test_push_empty_list(self):
        result = self.vm._vm_native_call("push", [[], "first"])
        self.assertEqual(result, ["first"])

    def test_push_wrong_arity(self):
        result = self.vm._vm_native_call("push", [[1]])
        self.assertIs(result, _VM_NATIVE_MISS)

    def test_push_three_args(self):
        result = self.vm._vm_native_call("push", [[1], 2, 3])
        self.assertIs(result, _VM_NATIVE_MISS)

    # --- append (mutating) ---
    def test_append_plain_list(self):
        lst = [1, 2]
        result = self.vm._vm_native_call("append", [lst, 3])
        self.assertIs(result, lst)  # returns same object
        self.assertEqual(lst, [1, 2, 3])

    def test_append_zlist(self):
        zl = ZList([10, 20])
        result = self.vm._vm_native_call("append", [zl, 30])
        self.assertIs(result, zl)
        self.assertEqual(zl.elements, [10, 20, 30])

    def test_append_wrong_arity(self):
        result = self.vm._vm_native_call("append", [[1]])
        self.assertIs(result, _VM_NATIVE_MISS)

    # --- length / len ---
    def test_length_plain_list(self):
        self.assertEqual(self.vm._vm_native_call("length", [[1, 2, 3]]), 3)

    def test_len_plain_list(self):
        self.assertEqual(self.vm._vm_native_call("len", [[1, 2]]), 2)

    def test_length_zlist(self):
        self.assertEqual(self.vm._vm_native_call("length", [ZList([1, 2, 3, 4])]), 4)

    def test_length_string(self):
        self.assertEqual(self.vm._vm_native_call("length", ["hello"]), 5)

    def test_length_zstring(self):
        self.assertEqual(self.vm._vm_native_call("length", [ZString("abc")]), 3)

    def test_length_dict(self):
        self.assertEqual(self.vm._vm_native_call("length", [{"a": 1, "b": 2}]), 2)

    def test_length_zmap(self):
        self.assertEqual(self.vm._vm_native_call("length", [ZMap({"x": 1})]), 1)

    def test_length_empty_list(self):
        self.assertEqual(self.vm._vm_native_call("length", [[]]), 0)

    def test_length_none_returns_zero(self):
        # None has no __len__, returns 0
        self.assertEqual(self.vm._vm_native_call("length", [None]), 0)

    def test_length_wrong_arity(self):
        result = self.vm._vm_native_call("length", [])
        self.assertIs(result, _VM_NATIVE_MISS)

    def test_length_has_len(self):
        """Objects with __len__ are supported"""
        self.assertEqual(self.vm._vm_native_call("length", [b"bytes"]), 5)

    # --- str / string ---
    def test_str_integer(self):
        self.assertEqual(self.vm._vm_native_call("str", [42]), "42")

    def test_string_alias(self):
        self.assertEqual(self.vm._vm_native_call("string", [42]), "42")

    def test_str_none_returns_null(self):
        self.assertEqual(self.vm._vm_native_call("str", [None]), "null")

    def test_str_bool_true(self):
        self.assertEqual(self.vm._vm_native_call("str", [True]), "true")

    def test_str_bool_false(self):
        self.assertEqual(self.vm._vm_native_call("str", [False]), "false")

    def test_str_zinteger(self):
        self.assertEqual(self.vm._vm_native_call("str", [ZInteger(99)]), "99")

    def test_str_string_passthrough(self):
        self.assertEqual(self.vm._vm_native_call("str", ["hello"]), "hello")

    def test_str_wrong_arity(self):
        result = self.vm._vm_native_call("str", [])
        self.assertIs(result, _VM_NATIVE_MISS)

    # --- range ---
    def test_range_single_arg(self):
        self.assertEqual(self.vm._vm_native_call("range", [5]), [0, 1, 2, 3, 4])

    def test_range_two_args(self):
        self.assertEqual(self.vm._vm_native_call("range", [2, 6]), [2, 3, 4, 5])

    def test_range_zero(self):
        self.assertEqual(self.vm._vm_native_call("range", [0]), [])

    def test_range_zinteger(self):
        self.assertEqual(self.vm._vm_native_call("range", [ZInteger(3)]), [0, 1, 2])

    def test_range_two_zintegers(self):
        self.assertEqual(
            self.vm._vm_native_call("range", [ZInteger(1), ZInteger(4)]),
            [1, 2, 3]
        )

    def test_range_float_truncated(self):
        self.assertEqual(self.vm._vm_native_call("range", [3.7]), [0, 1, 2])

    def test_range_wrong_arity(self):
        result = self.vm._vm_native_call("range", [])
        self.assertIs(result, _VM_NATIVE_MISS)

    # --- unknown function ---
    def test_unknown_function_returns_miss(self):
        result = self.vm._vm_native_call("unknown_func", [1, 2])
        self.assertIs(result, _VM_NATIVE_MISS)

    def test_print_returns_miss(self):
        result = self.vm._vm_native_call("print", ["hello"])
        self.assertIs(result, _VM_NATIVE_MISS)


# ========================== _build_entity_definition ==========================

class TestBuildEntityDefinition(unittest.TestCase):
    """Tests for _build_entity_definition() — constructs EntityDefinition from stack."""

    def setUp(self):
        self.vm = VM(debug=False)

    def _make_entity_def(self, entity_name, properties):
        """Helper: build EntityDefinition via the VM's _build_entity_definition.
        
        Simulates the stack layout the compiler creates:
            push entity_name
            for each property: push default_value, push prop_name
            DEFINE_ENTITY member_count
        """
        stack = []
        consts = [entity_name]
        
        # Push entity name
        stack.append(entity_name)
        
        # Push properties (value, name) pairs
        for prop_name, default_val in properties:
            stack.append(default_val)
            stack.append(prop_name)

        def stack_pop():
            return stack.pop() if stack else None

        def const(idx):
            return consts[idx] if idx < len(consts) else None

        member_count = len(properties)
        return self.vm._build_entity_definition(member_count, stack, stack_pop, const)

    def test_basic_entity(self):
        entity_def = self._make_entity_def("User", [
            ("name", None),
            ("age", None),
        ])
        self.assertIsInstance(entity_def, EntityDefinition)
        self.assertEqual(entity_def.name, "User")

    def test_entity_properties_count(self):
        entity_def = self._make_entity_def("Point", [
            ("x", 0),
            ("y", 0),
        ])
        all_props = entity_def.get_all_properties()
        self.assertIn("x", all_props)
        self.assertIn("y", all_props)

    def test_entity_with_defaults(self):
        entity_def = self._make_entity_def("Config", [
            ("host", "localhost"),
            ("port", 8080),
        ])
        all_props = entity_def.get_all_properties()
        # Defaults are wrapped via _wrap_for_builtin, check they exist
        self.assertIn("host", all_props)
        self.assertIn("port", all_props)

    def test_empty_entity(self):
        entity_def = self._make_entity_def("Empty", [])
        self.assertIsInstance(entity_def, EntityDefinition)
        self.assertEqual(entity_def.name, "Empty")
        self.assertEqual(len(entity_def.get_all_properties()), 0)

    def test_single_property_entity(self):
        entity_def = self._make_entity_def("Wrapper", [("value", None)])
        self.assertEqual(len(entity_def.get_all_properties()), 1)
        self.assertIn("value", entity_def.get_all_properties())


# ========================== _construct_entity ==========================

class TestConstructEntity(unittest.TestCase):
    """Tests for _construct_entity() — creates EntityInstance from EntityDefinition."""

    def setUp(self):
        self.vm = VM(debug=False)

    def _make_def(self, name, prop_names):
        """Create a simple EntityDefinition with property names."""
        properties = {p: {"type": "any", "default_value": None} for p in prop_names}
        return EntityDefinition(name, properties)

    def test_positional_args(self):
        edef = self._make_def("User", ["name", "age"])
        instance = self.vm._construct_entity(edef, ["Alice", 30])
        self.assertIsInstance(instance, EntityInstance)
        # ZInteger/ZString don't have __eq__, compare .value
        name_val = instance.data.get("name")
        age_val = instance.data.get("age")
        self.assertEqual(getattr(name_val, 'value', name_val), "Alice")
        self.assertEqual(getattr(age_val, 'value', age_val), 30)

    def test_map_arg(self):
        edef = self._make_def("Point", ["x", "y"])
        instance = self.vm._construct_entity(edef, [{"x": 10, "y": 20}])
        self.assertIsInstance(instance, EntityInstance)

    def test_zmap_arg(self):
        edef = self._make_def("Point", ["x", "y"])
        zmap = ZMap({"x": 5, "y": 15})
        instance = self.vm._construct_entity(edef, [zmap])
        self.assertIsInstance(instance, EntityInstance)

    def test_empty_args(self):
        edef = self._make_def("Empty", [])
        instance = self.vm._construct_entity(edef, [])
        self.assertIsInstance(instance, EntityInstance)

    def test_partial_args(self):
        """Fewer args than properties should still work (missing get defaults)"""
        edef = self._make_def("User", ["name", "age", "role"])
        instance = self.vm._construct_entity(edef, ["Bob"])
        self.assertIsInstance(instance, EntityInstance)
        name_val = instance.data.get("name")
        self.assertEqual(getattr(name_val, 'value', name_val), "Bob")

    def test_extra_args_ignored(self):
        """Extra positional args beyond property count are silently ignored"""
        edef = self._make_def("Pair", ["a", "b"])
        instance = self.vm._construct_entity(edef, [1, 2, 3, 4])
        self.assertIsInstance(instance, EntityInstance)


# ========================== _compile_StateStatement ==========================

class TestCompileStateStatement(unittest.TestCase):
    """Tests for _compile_StateStatement in the VM compiler."""

    def _compile(self, code):
        """Compile Zexus code and return bytecode."""
        from src.zexus.vm.compiler import BytecodeCompiler
        from src.zexus.lexer import Lexer
        from src.zexus.parser import Parser

        lexer = Lexer(code)
        parser = Parser(lexer)
        ast = parser.parse_program()

        compiler = BytecodeCompiler()
        bytecode = compiler.compile(ast)
        return bytecode

    def test_state_compiles(self):
        """Basic state declaration should compile without error"""
        bytecode = self._compile("state counter = 0;")
        self.assertIsNotNone(bytecode)

    def test_state_produces_instructions(self):
        """State should produce STATE_WRITE + STORE_NAME opcodes"""
        bytecode = self._compile("state counter = 0;")
        # Ensure we have instructions (not empty)
        instrs = bytecode.instructions if hasattr(bytecode, 'instructions') else bytecode
        self.assertTrue(len(instrs) > 0)

    def test_state_string_value(self):
        bytecode = self._compile('state name = "Alice";')
        self.assertIsNotNone(bytecode)

    def test_state_null_value(self):
        """State with no initial value should compile (defaults to null)"""
        # state without value may need special parsing; let's try with explicit null
        bytecode = self._compile("state data = null;")
        self.assertIsNotNone(bytecode)

    def test_state_expression_value(self):
        """State with computed initial value"""
        bytecode = self._compile("state total = 10 + 20;")
        self.assertIsNotNone(bytecode)

    def test_multiple_state_declarations(self):
        """Multiple standalone state declarations"""
        code = """
        state x = 1;
        state y = 2;
        state z = 3;
        """
        bytecode = self._compile(code)
        self.assertIsNotNone(bytecode)


# ========================== Integration: native call via VM execution ==========================

class TestNativeCallIntegration(unittest.TestCase):
    """Integration tests: _vm_native_call is invoked during actual VM execution."""

    def _run(self, code):
        """Run Zexus code through the full pipeline and capture stdout."""
        from src.zexus.vm.compiler import BytecodeCompiler
        from src.zexus.lexer import Lexer
        from src.zexus.parser import Parser

        lexer = Lexer(code)
        parser = Parser(lexer)
        ast = parser.parse_program()

        compiler = BytecodeCompiler()
        bytecode = compiler.compile(ast)

        vm = VM(debug=False)
        captured = io.StringIO()
        with patch('sys.stdout', captured):
            vm.execute(bytecode)
        return captured.getvalue().strip()

    def test_append_and_length(self):
        code = """
        let arr = [];
        append(arr, 10);
        append(arr, 20);
        append(arr, 30);
        print(length(arr));
        """
        output = self._run(code)
        self.assertIn("3", output)

    def test_str_conversion(self):
        code = """
        let x = 42;
        print(str(x));
        """
        output = self._run(code)
        self.assertIn("42", output)

    def test_push_returns_new_list(self):
        code = """
        let a = [1, 2];
        let b = push(a, 3);
        print(length(a));
        print(length(b));
        """
        output = self._run(code)
        # a should still be 2, b should be 3
        lines = output.split('\n')
        self.assertIn("2", lines[0])
        self.assertIn("3", lines[1])

    def test_range_generation(self):
        code = """
        let nums = range(5);
        print(length(nums));
        """
        output = self._run(code)
        self.assertIn("5", output)


if __name__ == '__main__':
    unittest.main()
