"""Tests for the Zexus Kernel extension layer.

Covers: DomainRegistry, Kernel hooks, ZIR opcode catalogue, and the
stdlib "kernel" module integration.
"""

import unittest
from src.zexus.kernel.registry import DomainRegistry, DomainDescriptor
from src.zexus.kernel.hooks import Kernel, KernelEvent, get_kernel
from src.zexus.kernel.zir import CoreOpcode, is_core_opcode, resolve_opcode_name, validate_zir


# =========================================================================
# DomainRegistry
# =========================================================================

class TestDomainRegistry(unittest.TestCase):
    """Unit tests for DomainRegistry."""

    def setUp(self):
        self.registry = DomainRegistry()

    def tearDown(self):
        self.registry.reset()

    # -- Registration -------------------------------------------------------

    def test_register_domain_basic(self):
        desc = self.registry.register_domain(name="test", version="1.0.0")
        self.assertIsInstance(desc, DomainDescriptor)
        self.assertEqual(desc.name, "test")
        self.assertEqual(desc.version, "1.0.0")

    def test_register_domain_with_opcodes(self):
        opcodes = {0x5000: "OP_A", 0x5001: "OP_B"}
        desc = self.registry.register_domain(name="demo", opcodes=opcodes)
        self.assertEqual(desc.opcodes, opcodes)

    def test_register_duplicate_raises(self):
        self.registry.register_domain(name="dup")
        with self.assertRaises(ValueError):
            self.registry.register_domain(name="dup")

    def test_register_opcode_collision_raises(self):
        self.registry.register_domain(name="a", opcodes={0x5000: "OP_X"})
        with self.assertRaises(ValueError):
            self.registry.register_domain(name="b", opcodes={0x5000: "OP_Y"})

    def test_register_with_description(self):
        desc = self.registry.register_domain(
            name="desc_test", description="A test domain"
        )
        self.assertEqual(desc.description, "A test domain")

    def test_register_with_dependencies(self):
        desc = self.registry.register_domain(
            name="dep_test", dependencies=["core", "crypto"]
        )
        self.assertEqual(desc.dependencies, ["core", "crypto"])

    # -- Unregistration -----------------------------------------------------

    def test_unregister_domain(self):
        self.registry.register_domain(name="temp", opcodes={0x6000: "OP_Z"})
        self.assertIsNotNone(self.registry.get_domain("temp"))
        self.registry.unregister_domain("temp")
        self.assertIsNone(self.registry.get_domain("temp"))
        # Opcode should be freed
        self.assertIsNone(self.registry.resolve_opcode(0x6000))

    def test_unregister_nonexistent_is_noop(self):
        self.registry.unregister_domain("nope")  # should not raise

    # -- Query --------------------------------------------------------------

    def test_get_domain_existing(self):
        self.registry.register_domain(name="q1")
        self.assertIsNotNone(self.registry.get_domain("q1"))

    def test_get_domain_missing(self):
        self.assertIsNone(self.registry.get_domain("nonexistent"))

    def test_list_domains(self):
        self.registry.register_domain(name="d1")
        self.registry.register_domain(name="d2")
        names = {d.name for d in self.registry.list_domains()}
        self.assertEqual(names, {"d1", "d2"})

    def test_domain_names_property(self):
        self.registry.register_domain(name="x")
        self.assertIn("x", self.registry.domain_names)

    def test_resolve_opcode(self):
        self.registry.register_domain(name="owner", opcodes={0x7000: "OP_FOO"})
        self.assertEqual(self.registry.resolve_opcode(0x7000), "owner")

    def test_resolve_unknown_opcode(self):
        self.assertIsNone(self.registry.resolve_opcode(0xFFFF))

    # -- Dependencies -------------------------------------------------------

    def test_check_dependencies_satisfied(self):
        self.registry.register_domain(name="base")
        self.registry.register_domain(name="child", dependencies=["base"])
        self.assertEqual(self.registry.check_dependencies("child"), [])

    def test_check_dependencies_missing(self):
        self.registry.register_domain(name="child2", dependencies=["missing_dep"])
        self.assertEqual(self.registry.check_dependencies("child2"), ["missing_dep"])

    def test_check_dependencies_unknown_domain(self):
        self.assertEqual(self.registry.check_dependencies("ghost"), ["ghost"])

    # -- Listeners ----------------------------------------------------------

    def test_on_domain_registered_listener(self):
        captured = []
        self.registry.on_domain_registered(lambda desc: captured.append(desc.name))
        self.registry.register_domain(name="listen_test")
        self.assertEqual(captured, ["listen_test"])

    def test_listener_exception_does_not_break_registration(self):
        def bad_listener(desc):
            raise RuntimeError("boom")

        self.registry.on_domain_registered(bad_listener)
        # Should still succeed
        desc = self.registry.register_domain(name="resilient")
        self.assertIsNotNone(desc)

    # -- Reset --------------------------------------------------------------

    def test_reset_clears_everything(self):
        self.registry.register_domain(name="r1", opcodes={0x8000: "OP"})
        self.registry.reset()
        self.assertEqual(len(self.registry.list_domains()), 0)
        self.assertIsNone(self.registry.resolve_opcode(0x8000))

    # -- Repr ---------------------------------------------------------------

    def test_repr(self):
        self.registry.register_domain(name="alpha")
        r = repr(self.registry)
        self.assertIn("alpha", r)
        self.assertIn("DomainRegistry", r)


# =========================================================================
# Kernel
# =========================================================================

class TestKernel(unittest.TestCase):
    """Unit tests for the Kernel extension layer."""

    def setUp(self):
        self.registry = DomainRegistry()
        self.kernel = Kernel(registry=self.registry)

    def tearDown(self):
        self.registry.reset()

    # -- Boot ---------------------------------------------------------------

    def test_boot_registers_builtin_domains(self):
        self.kernel.boot()
        self.assertTrue(self.kernel.is_booted)
        names = self.registry.domain_names
        self.assertIn("blockchain", names)
        self.assertIn("web", names)
        self.assertIn("system", names)
        self.assertIn("ui", names)

    def test_boot_idempotent(self):
        self.kernel.boot()
        self.kernel.boot()  # second call is a no-op
        self.assertEqual(len(self.registry.list_domains()), 4)

    def test_boot_returns_self(self):
        result = self.kernel.boot()
        self.assertIs(result, self.kernel)

    # -- Opcode resolution --------------------------------------------------

    def test_register_and_get_opcode_handler(self):
        handler = lambda *a: "result"
        self.kernel.register_opcode_handler(0x1000, handler)
        self.assertIs(self.kernel.get_opcode_handler(0x1000), handler)

    def test_get_opcode_handler_missing(self):
        self.assertIsNone(self.kernel.get_opcode_handler(0xDEAD))

    def test_resolve_opcode_domain(self):
        self.kernel.boot()
        self.assertEqual(self.kernel.resolve_opcode_domain(0x1000), "blockchain")
        self.assertEqual(self.kernel.resolve_opcode_domain(0x1100), "web")
        self.assertIsNone(self.kernel.resolve_opcode_domain(0xFFFF))

    # -- Security -----------------------------------------------------------

    def test_check_security_no_policies(self):
        """With no domains having policies, everything is approved."""
        self.kernel.boot()
        self.assertTrue(self.kernel.check_security("anything"))

    def test_check_security_rejecting_policy(self):
        class RejectAll:
            def check(self, op, ctx):
                return False

        self.registry.register_domain(
            name="strict", security_policy=RejectAll()
        )
        self.assertFalse(self.kernel.check_security("write"))

    def test_check_security_approving_policy(self):
        class ApproveAll:
            def check(self, op, ctx):
                return True

        self.registry.register_domain(
            name="lenient", security_policy=ApproveAll()
        )
        self.assertTrue(self.kernel.check_security("read"))

    def test_check_security_exception_in_policy_rejects(self):
        class Broken:
            def check(self, op, ctx):
                raise RuntimeError("crash")

        self.registry.register_domain(name="broken", security_policy=Broken())
        self.assertFalse(self.kernel.check_security("op"))

    # -- Middleware ----------------------------------------------------------

    def test_middleware_transforms_data(self):
        self.kernel.use(lambda stage, data: data.upper() if isinstance(data, str) else data)
        result = self.kernel.run_middleware("compile", "hello")
        self.assertEqual(result, "HELLO")

    def test_middleware_chain(self):
        self.kernel.use(lambda stage, data: data + "_a")
        self.kernel.use(lambda stage, data: data + "_b")
        result = self.kernel.run_middleware("test", "start")
        self.assertEqual(result, "start_a_b")

    def test_middleware_exception_is_swallowed(self):
        self.kernel.use(lambda stage, data: (_ for _ in ()).throw(RuntimeError("boom")))
        self.kernel.use(lambda stage, data: data + "_ok")
        result = self.kernel.run_middleware("test", "x")
        # First middleware fails, second still runs
        self.assertEqual(result, "x_ok")

    # -- Events -------------------------------------------------------------

    def test_boot_emits_event(self):
        events = []
        self.kernel.on("kernel.booted", lambda e: events.append(e))
        self.kernel.boot()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, "kernel.booted")

    def test_wildcard_listener(self):
        events = []
        self.kernel.on("*", lambda e: events.append(e.name))
        self.kernel.boot()
        self.assertIn("kernel.booted", events)

    def test_event_listener_exception_is_swallowed(self):
        self.kernel.on("kernel.booted", lambda e: 1 / 0)
        self.kernel.boot()  # should not raise

    # -- Status -------------------------------------------------------------

    def test_status_before_boot(self):
        status = self.kernel.status()
        self.assertFalse(status["booted"])
        self.assertEqual(status["domain_count"], 0)

    def test_status_after_boot(self):
        self.kernel.boot()
        status = self.kernel.status()
        self.assertTrue(status["booted"])
        self.assertEqual(status["domain_count"], 4)
        self.assertIn("blockchain", status["domains"])

    # -- Repr ---------------------------------------------------------------

    def test_repr_idle(self):
        r = repr(self.kernel)
        self.assertIn("idle", r)

    def test_repr_booted(self):
        self.kernel.boot()
        r = repr(self.kernel)
        self.assertIn("booted", r)
        self.assertIn("4 domains", r)


# =========================================================================
# KernelEvent
# =========================================================================

class TestKernelEvent(unittest.TestCase):
    def test_event_creation(self):
        e = KernelEvent("test.event", {"key": "val"})
        self.assertEqual(e.name, "test.event")
        self.assertEqual(e.data["key"], "val")
        self.assertGreater(e.timestamp, 0)

    def test_event_repr(self):
        e = KernelEvent("x")
        self.assertIn("x", repr(e))


# =========================================================================
# ZIR — CoreOpcode
# =========================================================================

class TestCoreOpcode(unittest.TestCase):
    def test_load_const_value(self):
        self.assertEqual(CoreOpcode.LOAD_CONST, 0x0001)

    def test_all_opcodes_unique(self):
        values = [op.value for op in CoreOpcode]
        self.assertEqual(len(values), len(set(values)))

    def test_is_core_opcode_true(self):
        self.assertTrue(is_core_opcode(0x0001))
        self.assertTrue(is_core_opcode(0x0100))

    def test_is_core_opcode_false(self):
        self.assertFalse(is_core_opcode(0x9999))
        self.assertFalse(is_core_opcode(0x1000))  # domain opcode

    def test_resolve_opcode_name_core(self):
        self.assertEqual(resolve_opcode_name(0x0001), "LOAD_CONST")
        self.assertEqual(resolve_opcode_name(0x0100), "JUMP")

    def test_resolve_opcode_name_unknown(self):
        self.assertIsNone(resolve_opcode_name(0x9999))


# =========================================================================
# ZIR — validate_zir
# =========================================================================

class TestValidateZir(unittest.TestCase):
    def setUp(self):
        # validate_zir uses the global registry, so boot the global kernel
        from src.zexus.kernel import get_kernel
        self._kernel = get_kernel()
        self._kernel.boot()

    def test_valid_core_only(self):
        instructions = [(0x0001,), (0x0010,), (0x0100,)]
        self.assertEqual(validate_zir(instructions), [])

    def test_unknown_opcode(self):
        errors = validate_zir([(0x9999,)])
        self.assertEqual(len(errors), 1)
        self.assertIn("0x9999", errors[0])

    def test_domain_restricted_pass(self):
        instructions = [(0x1000,)]  # blockchain opcode
        errors = validate_zir(instructions, allowed_domains={"blockchain"})
        self.assertEqual(errors, [])

    def test_domain_restricted_fail(self):
        instructions = [(0x1100,)]  # web opcode
        errors = validate_zir(instructions, allowed_domains={"blockchain"})
        self.assertEqual(len(errors), 1)
        self.assertIn("web", errors[0])

    def test_mixed_instructions(self):
        instructions = [(0x0001,), (0x1000,), (0x1100,), (0x9999,)]
        errors = validate_zir(instructions)
        # Only 0x9999 is unknown
        self.assertEqual(len(errors), 1)

    def test_empty_instructions(self):
        self.assertEqual(validate_zir([]), [])


# =========================================================================
# Singleton
# =========================================================================

class TestSingleton(unittest.TestCase):
    def test_get_kernel_returns_same_instance(self):
        k1 = get_kernel()
        k2 = get_kernel()
        self.assertIs(k1, k2)

    def test_get_kernel_is_bootable(self):
        k = get_kernel()
        k.boot()
        self.assertTrue(k.is_booted)


# =========================================================================
# DomainDescriptor
# =========================================================================

class TestDomainDescriptor(unittest.TestCase):
    def test_default_values(self):
        d = DomainDescriptor(name="t")
        self.assertEqual(d.version, "0.0.0")
        self.assertEqual(d.opcodes, {})
        self.assertIsNone(d.security_policy)
        self.assertEqual(d.dependencies, [])

    def test_repr(self):
        d = DomainDescriptor(name="x", version="2.0", opcodes={1: "A", 2: "B"})
        r = repr(d)
        self.assertIn("x", r)
        self.assertIn("2.0", r)
        self.assertIn("2", r)  # 2 opcodes


if __name__ == "__main__":
    unittest.main()
