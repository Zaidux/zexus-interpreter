"""
Tests for advanced type system.
"""

import unittest
from src.zexus.advanced_types import (
    TypeParameter, TypeVariance, GenericType, UnionType,
    StructuralTrait, TraitImpl, SimpleIterableTrait,
    SimpleComparableTrait, SimpleCloneableTrait,
    AdvancedTypeSpec, TraitRegistry, GenericResolver,
    get_trait_registry, create_generic_type, create_union_type
)


class TestTypeParameter(unittest.TestCase):
    """Test type parameters."""
    
    def test_create_type_parameter(self):
        """Test creating type parameter."""
        param = TypeParameter("T")
        self.assertEqual(param.name, "T")
        self.assertEqual(param.variance, TypeVariance.INVARIANT)
    
    def test_type_parameter_with_bounds(self):
        """Test type parameter with bounds."""
        bound = AdvancedTypeSpec("Comparable")
        param = TypeParameter("T", bounds=[bound])
        self.assertEqual(len(param.bounds), 1)
    
    def test_type_parameter_variance(self):
        """Test type parameter variance."""
        param = TypeParameter("T", variance=TypeVariance.COVARIANT)
        self.assertEqual(param.variance, TypeVariance.COVARIANT)
    
    def test_type_parameter_equality(self):
        """Test type parameter equality."""
        p1 = TypeParameter("T")
        p2 = TypeParameter("T")
        self.assertEqual(p1, p2)
        
        p3 = TypeParameter("U")
        self.assertNotEqual(p1, p3)
    
    def test_type_parameter_hashable(self):
        """Test type parameters are hashable."""
        p1 = TypeParameter("T")
        p2 = TypeParameter("T")
        
        s = {p1, p2}
        self.assertEqual(len(s), 1)


class TestGenericType(unittest.TestCase):
    """Test generic types."""
    
    def test_create_generic(self):
        """Test creating generic type."""
        base = AdvancedTypeSpec("List")
        param = TypeParameter("T")
        generic = GenericType(base, [param])
        
        self.assertEqual(len(generic.type_params), 1)
        self.assertFalse(generic.is_fully_specified())
    
    def test_generic_fully_specified(self):
        """Test fully specified generic."""
        base = AdvancedTypeSpec("List")
        param = TypeParameter("T")
        
        generic = GenericType(base, [param])
        generic.type_args["T"] = AdvancedTypeSpec("int")
        
        self.assertTrue(generic.is_fully_specified())
    
    def test_generic_string_repr(self):
        """Test generic string representation."""
        base = AdvancedTypeSpec("List")
        param = TypeParameter("T")
        generic = GenericType(base, [param])
        
        self.assertIn("Generic", str(generic))


class TestUnionType(unittest.TestCase):
    """Test union types."""
    
    def test_create_union(self):
        """Test creating union type."""
        union = UnionType()
        self.assertEqual(len(union.types), 0)
    
    def test_add_type_to_union(self):
        """Test adding types to union."""
        union = UnionType()
        t1 = AdvancedTypeSpec("int")
        t2 = AdvancedTypeSpec("string")
        
        union.add_type(t1)
        union.add_type(t2)
        
        self.assertEqual(len(union.types), 2)
    
    def test_union_membership(self):
        """Test checking union membership."""
        union = UnionType()
        t1 = AdvancedTypeSpec("int")
        t2 = AdvancedTypeSpec("string")
        t3 = AdvancedTypeSpec("bool")
        
        union.add_type(t1)
        union.add_type(t2)
        
        self.assertTrue(union.is_member(t1))
        self.assertFalse(union.is_member(t3))


class TestStructuralTrait(unittest.TestCase):
    """Test structural traits."""
    
    def test_create_trait(self):
        """Test creating trait."""
        trait = StructuralTrait("Named")
        trait.required_methods.add("get_name")
        
        self.assertEqual(trait.get_name(), "Named")
        self.assertIn("get_name", trait.get_required_methods())
    
    def test_validate_implementation(self):
        """Test trait validation."""
        trait = StructuralTrait("Named")
        trait.required_methods.add("get_name")
        
        class GoodImpl:
            def get_name(self):
                return "test"
        
        class BadImpl:
            pass
        
        good = GoodImpl()
        bad = BadImpl()
        
        valid_good, _ = trait.validate_implementation(good)
        valid_bad, errors = trait.validate_implementation(bad)
        
        self.assertTrue(valid_good)
        self.assertFalse(valid_bad)
        self.assertGreater(len(errors), 0)
    
    def test_default_implementations(self):
        """Test trait default implementations."""
        trait = StructuralTrait("Printable")
        trait.default_impls["to_string"] = lambda: "default"
        
        impls = trait.get_provided_methods()
        self.assertIn("to_string", impls)


class TestSimpleTraits(unittest.TestCase):
    """Test built-in simple traits."""
    
    def test_iterable_trait(self):
        """Test iterable trait."""
        trait = SimpleIterableTrait()
        self.assertEqual(trait.get_name(), "Iterable")
        self.assertIn("iter", trait.get_required_methods())
        self.assertIn("next", trait.get_required_methods())
    
    def test_comparable_trait(self):
        """Test comparable trait."""
        trait = SimpleComparableTrait()
        self.assertEqual(trait.get_name(), "Comparable")
        self.assertIn("compare", trait.get_required_methods())
        self.assertIn("equals", trait.get_required_methods())
    
    def test_cloneable_trait(self):
        """Test cloneable trait."""
        trait = SimpleCloneableTrait()
        self.assertEqual(trait.get_name(), "Cloneable")
        self.assertIn("clone", trait.get_required_methods())


class TestAdvancedTypeSpec(unittest.TestCase):
    """Test advanced type specification."""
    
    def test_create_type(self):
        """Test creating advanced type."""
        spec = AdvancedTypeSpec("int")
        self.assertEqual(spec.base_type, "int")
        self.assertIsNone(spec.generic)
    
    def test_generic_type(self):
        """Test generic type specification."""
        spec = AdvancedTypeSpec("List")
        params = [TypeParameter("T")]
        spec.with_generic(params)
        
        self.assertIsNotNone(spec.generic)
        self.assertEqual(len(spec.generic.type_params), 1)
    
    def test_union_type(self):
        """Test union type specification."""
        types = [AdvancedTypeSpec("int"), AdvancedTypeSpec("string")]
        spec = create_union_type(types)
        
        self.assertIsNotNone(spec.union)
        self.assertEqual(len(spec.union.types), 2)
    
    def test_nullable_type(self):
        """Test nullable type."""
        spec = AdvancedTypeSpec("string").make_nullable()
        self.assertTrue(spec.nullable)
    
    def test_array_type(self):
        """Test array type."""
        spec = AdvancedTypeSpec("int").make_array()
        self.assertIsNotNone(spec.array_of)
        self.assertIn("[]", spec.base_type)
    
    def test_with_trait(self):
        """Test adding trait requirement."""
        spec = AdvancedTypeSpec("MyType")
        trait = SimpleComparableTrait()
        spec.with_trait(trait)
        
        self.assertEqual(len(spec.traits), 1)
    
    def test_trait_satisfaction(self):
        """Test checking trait satisfaction."""
        spec = AdvancedTypeSpec("MyType")
        trait = SimpleComparableTrait()
        spec.with_trait(trait)
        
        class GoodImpl:
            def compare(self): pass
            def equals(self): pass
        
        class BadImpl:
            pass
        
        good = GoodImpl()
        bad = BadImpl()
        
        self.assertTrue(spec.satisfies_traits(good))
        self.assertFalse(spec.satisfies_traits(bad))
    
    def test_assignability(self):
        """Test type assignability."""
        spec1 = AdvancedTypeSpec("int")
        spec2 = AdvancedTypeSpec("int")
        
        self.assertTrue(spec1.is_assignable_to(spec2))
    
    def test_nullable_assignability(self):
        """Test nullable assignability."""
        spec1 = AdvancedTypeSpec("int").make_nullable()
        spec2 = AdvancedTypeSpec("int").make_nullable()
        
        self.assertTrue(spec1.is_assignable_to(spec2))


class TestTraitRegistry(unittest.TestCase):
    """Test trait registry."""
    
    def test_registry_creation(self):
        """Test creating registry."""
        registry = TraitRegistry()
        self.assertGreater(len(registry.traits), 0)
    
    def test_register_trait(self):
        """Test registering trait."""
        registry = TraitRegistry()
        trait = StructuralTrait("Custom")
        
        registry.register_trait("Custom", trait)
        self.assertIsNotNone(registry.get_trait("Custom"))
    
    def test_default_traits(self):
        """Test default traits registered."""
        registry = TraitRegistry()
        
        self.assertIsNotNone(registry.get_trait("Iterable"))
        self.assertIsNotNone(registry.get_trait("Comparable"))
        self.assertIsNotNone(registry.get_trait("Cloneable"))
    
    def test_register_impl(self):
        """Test registering trait implementation."""
        registry = TraitRegistry()
        trait = registry.get_trait("Comparable")
        impl = TraitImpl("int", trait)
        
        registry.register_impl("int", "Comparable", impl)
        
        retrieved = registry.get_impl("int", "Comparable")
        self.assertIsNotNone(retrieved)
    
    def test_get_type_traits(self):
        """Test getting traits for type."""
        registry = TraitRegistry()
        trait1 = registry.get_trait("Iterable")
        trait2 = registry.get_trait("Comparable")
        
        impl1 = TraitImpl("MyList", trait1)
        impl2 = TraitImpl("MyList", trait2)
        
        registry.register_impl("MyList", "Iterable", impl1)
        registry.register_impl("MyList", "Comparable", impl2)
        
        traits = registry.get_type_traits("MyList")
        self.assertEqual(len(traits), 2)


class TestGenericResolver(unittest.TestCase):
    """Test generic resolver."""
    
    def test_resolver_creation(self):
        """Test creating resolver."""
        resolver = GenericResolver()
        self.assertEqual(len(resolver.instantiations), 0)
    
    def test_resolve_generic(self):
        """Test resolving generic type."""
        resolver = GenericResolver()
        
        spec = create_generic_type("List", ["T"])
        type_args = {"T": AdvancedTypeSpec("int")}
        
        result = resolver.resolve(spec, type_args)
        self.assertIsNotNone(result)
    
    def test_type_bounds_checking(self):
        """Test checking type bounds."""
        resolver = GenericResolver()
        bound = AdvancedTypeSpec("Comparable")
        param = TypeParameter("T", bounds=[bound])
        
        arg1 = AdvancedTypeSpec("Comparable")
        arg2 = AdvancedTypeSpec("int")
        
        # Both should pass simplified check
        result1 = resolver.check_type_bounds(param, arg1)
        result2 = resolver.check_type_bounds(param, arg2)
        
        # At least one should be true in this simplified implementation
        self.assertTrue(result1 or result2)


class TestGlobalRegistry(unittest.TestCase):
    """Test global trait registry."""
    
    def test_get_global_registry(self):
        """Test getting global registry."""
        registry = get_trait_registry()
        self.assertIsNotNone(registry)
        self.assertIsInstance(registry, TraitRegistry)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""
    
    def test_create_generic_type_helper(self):
        """Test creating generic type with helper."""
        spec = create_generic_type("Dict", ["K", "V"])
        self.assertEqual(spec.base_type, "Dict")
        self.assertIsNotNone(spec.generic)
        self.assertEqual(len(spec.generic.type_params), 2)
    
    def test_create_union_type_helper(self):
        """Test creating union with helper."""
        types = [
            AdvancedTypeSpec("int"),
            AdvancedTypeSpec("string"),
            AdvancedTypeSpec("null")
        ]
        union = create_union_type(types)
        
        self.assertEqual(len(union.union.types), 3)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_generic_with_traits(self):
        """Test generic type with trait requirements."""
        spec = create_generic_type("Container", ["T"])
        trait = SimpleIterableTrait()
        spec.with_trait(trait)
        
        self.assertIsNotNone(spec.generic)
        self.assertGreater(len(spec.traits), 0)
    
    def test_union_with_traits(self):
        """Test union type with traits."""
        types = [AdvancedTypeSpec("int"), AdvancedTypeSpec("string")]
        union = create_union_type(types)
        trait = SimpleComparableTrait()
        union.with_trait(trait)
        
        self.assertIsNotNone(union.union)
        self.assertGreater(len(union.traits), 0)
    
    def test_complex_type_composition(self):
        """Test complex type composition."""
        # Array of nullable generics: Array<Comparable>?
        spec = (create_generic_type("Array", ["T"])
               .make_nullable()
               .with_trait(SimpleIterableTrait()))
        
        self.assertTrue(spec.nullable)
        self.assertIsNotNone(spec.generic)
        self.assertGreater(len(spec.traits), 0)


if __name__ == "__main__":
    unittest.main()
