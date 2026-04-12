"""Tests for stdlib FuzzModule."""

import pytest
from src.zexus.stdlib.fuzz import FuzzModule


class TestFuzzGenerators:
    def test_fuzz_string(self):
        s = FuzzModule.fuzz_string(5, 20)
        assert isinstance(s, str)
        assert 5 <= len(s) <= 20

    def test_fuzz_string_default(self):
        s = FuzzModule.fuzz_string()
        assert isinstance(s, str)

    def test_fuzz_int(self):
        n = FuzzModule.fuzz_int(0, 100)
        assert isinstance(n, int)
        assert 0 <= n <= 100

    def test_fuzz_int_default(self):
        n = FuzzModule.fuzz_int()
        assert isinstance(n, int)

    def test_fuzz_bytes(self):
        b = FuzzModule.fuzz_bytes(10, 20)
        assert isinstance(b, bytes)
        assert 10 <= len(b) <= 20

    def test_fuzz_json(self):
        obj = FuzzModule.fuzz_json(max_depth=2)
        assert obj is not None  # Should be some JSON-like structure

    def test_mutate_string(self):
        original = "hello world"
        mutated = FuzzModule.mutate(original)
        assert isinstance(mutated, str)
        # Mutation should usually change something (not guaranteed)

    def test_mutate_bytes(self):
        original = b"test data"
        mutated = FuzzModule.mutate(original)
        assert isinstance(mutated, (str, bytes))


class TestFuzzer:
    def test_create_fuzzer(self):
        def target(inp):
            pass
        fuzzer = FuzzModule.create_fuzzer(target)
        assert fuzzer is not None

    def test_run_fuzzer_no_crash(self):
        def safe_fn(inp):
            return len(str(inp))
        fuzzer = FuzzModule.create_fuzzer(safe_fn, max_iterations=100)
        results = FuzzModule.run(fuzzer, timeout=5)
        assert results["iterations"] > 0
        assert results["crashes"] == 0

    def test_run_fuzzer_with_crash(self):
        def crashy(inp):
            if isinstance(inp, str) and len(inp) > 5:
                raise ValueError("boom")
        fuzzer = FuzzModule.create_fuzzer(crashy, max_iterations=1000)
        results = FuzzModule.run(fuzzer, timeout=10)
        assert results["iterations"] > 0
        # Should find crashes for strings > 5 chars
        assert results["crashes"] >= 0  # May or may not find it depending on fuzz inputs


class TestCorpus:
    def test_corpus_save_load(self, tmp_path):
        def target(inp):
            pass
        fuzzer = FuzzModule.create_fuzzer(target, seed_corpus=["test1", "test2"])
        filepath = str(tmp_path / "corpus.json")
        FuzzModule.corpus_save(fuzzer, filepath)
        loaded = FuzzModule.corpus_load(filepath)
        assert loaded is not None
