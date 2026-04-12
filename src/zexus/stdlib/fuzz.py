"""Coverage-guided fuzzing module for Zexus standard library."""

import random
import os
import json
import time
import hashlib
import traceback
import string


class FuzzModule:
    """Provides coverage-guided fuzzing utilities for testing and security analysis."""

    @staticmethod
    def create_fuzzer(target_fn, seed_corpus=None, max_iterations=10000):
        """Create a fuzzer context targeting a callable.

        Args:
            target_fn: The function to fuzz.
            seed_corpus: Optional list of initial inputs to seed the fuzzer.
            max_iterations: Maximum number of fuzzing iterations.

        Returns:
            A fuzzer context dict.
        """
        return {
            "target_fn": target_fn,
            "seed_corpus": list(seed_corpus) if seed_corpus else [],
            "corpus": list(seed_corpus) if seed_corpus else [],
            "max_iterations": max_iterations,
            "crashes": [],
            "crash_hashes": set(),
            "coverage": set(),
            "iterations": 0,
            "start_time": None,
            "end_time": None,
        }

    @staticmethod
    def run(fuzzer, timeout=60):
        """Run the fuzzer until max_iterations or timeout is reached.

        Args:
            fuzzer: A fuzzer context dict from create_fuzzer.
            timeout: Maximum runtime in seconds.

        Returns:
            A results dict with iterations, crashes, unique_crashes,
            coverage_pct, and crash_details.
        """
        target_fn = fuzzer["target_fn"]
        max_iterations = fuzzer["max_iterations"]
        corpus = fuzzer["corpus"]
        crashes = fuzzer["crashes"]
        crash_hashes = fuzzer["crash_hashes"]
        coverage = fuzzer["coverage"]

        fuzzer["start_time"] = time.time()
        deadline = fuzzer["start_time"] + timeout

        for i in range(max_iterations):
            if time.time() >= deadline:
                break

            fuzzer["iterations"] = i + 1

            if corpus:
                base_input = random.choice(corpus)
                test_input = FuzzModule.mutate(base_input)
            else:
                test_input = FuzzModule.fuzz_string(min_len=1, max_len=256)

            try:
                result = target_fn(test_input)
                input_hash = hashlib.sha256(
                    repr(test_input).encode()
                ).hexdigest()[:16]
                coverage.add(input_hash)

                if len(corpus) < 10000:
                    corpus.append(test_input)
            except Exception as e:
                tb = traceback.format_exc()
                crash_hash = hashlib.sha256(tb.encode()).hexdigest()

                crash_detail = {
                    "input": repr(test_input),
                    "exception": str(e),
                    "exception_type": type(e).__name__,
                    "traceback_hash": crash_hash,
                    "iteration": i + 1,
                    "timestamp": time.time(),
                }

                crashes.append(crash_detail)
                crash_hashes.add(crash_hash)

        fuzzer["end_time"] = time.time()

        total_possible = max(fuzzer["iterations"], 1)
        coverage_pct = round((len(coverage) / total_possible) * 100, 2)

        return {
            "iterations": fuzzer["iterations"],
            "crashes": len(crashes),
            "unique_crashes": len(crash_hashes),
            "coverage_pct": coverage_pct,
            "crash_details": list(crashes),
            "duration": fuzzer["end_time"] - fuzzer["start_time"],
        }

    @staticmethod
    def fuzz_string(min_len=0, max_len=1000):
        """Generate a random string input for fuzzing.

        Args:
            min_len: Minimum string length.
            max_len: Maximum string length.

        Returns:
            A random string.
        """
        length = random.randint(min_len, max_len)
        charset = string.printable
        return "".join(random.choice(charset) for _ in range(length))

    @staticmethod
    def fuzz_int(min_val=-(2**31), max_val=2**31):
        """Generate a random integer input for fuzzing.

        Args:
            min_val: Minimum integer value.
            max_val: Maximum integer value.

        Returns:
            A random integer, occasionally picking boundary values.
        """
        boundary_values = [0, 1, -1, min_val, max_val, min_val + 1, max_val - 1]
        if random.random() < 0.2:
            return random.choice(boundary_values)
        return random.randint(min_val, max_val)

    @staticmethod
    def fuzz_bytes(min_len=0, max_len=1000):
        """Generate random bytes for fuzzing.

        Args:
            min_len: Minimum byte length.
            max_len: Maximum byte length.

        Returns:
            Random bytes.
        """
        length = random.randint(min_len, max_len)
        return os.urandom(length)

    @staticmethod
    def fuzz_json(max_depth=3):
        """Generate a random JSON-like structure for fuzzing.

        Args:
            max_depth: Maximum nesting depth.

        Returns:
            A random JSON-compatible Python structure.
        """
        return FuzzModule._random_json_value(max_depth)

    @staticmethod
    def _random_json_value(depth):
        """Recursively build a random JSON value."""
        if depth <= 0:
            return random.choice([
                random.randint(-1000, 1000),
                random.random(),
                FuzzModule.fuzz_string(min_len=0, max_len=20),
                True,
                False,
                None,
            ])

        kind = random.choice(["object", "array", "string", "number", "bool", "null"])

        if kind == "object":
            size = random.randint(0, 5)
            return {
                FuzzModule.fuzz_string(min_len=1, max_len=10): FuzzModule._random_json_value(depth - 1)
                for _ in range(size)
            }
        elif kind == "array":
            size = random.randint(0, 5)
            return [FuzzModule._random_json_value(depth - 1) for _ in range(size)]
        elif kind == "string":
            return FuzzModule.fuzz_string(min_len=0, max_len=50)
        elif kind == "number":
            return random.choice([random.randint(-10000, 10000), random.random()])
        elif kind == "bool":
            return random.choice([True, False])
        else:
            return None

    @staticmethod
    def mutate(data):
        """Mutate existing input data using various strategies.

        Strategies: bit flip, byte insert, byte delete, boundary value substitution.

        Args:
            data: Input data (str or bytes) to mutate.

        Returns:
            Mutated data of the same type as input.
        """
        if isinstance(data, bytes):
            return FuzzModule._mutate_bytes(data)
        elif isinstance(data, str):
            return FuzzModule._mutate_string(data)
        elif isinstance(data, int):
            return FuzzModule._mutate_int(data)
        else:
            return data

    @staticmethod
    def _mutate_bytes(data):
        """Apply mutation strategies to bytes."""
        if not data:
            return os.urandom(random.randint(1, 10))

        buf = bytearray(data)
        strategy = random.choice(["bit_flip", "byte_insert", "byte_delete", "boundary"])

        if strategy == "bit_flip" and buf:
            idx = random.randint(0, len(buf) - 1)
            bit = 1 << random.randint(0, 7)
            buf[idx] ^= bit
        elif strategy == "byte_insert":
            idx = random.randint(0, len(buf))
            buf.insert(idx, random.randint(0, 255))
        elif strategy == "byte_delete" and len(buf) > 1:
            idx = random.randint(0, len(buf) - 1)
            del buf[idx]
        elif strategy == "boundary" and buf:
            idx = random.randint(0, len(buf) - 1)
            buf[idx] = random.choice([0x00, 0xFF, 0x7F, 0x80])

        return bytes(buf)

    @staticmethod
    def _mutate_string(data):
        """Apply mutation strategies to strings."""
        if not data:
            return FuzzModule.fuzz_string(min_len=1, max_len=10)

        chars = list(data)
        strategy = random.choice(["bit_flip", "byte_insert", "byte_delete", "boundary"])

        if strategy == "bit_flip" and chars:
            idx = random.randint(0, len(chars) - 1)
            c = ord(chars[idx])
            bit = 1 << random.randint(0, 6)
            chars[idx] = chr((c ^ bit) % 0x110000)
        elif strategy == "byte_insert":
            idx = random.randint(0, len(chars))
            chars.insert(idx, random.choice(string.printable))
        elif strategy == "byte_delete" and len(chars) > 1:
            idx = random.randint(0, len(chars) - 1)
            del chars[idx]
        elif strategy == "boundary":
            idx = random.randint(0, len(chars) - 1)
            chars[idx] = random.choice(["\x00", "\xff", "\n", "\r", "'", '"', "\\"])

        return "".join(chars)

    @staticmethod
    def _mutate_int(data):
        """Apply mutation strategies to integers."""
        strategy = random.choice(["bit_flip", "boundary", "arithmetic"])

        if strategy == "bit_flip":
            bit = 1 << random.randint(0, 31)
            return data ^ bit
        elif strategy == "boundary":
            return random.choice([0, 1, -1, 2**31 - 1, -(2**31), 2**32 - 1])
        else:
            delta = random.choice([-1, 1, -256, 256, -65536, 65536])
            return data + delta

    @staticmethod
    def create_contract_fuzzer(contract_abi, seed_inputs=None):
        """Create a fuzzer context for smart contract ABIs.

        Args:
            contract_abi: A list of ABI entries (dicts with 'name', 'inputs', etc.).
            seed_inputs: Optional dict mapping function names to lists of seed inputs.

        Returns:
            A contract fuzzer context dict.
        """
        functions = []
        for entry in contract_abi:
            if entry.get("type", "function") == "function":
                functions.append({
                    "name": entry.get("name", ""),
                    "inputs": entry.get("inputs", []),
                })

        return {
            "contract_abi": contract_abi,
            "functions": functions,
            "seed_inputs": seed_inputs or {},
            "corpus": {},
            "crashes": [],
            "crash_hashes": set(),
            "iterations": 0,
            "call_sequences": [],
        }

    @staticmethod
    def corpus_save(fuzzer, filepath):
        """Save the fuzzer corpus to a JSON file on disk.

        Args:
            fuzzer: A fuzzer context dict.
            filepath: Path to write the corpus file.
        """
        serializable_corpus = []
        for item in fuzzer.get("corpus", []):
            if isinstance(item, bytes):
                serializable_corpus.append({"type": "bytes", "data": item.hex()})
            else:
                serializable_corpus.append({"type": "other", "data": repr(item)})

        payload = {
            "corpus": serializable_corpus,
            "iterations": fuzzer.get("iterations", 0),
            "crash_hashes": list(fuzzer.get("crash_hashes", set())),
            "crashes": fuzzer.get("crashes", []),
        }

        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def corpus_load(filepath):
        """Load a corpus from a JSON file on disk.

        Args:
            filepath: Path to the corpus file.

        Returns:
            A dict with corpus data and metadata.
        """
        with open(filepath, "r") as f:
            payload = json.load(f)

        corpus = []
        for item in payload.get("corpus", []):
            if item.get("type") == "bytes":
                corpus.append(bytes.fromhex(item["data"]))
            else:
                corpus.append(item.get("data", ""))

        return {
            "corpus": corpus,
            "iterations": payload.get("iterations", 0),
            "crash_hashes": set(payload.get("crash_hashes", [])),
            "crashes": payload.get("crashes", []),
        }
