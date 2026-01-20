#!/usr/bin/env python3
"""Profile Zexus interpreter + VM on large blockchain workloads.

Targets by default:
- blockchain_test/perf_full_network_10k.zx
- blockchain_test/full_network_chain/full_network_blockchain.zx

Outputs:
- Console summary
- JSON reports per run
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure local src is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from zexus.lexer import Lexer
from zexus.parser import UltimateParser
from zexus.evaluator import evaluate, Evaluator
from zexus.object import Environment, String
from zexus.vm.vm import VM
from zexus.vm.profiler import ProfilingLevel
import zexus.vm.vm as vm_module

DEFAULT_FILES = [
    REPO_ROOT / "blockchain_test" / "perf_full_network_10k.zx",
    REPO_ROOT / "blockchain_test" / "full_network_chain" / "full_network_blockchain.zx",
]


def _disable_fastops():
    """Disable Cython fastops to avoid segfaults during profiling."""
    try:
        vm_module._FASTOPS_AVAILABLE = False
        vm_module._fastops = None
    except Exception:
        pass


@dataclass
class ProfileConfig:
    syntax_style: str
    top_n: int
    enable_memory: bool
    enable_vm_profiler: bool
    vm_profiler_level: str
    vm_profiler_sample_rate: float
    enable_opcode_profile: bool
    enable_gas_metering: bool
    gas_limit: Optional[int]


@dataclass
class ProfileSummary:
    name: str
    elapsed_ms: float
    memory_peak_mb: float
    top_cumulative: List[Dict[str, Any]]
    top_self: List[Dict[str, Any]]


@dataclass
class VmExtras:
    vm_stats: Dict[str, Any]
    vm_profiler_summary: Optional[Dict[str, Any]]
    vm_hottest_instructions: List[Dict[str, Any]]
    vm_slowest_instructions: List[Dict[str, Any]]
    vm_hot_loops: List[Dict[str, Any]]
    vm_opcode_profile: List[Dict[str, Any]]


@dataclass
class FileReport:
    file: str
    parse_time_ms: float
    interpreter: Optional[ProfileSummary]
    vm: Optional[ProfileSummary]
    vm_extras: Optional[VmExtras]
    errors: List[str]


def _parse_file(path: Path, syntax_style: str) -> Tuple[Any, str, float]:
    source = path.read_text(encoding="utf-8")
    lexer = Lexer(source)
    setattr(lexer, "filename", str(path))
    parser = UltimateParser(lexer, syntax_style)
    start = time.perf_counter()
    program = parser.parse_program()
    parse_time = (time.perf_counter() - start) * 1000
    if parser.errors:
        raise RuntimeError(f"Parse errors: {parser.errors}")
    return program, source, parse_time


def _extract_top(stats: pstats.Stats, sort_key: str, top_n: int) -> List[Dict[str, Any]]:
    stats.sort_stats(sort_key)
    results = []
    for func in stats.fcn_list[:top_n]:
        cc, nc, tt, ct, callers = stats.stats[func]
        filename, line, name = func
        results.append({
            "function": name,
            "file": filename,
            "line": line,
            "callcount": nc,
            "primitive_calls": cc,
            "self_time_s": tt,
            "cumulative_time_s": ct,
        })
    return results


def _profile_call(fn, enable_memory: bool) -> Tuple[float, float, pstats.Stats]:
    if enable_memory:
        tracemalloc.start()
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    fn()
    profiler.disable()
    elapsed_ms = (time.perf_counter() - start) * 1000
    memory_peak = 0.0
    if enable_memory:
        _, peak = tracemalloc.get_traced_memory()
        memory_peak = peak / 1024 / 1024
        tracemalloc.stop()
    stats = pstats.Stats(profiler)
    return elapsed_ms, memory_peak, stats


def _build_env(file_path: Path) -> Environment:
    env = Environment()
    try:
        env.set("__file__", String(str(file_path)))
    except Exception:
        pass
    return env


def _profile_interpreter(program, file_path: Path, cfg: ProfileConfig) -> ProfileSummary:
    env = _build_env(file_path)
    setattr(env, "disable_vm", True)

    def run():
        evaluate(program, env, use_vm=False)

    elapsed_ms, memory_peak_mb, stats = _profile_call(run, cfg.enable_memory)
    return ProfileSummary(
        name="interpreter",
        elapsed_ms=elapsed_ms,
        memory_peak_mb=memory_peak_mb,
        top_cumulative=_extract_top(stats, "cumulative", cfg.top_n),
        top_self=_extract_top(stats, "time", cfg.top_n),
    )


def _profile_vm(program, file_path: Path, cfg: ProfileConfig) -> Tuple[ProfileSummary, VmExtras]:
    env = _build_env(file_path)
    evaluator = Evaluator(use_vm=True)

    vm_profiler_level = getattr(ProfilingLevel, cfg.vm_profiler_level, ProfilingLevel.DETAILED)
    vm = VM(
        use_jit=True,
        jit_threshold=100,
        enable_gas_metering=cfg.enable_gas_metering,
        gas_limit=cfg.gas_limit,
        enable_profiling=cfg.enable_vm_profiler,
        profiling_level=vm_profiler_level.name,
        profiling_sample_rate=cfg.vm_profiler_sample_rate,
        profiling_track_overhead=False,
    )
    evaluator.vm_instance = vm
    evaluator.use_vm = True

    opcode_env_key = "ZEXUS_VM_PROFILE_OPS"
    prev_opcode = os.environ.get(opcode_env_key)

    if cfg.enable_opcode_profile:
        os.environ[opcode_env_key] = "1"

    def run():
        evaluator.eval_with_vm_support(program, env)

    try:
        elapsed_ms, memory_peak_mb, stats = _profile_call(run, cfg.enable_memory)
    finally:
        if cfg.enable_opcode_profile:
            if prev_opcode is None:
                os.environ.pop(opcode_env_key, None)
            else:
                os.environ[opcode_env_key] = prev_opcode

    summary = ProfileSummary(
        name="vm",
        elapsed_ms=elapsed_ms,
        memory_peak_mb=memory_peak_mb,
        top_cumulative=_extract_top(stats, "cumulative", cfg.top_n),
        top_self=_extract_top(stats, "time", cfg.top_n),
    )

    profiler_summary = None
    hottest: List[Dict[str, Any]] = []
    slowest: List[Dict[str, Any]] = []
    hot_loops: List[Dict[str, Any]] = []
    if vm.profiler and vm.profiler.enabled:
        profiler_summary = vm.profiler.get_summary()
        hottest = [stat.to_dict() for stat in vm.profiler.get_hottest_instructions(cfg.top_n)]
        slowest = [stat.to_dict() for stat in vm.profiler.get_slowest_instructions(cfg.top_n)]
        hot_loops = [loop.to_dict() for loop in vm.profiler.get_hot_loops()]

    opcode_profile: List[Dict[str, Any]] = []
    if getattr(vm, "_last_opcode_profile", None):
        for op_name, count in vm._last_opcode_profile[: cfg.top_n]:
            opcode_profile.append({"opcode": op_name, "count": count})

    vm_extras = VmExtras(
        vm_stats=evaluator.get_full_vm_statistics(),
        vm_profiler_summary=profiler_summary,
        vm_hottest_instructions=hottest,
        vm_slowest_instructions=slowest,
        vm_hot_loops=hot_loops,
        vm_opcode_profile=opcode_profile,
    )

    return summary, vm_extras


def _write_report(output_dir: Path, report: FileReport) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(report.file).stem}_profile.json"
    payload = {
        "file": report.file,
        "parse_time_ms": report.parse_time_ms,
        "interpreter": asdict(report.interpreter) if report.interpreter else None,
        "vm": asdict(report.vm) if report.vm else None,
        "vm_extras": asdict(report.vm_extras) if report.vm_extras else None,
        "errors": report.errors,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _print_summary(report: FileReport, output_path: Path):
    print("=" * 90)
    print(f"File: {report.file}")
    print(f"Parse time: {report.parse_time_ms:.2f} ms")
    if report.interpreter:
        print(f"Interpreter: {report.interpreter.elapsed_ms:.2f} ms | Peak MB: {report.interpreter.memory_peak_mb:.2f}")
    if report.vm:
        print(f"VM: {report.vm.elapsed_ms:.2f} ms | Peak MB: {report.vm.memory_peak_mb:.2f}")
    if report.vm_extras and report.vm_extras.vm_opcode_profile:
        print("Top VM opcodes:")
        for entry in report.vm_extras.vm_opcode_profile[:10]:
            print(f"  {entry['opcode']}: {entry['count']}")
    if report.errors:
        print("Errors:")
        for err in report.errors:
            print(f"  - {err}")
    print(f"Report written: {output_path}")
    print("=" * 90)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile Zexus VM + interpreter on large blockchain workloads")
    parser.add_argument("files", nargs="*", help="ZX files to profile")
    parser.add_argument("--syntax-style", default="auto", choices=["auto", "universal", "tolerable"])
    parser.add_argument("--top", type=int, default=25, help="Top N functions/opcodes to report")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory profiling")
    parser.add_argument("--no-vm-profiler", action="store_true", help="Disable VM instruction profiler")
    parser.add_argument("--vm-profiler-level", default="DETAILED", choices=["BASIC", "DETAILED", "FULL"])
    parser.add_argument("--vm-profiler-sample-rate", type=float, default=1.0)
    parser.add_argument("--no-opcode-profile", action="store_true", help="Disable VM opcode counting")
    parser.add_argument("--no-gas", action="store_true", help="Disable gas metering in VM")
    parser.add_argument("--gas-limit", type=int, default=None, help="Override gas limit for VM")
    parser.add_argument("--skip-interpreter", action="store_true", help="Skip interpreter profiling")
    parser.add_argument("--skip-vm", action="store_true", help="Skip VM profiling")
    parser.add_argument("--parse-only", action="store_true", help="Only parse files (no execution)")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "tmp" / "perf_reports"))
    args = parser.parse_args()

    _disable_fastops()

    targets = [Path(p) for p in args.files] if args.files else list(DEFAULT_FILES)
    output_base = Path(args.output_dir)
    output_dir = output_base / datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = ProfileConfig(
        syntax_style=args.syntax_style,
        top_n=args.top,
        enable_memory=not args.no_memory,
        enable_vm_profiler=not args.no_vm_profiler,
        vm_profiler_level=args.vm_profiler_level,
        vm_profiler_sample_rate=args.vm_profiler_sample_rate,
        enable_opcode_profile=not args.no_opcode_profile,
        enable_gas_metering=not args.no_gas,
        gas_limit=args.gas_limit,
    )

    for path in targets:
        errors: List[str] = []
        interpreter_summary = None
        vm_summary = None
        vm_extras = None
        parse_time_ms = 0.0
        if not path.exists():
            errors.append(f"Missing file: {path}")
            report = FileReport(
                file=str(path),
                parse_time_ms=parse_time_ms,
                interpreter=interpreter_summary,
                vm=vm_summary,
                vm_extras=vm_extras,
                errors=errors,
            )
            output_path = _write_report(output_dir, report)
            _print_summary(report, output_path)
            continue

        try:
            program, _, parse_time_ms = _parse_file(path, cfg.syntax_style)
        except Exception as exc:
            errors.append(str(exc))
            report = FileReport(
                file=str(path),
                parse_time_ms=parse_time_ms,
                interpreter=interpreter_summary,
                vm=vm_summary,
                vm_extras=vm_extras,
                errors=errors,
            )
            output_path = _write_report(output_dir, report)
            _print_summary(report, output_path)
            continue

        if args.parse_only:
            report = FileReport(
                file=str(path),
                parse_time_ms=parse_time_ms,
                interpreter=interpreter_summary,
                vm=vm_summary,
                vm_extras=vm_extras,
                errors=errors,
            )
            output_path = _write_report(output_dir, report)
            _print_summary(report, output_path)
            continue

        if not args.skip_interpreter:
            try:
                interpreter_summary = _profile_interpreter(program, path, cfg)
            except Exception as exc:
                errors.append(f"Interpreter error: {exc}")

        if not args.skip_vm:
            try:
                vm_summary, vm_extras = _profile_vm(program, path, cfg)
            except Exception as exc:
                errors.append(f"VM error: {exc}")

        report = FileReport(
            file=str(path),
            parse_time_ms=parse_time_ms,
            interpreter=interpreter_summary,
            vm=vm_summary,
            vm_extras=vm_extras,
            errors=errors,
        )
        output_path = _write_report(output_dir, report)
        _print_summary(report, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
