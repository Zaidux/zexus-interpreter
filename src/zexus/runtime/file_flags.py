"""Parse and apply inline file flags for Zexus execution.

Supported directive formats (first 25 lines):
- // @zexus: {"use_vm": true, "vm_mode": "stack", "no_optimize": false}
- # @zexus: use_vm=true; vm_mode=auto; debug=full

Values accept booleans, ints, floats, or strings.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import json
import re

_MAX_SCAN_LINES = 25


def parse_file_flags(source: str) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    if not source:
        return flags

    lines = source.splitlines()[:_MAX_SCAN_LINES]
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "@zexus" not in stripped:
            continue

        # Strip leading comment markers
        directive = stripped
        for prefix in ("//", "#", "/*", "*/"):
            if directive.startswith(prefix):
                directive = directive[len(prefix):].strip()
        # Remove leading @zexus marker
        if directive.lower().startswith("@zexus"):
            directive = directive[len("@zexus"):].strip()
        if directive.startswith(":"):
            directive = directive[1:].strip()

        # JSON object form
        if "{" in directive:
            json_part = directive[directive.find("{"):].strip()
            try:
                parsed = json.loads(json_part)
                if isinstance(parsed, dict):
                    flags.update(parsed)
            except Exception:
                pass
            continue

        # key=value form (semicolon or comma separated)
        for part in re.split(r"[;,]", directive):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                continue
            key, raw_val = part.split("=", 1)
            key = key.strip()
            raw_val = raw_val.strip()
            flags[key] = _parse_value(raw_val)

    return flags


def _parse_value(raw: str) -> Any:
    if not raw:
        return raw
    lowered = raw.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    # numbers
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except Exception:
        pass
    # quoted string
    if (raw.startswith("\"") and raw.endswith("\"")) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def apply_vm_config(vm, vm_config: Dict[str, Any]) -> None:
    if not vm_config:
        return

    mode_value = vm_config.get("mode")
    if isinstance(mode_value, str):
        try:
            from ..vm.vm import VMMode
            vm.mode = VMMode(mode_value.lower())
        except Exception:
            pass

    # Explicit gas metering disable
    if vm_config.get("enable_gas_metering") is False:
        try:
            vm.enable_gas_metering = False
            vm.gas_metering = None
        except Exception:
            pass

    # perf fast dispatch
    if vm_config.get("perf_fast_dispatch") is True:
        try:
            vm._perf_fast_dispatch = True
        except Exception:
            pass

    if vm_config.get("prefer_register") is True:
        try:
            vm.prefer_register = True
        except Exception:
            pass

    if vm_config.get("prefer_parallel") is True:
        try:
            vm.prefer_parallel = True
        except Exception:
            pass

    # generic attribute setters
    for key, value in vm_config.items():
        if key == "mode":
            continue
        if hasattr(vm, key):
            try:
                setattr(vm, key, value)
            except Exception:
                continue
