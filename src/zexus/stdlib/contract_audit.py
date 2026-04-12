"""Smart-contract security analyser for Zexus source code.

Provides rule-based auditing of Zexus smart-contract files, targeting
common vulnerability classes such as reentrancy, integer overflow, missing
access control, and denial-of-service patterns.
"""

import json
import os
import re


class ContractAuditModule:
    """Security auditor for Zexus smart contracts.

    All methods are static.  The primary entry point is ``audit()``, which
    runs every built-in contract-security rule against the supplied source
    and returns a list of structured finding dictionaries.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def audit(source_code, filename="<contract>"):
        """Run all contract security checks on Zexus source code.

        Args:
            source_code: String containing Zexus contract source code.
            filename: Logical filename attached to every finding.

        Returns:
            A list of finding dicts, each containing:
                rule      – str identifier (e.g. ``"reentrancy"``).
                severity  – ``"critical"`` | ``"high"`` | ``"medium"`` |
                            ``"low"`` | ``"info"``.
                line      – 1-based line number.
                column    – 1-based column number.
                message   – Human-readable description.
                evidence  – Source snippet that triggered the rule.
                file      – Filename associated with the finding.
        """
        checks = [
            ContractAuditModule._check_reentrancy,
            ContractAuditModule._check_integer_overflow,
            ContractAuditModule._check_access_control,
            ContractAuditModule._check_gas_limits,
            ContractAuditModule._check_unchecked_return,
            ContractAuditModule._check_timestamp_dependence,
            ContractAuditModule._check_front_running,
            ContractAuditModule._check_denial_of_service,
        ]
        findings = []
        for check in checks:
            findings.extend(check(source_code, filename))
        return findings

    @staticmethod
    def audit_file(filepath):
        """Audit a single Zexus contract file.

        Args:
            filepath: Path to the contract file on disk.

        Returns:
            List of finding dicts (see ``audit``).

        Raises:
            FileNotFoundError: If *filepath* does not exist.
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            source = fh.read()
        return ContractAuditModule.audit(source, filename=filepath)

    @staticmethod
    def audit_directory(dirpath):
        """Recursively audit all contract files in a directory.

        Scans files ending in ``.zx``, ``.zxc``, and ``.contract``.

        Args:
            dirpath: Root directory to walk.

        Returns:
            Aggregated list of finding dicts.
        """
        extensions = (".zx", ".zxc", ".contract")
        findings = []
        for root, _dirs, files in os.walk(dirpath):
            for name in files:
                if any(name.endswith(ext) for ext in extensions):
                    full = os.path.join(root, name)
                    findings.extend(ContractAuditModule.audit_file(full))
        return findings

    @staticmethod
    def summary(findings):
        """Return a summary dict with counts grouped by severity.

        Args:
            findings: List of finding dicts.

        Returns:
            Dict mapping severity strings to integer counts, plus a
            ``"total"`` key.
        """
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for f in findings:
            sev = f.get("severity", "info")
            if sev in counts:
                counts[sev] += 1
        counts["total"] = len(findings)
        return counts

    @staticmethod
    def format_report(findings, format="text"):
        """Format findings as a human-readable report.

        Args:
            findings: List of finding dicts.
            format: ``"text"`` for plain-text or ``"json"`` for JSON.

        Returns:
            Formatted string.
        """
        if format == "json":
            return json.dumps(findings, indent=2)

        if not findings:
            return "No contract security findings detected."

        lines = []
        for f in findings:
            lines.append(
                f"[{f['severity'].upper()}] {f['rule']} "
                f"at {f['file']}:{f['line']}:{f['column']}"
            )
            lines.append(f"  {f['message']}")
            lines.append(f"  Evidence: {f['evidence']}")
            lines.append("")

        s = ContractAuditModule.summary(findings)
        lines.append(
            f"Summary: {s['total']} finding(s) — "
            f"critical={s['critical']}, high={s['high']}, "
            f"medium={s['medium']}, low={s['low']}, info={s['info']}"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal rules
    # ------------------------------------------------------------------

    @staticmethod
    def _check_reentrancy(source, filename):
        """Detect state writes that occur after external calls.

        The classic reentrancy pattern: a contract calls an external
        address and then modifies its own state *after* the call returns,
        allowing the callee to re-enter the function before the state
        update takes effect.
        """
        findings = []
        # Identify action/function bodies
        block_re = re.compile(
            r'(?i)(?:action|function)\s+(\w+)\s*\([^)]*\)\s*\{',
        )
        state_write_re = re.compile(
            r'(?i)(state\s*\.\s*\w+\s*=|self\s*\.\s*\w+\s*=|'
            r'balances\s*\[\s*\w+\s*\]\s*=|storage\s*\.\s*\w+\s*=)'
        )
        external_call_re = re.compile(
            r'(?i)(call\s*\(|send\s*\(|transfer\s*\(|external\s*\.\s*\w+\s*\(|'
            r'invoke\s*\()'
        )

        for block_match in block_re.finditer(source):
            func_name = block_match.group(1)
            brace_start = source.find("{", block_match.end() - 1)
            if brace_start == -1:
                continue
            body, body_offset = _extract_block(source, brace_start)

            ext_calls = list(external_call_re.finditer(body))
            if not ext_calls:
                continue
            last_call_end = max(m.end() for m in ext_calls)

            for sw in state_write_re.finditer(body):
                if sw.start() > last_call_end:
                    abs_pos = body_offset + sw.start()
                    line_no = source[:abs_pos].count("\n") + 1
                    col = abs_pos - source.rfind("\n", 0, abs_pos)
                    findings.append(_finding(
                        rule="reentrancy",
                        severity="critical",
                        line=line_no,
                        column=col,
                        message=(
                            f"Possible reentrancy in '{func_name}': state "
                            "write occurs after external call"
                        ),
                        evidence=sw.group(0).strip(),
                        filename=filename,
                    ))
        return findings

    @staticmethod
    def _check_integer_overflow(source, filename):
        """Detect arithmetic operations without overflow checks.

        Flags basic arithmetic (``+``, ``-``, ``*``) on balance-like or
        amount-like variables when there is no preceding
        ``safe_add`` / ``safe_sub`` / ``safe_mul`` / ``checked_`` guard.
        """
        patterns = [
            (
                r'(?i)(balance|amount|total|supply|value)\s*[=]\s*'
                r'(balance|amount|total|supply|value)\s*[\+\-\*]',
                "integer-overflow",
                "high",
                "Arithmetic on value variable without overflow check",
            ),
            (
                r'(?i)(balance|amount|total|supply|value)\s*'
                r'[\+\-\*]=\s*\w+',
                "integer-overflow",
                "high",
                "Compound assignment on value variable without overflow check",
            ),
        ]
        raw = _run_patterns(patterns, source, filename)

        # Filter out findings where a safe-math wrapper is evident nearby
        safe_re = re.compile(
            r'(?i)(safe_add|safe_sub|safe_mul|checked_add|checked_sub|'
            r'checked_mul|overflow_check)'
        )
        filtered = []
        lines = source.split("\n")
        for f in raw:
            line_idx = f["line"] - 1
            context_start = max(0, line_idx - 3)
            context_end = min(len(lines), line_idx + 2)
            context = "\n".join(lines[context_start:context_end])
            if not safe_re.search(context):
                filtered.append(f)
        return filtered

    @staticmethod
    def _check_access_control(source, filename):
        """Detect missing access-control checks on state-modifying actions.

        Any ``action`` that writes to ``state.``, ``self.``, ``balances``,
        or ``storage`` without calling ``require_owner``, ``has_role``, or
        ``has_permission`` is flagged.
        """
        findings = []
        auth_re = re.compile(
            r'(?i)(require_owner|has_role|has_permission|only_owner|'
            r'require_role|is_admin|check_permission|require_auth)'
        )
        state_write_re = re.compile(
            r'(?i)(state\s*\.\s*\w+\s*=|self\s*\.\s*\w+\s*=|'
            r'balances\s*\[|storage\s*\.\s*\w+\s*=)'
        )
        action_re = re.compile(
            r'(?i)action\s+(\w+)\s*\([^)]*\)\s*\{',
        )

        for m in action_re.finditer(source):
            action_name = m.group(1)
            brace_start = source.find("{", m.end() - 1)
            if brace_start == -1:
                continue
            body, _offset = _extract_block(source, brace_start)

            if state_write_re.search(body) and not auth_re.search(body):
                line_no = source[:m.start()].count("\n") + 1
                col = m.start() - source.rfind("\n", 0, m.start())
                findings.append(_finding(
                    rule="missing-access-control",
                    severity="high",
                    line=line_no,
                    column=col,
                    message=(
                        f"Action '{action_name}' modifies state without "
                        "access control (require_owner / has_role / "
                        "has_permission)"
                    ),
                    evidence=m.group(0).strip(),
                    filename=filename,
                ))
        return findings

    @staticmethod
    def _check_gas_limits(source, filename):
        """Detect unbounded loops that could exhaust gas.

        Flags ``for`` loops whose iteration bound is not a numeric literal,
        which may iterate an unpredictable number of times.
        """
        findings = []
        # Match "for" with a non-literal upper bound
        loop_re = re.compile(
            r'\bfor\s*\(\s*\w+\s*(?:=|in)\s*[^)]*\)',
        )
        bounded_re = re.compile(r'\b\d+\b')

        for m in loop_re.finditer(source):
            loop_header = m.group(0)
            if not bounded_re.search(loop_header):
                line_no = source[:m.start()].count("\n") + 1
                col = m.start() - source.rfind("\n", 0, m.start())
                findings.append(_finding(
                    rule="unbounded-loop",
                    severity="medium",
                    line=line_no,
                    column=col,
                    message="Unbounded loop may consume excessive gas",
                    evidence=loop_header.strip(),
                    filename=filename,
                ))
        return findings

    @staticmethod
    def _check_unchecked_return(source, filename):
        """Detect unchecked return values from external calls.

        Flags ``call()``, ``send()``, and ``transfer()`` whose return
        value is not captured (i.e. the call is a standalone statement,
        not on the right-hand side of an assignment or ``if`` condition).
        """
        call_re = re.compile(
            r'(?i)(?:call|send|transfer|invoke)\s*\([^)]*\)\s*;'
        )
        findings = []
        lines = source.split("\n")
        for m in call_re.finditer(source):
            line_no = source[:m.start()].count("\n") + 1
            col = m.start() - source.rfind("\n", 0, m.start())
            line_text = lines[line_no - 1] if line_no <= len(lines) else ""
            stripped = line_text.strip()
            # Skip if part of an assignment or condition
            prefix = stripped.split("(")[0]
            if "=" in prefix or stripped.startswith("if"):
                continue
            findings.append(_finding(
                rule="unchecked-return",
                severity="high",
                line=line_no,
                column=col,
                message="Return value of external call is not checked",
                evidence=m.group(0).strip(),
                filename=filename,
            ))
        return findings

    @staticmethod
    def _check_timestamp_dependence(source, filename):
        """Detect reliance on ``TX.timestamp`` for critical logic.

        Miners can manipulate block timestamps within a small range, so
        using them for randomness or access control is risky.
        """
        patterns = [
            (
                r'(?i)TX\.timestamp\s*[<>=!]+',
                "timestamp-dependence",
                "medium",
                "Critical logic depends on TX.timestamp which can be manipulated",
            ),
            (
                r'(?i)block\.timestamp\s*[<>=!]+',
                "timestamp-dependence",
                "medium",
                "Critical logic depends on block.timestamp which can be manipulated",
            ),
            (
                r'(?i)(random|seed)\s*=\s*TX\.timestamp',
                "timestamp-as-random",
                "high",
                "TX.timestamp used as randomness source — easily predictable",
            ),
        ]
        return _run_patterns(patterns, source, filename)

    @staticmethod
    def _check_front_running(source, filename):
        """Detect patterns vulnerable to front-running attacks.

        Flags operations where a price or rate is read and then used in a
        subsequent trade/swap without slippage or commit-reveal protection.
        """
        patterns = [
            (
                r'(?i)(price|rate)\s*=\s*get_(price|rate)\s*\(.*\)\s*;'
                r'[\s\S]{0,200}?(swap|trade|exchange|buy|sell)\s*\(',
                "front-running",
                "high",
                "Price-dependent operation may be vulnerable to front-running",
            ),
            (
                r'(?i)(approve|allowance)\s*\(\s*[^)]+\)\s*;'
                r'[\s\S]{0,200}?transfer_from\s*\(',
                "front-running-approval",
                "medium",
                "Approve + transferFrom pattern may be front-run",
            ),
        ]
        return _run_patterns(patterns, source, filename)

    @staticmethod
    def _check_denial_of_service(source, filename):
        """Detect patterns that could cause denial of service.

        Examples include sending tokens to unknown addresses inside a loop
        (one revert blocks all), or unbounded array iteration.
        """
        findings = []
        # Sending inside a loop
        loop_send_re = re.compile(
            r'(?i)for\s*\([^)]*\)\s*\{[^}]*(send|transfer|call)\s*\(',
            re.DOTALL,
        )
        for m in loop_send_re.finditer(source):
            line_no = source[:m.start()].count("\n") + 1
            col = m.start() - source.rfind("\n", 0, m.start())
            findings.append(_finding(
                rule="dos-loop-send",
                severity="high",
                line=line_no,
                column=col,
                message=(
                    "Sending funds inside a loop: a single revert could "
                    "block all subsequent iterations"
                ),
                evidence=m.group(0).strip()[:120],
                filename=filename,
            ))

        # Unbounded array used as loop bound
        array_loop_re = re.compile(
            r'(?i)for\s*\([^)]*\b(\w+)\.length\b',
        )
        for m in array_loop_re.finditer(source):
            line_no = source[:m.start()].count("\n") + 1
            col = m.start() - source.rfind("\n", 0, m.start())
            findings.append(_finding(
                rule="dos-unbounded-array",
                severity="medium",
                line=line_no,
                column=col,
                message=(
                    f"Loop iterates over '{m.group(1)}.length' which may "
                    "grow unboundedly and cause out-of-gas"
                ),
                evidence=m.group(0).strip(),
                filename=filename,
            ))

        return findings


# ------------------------------------------------------------------
# Module-private helpers
# ------------------------------------------------------------------

def _finding(rule, severity, line, column, message, evidence, filename):
    """Build a normalised finding dict."""
    return {
        "rule": rule,
        "severity": severity,
        "line": line,
        "column": column,
        "message": message,
        "evidence": evidence,
        "file": filename,
    }


def _run_patterns(patterns, source, filename):
    """Apply a list of ``(regex, rule, severity, message)`` tuples and
    return all matches as finding dicts."""
    findings = []
    for regex, rule, severity, message in patterns:
        for m in re.finditer(regex, source):
            line_no = source[:m.start()].count("\n") + 1
            col = m.start() - source.rfind("\n", 0, m.start())
            findings.append(_finding(
                rule=rule,
                severity=severity,
                line=line_no,
                column=col,
                message=message,
                evidence=m.group(0).strip()[:120],
                filename=filename,
            ))
    return findings


def _extract_block(source, brace_pos):
    """Extract a brace-delimited block starting at *brace_pos*.

    Returns ``(body_text, body_start_offset)`` where *body_text* is the
    content between the opening and closing braces (exclusive) and
    *body_start_offset* is the absolute position of the first character
    inside the block.
    """
    depth = 1
    idx = brace_pos + 1
    while idx < len(source) and depth > 0:
        if source[idx] == "{":
            depth += 1
        elif source[idx] == "}":
            depth -= 1
        idx += 1
    return source[brace_pos + 1 : idx - 1], brace_pos + 1
