"""Static analysis / SAST scanner for Zexus source code.

Provides rule-based security scanning that detects common vulnerability
patterns in Zexus (.zx) source files. Findings can be exported in plain
text, JSON, or SARIF format for integration with GitHub Advanced Security.
"""

import json
import os
import re


class AuditModule:
    """Static Application Security Testing (SAST) scanner for Zexus code.

    All methods are static.  The primary entry point is ``scan()``, which
    runs every built-in security rule against the supplied source code and
    returns a list of structured finding dictionaries.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def scan(source_code, filename="<input>"):
        """Run all security checks on Zexus source code.

        Args:
            source_code: String containing Zexus source code to analyse.
            filename: Logical filename attached to every finding.

        Returns:
            A list of finding dicts, each containing:
                rule      – str identifier (e.g. ``"hardcoded-secret"``).
                severity  – ``"critical"`` | ``"high"`` | ``"medium"`` |
                            ``"low"`` | ``"info"``.
                line      – 1-based line number.
                column    – 1-based column number.
                message   – Human-readable description.
                evidence  – Source snippet that triggered the rule.
                file      – Filename associated with the finding.
        """
        checks = [
            AuditModule._check_hardcoded_secrets,
            AuditModule._check_sql_injection,
            AuditModule._check_xss,
            AuditModule._check_command_injection,
            AuditModule._check_path_traversal,
            AuditModule._check_unsafe_deserialization,
            AuditModule._check_missing_auth,
            AuditModule._check_taint_flow,
        ]
        findings = []
        for check in checks:
            findings.extend(check(source_code, filename))
        return findings

    @staticmethod
    def scan_file(filepath):
        """Scan a single Zexus source file.

        Args:
            filepath: Path to the ``.zx`` file on disk.

        Returns:
            List of finding dicts (see ``scan``).

        Raises:
            FileNotFoundError: If *filepath* does not exist.
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            source = fh.read()
        return AuditModule.scan(source, filename=filepath)

    @staticmethod
    def scan_directory(dirpath, extensions=None):
        """Recursively scan a directory for Zexus source files.

        Args:
            dirpath: Root directory to walk.
            extensions: Iterable of file extensions to include
                (e.g. ``[".zx", ".zexus"]``).  Defaults to ``[".zx"]``.

        Returns:
            List of finding dicts aggregated from all matched files.
        """
        if extensions is None:
            extensions = [".zx"]
        findings = []
        for root, _dirs, files in os.walk(dirpath):
            for name in files:
                if any(name.endswith(ext) for ext in extensions):
                    full = os.path.join(root, name)
                    findings.extend(AuditModule.scan_file(full))
        return findings

    @staticmethod
    def to_sarif(findings):
        """Convert findings to SARIF v2.1.0 JSON format.

        Args:
            findings: List of finding dicts produced by ``scan``.

        Returns:
            A dict representing a valid SARIF log that can be uploaded to
            GitHub Advanced Security via ``code-scanning/sarifs``.
        """
        rules_map = {}
        results = []
        for f in findings:
            rule_id = f["rule"]
            if rule_id not in rules_map:
                rules_map[rule_id] = {
                    "id": rule_id,
                    "shortDescription": {"text": f["message"]},
                    "defaultConfiguration": {
                        "level": _sarif_level(f["severity"]),
                    },
                }
            results.append({
                "ruleId": rule_id,
                "level": _sarif_level(f["severity"]),
                "message": {"text": f["message"]},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": f["file"]},
                            "region": {
                                "startLine": f["line"],
                                "startColumn": f["column"],
                            },
                        }
                    }
                ],
            })

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "zexus-audit",
                            "version": "1.0.0",
                            "rules": list(rules_map.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

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
            return "No security findings detected."

        lines = []
        for f in findings:
            lines.append(
                f"[{f['severity'].upper()}] {f['rule']} "
                f"at {f['file']}:{f['line']}:{f['column']}"
            )
            lines.append(f"  {f['message']}")
            lines.append(f"  Evidence: {f['evidence']}")
            lines.append("")

        s = AuditModule.summary(findings)
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
    def _check_hardcoded_secrets(source, filename):
        """Detect hardcoded API keys, passwords, tokens, and secrets.

        Flags assignments where a secret-like variable name is bound to a
        string literal (e.g. ``password = "hunter2"``).
        """
        patterns = [
            (
                r'(?i)(password|passwd|pwd)\s*=\s*"[^"]+"',
                "hardcoded-password",
                "critical",
                "Hardcoded password detected",
            ),
            (
                r'(?i)(api_key|apikey)\s*=\s*"[^"]+"',
                "hardcoded-api-key",
                "critical",
                "Hardcoded API key detected",
            ),
            (
                r'(?i)(secret|secret_key)\s*=\s*"[^"]+"',
                "hardcoded-secret",
                "critical",
                "Hardcoded secret detected",
            ),
            (
                r'(?i)(token|access_token|auth_token)\s*=\s*"[^"]+"',
                "hardcoded-token",
                "high",
                "Hardcoded token detected",
            ),
            (
                r'(?i)(private_key|priv_key)\s*=\s*"[^"]+"',
                "hardcoded-private-key",
                "critical",
                "Hardcoded private key detected",
            ),
        ]
        return _run_patterns(patterns, source, filename)

    @staticmethod
    def _check_sql_injection(source, filename):
        """Detect string concatenation in SQL-like query patterns.

        Looks for SQL keywords combined with the Zexus concatenation
        operator (``++``) which can indicate unsanitised user input being
        interpolated into queries.
        """
        patterns = [
            (
                r'(?i)"SELECT\s+.*"\s*\+\+\s*\w+',
                "sql-injection",
                "critical",
                "Potential SQL injection via string concatenation in SELECT",
            ),
            (
                r'(?i)"INSERT\s+.*"\s*\+\+\s*\w+',
                "sql-injection",
                "critical",
                "Potential SQL injection via string concatenation in INSERT",
            ),
            (
                r'(?i)"UPDATE\s+.*"\s*\+\+\s*\w+',
                "sql-injection",
                "critical",
                "Potential SQL injection via string concatenation in UPDATE",
            ),
            (
                r'(?i)"DELETE\s+.*"\s*\+\+\s*\w+',
                "sql-injection",
                "critical",
                "Potential SQL injection via string concatenation in DELETE",
            ),
            (
                r'(?i)query\s*\(\s*"[^"]*"\s*\+\+',
                "sql-injection",
                "high",
                "Potential SQL injection via dynamic query construction",
            ),
        ]
        return _run_patterns(patterns, source, filename)

    @staticmethod
    def _check_xss(source, filename):
        """Detect unsanitized output to HTML contexts.

        Flags calls that write user-controlled data directly into HTML
        without escaping.
        """
        patterns = [
            (
                r'(?i)html\s*\(\s*.*\binput\b',
                "xss-unsanitized-input",
                "high",
                "Potential XSS: unsanitized input passed to HTML context",
            ),
            (
                r'(?i)render\s*\(\s*"<[^>]*>"\s*\+\+\s*\w+',
                "xss-string-concat",
                "high",
                "Potential XSS: string concatenation in HTML rendering",
            ),
            (
                r'(?i)innerHTML\s*=\s*\w+',
                "xss-innerhtml",
                "high",
                "Potential XSS: direct assignment to innerHTML",
            ),
            (
                r'(?i)document\.write\s*\(',
                "xss-document-write",
                "medium",
                "Potential XSS: use of document.write",
            ),
        ]
        return _run_patterns(patterns, source, filename)

    @staticmethod
    def _check_command_injection(source, filename):
        """Detect shell execution with potentially unsanitised user input.

        Flags patterns like ``exec("cmd " ++ variable)`` or ``shell(input)``.
        """
        patterns = [
            (
                r'(?i)exec\s*\(\s*"[^"]*"\s*\+\+\s*\w+',
                "command-injection",
                "critical",
                "Potential command injection via string concatenation in exec",
            ),
            (
                r'(?i)shell\s*\(\s*\w+\s*\)',
                "command-injection",
                "critical",
                "Potential command injection: variable passed directly to shell",
            ),
            (
                r'(?i)system\s*\(\s*"[^"]*"\s*\+\+\s*\w+',
                "command-injection",
                "critical",
                "Potential command injection via string concatenation in system call",
            ),
            (
                r'(?i)run_command\s*\(\s*\w+\s*\)',
                "command-injection",
                "high",
                "Potential command injection: variable passed to run_command",
            ),
        ]
        return _run_patterns(patterns, source, filename)

    @staticmethod
    def _check_path_traversal(source, filename):
        """Detect file operations with unsanitised paths.

        Flags calls like ``read_file(user_input)`` where the path argument
        may come directly from external input.
        """
        patterns = [
            (
                r'(?i)(read_file|write_file|open)\s*\(\s*"[^"]*"\s*\+\+\s*\w+',
                "path-traversal",
                "high",
                "Potential path traversal via concatenated file path",
            ),
            (
                r'(?i)(read_file|write_file|open)\s*\(\s*\w*input\w*\s*\)',
                "path-traversal",
                "high",
                "Potential path traversal: input variable used as file path",
            ),
            (
                r'\.\./\.\.',
                "path-traversal-literal",
                "medium",
                "Literal path traversal sequence detected",
            ),
        ]
        return _run_patterns(patterns, source, filename)

    @staticmethod
    def _check_unsafe_deserialization(source, filename):
        """Detect unsafe JSON / data parsing from untrusted sources.

        Flags patterns where deserialization functions receive external
        data without validation.
        """
        patterns = [
            (
                r'(?i)json_parse\s*\(\s*\w*input\w*\s*\)',
                "unsafe-deserialization",
                "high",
                "Unsafe deserialization: parsing JSON from input without validation",
            ),
            (
                r'(?i)deserialize\s*\(\s*\w*input\w*\s*\)',
                "unsafe-deserialization",
                "high",
                "Unsafe deserialization: deserializing untrusted input",
            ),
            (
                r'(?i)eval\s*\(\s*\w+\s*\)',
                "unsafe-eval",
                "critical",
                "Unsafe eval: executing dynamic code",
            ),
            (
                r'(?i)from_bytes\s*\(\s*\w*input\w*',
                "unsafe-deserialization",
                "medium",
                "Potential unsafe deserialization from raw bytes",
            ),
        ]
        return _run_patterns(patterns, source, filename)

    @staticmethod
    def _check_missing_auth(source, filename):
        """Detect public actions that lack access-control checks.

        A public ``action`` block that contains no call to
        ``require_auth``, ``require_role``, ``check_permission``, or
        ``has_role`` is flagged.
        """
        findings = []
        auth_keywords = re.compile(
            r'(?i)(require_auth|require_role|check_permission|has_role|'
            r'is_authenticated|require_owner|has_permission)'
        )
        action_pattern = re.compile(
            r'(?i)^(\s*)(public\s+)?action\s+(async\s+)?(\w+)\s*\(',
            re.MULTILINE,
        )
        for match in action_pattern.finditer(source):
            action_name = match.group(4)
            start_pos = match.start()
            line_no = source[:start_pos].count("\n") + 1
            col = match.start() - source.rfind("\n", 0, match.start())

            # Collect the body — simple brace / indent heuristic
            body_start = source.find("{", match.end())
            if body_start == -1:
                continue
            depth = 1
            idx = body_start + 1
            while idx < len(source) and depth > 0:
                if source[idx] == "{":
                    depth += 1
                elif source[idx] == "}":
                    depth -= 1
                idx += 1
            body = source[body_start:idx]

            if not auth_keywords.search(body):
                findings.append(_finding(
                    rule="missing-auth",
                    severity="medium",
                    line=line_no,
                    column=col,
                    message=(
                        f"Action '{action_name}' has no authentication or "
                        "authorization check"
                    ),
                    evidence=match.group(0).strip(),
                    filename=filename,
                ))
        return findings

    @staticmethod
    def _check_taint_flow(source, filename):
        """Simple taint analysis tracking ``input()`` calls to sensitive sinks.

        Identifies variables that receive a value from ``input()`` and then
        flow—without sanitisation—into sinks such as ``exec``, ``query``,
        ``shell``, ``eval``, ``system``, ``render``, or ``html``.
        """
        findings = []
        # Step 1: collect tainted variable names
        taint_re = re.compile(r'(\w+)\s*=\s*input\s*\(')
        tainted = set()
        for m in taint_re.finditer(source):
            tainted.add(m.group(1))

        if not tainted:
            return findings

        # Step 2: look for tainted vars reaching sinks
        sinks = ["exec", "query", "shell", "eval", "system", "render", "html",
                  "run_command", "write_file", "read_file"]
        for var in tainted:
            escaped_var = re.escape(var)
            for sink in sinks:
                sink_re = re.compile(
                    rf'(?i){re.escape(sink)}\s*\([^)]*\b{escaped_var}\b'
                )
                for m in sink_re.finditer(source):
                    line_no = source[:m.start()].count("\n") + 1
                    col = m.start() - source.rfind("\n", 0, m.start())
                    findings.append(_finding(
                        rule="taint-flow",
                        severity="critical",
                        line=line_no,
                        column=col,
                        message=(
                            f"Tainted variable '{var}' from input() flows "
                            f"into sink '{sink}' without sanitization"
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
                evidence=m.group(0).strip(),
                filename=filename,
            ))
    return findings


_SARIF_SEVERITY_MAP = {
    "critical": "error",
    "high": "error",
    "medium": "warning",
    "low": "note",
    "info": "note",
}


def _sarif_level(severity):
    """Map an internal severity string to a SARIF level."""
    return _SARIF_SEVERITY_MAP.get(severity, "warning")
