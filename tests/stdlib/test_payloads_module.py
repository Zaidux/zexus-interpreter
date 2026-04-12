"""Tests for stdlib PayloadsModule."""

import pytest
from src.zexus.stdlib.payloads import PayloadsModule


class TestXSS:
    def test_xss_all(self):
        payloads = PayloadsModule.xss("all")
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        assert any("<script>" in p.lower() for p in payloads)

    def test_xss_basic(self):
        payloads = PayloadsModule.xss("basic")
        assert isinstance(payloads, list)
        assert len(payloads) > 0

    def test_xss_event(self):
        payloads = PayloadsModule.xss("event")
        assert isinstance(payloads, list)

    def test_xss_encoded(self):
        payloads = PayloadsModule.xss("encoded")
        assert isinstance(payloads, list)


class TestSQLi:
    def test_sqli_all(self):
        payloads = PayloadsModule.sqli("all")
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        assert any("'" in p for p in payloads)

    def test_sqli_union(self):
        payloads = PayloadsModule.sqli("union")
        assert isinstance(payloads, list)

    def test_sqli_blind(self):
        payloads = PayloadsModule.sqli("blind")
        assert isinstance(payloads, list)


class TestSSRF:
    def test_ssrf(self):
        payloads = PayloadsModule.ssrf()
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        assert any("localhost" in p or "127.0.0.1" in p for p in payloads)


class TestPathTraversal:
    def test_path_traversal(self):
        payloads = PayloadsModule.path_traversal()
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        assert any(".." in p for p in payloads)


class TestCommandInjection:
    def test_command_injection(self):
        payloads = PayloadsModule.command_injection()
        assert isinstance(payloads, list)
        assert len(payloads) > 0


class TestXXE:
    def test_xxe(self):
        payloads = PayloadsModule.xxe()
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        assert any("<!ENTITY" in p or "ENTITY" in p for p in payloads)


class TestEncodePayload:
    def test_url_encode(self):
        result = PayloadsModule.encode_payload("<script>alert(1)</script>", "url")
        assert "%" in result

    def test_base64_encode(self):
        result = PayloadsModule.encode_payload("test", "base64")
        assert isinstance(result, str)

    def test_hex_encode(self):
        result = PayloadsModule.encode_payload("ABC", "hex")
        assert isinstance(result, str)

    def test_html_encode(self):
        result = PayloadsModule.encode_payload("<b>", "html")
        assert "&lt;" in result


class TestWordlist:
    def test_common_wordlist(self):
        words = PayloadsModule.generate_wordlist("common")
        assert isinstance(words, list)
        assert len(words) > 0

    def test_directories_wordlist(self):
        words = PayloadsModule.generate_wordlist("directories")
        assert isinstance(words, list)
        assert len(words) > 0


class TestHeaderInjection:
    def test_header_injection(self):
        payloads = PayloadsModule.header_injection()
        assert isinstance(payloads, list)
        assert len(payloads) > 0


class TestTemplateInjection:
    def test_template_injection(self):
        payloads = PayloadsModule.template_injection()
        assert isinstance(payloads, list)
        assert len(payloads) > 0
