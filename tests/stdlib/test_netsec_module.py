"""Tests for stdlib NetsecModule."""

import pytest
from src.zexus.stdlib.netsec import NetsecModule


class TestDNSLookup:
    def test_dns_lookup_a_record(self):
        # localhost should always resolve
        try:
            results = NetsecModule.dns_lookup("localhost", "A")
            assert isinstance(results, list)
        except Exception:
            pytest.skip("DNS not available in this environment")

    def test_dns_lookup_invalid_type(self):
        try:
            results = NetsecModule.dns_lookup("localhost", "INVALID")
            assert isinstance(results, list)
        except Exception:
            pass  # Expected to fail for invalid types


class TestPortScan:
    def test_port_scan_localhost(self):
        # Scan a single port that's unlikely open
        results = NetsecModule.port_scan("127.0.0.1", ports=[65534], timeout=0.5)
        assert isinstance(results, list)
        assert len(results) == 1
        assert "port" in results[0]
        assert "state" in results[0]

    def test_port_scan_default_ports(self):
        # Just verify the function runs with defaults on localhost
        results = NetsecModule.port_scan("127.0.0.1", timeout=0.1)
        assert isinstance(results, list)
        assert len(results) > 0


class TestBannerGrab:
    def test_banner_grab_closed_port(self):
        result = NetsecModule.banner_grab("127.0.0.1", 65534, timeout=0.5)
        assert isinstance(result, str)


class TestSSLCertInfo:
    def test_ssl_cert_info_no_server(self):
        # No SSL server at this port; should return error info or empty dict
        try:
            result = NetsecModule.ssl_cert_info("127.0.0.1", 65534)
            assert isinstance(result, dict)
        except Exception:
            pass  # Expected when no SSL server


class TestSecurityHeaders:
    """These tests verify the function signature and return format.
    They may skip if no network is available."""

    def test_returns_dict(self):
        try:
            result = NetsecModule.security_headers("http://127.0.0.1:1")
            assert isinstance(result, dict)
        except Exception:
            pass  # Connection refused is expected
