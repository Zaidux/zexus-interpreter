"""Network security tools module for Zexus standard library."""

import socket
import ssl
import urllib.request
import urllib.error
import http.client
import json
import struct
import datetime


# Common port-to-service mapping
_COMMON_SERVICES = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
    80: "http", 110: "pop3", 143: "imap", 443: "https", 445: "smb",
    993: "imaps", 995: "pop3s", 3306: "mysql", 3389: "rdp",
    5432: "postgresql", 8080: "http-proxy", 8443: "https-alt",
}

_DEFAULT_PORTS = list(_COMMON_SERVICES.keys())

_RECOMMENDED_SECURITY_HEADERS = [
    "Strict-Transport-Security",
    "Content-Security-Policy",
    "X-Frame-Options",
    "X-Content-Type-Options",
    "X-XSS-Protection",
    "Referrer-Policy",
    "Permissions-Policy",
]


class NetsecModule:
    """Provides network security scanning and analysis utilities.

    All methods are static and intended for authorized security testing
    and educational purposes only. Always obtain proper authorization
    before scanning or testing any systems you do not own.
    """

    @staticmethod
    def port_scan(host, ports=None, timeout=1.0):
        """Scan TCP ports on a target host.

        Args:
            host: Target hostname or IP address.
            ports: List of port numbers to scan. Defaults to common ports.
            timeout: Connection timeout in seconds.

        Returns:
            List of dicts with keys: port, state, service.
        """
        if ports is None:
            ports = _DEFAULT_PORTS

        results = []
        for port in ports:
            service = _COMMON_SERVICES.get(port, "unknown")
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                if result == 0:
                    state = "open"
                else:
                    state = "closed"
                sock.close()
            except socket.timeout:
                state = "filtered"
            except OSError:
                state = "filtered"

            results.append({
                "port": port,
                "state": state,
                "service": service,
            })

        return results

    @staticmethod
    def tls_check(host, port=443):
        """Check TLS configuration of a remote host.

        Args:
            host: Target hostname.
            port: Target port (default 443).

        Returns:
            Dict with version, cipher, cert_subject, cert_issuer,
            cert_expiry, days_until_expiry, and is_expired.
        """
        context = ssl.create_default_context()
        try:
            with socket.create_connection((host, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=host) as tls_sock:
                    cert = tls_sock.getpeercert()
                    cipher_info = tls_sock.cipher()
                    version = tls_sock.version()

                    subject = dict(x[0] for x in cert.get("subject", ()))
                    issuer = dict(x[0] for x in cert.get("issuer", ()))

                    not_after = cert.get("notAfter", "")
                    expiry_dt = datetime.datetime.strptime(
                        not_after, "%b %d %H:%M:%S %Y %Z"
                    )
                    now = datetime.datetime.utcnow()
                    days_until = (expiry_dt - now).days

                    return {
                        "version": version,
                        "cipher": cipher_info[0] if cipher_info else None,
                        "cert_subject": subject,
                        "cert_issuer": issuer,
                        "cert_expiry": not_after,
                        "days_until_expiry": days_until,
                        "is_expired": days_until < 0,
                    }
        except ssl.SSLError as e:
            return {"error": f"SSL error: {e}"}
        except socket.timeout:
            return {"error": "Connection timed out"}
        except OSError as e:
            return {"error": f"Connection failed: {e}"}

    @staticmethod
    def dns_lookup(domain, record_type="A"):
        """Perform a DNS lookup for the given domain.

        Args:
            domain: The domain name to look up.
            record_type: DNS record type — A, AAAA, MX, NS, TXT, CNAME, SOA.

        Returns:
            List of DNS record strings.
        """
        record_type = record_type.upper()
        results = []

        try:
            if record_type == "A":
                infos = socket.getaddrinfo(domain, None, socket.AF_INET)
                results = list({info[4][0] for info in infos})

            elif record_type == "AAAA":
                infos = socket.getaddrinfo(domain, None, socket.AF_INET6)
                results = list({info[4][0] for info in infos})

            elif record_type in ("MX", "NS", "TXT", "CNAME", "SOA"):
                # Use a basic DNS query over UDP to the system resolver
                results = NetsecModule._dns_query(domain, record_type)

            else:
                return [f"Unsupported record type: {record_type}"]

        except socket.gaierror as e:
            return [f"DNS lookup failed: {e}"]
        except OSError as e:
            return [f"DNS lookup error: {e}"]

        return results

    @staticmethod
    def _dns_query(domain, record_type):
        """Perform a raw DNS query via UDP to the system resolver.

        This is a minimal implementation for record types not directly
        supported by socket.getaddrinfo.

        Args:
            domain: Domain to query.
            record_type: One of MX, NS, TXT, CNAME, SOA.

        Returns:
            List of record strings.
        """
        type_map = {
            "A": 1, "NS": 2, "CNAME": 5, "SOA": 6,
            "MX": 15, "TXT": 16, "AAAA": 28,
        }
        qtype = type_map.get(record_type, 1)

        # Build DNS query packet
        transaction_id = b'\xaa\xbb'
        flags = b'\x01\x00'  # standard query, recursion desired
        counts = struct.pack(">HHHH", 1, 0, 0, 0)

        qname = b""
        for part in domain.split("."):
            qname += bytes([len(part)]) + part.encode()
        qname += b'\x00'

        question = qname + struct.pack(">HH", qtype, 1)  # class IN
        packet = transaction_id + flags + counts + question

        # Determine system resolver
        nameserver = "8.8.8.8"
        try:
            with open("/etc/resolv.conf", "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("nameserver"):
                        ns = line.split()[1]
                        if ns not in ("127.0.0.53",):
                            nameserver = ns
                            break
        except (OSError, IndexError):
            pass

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)
        try:
            sock.sendto(packet, (nameserver, 53))
            data, _ = sock.recvfrom(4096)
        except (socket.timeout, OSError):
            return [f"DNS query timed out for {domain} {record_type}"]
        finally:
            sock.close()

        return NetsecModule._parse_dns_response(data, record_type)

    @staticmethod
    def _parse_dns_response(data, record_type):
        """Parse a raw DNS response packet.

        Args:
            data: Raw bytes of the DNS response.
            record_type: The record type that was queried.

        Returns:
            List of record strings.
        """
        results = []
        if len(data) < 12:
            return results

        ancount = struct.unpack(">H", data[6:8])[0]

        # Skip header (12 bytes) and question section
        offset = 12
        # Skip question QNAME
        while offset < len(data) and data[offset] != 0:
            length = data[offset]
            if length >= 192:  # pointer
                offset += 2
                break
            offset += 1 + length
        else:
            offset += 1  # null terminator
        offset += 4  # QTYPE + QCLASS

        for _ in range(ancount):
            if offset >= len(data):
                break

            # Parse name (may be a pointer)
            if data[offset] >= 192:
                offset += 2
            else:
                while offset < len(data) and data[offset] != 0:
                    offset += 1 + data[offset]
                offset += 1

            if offset + 10 > len(data):
                break

            rtype = struct.unpack(">H", data[offset:offset + 2])[0]
            rdlength = struct.unpack(">H", data[offset + 8:offset + 10])[0]
            offset += 10

            if offset + rdlength > len(data):
                break

            rdata = data[offset:offset + rdlength]
            offset += rdlength

            type_map = {"MX": 15, "NS": 2, "TXT": 16, "CNAME": 5, "SOA": 6, "A": 1}
            expected = type_map.get(record_type, 0)
            if rtype != expected:
                continue

            if record_type == "A" and rdlength == 4:
                results.append(socket.inet_ntoa(rdata))
            elif record_type == "MX" and rdlength >= 4:
                priority = struct.unpack(">H", rdata[:2])[0]
                name = NetsecModule._decode_dns_name(data, offset - rdlength + 2)
                results.append(f"{priority} {name}")
            elif record_type in ("NS", "CNAME"):
                name = NetsecModule._decode_dns_name(data, offset - rdlength)
                results.append(name)
            elif record_type == "TXT":
                txt = ""
                i = 0
                while i < len(rdata):
                    tlen = rdata[i]
                    txt += rdata[i + 1:i + 1 + tlen].decode("utf-8", errors="replace")
                    i += 1 + tlen
                results.append(txt)
            elif record_type == "SOA":
                results.append(f"SOA record ({rdlength} bytes)")

        return results

    @staticmethod
    def _decode_dns_name(data, offset):
        """Decode a DNS domain name from packet data, handling pointers.

        Args:
            data: Full DNS packet bytes.
            offset: Starting offset of the name.

        Returns:
            Decoded domain name string.
        """
        parts = []
        seen = set()
        while offset < len(data):
            if offset in seen:
                break
            seen.add(offset)

            length = data[offset]
            if length == 0:
                break
            if length >= 192:
                pointer = struct.unpack(">H", data[offset:offset + 2])[0] & 0x3FFF
                offset = pointer
                continue
            parts.append(data[offset + 1:offset + 1 + length].decode("utf-8", errors="replace"))
            offset += 1 + length

        return ".".join(parts)

    @staticmethod
    def security_headers(url):
        """Analyze HTTP security headers of a given URL.

        Args:
            url: The URL to check (should include scheme).

        Returns:
            Dict with present, missing, grade, and details.
        """
        try:
            req = urllib.request.Request(url, method="GET")
            req.add_header("User-Agent", "Zexus-SecurityScanner/1.0")
            with urllib.request.urlopen(req, timeout=10) as resp:
                headers = {k.lower(): v for k, v in resp.getheaders()}
        except (urllib.error.URLError, OSError) as e:
            return {"error": f"Failed to fetch URL: {e}"}

        present = []
        missing = []
        details = {}

        for hdr in _RECOMMENDED_SECURITY_HEADERS:
            hdr_lower = hdr.lower()
            if hdr_lower in headers:
                present.append(hdr)
                details[hdr] = headers[hdr_lower]
            else:
                missing.append(hdr)

        # Grade based on how many recommended headers are present
        total = len(_RECOMMENDED_SECURITY_HEADERS)
        found = len(present)
        ratio = found / total if total > 0 else 0

        if ratio >= 1.0:
            grade = "A"
        elif ratio >= 0.85:
            grade = "B"
        elif ratio >= 0.7:
            grade = "C"
        elif ratio >= 0.5:
            grade = "D"
        elif ratio >= 0.3:
            grade = "E"
        else:
            grade = "F"

        return {
            "present": present,
            "missing": missing,
            "grade": grade,
            "details": details,
        }

    @staticmethod
    def whois_lookup(domain):
        """Perform a basic WHOIS lookup for a domain.

        Connects to the appropriate WHOIS server and retrieves
        registrar, creation date, and expiry date information.

        Args:
            domain: The domain to look up.

        Returns:
            Dict with registrar, creation_date, expiry_date, and raw response.
        """
        tld = domain.rsplit(".", 1)[-1]
        whois_servers = {
            "com": "whois.verisign-grs.com",
            "net": "whois.verisign-grs.com",
            "org": "whois.pir.org",
            "io": "whois.nic.io",
            "dev": "whois.nic.google",
        }
        server = whois_servers.get(tld, f"whois.nic.{tld}")

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((server, 43))
            sock.sendall((domain + "\r\n").encode())

            response = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
            sock.close()

            raw = response.decode("utf-8", errors="replace")

            info = {
                "registrar": None,
                "creation_date": None,
                "expiry_date": None,
                "raw": raw,
            }

            for line in raw.splitlines():
                lower = line.lower().strip()
                if "registrar:" in lower and info["registrar"] is None:
                    info["registrar"] = line.split(":", 1)[1].strip()
                elif "creation date:" in lower and info["creation_date"] is None:
                    info["creation_date"] = line.split(":", 1)[1].strip()
                elif "expiry date:" in lower or "expiration date:" in lower:
                    if info["expiry_date"] is None:
                        info["expiry_date"] = line.split(":", 1)[1].strip()

            return info

        except socket.timeout:
            return {"error": "WHOIS query timed out"}
        except OSError as e:
            return {"error": f"WHOIS query failed: {e}"}

    @staticmethod
    def banner_grab(host, port, timeout=3.0):
        """Grab the service banner from an open port.

        Args:
            host: Target hostname or IP.
            port: Target port number.
            timeout: Connection timeout in seconds.

        Returns:
            The banner string, or an error message.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            # Send a minimal probe for HTTP-like services
            if port in (80, 8080, 8443, 443):
                sock.sendall(b"HEAD / HTTP/1.0\r\nHost: " + host.encode() + b"\r\n\r\n")
            banner = sock.recv(4096)
            sock.close()
            return banner.decode("utf-8", errors="replace").strip()
        except socket.timeout:
            return "Error: Connection timed out"
        except OSError as e:
            return f"Error: {e}"

    @staticmethod
    def http_methods(url):
        """Check which HTTP methods a URL supports via an OPTIONS request.

        Args:
            url: The URL to test.

        Returns:
            List of allowed HTTP method strings.
        """
        try:
            req = urllib.request.Request(url, method="OPTIONS")
            req.add_header("User-Agent", "Zexus-SecurityScanner/1.0")
            with urllib.request.urlopen(req, timeout=10) as resp:
                allow = resp.getheader("Allow", "")
                if allow:
                    return [m.strip() for m in allow.split(",")]
                return []
        except (urllib.error.URLError, OSError) as e:
            return [f"Error: {e}"]

    @staticmethod
    def ssl_cert_info(host, port=443):
        """Get detailed SSL/TLS certificate information.

        Args:
            host: Target hostname.
            port: Target port (default 443).

        Returns:
            Dict containing subject, issuer, serial_number, version,
            not_before, not_after, san (Subject Alternative Names),
            signature_algorithm, and days_until_expiry.
        """
        context = ssl.create_default_context()
        try:
            with socket.create_connection((host, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=host) as tls_sock:
                    cert = tls_sock.getpeercert()
                    der = tls_sock.getpeercert(binary_form=True)

                    subject = dict(x[0] for x in cert.get("subject", ()))
                    issuer = dict(x[0] for x in cert.get("issuer", ()))

                    san = []
                    for san_type, san_value in cert.get("subjectAltName", ()):
                        san.append(f"{san_type}:{san_value}")

                    not_before = cert.get("notBefore", "")
                    not_after = cert.get("notAfter", "")

                    days_until = None
                    if not_after:
                        expiry_dt = datetime.datetime.strptime(
                            not_after, "%b %d %H:%M:%S %Y %Z"
                        )
                        days_until = (expiry_dt - datetime.datetime.utcnow()).days

                    return {
                        "subject": subject,
                        "issuer": issuer,
                        "serial_number": cert.get("serialNumber", ""),
                        "version": cert.get("version", ""),
                        "not_before": not_before,
                        "not_after": not_after,
                        "san": san,
                        "days_until_expiry": days_until,
                        "der_length": len(der) if der else 0,
                    }
        except ssl.SSLError as e:
            return {"error": f"SSL error: {e}"}
        except socket.timeout:
            return {"error": "Connection timed out"}
        except OSError as e:
            return {"error": f"Connection failed: {e}"}

    @staticmethod
    def check_open_redirect(url):
        """Test a URL for open redirect vulnerability.

        Sends requests with common redirect parameters pointing to an
        external domain and checks if the server follows the redirect.

        Args:
            url: The base URL to test.

        Returns:
            Dict with vulnerable (bool) and details (str).
        """
        redirect_params = [
            "url", "redirect", "redirect_url", "next", "return",
            "returnTo", "return_url", "dest", "destination", "redir",
            "redirect_uri", "continue", "go", "target", "out",
        ]
        external_domain = "https://example.com"
        separator = "&" if "?" in url else "?"

        findings = []
        for param in redirect_params:
            test_url = f"{url}{separator}{param}={external_domain}"
            try:
                req = urllib.request.Request(test_url, method="GET")
                req.add_header("User-Agent", "Zexus-SecurityScanner/1.0")
                # Use a non-redirecting opener to inspect Location header
                opener = urllib.request.build_opener(
                    urllib.request.HTTPRedirectHandler
                )
                resp = opener.open(req, timeout=10)
                final_url = resp.geturl()
                if "example.com" in final_url:
                    findings.append(
                        f"Parameter '{param}' redirected to {final_url}"
                    )
                resp.close()
            except urllib.error.HTTPError as e:
                location = e.headers.get("Location", "")
                if "example.com" in location:
                    findings.append(
                        f"Parameter '{param}' returned redirect to {location}"
                    )
            except (urllib.error.URLError, OSError):
                continue

        return {
            "vulnerable": len(findings) > 0,
            "details": findings if findings else "No open redirect detected",
        }
