# Zexus Quick Start

Everything in this file is verified against zexus **2.0.0** — every
command was executed as written. If something here fails, it's a bug:
please open an issue.

## 1. Install

```bash
pip install zexus==2.0.0
```

Verify:

```bash
zx --help
```

## 2. Hello, world

Create `hello.zx`:

```zexus
print("Hello, Zexus!")
```

Run it:

```bash
zx run hello.zx
```

**Output:** `Hello, Zexus!`

## 3. The safety model (try it)

Create `safety.zx`:

```zexus
// file_read_text is DENIED until granted — safe by default
let r = file_read_text("hello.zx")
print(r)
```

Run it — you get a clean error and exit code 1. Now grant:

```zexus
grant self io_full
print(file_read_text("hello.zx"))
revoke self io_full
```

Works, then revoked.

## 4. Language tour (canonical grammar)

```zexus
// fn is the canonical keyword (function also accepted in 2.x)
fn fib(n) {
    if n < 2 { return n }
    return fib(n - 1) + fib(n - 2)
}
print(fib(10))                       // 55

// match with _ wildcard — statement or value position
let desc = match 5 % 2 { 0 => "even" _ => "odd" }
print(desc)                          // odd

// ranges: exclusive end, Python semantics
let total = 0
for i in 0..4 { total = total + i }
print(total)                         // 6

// bytes with raw escapes (0xff is the BYTE, not the codepoint)
let probe = b"\x00\x01\xff"
print(probe.len())                   // 3
print(probe.to_hex())                // 0001ff

// string methods
print("  hi  ".trim().upper())       // HI
print("a-b-c".split("-").len())      // 3
```

## 5. Contracts (crypto category)

```zexus
contract Counter {
    state { count: 0 }
    action increment() { this.count = this.count + 1 }
    action get() { return this.count }
}

let c = Counter()
c.increment()
c.increment()
print(c.get())                       // 2 — identical on VM and tree-walk
```

Force an engine: `zx run --use-vm file.zx` / `zx run --no-vm file.zx` —
both produce the same output (enforced by CI).

## 6. Security work (the point)

A minimal recon probe:

```zexus
grant self network
use "netsec"

let headers = security_headers("https://example.com")
print(headers.missing)

let tls = tls_check("example.com")
print(tls.version)
```

Full working tools: [examples/recon_demo.zx](examples/recon_demo.zx)
(live recon: DNS/TLS/headers/cookies/fingerprint/endpoints/subdomains)
and [examples/api_server_demo.zx](examples/api_server.zx) (a JSON API server
in pure Zexus on raw sockets).

## 7. Where to go next

- [GRAMMAR.md](GRAMMAR.md) — the language contract (canonical forms,
  migration table)
- [README.md](README.md) — feature status, honest limitations
- [ROADMAP.md](ROADMAP.md) — what's done and what's next
- [examples/](examples/) — runnable programs
