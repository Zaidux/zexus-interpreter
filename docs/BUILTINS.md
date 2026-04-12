# Zexus Built-in Functions Reference

Complete list of built-in functions available in Zexus v1.8.4.

Functions marked with **[I]** are interpreter-only, **[VM]** are VM-optimized,
and **[Both]** work in both execution modes.

> **⚠️ Deprecation Notice (v1.8.4+):** Several builtins have been moved to
> dedicated modules. The global builtins still work but emit deprecation
> warnings.  See [Migration Guide](#migration-guide) at the bottom of this
> document.

---

## Type Conversions & Inspection

| Function | Description | Mode |
|----------|-------------|------|
| `str(value)` / `string(value)` | Convert to string | Both |
| `int(value)` | Convert to integer | Both |
| `float(value)` | Convert to float | Both |
| `typeof(value)` / `type(value)` | Get type name as string | Both |
| `len(value)` / `length(value)` | Get length of string, list, or map | Both |
| `abs(value)` | Absolute value | Both |
| `range(end)` / `range(start, end)` / `range(start, end, step)` | Generate list of integers | Both |

## Output

| Function | Description | Mode |
|----------|-------------|------|
| `print(value, ...)` | Print values to stdout | Both |
| `debug(message)` | Print debug output | I |
| `input(prompt?)` | Read user input from stdin | I |

## List Operations

| Function | Description | Mode |
|----------|-------------|------|
| `push(list, item)` / `list.push(item)` | Append item to list | Both |
| `append(list, item)` | Append item to list (alias) | Both |
| `first(list)` | Get first element | I |
| `rest(list)` | Get all elements except first | I |
| `extend(list, other)` | Extend list with another list | I |
| `slice(list, start, end?)` | Get sub-list | I |
| `sort(list)` | Sort list in place | I |
| `list.map(fn)` | Map function over list | I |
| `list.filter(fn)` | Filter list with predicate | I |
| `list.reduce(fn, initial)` | Reduce list to single value | I |
| `list.indexOf(item)` | Find index of item (-1 if missing) | I |
| `list.contains(item)` | Check if list contains item | I |
| `list.reverse()` | Reverse list | I |
| `list.join(separator)` | Join list elements as string | I |
| `list.flatten()` | Flatten nested lists | I |
| `list.first()` | Get first element | I |
| `list.last()` | Get last element | I |
| `list.pop()` | Remove and return last element | I |
| `list.count(item)` | Count occurrences | I |
| `list.is_empty()` | Check if list is empty | I |
| `list.length()` / `list.size()` | Get list length | I |
| `list.slice(start, end?)` | Get sub-list | I |
| `list.sort()` | Sort list | I |

## Map/Dictionary Operations

| Function | Description | Mode |
|----------|-------------|------|
| `keys(map)` / `map.keys()` | Get list of keys | I |
| `values(map)` / `map.values()` | Get list of values | I |
| `entries(map)` / `map.entries()` | Get list of [key, value] pairs | I |
| `map.has(key)` | Check if key exists | I |
| `map.delete(key)` | Remove a key | I |
| `map.contains(key)` | Check if key exists | I |
| `map.size()` | Get number of entries | I |

## String Operations

| Function | Description | Mode |
|----------|-------------|------|
| `uppercase(str)` / `str.toUpperCase()` | Convert to uppercase | I |
| `lowercase(str)` / `str.toLowerCase()` | Convert to lowercase | I |
| `split(str, separator)` / `str.split(sep)` | Split string into list | I |
| `str.trim()` | Remove leading/trailing whitespace | I |
| `str.replace(old, new)` | Replace occurrences | I |
| `str.startsWith(prefix)` | Check prefix | I |
| `str.endsWith(suffix)` | Check suffix | I |
| `str.substring(start, end?)` | Extract substring | I |
| `str.charAt(index)` | Get character at index | I |
| `str.repeat(count)` | Repeat string N times | I |
| `matches_pattern(str, pattern)` | Regex pattern match | I |

## Math

| Function | Description | Mode |
|----------|-------------|------|
| `abs(n)` | Absolute value | I |
| `random()` | Random float 0-1 | I |
| `math.random_int(min, max)` | Random integer in range | I |
| `math.sqrt(n)` | Square root | I |
| `math.min(a, b)` | Minimum of two values | I |
| `math.max(a, b)` | Maximum of two values | I |

## Time & Date

| Function | Description | Mode |
|----------|-------------|------|
| `now()` | Current datetime object | I |
| `timestamp()` | Current Unix timestamp | I |
| `time()` | Current time (alias) | I |
| `sleep(seconds)` | Pause execution | I |

## Cryptography

| Function | Description | Mode |
|----------|-------------|------|
| `hash(data)` | Hash data (default algo) | I |
| ~~`hash_password(password)`~~ | **⛔ DEPRECATED** — use `password.hash()` | I |
| ~~`verify_password(hash, password)`~~ | **⛔ DEPRECATED** — use `password.verify()` | I |
| `crypto_random(length)` | Cryptographic random bytes | I |
| `keccak256(data)` | Keccak-256 hash | I |
| `to_hex(data)` | Convert to hex string | I |
| `from_hex(hex_str)` | Convert from hex string | I |
| `crypto.sha256(data)` | SHA-256 hash | I |
| `crypto.aes_encrypt(data, key)` | AES encryption | I |
| `crypto.aes_decrypt(data, key)` | AES decryption | I |
| `crypto.generate_keypair(algo?)` | Generate key pair | I |

## File System

| Function | Description | Mode |
|----------|-------------|------|
| `read_file(path)` | Read file contents | I |
| `file_read_text(path)` | Read file as text | I |
| `file_write_text(path, content)` | Write text to file | I |
| `file_exists(path)` | Check if file exists | I |
| `file_read_json(path)` | Read and parse JSON file | I |
| `file_write_json(path, data)` | Write data as JSON file | I |
| `file_append(path, content)` | Append to file | I |
| `file_list_dir(path)` | List directory contents | I |
| `fs_is_file(path)` | Check if path is a file | I |
| `fs_is_dir(path)` | Check if path is a directory | I |
| `fs_mkdir(path)` | Create directory | I |
| `fs_remove(path)` | Remove file | I |
| `fs_rename(old, new)` | Rename file/directory | I |
| `fs_copy(src, dest)` | Copy file | I |

## HTTP / Networking

| Function | Description | Mode |
|----------|-------------|------|
| `http_get(url)` | HTTP GET request | I |
| `http_post(url, body)` | HTTP POST request | I |
| `http_put(url, body)` | HTTP PUT request | I |
| `http_delete(url)` | HTTP DELETE request | I |
| `http_request(method, url, opts)` | Custom HTTP request | I |

## JSON

| Function | Description | Mode |
|----------|-------------|------|
| `json.parse(str)` | Parse JSON string to object | I |
| `json.stringify(obj)` | Serialize object to JSON | I |
| `json.pretty(obj)` | Pretty-print JSON | I |

## Validation

> **⚠️ DEPRECATED** — The functions below still work but emit deprecation
> warnings.  Migrate to `use "validation"`:
>
> ```zexus
> use "validation"
> let ok = validation.is_email("user@example.com")
> ```
>
> The generic `matches_pattern()`, `is_numeric()`, and `validate_length()`
> remain as core builtins because they are general-purpose.

| Function | Description | Mode |
|----------|-------------|------|
| ~~`is_email(str)`~~ | **⛔ DEPRECATED** — use `validation.is_email()` | I |
| ~~`is_url(str)`~~ | **⛔ DEPRECATED** — use `validation.is_url()` | I |
| ~~`is_phone(str)`~~ | **⛔ DEPRECATED** — use `validation.is_phone()` | I |
| `is_numeric(str)` | Check if string is numeric | I |
| `validate_length(str, min, max)` | Validate string length | I |
| ~~`password_strength(str)`~~ | **⛔ DEPRECATED** — use `validation.password_strength()` | I |

## Environment

| Function | Description | Mode |
|----------|-------------|------|
| `env_get(name)` | Get environment variable | I |
| `env_set(name, value)` | Set environment variable — **🔒 requires `sys.env` capability** | I |
| `env_exists(name)` | Check if env var exists | I |

> **Security note:** `env_set()` is now gated behind the `sys.env`
> capability.  In sandboxed or contract execution contexts where the
> capability is not granted, calls to `env_set()` will be denied.
> `env_get()` and `env_exists()` remain unrestricted for read-only access.

## Persistence

| Function | Description | Mode |
|----------|-------------|------|
| `persist_set(key, value)` | Store persistent value | I |
| `persist_get(key)` | Retrieve persistent value | I |
| `persistent_delete(key)` | Delete persistent value | I |

## Access Control

| Function | Description | Mode |
|----------|-------------|------|
| `has_role(entity, role)` | Check if entity has role | I |
| `has_permission(entity, perm)` | Check if entity has permission | I |
| `grant_role(entity, role)` | Grant role to entity | I |
| `revoke_role(entity, role)` | Revoke role from entity | I |
| `require_owner(entity)` | Assert entity is owner | I |

## Concurrency

| Function | Description | Mode |
|----------|-------------|------|
| `spawn(fn)` | Spawn a concurrent task | I |
| `send(channel, value)` | Send value to channel | Both |
| `receive(channel)` | Receive value from channel | Both |
| `close_channel(channel)` | Close a channel | Both |

## Program Control

| Function | Description | Mode |
|----------|-------------|------|
| `exit_program(code?)` | Exit with optional code | I |
| `is_main()` | Check if current file is main | I |
| `schedule(fn, interval)` | Schedule repeated execution | I |

---

## Method Chaining

Lists, maps, and strings support method calls via dot notation:

```zexus
let items = [3, 1, 4, 1, 5]
let sorted = items.sort()
let joined = items.join(", ")
print(joined)

let text = "Hello World"
let upper = text.toUpperCase()
let parts = text.split(" ")
```

## Module Functions

Import module functions with `use`:

```zexus
use "crypto"
let h = crypto.sha256("data")

use "datetime"
let now_val = datetime.now()

use "math"
let r = math.random_int(1, 100)

use "validation"
let ok = validation.is_email("user@test.com")

use "password"
let hashed = password.hash("secret")
let verified = password.verify("secret", hashed)
```

---

## Migration Guide

The following builtins are deprecated starting in v1.8.4 and will be removed
in a future release.  They continue to work but emit a `DeprecationWarning`.

### Validation builtins → `use "validation"`

| Old (deprecated) | New (module) |
|-------------------|--------------|
| `is_email(s)` | `validation.is_email(s)` |
| `is_url(s)` | `validation.is_url(s)` |
| `is_phone(s)` | `validation.is_phone(s)` |
| `password_strength(s)` | `validation.password_strength(s)` |

**Why?** These are application-level validations, not language primitives.
The general-purpose `matches_pattern(str, regex)` builtin covers custom
pattern matching.  Keeping them in a module reduces the global namespace and
makes the language core leaner.

### Password crypto builtins → `use "password"`

| Old (deprecated) | New (module) |
|-------------------|--------------|
| `hash_password(pw)` | `password.hash(pw)` |
| `verify_password(hash, password)` | `password.verify(hash, password)` |

**Why?** Password hashing is application-level crypto.  Core crypto
builtins (`hash()`, `keccak256()`, `crypto.sha256()`) remain global.

### `env_set()` — now capability-gated

`env_set()` is **not** removed but now requires the `sys.env` capability.
In default (AllowAll) mode this is transparent.  In sandboxed or contract
execution contexts the capability must be explicitly granted:

```zexus
// Sandbox blocks env_set by default
sandbox {
    env_set("FOO", "bar")  // ⛔ denied
}

// Grant the capability first
grant "sys.env" to current_context
env_set("FOO", "bar")  // ✅ allowed
```
