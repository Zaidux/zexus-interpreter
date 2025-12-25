# Zexus Standard Library

The Zexus standard library provides essential modules for common programming tasks.

## Modules

### fs - File System Operations
### http - HTTP Client
### json - JSON Parsing and Serialization  
### datetime - Date and Time Operations
### crypto - Cryptographic Functions
### blockchain - Blockchain Utilities

## Installation

The standard library is included with Zexus. No additional installation required.

## Usage

Import modules in your Zexus code:

```zexus
# Import entire module
use "stdlib/fs" as fs
use "stdlib/http" as http
use "stdlib/json" as json
use "stdlib/datetime" as datetime
use "stdlib/crypto" as crypto
use "stdlib/blockchain" as blockchain
```

Or import specific functions:

```zexus
use {read_file, write_file} from "stdlib/fs"
use {get, post} from "stdlib/http"
use {parse, stringify} from "stdlib/json"
use {now, timestamp} from "stdlib/datetime"
use {hash_sha256, keccak256} from "stdlib/crypto"
use {create_address, validate_address} from "stdlib/blockchain"
```

You can also use shorthand without the "stdlib/" prefix:

```zexus
use {read_file, write_file} from "fs"
use {hash_sha256} from "crypto"
use {create_address} from "blockchain"
```

## Module Documentation

See individual module documentation:
- [crypto Module](CRYPTO_MODULE.md) - Cryptographic functions (15+ functions)
- [blockchain Module](BLOCKCHAIN_MODULE.md) - Blockchain utilities (12+ functions)

Note: Additional modules (fs, http, json, datetime) are available in the standard library but documentation is pending.

## Quick Examples

### File System
```zexus
use {read_file, write_file, exists} from "fs"

write_file("hello.txt", "Hello, World!")
let content = read_file("hello.txt")
print(content)  # Outputs: Hello, World!
```

### HTTP
```zexus
use {get, post} from "http"

let response = get("https://api.example.com/data")
print(response.body)
```

### JSON
```zexus
use {parse, stringify} from "json"

let data = {name: "Alice", age: 30}
let json_str = stringify(data)
print(json_str)  # {"name":"Alice","age":30}

let parsed = parse(json_str)
print(parsed.name)  # Alice
```

### DateTime
```zexus
use {now, timestamp, format} from "datetime"

let current = now()
print(current)

let ts = timestamp()
print(ts)  # 1735136873.308
```

### Crypto
```zexus
use {hash_sha256, keccak256, random_bytes} from "crypto"

let hash = hash_sha256("Hello World")
print(hash)

let keccak_hash = keccak256("Hello World")
print(keccak_hash)

let random = random_bytes(32)
print(random)
```

### Blockchain
```zexus
use {create_address, validate_address, calculate_merkle_root} from "blockchain"

let address = create_address("my_public_key")
print(address)  # 0x...

let is_valid = validate_address(address)
print(is_valid)  # true

let hashes = ["hash1", "hash2", "hash3"]
let merkle_root = calculate_merkle_root(hashes)
print(merkle_root)
```

## Complete Module List

**Total: 100+ functions across 6 modules**

- **fs**: 30+ file system operations
- **http**: 5 HTTP methods
- **json**: 7 JSON utilities
- **datetime**: 25+ date/time functions
- **crypto**: 15+ cryptographic functions
- **blockchain**: 12+ blockchain utilities
