# Crypto Module

The `crypto` module provides cryptographic operations including hashing, random number generation, and key derivation.

## Usage

```zexus
use {hash_sha256, keccak256, random_bytes} from "crypto"
# or
use {hash_sha256, keccak256, random_bytes} from "stdlib/crypto"
```

## Functions

### Hashing Functions

#### `hash_sha256(data: string) -> string`
Calculate SHA-256 hash of data.

```zexus
let hash = hash_sha256("Hello World")
print(hash)  # a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
```

#### `hash_sha512(data: string) -> string`
Calculate SHA-512 hash of data.

```zexus
let hash = hash_sha512("Hello World")
```

#### `hash_md5(data: string) -> string`
Calculate MD5 hash of data (not recommended for security).

#### `hash_blake2b(data: string, digest_size: int = 32) -> string`
Calculate BLAKE2b hash.

#### `hash_blake2s(data: string, digest_size: int = 32) -> string`
Calculate BLAKE2s hash.

#### `keccak256(data: string) -> string`
Calculate Keccak-256 hash (Ethereum-style).

```zexus
let hash = keccak256("Hello World")
print(hash)  # e167f68d6563d75bb25f3aa49c29ef612d41352dc00606de7cbd630bb2665f51
```

#### `sha3_256(data: string) -> string`
Calculate SHA3-256 hash.

#### `sha3_512(data: string) -> string`
Calculate SHA3-512 hash.

### HMAC Functions

#### `hmac_sha256(data: string, key: string) -> string`
Calculate HMAC-SHA256.

```zexus
let hmac = hmac_sha256("message", "secret_key")
```

#### `hmac_sha512(data: string, key: string) -> string`
Calculate HMAC-SHA512.

### Random Generation

#### `random_bytes(size: int = 32) -> string`
Generate cryptographically secure random bytes (hex encoded).

```zexus
let random = random_bytes(32)
print(random)  # 64 hex characters (32 bytes)
```

#### `random_int(min: int = 0, max: int = 2^32) -> int`
Generate cryptographically secure random integer.

```zexus
let num = random_int(1, 100)
```

#### `generate_salt(size: int = 16) -> string`
Generate random salt for password hashing.

```zexus
let salt = generate_salt()
```

### Key Derivation

#### `pbkdf2(password: string, salt: string, iterations: int = 100000, key_length: int = 32, algorithm: string = 'sha256') -> string`
Derive key from password using PBKDF2.

```zexus
let salt = generate_salt()
let key = pbkdf2("my_password", salt, 100000)
```

### Utility Functions

#### `compare_digest(a: string, b: string) -> bool`
Constant-time string comparison (prevents timing attacks).

```zexus
let is_equal = compare_digest(hash1, hash2)
```

#### `constant_time_compare(a: string, b: string) -> bool`
Alias for `compare_digest`.

## Complete Example

```zexus
use {hash_sha256, keccak256, random_bytes, pbkdf2, generate_salt} from "crypto"

# Hash data
let data = "Hello World"
let sha_hash = hash_sha256(data)
let keccak_hash = keccak256(data)

print("SHA-256: " + sha_hash)
print("Keccak-256: " + keccak_hash)

# Generate random data
let random = random_bytes(32)
print("Random: " + random)

# Derive key from password
let salt = generate_salt()
let password = "my_secure_password"
let derived_key = pbkdf2(password, salt, 100000)

print("Derived Key: " + derived_key)
```

## Security Notes

- Use SHA-256 or SHA-512 for general hashing
- Use Keccak-256 for Ethereum compatibility
- Use PBKDF2 for password hashing with at least 100,000 iterations
- Use `random_bytes` for generating cryptographic keys or tokens
- Use `compare_digest` when comparing hashes to prevent timing attacks
- Avoid MD5 for security-sensitive applications

## Function Count

**Total: 15 functions**

- Hashing: 8 functions
- HMAC: 2 functions
- Random: 3 functions
- Key Derivation: 1 function
- Utilities: 2 functions
