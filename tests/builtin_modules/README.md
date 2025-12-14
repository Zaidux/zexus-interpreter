# Builtin Modules Tests

Tests for the builtin module system (crypto, datetime, math).

## Test Files

- `test_crypto_basic.zx` - Basic crypto module import and keccak256 usage

## Running Tests

```bash
# Using development mode
./zx-dev run tests/builtin_modules/test_crypto_basic.zx

# Using production command
zx run tests/builtin_modules/test_crypto_basic.zx
```

## Expected Output

```
Testing crypto...
Result: 0x36f028580bb02cc8272a9a020f4200e346e276ae664e45ee80745574e2f5ab80
```
