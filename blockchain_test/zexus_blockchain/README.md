# Zexus Production Blockchain

This directory contains the **fastest and most secure blockchain implementation** that Zexus can offer.

## Features

### Performance
- **High Throughput**: Optimized for maximum transactions per second
- **Low Latency**: Sub-second block confirmation
- **Efficient Storage**: Batch operations and optimized data structures
- **VM Acceleration**: Automatic compilation for performance-critical paths

### Security
- **Memory Safety**: Bounds checking, overflow protection
- **Access Control**: msg.sender validation, require() guards
- **Reentrancy Protection**: Built-in reentrancy guards
- **Input Validation**: Comprehensive parameter checking
- **Secure Storage**: Immutable ledger with cryptographic integrity

### Architecture
1. **Consensus**: Proof-of-Work with adjustable difficulty
2. **Block Structure**: Header + Transactions + State Root
3. **State Management**: Merkle tree for efficient verification
4. **Transaction Pool**: Prioritized mempool with gas-based ordering
5. **Smart Contracts**: Full Zexus language support with VM execution

## Files

- `blockchain.zx` - Core blockchain implementation
- `consensus.zx` - Consensus mechanism (PoW)
- `merkle.zx` - Merkle tree for state verification
- `mempool.zx` - Transaction pool management
- `validator.zx` - Block and transaction validation
- `run_blockchain.zx` - Main entry point to start blockchain
- `test_blockchain.zx` - Comprehensive test suite

## Security Tests

The `security_tests/` directory contains exploitation attempts:
- `test_reentrancy.zx` - Reentrancy attack attempts
- `test_overflow.zx` - Integer overflow/underflow attacks
- `test_unauthorized.zx` - Access control bypass attempts
- `test_dos.zx` - Denial of service attacks
- `test_front_running.zx` - Front-running and MEV attacks

All tests should **FAIL** (i.e., the attacks should be prevented).

## Usage

```bash
# Run the blockchain
./zx-run blockchain_test/zexus_blockchain/run_blockchain.zx

# Run security tests
./zx-run blockchain_test/zexus_blockchain/security_tests/test_all.zx
```

## Performance Benchmarks

Target metrics:
- **TPS**: >500 transactions/second (with optimizer)
- **Block Time**: 1-5 seconds
- **Finality**: 6 confirmations (~30-60 seconds)
- **Contract Execution**: <100ms per contract call

## Security Guarantees

1. ✅ Memory safe (safer than Rust)
2. ✅ Reentrancy protected
3. ✅ Overflow protected
4. ✅ Access control enforced
5. ✅ Input validation required
6. ✅ State consistency guaranteed
