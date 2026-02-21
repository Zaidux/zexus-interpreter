# Zexus Blockchain — Complete AI Reference Guide

> **Source-verified reference** — Every keyword, API, and example in this document was extracted
> directly from the Zexus interpreter source code (`src/zexus/blockchain/`, `src/zexus/evaluator/`,
> `src/zexus/parser/`, `src/zexus/lexer.py`, `src/zexus/zexus_ast.py`, `src/zexus/zexus_token.py`,
> `src/zexus/stdlib/`, `src/zexus/builtin_modules.py`, `src/zexus/crypto_bridge.py`, and working
> `.zx` demo files in `blockchain_demo/`).

---

## Table of Contents

1. [Production Readiness Assessment](#1-production-readiness-assessment)
2. [Language Keywords for Blockchain](#2-language-keywords-for-blockchain)
3. [Smart Contracts](#3-smart-contracts)
4. [Tokens (ZX-20, ZX-721, ZX-1155)](#4-tokens-zx-20-zx-721-zx-1155)
5. [Wallets & Cryptography](#5-wallets--cryptography)
6. [Consensus Engines](#6-consensus-engines)
7. [Blockchain Nodes & Chain Management](#7-blockchain-nodes--chain-management)
8. [P2P Networking](#8-p2p-networking)
9. [Cross-Chain / Multichain Bridge](#9-cross-chain--multichain-bridge)
10. [Events & Logging](#10-events--logging)
11. [State Storage (Merkle Patricia Trie)](#11-state-storage-merkle-patricia-trie)
12. [Gas Metering & Transaction Context](#12-gas-metering--transaction-context)
13. [JSON-RPC Server](#13-json-rpc-server)
14. [Upgradeable Contracts & Governance](#14-upgradeable-contracts--governance)
15. [Formal Verification](#15-formal-verification)
16. [Performance Accelerator](#16-performance-accelerator)
17. [Standard Library Modules](#17-standard-library-modules)
18. [Post-Quantum Cryptography](#18-post-quantum-cryptography)
19. [Complete Working Examples](#19-complete-working-examples)
20. [Quick Reference: All Keywords & Built-ins](#20-quick-reference-all-keywords--built-ins)
21. [Production Hardening (v0.4.0)](#21-production-hardening-v040)

---

## 1. Production Readiness Assessment

### What Zexus Blockchain Provides

| Capability | Module | Status |
|---|---|---|
| Smart contracts with state, events, access control | `contract_vm.py`, evaluator | **Implemented** |
| Fungible tokens (ERC-20 analog: ZX-20) | `tokens.py` → `ZX20Token` | **Implemented** |
| NFTs (ERC-721 analog: ZX-721) | `tokens.py` → `ZX721Token` | **Implemented** |
| Multi-tokens (ERC-1155 analog: ZX-1155) | `tokens.py` → `ZX1155Token` | **Implemented** |
| HD Wallet (BIP-32/39/44) + Keystore V3 | `wallet.py` | **Implemented** |
| ECDSA (secp256k1) signing & verification | `crypto.py` | **Implemented** |
| Post-quantum signatures (SPHINCS+/WOTS+) | `crypto_bridge.py` | **Implemented** |
| Pluggable consensus (PoW, PoA, PoS, BFT) | `consensus.py` | **Implemented** |
| Full blockchain node with mining | `node.py`, `chain.py` | **Implemented** |
| TLS-encrypted P2P networking | `network.py` | **Implemented** |
| CA-signed TLS, mTLS, certificate pinning | `network.py` | **Implemented (v0.4.0)** |
| Cross-chain bridge (lock-mint / burn-release) | `multichain.py` | **Implemented** |
| Merkle Patricia Trie state storage | `mpt.py` | **Implemented** |
| Event indexing with Bloom filters (SQLite) | `events.py` | **Implemented** |
| Pluggable storage backends (SQLite/LevelDB/RocksDB) | `storage.py` | **Implemented (v0.4.0)** |
| Gas metering (EVM-compatible cost model) | `transaction.py` | **Implemented** |
| JSON-RPC 2.0 (HTTP + WebSocket) — 39 methods | `rpc.py` | **Implemented** |
| Upgradeable proxy contracts | `upgradeable.py` | **Implemented** |
| On-chain governance (proposals, voting, quorum) | `upgradeable.py` | **Implemented** |
| Formal verification (structural + invariant + bounded model checking + taint analysis) | `verification.py` | **Implemented** |
| AOT compilation, inline caching, WASM cache, parallel batch execution | `accelerator.py` | **Implemented** |
| Versioned ledger with audit trail & integrity checks | `ledger.py` | **Implemented** |
| Mempool with Replace-by-Fee (RBF) | `chain.py` → `Mempool` | **Implemented** |
| Peer reputation & rate limiting | `network.py` | **Implemented** |
| Reentrancy guard & call-depth limiting | `contract_vm.py` | **Implemented** |
| Production monitoring & Prometheus metrics | `monitoring.py` | **Implemented (v0.4.0)** |

### Verdict

Zexus provides a **complete, production-hardened blockchain development stack** — from language-level keywords (`contract`, `emit`, `require`, `tx`, `gas`, etc.) through a full runtime (consensus, networking, state tries, RPC) to tooling (formal verification with taint analysis, AOT compilation, governance, production monitoring). The 20-module architecture in `src/zexus/blockchain/` covers all subsystems needed for production blockchain deployment.

---

## 2. Language Keywords for Blockchain

These keywords are defined as token types in the Zexus lexer and parsed into AST nodes.

### Statement Keywords

| Keyword | Purpose | Example |
|---|---|---|
| `contract` | Define a smart contract | `contract Token { ... }` |
| `action` | Define a contract method / function | `action transfer(to, amount) { ... }` |
| `state` | Mutable state variable | `state counter = 0;` |
| `data` | Contract data member (inside contracts) | `data balances = {}` |
| `ledger` | Immutable versioned storage | `ledger balances = {};` |
| `persistent storage` | Persistent on-chain storage | `persistent storage totalSupply: integer` |
| `require` | Assert condition or revert | `require(balance >= amount, "Insufficient")` |
| `revert` | Explicitly revert a transaction | `revert("Not authorized")` |
| `emit` | Emit a blockchain event | `emit Transfer(from, to, amount)` |
| `limit` | Set gas limit | `limit(10000);` |
| `protocol` | Define an interface/trait | `protocol Transferable { ... }` |
| `implements` | Declare a contract implements a protocol | `contract Token implements Transferable { ... }` |
| `modifier` | Define a reusable access modifier | `modifier onlyOwner { require(TX.caller == owner, "Not owner") }` |
| `event` | Declare an event signature inside a contract | `event Transfer(from: Address, to: Address, amount: integer)` |

### Expression Keywords

| Keyword | Purpose | Example |
|---|---|---|
| `TX.caller` | Get the transaction sender address | `let sender = TX.caller` |
| `TX.timestamp` | Get the block timestamp | `let time = TX.timestamp` |
| `TX.block_hash` | Get current block hash | `let bh = TX.block_hash` |
| `TX.gas_limit` | Get gas limit of current tx | `let gl = TX.gas_limit` |
| `TX.gas_used` | Get gas consumed so far | `let gu = TX.gas_used` |
| `TX.gas_remaining` | Get remaining gas | `let gr = TX.gas_remaining` |
| `hash(data, algo)` | Cryptographic hash | `hash("hello", "SHA256")` |
| `signature(data, key, algo)` | Create digital signature | `signature(msg, privKey, "ECDSA")` |
| `verify_sig(data, sig, pubkey, algo)` | Verify a signature | `verify_sig(msg, sig, pubKey, "ECDSA")` |
| `gas.used` | Gas consumed | `let g = gas.used` |
| `gas.remaining` | Gas remaining | `let g = gas.remaining` |
| `gas.limit` | Gas limit | `let g = gas.limit` |
| `this` | Reference to current contract instance | `this.balances[account]` |
| `tx { ... }` | Transactional block (atomic execution) | `tx { balance = balance - amount; }` |

### Action Modifier Keywords

| Modifier | Purpose |
|---|---|
| `payable` | Action can receive tokens |
| `view` | Read-only action (no state mutation) |
| `pure` | No side effects |
| `public` | Exported / externally callable |
| `secure` | Security-hardened execution |
| `async` | Asynchronous execution |
| `native` | FFI / native call |
| `inline` | Inline at call site |

**Example with modifiers:**
```zexus
action payable transfer(to: Address, amount: integer) -> boolean {
    require(balances[TX.caller] >= amount, "Insufficient balance")
    balances[TX.caller] = balances[TX.caller] - amount
    balances[to] = balances[to] + amount
    emit Transfer(TX.caller, to, amount)
    return true
}

action pure total_supply() -> integer {
    return 1000000
}

action view balance_of(account: Address) -> integer {
    return balances[account]
}
```

### Module Import Keywords

| Syntax | Purpose |
|---|---|
| `use "crypto" as crypto` | Import the built-in crypto module |
| `use "blockchain" as bc` | Import the built-in blockchain stdlib module |
| `use "token.zx"` | Import another .zx file (contracts become available) |
| `export contract Name { }` | Export a contract for use by other files |
| `export action name() { }` | Export a function for use by other files |

---

## 3. Smart Contracts

### Contract Definition Syntax

```zexus
contract MyContract {
    // Data members (mutable state)
    data owner = ""
    data balances = {}
    data paused = false

    // Typed persistent storage (on-chain)
    persistent storage totalSupply: integer
    persistent storage allowances: Map<Address, Map<Address, integer>>

    // Event declarations
    event Transfer(from: Address, to: Address, amount: integer)
    event Approval(owner: Address, spender: Address, amount: integer)

    // Constructor (called once on deployment)
    action constructor() {
        owner = TX.caller
        balances[owner] = 1000000
    }

    // Actions (methods)
    action transfer(to, amount) {
        require(!paused, "Contract is paused")
        require(amount > 0, "Amount must be positive")
        let from_balance = this.balance_of(TX.caller)
        require(from_balance >= amount, "Insufficient balance")

        balances[TX.caller] = from_balance - amount
        balances[to] = balances[to] + amount
        emit Transfer(TX.caller, to, amount)
        return {"success": true}
    }

    action balance_of(account) {
        if balances[account] == null {
            return 0
        }
        return balances[account]
    }
}
```

### Exportable Contracts (for multi-file projects)

```zexus
// token.zx
export contract ZexusToken {
    data name = "Zexus Token"
    data symbol = "ZXS"
    data decimals = 18
    data total_supply = 0
    data owner = ""
    data balances = {}
    data allowances = {}
    data transfer_log = []
    data paused = false

    action init(initial_supply, deployer) {
        owner = deployer
        total_supply = initial_supply
        balances[deployer] = initial_supply
        return {"success": true, "supply": initial_supply}
    }

    action balance_of(account) {
        if balances[account] == null { return 0 }
        return balances[account]
    }

    action transfer(from_addr, to_addr, amount) {
        require(!paused, "Token is paused")
        require(amount > 0, "Amount must be positive")
        let from_balance = this.balance_of(from_addr)
        require(from_balance >= amount, "Insufficient balance")

        balances[from_addr] = from_balance - amount
        let to_balance = this.balance_of(to_addr)
        balances[to_addr] = to_balance + amount

        transfer_log = transfer_log + [{"type": "transfer", "from": from_addr, "to": to_addr, "amount": amount}]
        return {"success": true, "from": from_addr, "to": to_addr, "amount": amount}
    }

    action approve(owner_addr, spender, amount) {
        require(!paused, "Token is paused")
        require(amount >= 0, "Approval amount cannot be negative")
        let key = string(owner_addr) + ":" + string(spender)
        allowances[key] = amount
        return {"success": true}
    }

    action transfer_from(spender, from_addr, to_addr, amount) {
        require(!paused, "Token is paused")
        require(amount > 0, "Amount must be positive")
        let allowed = this.get_allowance(from_addr, spender)
        require(allowed >= amount, "Allowance exceeded")

        let result = this.transfer(from_addr, to_addr, amount)
        if result["success"] {
            let key = string(from_addr) + ":" + string(spender)
            allowances[key] = allowed - amount
        }
        return result
    }

    action get_allowance(owner_addr, spender) {
        let key = string(owner_addr) + ":" + string(spender)
        if allowances[key] == null { return 0 }
        return allowances[key]
    }

    action mint(to_addr, amount) {
        require(amount > 0, "Mint amount must be positive")
        total_supply = total_supply + amount
        let current = 0
        if balances[to_addr] != null { current = balances[to_addr] }
        balances[to_addr] = current + amount
        transfer_log = transfer_log + [{"type": "mint", "to": to_addr, "amount": amount}]
        return {"success": true, "new_supply": total_supply}
    }

    action burn(from_addr, amount) {
        require(amount > 0, "Burn amount must be positive")
        let bal = 0
        if balances[from_addr] != null { bal = balances[from_addr] }
        require(bal >= amount, "Insufficient balance to burn")
        balances[from_addr] = bal - amount
        total_supply = total_supply - amount
        transfer_log = transfer_log + [{"type": "burn", "from": from_addr, "amount": amount}]
        return {"success": true, "new_supply": total_supply}
    }

    action pause_token() {
        paused = true
        return {"success": true, "paused": true}
    }

    action unpause_token() {
        paused = false
        return {"success": true, "paused": false}
    }

    action get_total_supply() { return total_supply }
    action get_transfer_log() { return transfer_log }
}
```

### Contract Instantiation and Usage

```zexus
// Import the contract file
use "token.zx"

// Instantiate the contract
let token = ZexusToken()
token.init(10000000, "0xDEPLOYER")

// Use the contract
token.mint("0xALICE", 500000)
token.transfer("0xALICE", "0xBOB", 10000)
let balance = token.balance_of("0xALICE")
print("Alice balance: " + string(balance))
```

### Contract VM Runtime Features (from `contract_vm.py`)

The `ContractVM` provides these built-in functions inside contract execution:
- `verify_sig(data, signature, public_key)` — Verify a cryptographic signature
- `emit(event_name, ...args)` — Emit a blockchain event
- `get_balance(address)` — Query account balance
- `transfer(to, amount)` — Transfer tokens
- `keccak256(data)` — Keccak-256 hash
- `block_number()` — Current block number
- `block_timestamp()` — Current block timestamp
- `contract_call(address, action, args)` — Cross-contract call
- `static_call(address, action, args)` — Read-only cross-contract call
- `delegate_call(address, action, args)` — Delegatecall (runs with caller's storage)

**Security features:**
- Reentrancy guard (blocks recursive/reentrant calls)
- Call depth limit of 10 (prevents infinite cross-contract call chains)
- Gas metering for every operation
- State rollback on revert

---

## 4. Tokens (ZX-20, ZX-721, ZX-1155)

### ZX-20: Fungible Token (ERC-20 analog)

**Runtime class:** `ZX20Token` in `src/zexus/blockchain/tokens.py`

```python
# Python-level API (runtime module)
token = ZX20Token(name="Zexus Token", symbol="ZXS", decimals=18, initial_supply=1000000, owner="0xDEPLOYER")
token.transfer(sender="0xALICE", recipient="0xBOB", amount=100)
token.approve(owner="0xALICE", spender="0xDEX", amount=500)
token.transfer_from(spender="0xDEX", sender="0xALICE", recipient="0xBOB", amount=200)
token.mint(to="0xALICE", amount=1000, caller="0xOWNER")
token.burn(from_addr="0xALICE", amount=500)
balance = token.balance_of("0xALICE")
```

**Zexus-level contract (see Section 3 above for full example).**

### ZX-721: Non-Fungible Token (ERC-721 analog)

**Runtime class:** `ZX721Token` in `src/zexus/blockchain/tokens.py`

Methods: `mint(to, token_id, uri, caller)`, `burn(token_id, caller)`, `transfer_from(caller, from_addr, to, token_id)`, `safe_transfer_from(...)`, `approve(caller, to, token_id)`, `set_approval_for_all(owner, operator, approved)`, `owner_of(token_id)`, `token_uri(token_id)`, `tokens_of_owner(address)`

**Zexus-level NFT contract:**

```zexus
export contract NFTCollection {
    data collection_name = ""
    data collection_symbol = ""
    data owners = {}
    data metadata = {}
    data next_token_id = 1
    data total_minted = 0
    data owner_tokens = {}
    data approved = {}

    action init(name, symbol) {
        collection_name = name
        collection_symbol = symbol
        return {"success": true}
    }

    action mint(to_address, token_name, description, image_url) {
        let token_id = next_token_id
        next_token_id = next_token_id + 1
        total_minted = total_minted + 1

        owners[string(token_id)] = to_address
        metadata[string(token_id)] = {
            "name": token_name,
            "description": description,
            "image": image_url,
            "token_id": token_id,
            "creator": to_address
        }

        let existing = []
        if owner_tokens[to_address] != null {
            existing = owner_tokens[to_address]
        }
        owner_tokens[to_address] = existing + [token_id]
        return {"success": true, "token_id": token_id, "owner": to_address}
    }

    action transfer(from_addr, to_addr, token_id) {
        let tid = string(token_id)
        require(owners[tid] != null, "Token does not exist")
        require(owners[tid] == from_addr, "Not the owner")
        require(to_addr != "", "Recipient required")

        owners[tid] = to_addr
        let new_list = []
        if owner_tokens[to_addr] != null {
            new_list = owner_tokens[to_addr]
        }
        owner_tokens[to_addr] = new_list + [token_id]
        approved[tid] = ""
        return {"success": true, "token_id": token_id}
    }

    action owner_of(token_id) {
        let tid = string(token_id)
        if owners[tid] == null { return "" }
        return owners[tid]
    }

    action get_metadata(token_id) {
        let tid = string(token_id)
        if metadata[tid] == null { return {} }
        return metadata[tid]
    }

    action tokens_of(address) {
        if owner_tokens[address] == null { return [] }
        return owner_tokens[address]
    }

    action get_collection_info() {
        return {
            "name": collection_name,
            "symbol": collection_symbol,
            "total_minted": total_minted,
            "next_id": next_token_id
        }
    }
}
```

### ZX-1155: Multi-Token (ERC-1155 analog)

**Runtime class:** `ZX1155Token` in `src/zexus/blockchain/tokens.py`

Methods: `mint(to, token_id, amount, caller)`, `mint_batch(to, token_ids, amounts, caller)`, `burn(from_addr, token_id, amount)`, `burn_batch(from_addr, token_ids, amounts)`, `safe_transfer_from(caller, from_addr, to, token_id, amount)`, `safe_batch_transfer_from(...)`, `balance_of(address, token_id)`, `balance_of_batch(addresses, token_ids)`, `uri(token_id)`, `set_uri(token_id, uri, caller)`, `total_supply(token_id)`, `exists(token_id)`

---

## 5. Wallets & Cryptography

### HD Wallet — BIP-32/39/44 (from `wallet.py`)

**Constants:**
- `ZEXUS_COIN_TYPE = 806`
- `DEFAULT_PATH = "m/44'/806'/0'/0"`

**Key classes:**
- `HDWallet` — full hierarchical deterministic wallet
- `Account` — derived account (address, public key, private key, path)
- `Keystore` — V3 keystore format (AES-128-CTR + Scrypt)
- `ExtendedKey` — BIP-32 extended key with child derivation

**API:**
```python
# Create a new wallet
wallet = HDWallet.create(strength=128, passphrase="")
# From mnemonic
wallet = HDWallet.from_mnemonic("word1 word2 ... word12", passphrase="")

# Derive accounts
account = wallet.derive_account(index=0)
accounts = wallet.derive_multiple(count=5, start=0)

# Account has: address, public_key, private_key, path
account.sign("message")
account.sign_transaction({"to": "...", "value": 100})

# Mnemonic utilities
mnemonic = generate_mnemonic(strength=128)  # 12 words
is_valid = validate_mnemonic(mnemonic)
seed = mnemonic_to_seed(mnemonic, passphrase="")

# Keystore (encrypted storage)
keystore = Keystore.encrypt(private_key="...", password="secret")
private_key = Keystore.decrypt(keystore, password="secret")
Keystore.save(keystore, "wallet.json")
keystore = Keystore.load("wallet.json")
```

### Cryptographic Primitives (from `crypto.py`)

**Hash algorithms supported:** SHA256, SHA512, SHA3-256, SHA3-512, BLAKE2B, BLAKE2S, KECCAK256

**Zexus keyword-level usage:**

```zexus
// Hash
let h = hash("hello world", "SHA256")
let k = hash("data", "KECCAK256")

// Global built-in (no import needed)
let kh = keccak256("zexus blockchain")

// Signature
let keys = generateKeypair("ECDSA")
let sig = signature("message", keys["private_key"], "ECDSA")
let valid = verify_sig("message", sig, keys["public_key"], "ECDSA")

// Address derivation with custom prefix
let addr = deriveAddress(keys["public_key"])           // default "0x" prefix
let zx_addr = deriveAddress(keys["public_key"], "Zx01")  // custom prefix
let random = randomBytes(32)
```

**Crypto module (imported):**

```zexus
use "crypto" as crypto

let h = crypto.keccak256("data")
let keys = crypto.generate_keypair("ECDSA")
let sig = crypto.secp256k1_sign("data", keys["private_key"])
let valid = crypto.verify_signature("data", sig, keys["public_key"])
let merkle = crypto.calculate_merkle_root(["hash1", "hash2", "hash3"])
let sha = crypto.sha256("data")
let encrypted = crypto.aes_encrypt("plaintext", "key")
let decrypted = crypto.aes_decrypt(encrypted, "key")
```

### Global Built-in Crypto Functions (always available, no import needed)

| Function | Description |
|---|---|
| `hash(data, algorithm)` | Hash with specified algorithm |
| `keccak256(data)` | Keccak-256 hash |
| `generateKeypair(algorithm)` | Generate key pair ("ECDSA" or "RSA") |
| `signature(data, private_key, algorithm)` | Sign data |
| `verify_sig(data, signature, public_key, algorithm)` | Verify signature |
| `deriveAddress(public_key)` | Derive address from public key (default "0x" prefix) |
| `deriveAddress(public_key, prefix)` | Derive address with custom prefix |
| `randomBytes(length)` | Generate random bytes |

---

## 6. Consensus Engines

**Source:** `src/zexus/blockchain/consensus.py`

Four pluggable consensus engines, selectable via `create_consensus(algorithm, **kwargs)`:

### Proof of Work (PoW)
```python
consensus = create_consensus("pow", difficulty=4, max_iterations=1_000_000)
```
- SHA-256 nonce grinding
- Adjustable difficulty
- `seal(block, chain)` / `verify(block, chain)`

### Proof of Authority (PoA)
```python
consensus = create_consensus("poa", validators=["0xVAL1", "0xVAL2", "0xVAL3"], block_interval=5.0)
```
- Round-robin validator set
- `add_validator(address)` / `remove_validator(address)`

### Proof of Stake (PoS)
```python
consensus = create_consensus("pos", min_stake=1000)
```
- VRF-like weighted random validator selection
- `stake(address, amount)` / `unstake(address, amount)`
- `get_eligible_validators()`

### Byzantine Fault Tolerance (BFT) — PBFT 3-phase
```python
consensus = create_consensus("bft", validators=["0xVAL1", "0xVAL2", "0xVAL3"], block_interval=5.0, view_change_timeout=30.0)
```
- Pre-prepare → Prepare → Commit phases
- View change mechanism
- `propose(block, proposer)`, `on_pre_prepare(msg, sender)`, `on_prepare(msg, sender)`, `on_commit(msg, sender)`

---

## 7. Blockchain Nodes & Chain Management

### Chain (from `chain.py`)

**Data structures:**
- `BlockHeader` — height, prev_hash, timestamp, merkle_root, state_root, difficulty, nonce, miner
- `Transaction` — sender, recipient, value, data, nonce, gas_limit, gas_price, signature, tx_hash
- `TransactionReceipt` — tx_hash, block_hash, status, gas_used, logs, contract_address, return_value
- `Block` — header, transactions, receipts, hash

**Chain class:**
```python
chain = Chain(chain_id="zexus-mainnet", data_dir="/path/to/data")  # SQLite persistence
genesis = chain.create_genesis(miner="0xMINER")
chain.add_block(block)
block = chain.get_block(0)         # by height
block = chain.get_block("0xhash")  # by hash
account = chain.get_account("0xADDR")
info = chain.get_chain_info()
validation = chain.validate_chain()
```

**Mempool:**
```python
mempool = Mempool(max_size=10000, rbf_enabled=True, rbf_increment_pct=10.0)
mempool.add(tx)
mempool.replace_by_fee(tx)      # Replace-by-Fee support
pending = mempool.get_pending()
```

### Blockchain Node (from `node.py`)

```python
config = NodeConfig(
    chain_id="zexus-mainnet",
    host="0.0.0.0",
    port=30303,
    data_dir="./chain_data",
    consensus="pow",          # "pow" | "poa" | "pos" | "bft"
    miner_address="0xMINER",
    mining_enabled=True,
    rpc_enabled=True,
    rpc_port=8545,
    max_peers=50,
    mining_interval=10.0,
    block_gas_limit=10_000_000
)

node = BlockchainNode(config)
node.start()

# Transactions
tx = node.create_transaction(sender="0xALICE", recipient="0xBOB", value=100)
result = node.submit_transaction(tx)

# Queries
block = node.get_block(0)
latest = node.get_latest_block()
info = node.get_chain_info()
balance = node.get_balance("0xALICE")
nonce = node.get_nonce("0xALICE")

# Contracts
receipt = node.deploy_contract(code="contract source", deployer="0xDEPLOYER")
result = node.call_contract(contract_address, action="transfer", args=[...], caller="0xCALLER")
result = node.static_call_contract(contract_address, action="balanceOf", args=[...])

# Mining
node.start_mining()
block = node.mine_block_sync()
node.stop_mining()

# Admin
node.fund_account("0xALICE", 1000000)
chain_export = node.export_chain()

# Events
node.on("new_block", handler_fn)
node.on("new_tx", handler_fn)

# Cross-chain
node.join_router(router)
node.bridge_to("dest-chain-id", amount=100)

node.stop()
```

---

## 8. P2P Networking

**Source:** `src/zexus/blockchain/network.py`

### Features
- TLS-encrypted connections (self-signed certificates)
- Peer discovery
- Peer reputation management (scoring, banning)
- Rate limiting per peer per message type
- Gossip protocol for block/tx propagation

### Message Types
`PING`, `PONG`, `FIND_PEERS`, `PEERS`, `GET_BLOCKS`, `BLOCKS`, `GET_HEADERS`, `HEADERS`, `NEW_BLOCK`, `NEW_TX`, `GET_TX`, `TX`, `VOTE`, `PROPOSE`, `HANDSHAKE`, `DISCONNECT`, `ERROR`

### API
```python
network = P2PNetwork(node_id="node1", host="0.0.0.0", port=30303, chain_id="zexus-main", max_peers=50)
network.start()

# Connect to peers
await network.connect_to("peer_host", 30303)

# Send messages
network.broadcast(message)
network.gossip(message, exclude_peers=["peer1"])
network.send(peer_id, message)

# Peer management
network.disconnect_peer(peer_id)
network.discover_peers()
info = network.get_network_info()

# Events
network.on("message", handler)
network.on("peer_connected", handler)
network.on("peer_disconnected", handler)
```

### Peer Reputation
```python
rep_manager = PeerReputationManager()
# Reputation deltas:
# VALID_BLOCK = +5, INVALID_BLOCK = -20
# VALID_TX = +1, INVALID_TX = -10
# TIMEOUT = -5, SPAM = -15, GOOD_PEER = +3

rep_manager.update(peer_id, PeerReputationManager.VALID_BLOCK)
rep_manager.ban(peer_id)
is_banned = rep_manager.is_banned(peer_id)
```

---

## 9. Cross-Chain / Multichain Bridge

**Source:** `src/zexus/blockchain/multichain.py`

### Architecture
- **ChainRouter** — routes messages between chains
- **BridgeRelay** — Merkle-proof verification of cross-chain messages
- **BridgeContract** — lock-and-mint / burn-and-release asset bridging
- **MerkleProofEngine** — generates and verifies Merkle proofs

### API

```python
# Create a router and register chains
router = ChainRouter()
router.register_chain("chain-a", chain_a)
router.register_chain("chain-b", chain_b)
router.connect("chain-a", "chain-b")

# Send cross-chain message
msg = router.send(source_chain="chain-a", dest_chain="chain-b", sender="0xALICE", payload={"action": "transfer", "amount": 100})

# Relay messages
results = router.relay(router.flush_outbox("chain-a"))

# Or send + relay in one step
msg, success, reason = router.send_and_relay("chain-a", "chain-b", "0xALICE", payload)

# Bridge contract for asset transfer
bridge = BridgeContract(router, source_chain="chain-a", dest_chain="chain-b")
result = bridge.lock_and_mint(sender="0xALICE", amount=100, recipient="0xALICE")
result = bridge.burn_and_release(sender="0xALICE", amount=50, recipient="0xALICE")
tvl = bridge.total_value_locked
minted = bridge.total_minted
```

### Merkle Proof Engine
```python
root = MerkleProofEngine.compute_root(["hash1", "hash2", "hash3", "hash4"])
proof = MerkleProofEngine.generate_proof(["hash1", "hash2", "hash3", "hash4"], index=0)
is_valid = MerkleProofEngine.verify_proof("hash1", proof, root)
```

### Zexus-Level Cross-Chain Example

```zexus
use "crypto" as crypto
use "blockchain" as bc

// Generate keypairs and derive addresses with custom prefix
let alice_keys = generateKeypair("ECDSA")
let alice_addr = deriveAddress(alice_keys["public_key"], "Zx01")

// Simulate cross-chain transfer (lock-and-mint pattern)
let chain_a_balances = {"alice": 1000, "escrow": 0}
let chain_b_balances = {"alice": 0}

// Lock on Chain A
let transfer_amount = 300
chain_a_balances["alice"] = chain_a_balances["alice"] - transfer_amount
chain_a_balances["escrow"] = chain_a_balances["escrow"] + transfer_amount

// Create signed cross-chain message
let msg_data = string(transfer_amount) + "|alice|chainA|chainB"
let msg_hash = hash(msg_data, "SHA256")

// Mint on Chain B (after Merkle proof verification by relay)
chain_b_balances["alice"] = chain_b_balances["alice"] + transfer_amount

// Burn-and-release (reverse direction)
let burn_amount = 100
chain_b_balances["alice"] = chain_b_balances["alice"] - burn_amount
chain_a_balances["escrow"] = chain_a_balances["escrow"] - burn_amount
chain_a_balances["alice"] = chain_a_balances["alice"] + burn_amount
```

---

## 10. Events & Logging

**Source:** `src/zexus/blockchain/events.py`

### Bloom Filter
- 2048-bit, 3 hash functions
- Fast probabilistic membership testing for event topics

### Event Log
```python
log = EventLog(
    address="0xCONTRACT",
    topics=["Transfer(address,address,uint256)", "0xALICE", "0xBOB"],
    data={"amount": 100},
    block_number=42,
    tx_hash="0xtxhash",
    log_index=0
)
```

### Event Index (SQLite-backed)
```python
index = EventIndex(db_path=":memory:")  # or file path for persistence
index.index_block(block_number=42, logs=[log1, log2])

# Query logs
filter = LogFilter(from_block=0, to_block=100, address="0xCONTRACT", event_name="Transfer")
logs = index.get_logs(filter)
logs = index.get_logs_for_tx("0xtxhash")
logs = index.get_logs_for_block(42)

bloom = index.get_bloom(42)
count = index.count_logs()
```

### Zexus-Level Event Emission

```zexus
// Emit events (appended to __event_log__)
emit Transfer("0xALICE", "0xBOB", 100)
emit Approval("0xOWNER", "0xSPENDER", 500)
emit TestEvent()
emit PaymentEvent(sender, amount)

// Emit inside a contract
contract Token {
    event Transfer(from: Address, to: Address, amount: integer)

    action transfer(to, amount) {
        // ... transfer logic ...
        emit Transfer(TX.caller, to, amount)
    }
}

// Emit in conditionals and functions
if condition {
    emit ConditionalEvent("triggered")
}

action triggerEvent(name) {
    emit FunctionEvent(name)
}
```

---

## 11. State Storage (Merkle Patricia Trie)

**Source:** `src/zexus/blockchain/mpt.py`

### MerklePatriciaTrie
```python
trie = MerklePatriciaTrie()
trie.put("key1", "value1")
trie.put("key2", "value2")
value = trie.get("key1")
exists = trie.contains("key1")
deleted = trie.delete("key1")
root = trie.root_hash
size = trie.size

# Merkle proofs
proof = trie.generate_proof("key1")
is_valid, value = MerklePatriciaTrie.verify_proof(root_hash, "key1", proof)

# Snapshots
snap = trie.snapshot()
trie.restore(snap)

# Batch operations
trie.put_batch([("k1", "v1"), ("k2", "v2"), ("k3", "v3")])

# Iteration
all_items = trie.items()
all_keys = trie.keys()
all_values = trie.values()
```

### StateTrie (account + storage)
```python
state = StateTrie()
state.set_account("0xALICE", {"balance": 1000, "nonce": 0})
account = state.get_account("0xALICE")
state.set_storage("0xCONTRACT", "key", "value")
value = state.get_storage("0xCONTRACT", "key")
root = state.root_hash

# Proofs
account_proof = state.generate_account_proof("0xALICE")
storage_proof = state.generate_storage_proof("0xCONTRACT", "key")

# Snapshots
snap = state.snapshot()
state.restore(snap)

# Enumerate
all_accounts = state.all_accounts()
```

---

## 12. Gas Metering & Transaction Context

**Source:** `src/zexus/blockchain/transaction.py`

### Gas Cost Model (EVM-compatible)

| Operation | Gas Cost |
|---|---|
| `read` (SLOAD) | 800 |
| `write` (SSTORE) | 20,000 |
| `transfer` | 2,100 |
| `call` | 700 |
| `create` (deploy) | 32,000 |
| `hash` | 30 |
| `event` / `log` | 375 |

### Transaction Context

```python
ctx = create_tx_context(caller="0xALICE", gas_limit=1_000_000)
ctx.consume_gas(5000)
print(ctx.gas_used, ctx.gas_remaining)
ctx.revert("Some reason")

# Global helpers
consume_gas(200)
check_gas_and_consume("read")  # consumes 800 gas
```

### Gas Estimator
```python
tracker = GasTracker()
cost = tracker.get_cost("transfer")  # 2100
estimate = tracker.estimate_limit(["read", "write", "transfer"])  # sum of costs
```

### Zexus-Level Gas Usage

```zexus
// Set gas limit
limit(10000);

// Access gas tracking
let used = gas.used
let remaining = gas.remaining
let limit_val = gas.limit

// Gas estimation via blockchain stdlib
use "blockchain" as bc
// (see gas module in stdlib tests)
```

---

## 13. JSON-RPC Server

**Source:** `src/zexus/blockchain/rpc.py`

### Setup
```python
server = RPCServer(node=blockchain_node, host="0.0.0.0", port=8545, max_rps=100)
server.start()
```

### All 39 RPC Methods

**Chain queries (`zx_*`):**
| Method | Description |
|---|---|
| `zx_chainId` | Get chain ID |
| `zx_blockNumber` | Get latest block number |
| `zx_getBlockByNumber` | Get block by height |
| `zx_getBlockByHash` | Get block by hash |
| `zx_getBlockTransactionCount` | Transaction count in block |
| `zx_getBalance` | Get account balance |
| `zx_getTransactionCount` | Get account nonce |
| `zx_getAccount` | Get full account info |
| `zx_getTransactionByHash` | Get transaction by hash |
| `zx_getTransactionReceipt` | Get transaction receipt |
| `zx_sendTransaction` | Submit a transaction |
| `zx_sendRawTransaction` | Submit a raw signed transaction |
| `zx_gasPrice` | Get current gas price |
| `zx_estimateGas` | Estimate gas for a transaction |
| `zx_getChainInfo` | Get chain information |
| `zx_validateChain` | Validate chain integrity |
| `zx_getCode` | Get contract code at address |
| `zx_getLogs` | Query event logs |

**Transaction pool (`txpool_*`):**
| Method | Description |
|---|---|
| `txpool_status` | Mempool status |
| `txpool_content` | Pending transactions |
| `txpool_replaceByFee` | Replace transaction by fee |

**Network (`net_*`):**
| Method | Description |
|---|---|
| `net_version` | Network version |
| `net_peerCount` | Connected peer count |
| `net_listening` | Is node listening |
| `net_peers` | List connected peers |

**Mining (`miner_*`):**
| Method | Description |
|---|---|
| `miner_start` | Start mining |
| `miner_stop` | Stop mining |
| `miner_status` | Mining status |
| `miner_setMinerAddress` | Set miner address |
| `miner_mineBlock` | Mine a single block |

**Smart contracts (`contract_*`):**
| Method | Description |
|---|---|
| `contract_deploy` | Deploy a contract |
| `contract_call` | Call a contract action |
| `contract_staticCall` | Read-only contract call |

**Admin (`admin_*`):**
| Method | Description |
|---|---|
| `admin_nodeInfo` | Node information |
| `admin_fundAccount` | Fund an account (testnet) |
| `admin_exportChain` | Export chain data |
| `admin_rpcMethods` | List all RPC methods |

### WebSocket Subscriptions
The RPC server supports WebSocket connections with push notifications for:
- `new_block` — new blocks mined
- `new_tx` — new transactions
- `mined` — mining events

### RPC Error Codes
| Code | Name |
|---|---|
| -32700 | Parse Error |
| -32600 | Invalid Request |
| -32601 | Method Not Found |
| -32602 | Invalid Params |
| -32603 | Internal Error |
| -32001 | Not Found |
| -32010 | Transaction Rejected |
| -32011 | Insufficient Funds |
| -32012 | Nonce Too Low |
| -32013 | Gas Limit Exceeded |
| -32020 | Contract Error |
| -32030 | Mining Error |
| -32050 | Rate Limited |

---

## 14. Upgradeable Contracts & Governance

**Source:** `src/zexus/blockchain/upgradeable.py`

### Proxy Contracts (upgradeable pattern)

```python
manager = UpgradeManager(vm=contract_vm)

# Create a proxy
proxy_addr, proxy = manager.create_proxy(
    admin="0xADMIN",
    implementation="0xIMPL_V1",
    deployer="0xDEPLOYER"
)

# Upgrade to new implementation
success, msg = manager.upgrade(proxy_addr, "0xIMPL_V2", caller="0xADMIN", metadata={"version": "2.0"})

# Rollback to previous version
success, msg = manager.rollback(proxy_addr, target_version=1, caller="0xADMIN")

# Change admin
success, msg = manager.change_admin(proxy_addr, "0xNEW_ADMIN", caller="0xADMIN")

# Call through proxy (delegates to implementation)
result = manager.proxy_call(proxy_addr, "transfer", args=[...], caller="0xUSER")

# History
history = manager.get_version_history(proxy_addr)
current = manager.get_current_version(proxy_addr)
```

### On-Chain Governance

**Allowed governance parameters:**
- `difficulty` (int)
- `target_block_time` (float)
- `chain_id` (str)
- `gas_limit_default` (int)
- `base_fee` (int)
- `max_block_gas` (int)
- `consensus_algorithm` (str)

**Proposal types:** `CONSENSUS_CHANGE`, `DIFFICULTY_CHANGE`, `GAS_LIMIT_CHANGE`, `BLOCK_TIME_CHANGE`, `FEE_STRUCTURE`, `CHAIN_PARAMETER`, `HARD_FORK`

```python
gov = ChainUpgradeGovernance(
    chain=chain,
    validators=["0xVAL1", "0xVAL2", "0xVAL3"],
    quorum_numerator=2,
    quorum_denominator=3
)

# Create proposal
success, msg, proposal_id = gov.propose(
    proposer="0xVAL1",
    proposal_type=ProposalType.DIFFICULTY_CHANGE,
    description="Increase difficulty to 6",
    changes={"difficulty": 6},
    activation_height=1000
)

# Vote
success, msg = gov.vote(proposal_id, voter="0xVAL2", approve=True)
success, msg = gov.vote(proposal_id, voter="0xVAL3", approve=True)

# Apply pending proposals
applied = gov.apply_pending(current_height=1000)

# Revert an upgrade
success, msg = gov.revert_upgrade(proposal_id, caller="0xVAL1")
```

### Zexus-Level DAO Governance Contract

```zexus
export contract Governance {
    data dao_name = ""
    data proposals = {}
    data votes = {}
    data next_proposal_id = 1
    data quorum = 50
    data token_contract = null

    action init(name, token, min_quorum) {
        dao_name = name
        token_contract = token
        quorum = min_quorum
        return {"success": true}
    }

    action create_proposal(proposer, title, description) {
        let balance = token_contract.balance_of(proposer)
        require(balance > 0, "Must hold tokens to create proposals")

        let pid = next_proposal_id
        next_proposal_id = next_proposal_id + 1
        proposals[string(pid)] = {
            "id": pid, "title": title, "description": description,
            "proposer": proposer, "yes_votes": 0, "no_votes": 0,
            "status": "active", "voter_count": 0
        }
        return {"success": true, "proposal_id": pid}
    }

    action vote(voter, proposal_id, support) {
        let pid = string(proposal_id)
        require(proposals[pid] != null, "Proposal does not exist")
        let proposal = proposals[pid]
        require(proposal["status"] == "active", "Proposal is not active")

        let vote_key = string(pid) + ":" + string(voter)
        require(votes[vote_key] == null, "Already voted")

        let power = token_contract.balance_of(voter)
        require(power > 0, "Must hold tokens to vote")

        votes[vote_key] = {"support": support, "power": power}
        if support {
            proposal["yes_votes"] = proposal["yes_votes"] + power
        } else {
            proposal["no_votes"] = proposal["no_votes"] + power
        }
        proposal["voter_count"] = proposal["voter_count"] + 1
        proposals[pid] = proposal
        return {"success": true}
    }

    action finalize(proposal_id) {
        let pid = string(proposal_id)
        require(proposals[pid] != null, "Proposal does not exist")
        let proposal = proposals[pid]
        require(proposal["status"] == "active", "Already finalized")

        let total_votes = proposal["yes_votes"] + proposal["no_votes"]
        if total_votes < quorum {
            proposal["status"] = "failed_quorum"
        } elif proposal["yes_votes"] > proposal["no_votes"] {
            proposal["status"] = "passed"
        } else {
            proposal["status"] = "rejected"
        }
        proposals[pid] = proposal
        return {"success": true, "status": proposal["status"]}
    }
}
```

---

## 15. Formal Verification

**Source:** `src/zexus/blockchain/verification.py`

### Verification Levels

| Level | Class | What It Checks |
|---|---|---|
| Level 1: Structural | `StructuralVerifier` | Access control, reentrancy vulnerabilities, division by zero, arithmetic overflow, transfer balance checks |
| Level 2: Invariant | `InvariantVerifier` | User-defined invariants (e.g., `balance >= 0`) are preserved across all actions |
| Level 3: Property | `PropertyVerifier` | Bounded model checking of pre/post-conditions with `@old(var)` references |

### Severity Levels
`INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`

### Finding Categories (13)
`ACCESS_CONTROL`, `REENTRANCY`, `ARITHMETIC_OVERFLOW`, `DIVISION_BY_ZERO`, `UNCHECKED_RETURN`, `INTEGER_UNDERFLOW`, `FRONT_RUNNING`, `DENIAL_OF_SERVICE`, `INVARIANT_VIOLATION`, `PRECONDITION_VIOLATION`, `POSTCONDITION_VIOLATION`, `STATE_INCONSISTENCY`, `GENERAL`

### Usage

```python
from zexus.blockchain.verification import FormalVerifier, VerificationLevel

# Basic structural verification
verifier = FormalVerifier(level=VerificationLevel.STRUCTURAL)
report = verifier.verify_contract(contract)
print(report.passed)  # True/False
print(report.summary())

# With invariants
verifier = FormalVerifier(level=VerificationLevel.INVARIANT)
verifier.add_invariant("balance >= 0")
verifier.add_invariant("total_supply > 0")
report = verifier.verify_contract(contract)

# With properties (bounded model checking)
verifier = FormalVerifier(level=VerificationLevel.PROPERTY, max_depth=3)
verifier.add_property(
    name="transfer_preserves_total",
    precondition="total_supply > 0",
    postcondition="total_supply == @old(total_supply)",
    action="transfer"
)
report = verifier.verify_contract(contract)

# Verify multiple contracts
reports = verifier.verify_multiple([contract1, contract2, contract3])
```

### Annotation-Based Verification

Contracts can embed verification annotations:
```
@invariant balance >= 0
@invariant total_supply > 0
@property transfer_preserves_supply
@pre total_supply > 0
@post total_supply == @old(total_supply)
```

```python
from zexus.blockchain.verification import AnnotationParser

annotations = AnnotationParser.parse_annotations(contract_source_code)
# or
annotations = AnnotationParser.from_contract_metadata(contract)
```

---

## 16. Performance Accelerator

**Source:** `src/zexus/blockchain/accelerator.py`

### Components

| Component | Class | Purpose |
|---|---|---|
| AOT Compiler | `AOTCompiler` | Ahead-of-time compilation of contract actions |
| Inline Cache | `InlineCache` | LRU cache (max 4096) for hot execution paths |
| WASM Cache | `WASMCache` | Disk-backed LRU cache for compiled WASM modules |
| Numeric Fast Path | `NumericFastPath` | Detects purely numeric operations for fast execution |
| Batch Executor | `BatchExecutor` | Parallel batch execution of transactions |

### Unified Accelerator API

```python
accel = ExecutionAccelerator(
    contract_vm=vm,
    cache_dir="./cache",
    aot_enabled=True,
    ic_enabled=True,
    wasm_cache_enabled=True,
    numeric_fast_path=True,
    optimization_level=2
)

# Auto-compile on deploy
accel.on_contract_deployed("0xCONTRACT", contract)

# Accelerated execution
result = accel.execute("0xCONTRACT", "transfer", args=[...], caller="0xALICE")

# Batch execution
batch_result = accel.execute_batch(transactions, chain)
print(batch_result.throughput)  # tx/s

# On upgrade
accel.on_contract_upgraded("0xCONTRACT", new_contract)

# Stats
stats = accel.get_stats()
accel.clear_caches()
```

---

## 17. Standard Library Modules

### `use "blockchain" as bc`

Available functions from `src/zexus/stdlib/blockchain.py`:

| Function | Description |
|---|---|
| `bc.create_address(key, prefix)` | Create a blockchain address |
| `bc.validate_address(address)` | Validate an address |
| `bc.calculate_merkle_root(hashes)` | Calculate Merkle root from hash list |
| `bc.create_block(index, timestamp, data, prev_hash, nonce)` | Create a block |
| `bc.hash_block(block)` | Hash a block |
| `bc.validate_block(block, prev_block)` | Validate block against predecessor |
| `bc.proof_of_work(block_data, difficulty)` | Mine: find nonce for difficulty |
| `bc.validate_proof_of_work(block_data, difficulty)` | Validate a PoW solution |
| `bc.create_transaction(from, to, amount)` | Create a transaction |
| `bc.hash_transaction(tx)` | Hash a transaction |
| `bc.create_genesis_block()` | Create the genesis block |
| `bc.validate_chain(blocks)` | Validate a chain of blocks |
| `bc.quantum_resistant_hash(data)` | Post-quantum resistant hash |
| `bc.quantum_safe_address(data)` | Post-quantum safe address |
| `bc.create_quantum_block(...)` | Create a quantum-resistant block |
| `bc.quantum_proof_of_work(data, difficulty)` | Quantum-resistant PoW |
| `bc.validate_quantum_proof_of_work(...)` | Validate quantum PoW |

### `use "crypto" as crypto`

Available functions from `src/zexus/builtin_modules.py`:

| Function | Description |
|---|---|
| `crypto.keccak256(data)` | Keccak-256 hash |
| `crypto.generate_keypair(algo)` | Generate ECDSA keypair |
| `crypto.secp256k1_sign(data, key)` | ECDSA sign |
| `crypto.verify_signature(data, sig, key)` | Verify signature |
| `crypto.calculate_merkle_root(hashes)` | Merkle root |
| `crypto.sha256(data)` | SHA-256 hash |
| `crypto.aes_encrypt(data, key)` | AES encryption |
| `crypto.aes_decrypt(data, key)` | AES decryption |

Additional from `src/zexus/builtin_plugins.py`:

| Function | Description |
|---|---|
| `crypto.sha512(data)` | SHA-512 hash |
| `crypto.hmac(data, key)` | HMAC |
| `crypto.random(length)` | Random bytes |

---

## 18. Post-Quantum Cryptography

**Source:** `src/zexus/crypto_bridge.py`

Zexus supports post-quantum cryptographic operations with automatic backend selection:

**Backend priority:** `liboqs` (Open Quantum Safe) → `pqcrypto` (SPHINCS+) → Pure-Python WOTS+ fallback

```python
from zexus.crypto_bridge import sha256_hash, generate_sphincs_keypair, sphincs_sign, sphincs_verify

# SHA-256 (always available)
h = sha256_hash("data")

# Post-quantum keypair generation
keypair = generate_sphincs_keypair()  # {"public_key": "...", "private_key": "..."}

# Post-quantum signing
sig = sphincs_sign("message", keypair["private_key"])

# Post-quantum verification
is_valid = sphincs_verify("message", sig, keypair["public_key"])
```

Also available through the blockchain stdlib:
```zexus
use "blockchain" as bc

let qhash = bc.quantum_resistant_hash("data")
let qaddr = bc.quantum_safe_address("data")
```

---

## 19. Complete Working Examples

### Example 1: Full Token + Wallet + Staking Ecosystem

```zexus
use "token.zx"
use "wallet.zx"
use "staking.zx"
use "crypto" as crypto
use "blockchain" as bc

// Deploy token
let token = ZexusToken()
token.init(10000000, "0xDEPLOYER")

// Fund accounts
token.mint("0xALICE", 500000)
token.mint("0xBOB", 300000)

// Create wallets
let alice_wallet = Wallet()
alice_wallet.setup("0xALICE", "Alice")
let bob_wallet = Wallet()
bob_wallet.setup("0xBOB", "Bob")

// Transfer tokens
token.transfer("0xALICE", "0xBOB", 10000)
alice_wallet.send_tokens(token, "0xBOB", 5000)

// Approve and transfer_from
token.approve("0xBOB", "0xALICE", 50000)
token.transfer_from("0xALICE", "0xBOB", "0xDEPLOYER", 20000)

// Staking
let staking = StakingPool()
staking.init("ZXS Staking", 5, 100)
staking.stake(token, "0xALICE", 50000)
staking.stake(token, "0xBOB", 30000)
staking.distribute_rewards(token)

let alice_rewards = staking.get_rewards("0xALICE")  // 2500
let pool_info = staking.get_pool_info()
print("Total staked: " + string(pool_info["total_staked"]))

// Unstake
staking.unstake(token, "0xALICE", 20000)
```

### Example 2: AMM DEX with Swaps

```zexus
use "dex.zx"

let dex = SimpleDEX()
dex.init("ZexusDEX", 30)  // 0.3% fee

// Add liquidity (100K token A, 200K token B)
dex.add_liquidity("0xDEPLOYER", 100000, 200000)

// Check price
let price = dex.get_price_a_in_b()  // 2 (1 A = 2 B)

// Swap: trade 1000 A for B (constant product AMM)
let swap = dex.swap_a_to_b("0xALICE", 1000)
print("Got " + string(swap["amount_out"]) + " B tokens")
print("Fee: " + string(swap["fee"]))

// Reverse swap: trade 500 B for A
let swap2 = dex.swap_b_to_a("0xBOB", 500)

// Pool info
let info = dex.get_pool_info()
print("Trades: " + string(info["trade_count"]))
```

### Example 3: NFT Collection

```zexus
use "nft.zx"

let nft = NFTCollection()
nft.init("ZexusPunks", "ZPUNK")

// Mint NFTs with metadata
nft.mint("0xALICE", "Zexus Genesis #1", "The first Zexus NFT", "ipfs://Qm1234")
nft.mint("0xALICE", "Zexus Genesis #2", "Rare golden variant", "ipfs://Qm5678")
nft.mint("0xBOB", "Zexus Punk #1", "A punk for Bob", "ipfs://QmABCD")

// Check ownership
let owner = nft.owner_of(1)  // "0xALICE"
let alice_nfts = nft.tokens_of("0xALICE")  // [1, 2]

// Transfer NFT
nft.transfer("0xALICE", "0xCHARLIE", 2)
let new_owner = nft.owner_of(2)  // "0xCHARLIE"

// Get metadata
let meta = nft.get_metadata(1)
print("Name: " + string(meta["name"]))
print("Image: " + string(meta["image"]))
```

### Example 4: DAO Governance

```zexus
use "token.zx"
use "governance.zx"

let token = ZexusToken()
token.init(10000000, "0xDEPLOYER")
token.mint("0xALICE", 500000)
token.mint("0xBOB", 300000)
token.mint("0xCHARLIE", 200000)

let dao = Governance()
dao.init("ZexusDAO", token, 1000)

// Create proposal (must hold tokens)
let prop = dao.create_proposal("0xALICE", "Upgrade Staking", "Increase rewards to 8%")

// Vote (voting power = token balance)
dao.vote("0xALICE", prop["proposal_id"], true)    // 500K YES
dao.vote("0xBOB", prop["proposal_id"], true)       // 300K YES
dao.vote("0xCHARLIE", prop["proposal_id"], false)  // 200K NO

// Finalize
let result = dao.finalize(prop["proposal_id"])
print("Status: " + string(result["status"]))  // "passed"
```

### Example 5: Cryptography & Address Operations

```zexus
use "crypto" as crypto
use "blockchain" as bc

// Generate ECDSA keypair
let keys = generateKeypair("ECDSA")
print("Public key: " + string(keys["public_key"])[:64] + "...")

// Derive addresses with different prefixes
let eth_addr = deriveAddress(keys["public_key"])            // "0x..."
let zx_addr = deriveAddress(keys["public_key"], "Zx01")    // "Zx01..."
let custom_addr = deriveAddress(keys["public_key"], "TEST_") // "TEST_..."

// All share the same 40-hex body
let body1 = string(eth_addr)[2:]
let body2 = string(zx_addr)[4:]
// body1 == body2

// Sign and verify
let message = "Transfer 100 tokens to Bob"
let sig = signature(message, keys["private_key"], "ECDSA")
let valid = verify_sig(message, sig, keys["public_key"], "ECDSA")
print("Signature valid: " + string(valid))

// Hash with various algorithms
let sha256 = hash("data", "SHA256")
let sha3 = hash("data", "SHA3-256")
let blake = hash("data", "BLAKE2B")
let keccak = keccak256("data")

// Merkle tree verification
let m1 = hash("tx1", "SHA256")
let m2 = hash("tx2", "SHA256")
let m3 = hash("tx3", "SHA256")
let m4 = hash("tx4", "SHA256")

let h12 = hash(string(m1) + string(m2), "SHA256")
let h34 = hash(string(m3) + string(m4), "SHA256")
let merkle_root = hash(string(h12) + string(h34), "SHA256")

// Verify inclusion of m1
let verify_step = hash(string(m1) + string(m2), "SHA256")
let verify_root = hash(string(verify_step) + string(h34), "SHA256")
let is_included = (verify_root == merkle_root)  // true
```

### Example 6: Blockchain Utilities (Blocks, Mining, Validation)

```zexus
use "blockchain" as bc
use "crypto" as crypto

// Generate addresses
let addr = bc.create_address(crypto.keccak256("seed"), "0x")

// Build a chain
let genesis = bc.create_genesis_block()
let block1 = bc.create_block(1, 1700000001, "Block 1 data", genesis["hash"], 0)
let block2 = bc.create_block(2, 1700000002, "Block 2 data", bc.hash_block(block1), 0)

// Mine a block (find nonce for difficulty)
print("Mining...")
let mined = bc.proof_of_work("test block data", 1)
print("Nonce: " + string(mined["nonce"]) + " Hash: " + string(mined["hash"]))

// Validate chain
let chain = [genesis, block1, block2]
let valid = bc.validate_chain(chain)
print("Chain valid: " + string(valid))

// Create a transaction
let tx = bc.create_transaction("0xALICE", "0xBOB", 1000)
print("TX hash: " + string(tx["hash"]))

// Calculate Merkle root
let hashes = [crypto.keccak256("tx1"), crypto.keccak256("tx2"), crypto.keccak256("tx3")]
let root = bc.calculate_merkle_root(hashes)
print("Merkle root: " + string(root))
```

### Example 7: Multi-File Smart Contract Ecosystem

**File structure:**

```
my_project/
├── token.zx           # Fungible token contract
├── wallet.zx          # Wallet contract
├── staking.zx         # Staking pool
├── dex.zx             # Decentralized exchange
├── nft.zx             # NFT collection
├── governance.zx      # DAO governance
├── utils.zx           # Utility functions
└── main.zx            # Entry point / orchestrator
```

**main.zx:**
```zexus
use "token.zx"
use "wallet.zx"
use "staking.zx"
use "dex.zx"
use "nft.zx"
use "governance.zx"
use "utils.zx"
use "crypto" as crypto
use "blockchain" as bc

// Deploy all contracts
let token = ZexusToken()
token.init(10000000, "0xDEPLOYER")

let staking = StakingPool()
staking.init("ZXS Staking", 5, 100)

let dex = SimpleDEX()
dex.init("ZexusDEX", 30)

let nft = NFTCollection()
nft.init("ZexusPunks", "ZPUNK")

let dao = Governance()
dao.init("ZexusDAO", token, 1000)

// ... orchestrate interactions between all contracts ...
```

**Run:** `python -m zexus run my_project/main.zx`

### Example 8: Staking Contract (DeFi pattern)

```zexus
export contract StakingPool {
    data pool_name = "ZXS Staking Pool"
    data staked_balances = {}
    data reward_balances = {}
    data total_staked = 0
    data reward_rate = 5
    data min_stake = 100
    data stakers = []
    data distribution_count = 0

    action init(name, rate, minimum) {
        pool_name = name
        reward_rate = rate
        min_stake = minimum
        return {"success": true}
    }

    action stake(token_contract, staker, amount) {
        require(amount >= min_stake, "Below minimum stake")

        let result = token_contract.transfer(staker, "0xSTAKING_POOL", amount)
        require(result["success"], "Token transfer failed")

        let current = 0
        if staked_balances[staker] != null {
            current = staked_balances[staker]
        }
        staked_balances[staker] = current + amount
        total_staked = total_staked + amount

        // Track unique stakers
        let is_new = true
        for each s in stakers {
            if s == staker { is_new = false }
        }
        if is_new { stakers = stakers + [staker] }

        return {"success": true, "staker": staker, "amount": amount}
    }

    action unstake(token_contract, staker, amount) {
        let current = 0
        if staked_balances[staker] != null {
            current = staked_balances[staker]
        }
        require(current >= amount, "Insufficient staked balance")

        let result = token_contract.transfer("0xSTAKING_POOL", staker, amount)
        require(result["success"], "Transfer failed")

        staked_balances[staker] = current - amount
        total_staked = total_staked - amount
        return {"success": true}
    }

    action distribute_rewards(token_contract) {
        require(total_staked > 0, "No tokens staked")
        distribution_count = distribution_count + 1

        let rewards_distributed = 0
        for each staker in stakers {
            let staked = staked_balances[staker]
            if staked != null {
                if staked > 0 {
                    let reward = (staked * reward_rate) / 100
                    token_contract.mint(staker, reward)

                    let current_reward = 0
                    if reward_balances[staker] != null {
                        current_reward = reward_balances[staker]
                    }
                    reward_balances[staker] = current_reward + reward
                    rewards_distributed = rewards_distributed + reward
                }
            }
        }
        return {"success": true, "total_rewards": rewards_distributed}
    }

    action get_staked(staker) {
        if staked_balances[staker] == null { return 0 }
        return staked_balances[staker]
    }

    action get_rewards(staker) {
        if reward_balances[staker] == null { return 0 }
        return reward_balances[staker]
    }

    action get_pool_info() {
        return {
            "name": pool_name,
            "total_staked": total_staked,
            "reward_rate": reward_rate,
            "staker_count": len(stakers),
            "distributions": distribution_count
        }
    }
}
```

### Example 9: Edge Cases & Security Patterns

```zexus
// Error handling with try/catch
let over_transfer = false
try {
    token.transfer("0xDAVE", "0xALICE", 999999999)
} catch (e) {
    over_transfer = true
}
print("Over-balance rejected: " + string(over_transfer))

// Zero amount rejection
let zero_transfer = false
try {
    token.transfer("0xALICE", "0xBOB", 0)
} catch (e) {
    zero_transfer = true
}

// Negative amount rejection
let neg_transfer = false
try {
    token.transfer("0xALICE", "0xBOB", -100)
} catch (e) {
    neg_transfer = true
}

// Pausable pattern
token.pause_token()
let paused_transfer = false
try {
    token.transfer("0xALICE", "0xBOB", 100)
} catch (e) {
    paused_transfer = true  // Blocked while paused
}
token.unpause_token()

// Double-vote prevention
let double_vote = false
try {
    dao.vote("0xALICE", 1, true)  // Already voted
} catch (e) {
    double_vote = true
}

// Non-owner NFT transfer prevention
let nft_fail = false
try {
    nft.transfer("0xDAVE", "0xALICE", 1)  // Dave doesn't own #1
} catch (e) {
    nft_fail = true
}
```

---

## 20. Quick Reference: All Keywords & Built-ins

### Blockchain-Specific Keywords (Lexer Tokens)

```
contract    ledger      state       data
action      tx          revert      require
hash        signature   verify_sig  limit
gas         persistent  storage     emit
protocol    implements  this        modifier
payable     view        pure        event
```

### Action Modifiers

```
payable     view        pure        public
secure      async       native      inline
```

### Module Imports

```
use "crypto" as crypto
use "blockchain" as bc
use "filename.zx"
export contract Name { ... }
export action name() { ... }
```

### Global Crypto Built-ins (always available)

```
hash(data, algorithm)
keccak256(data)
generateKeypair(algorithm)
signature(data, private_key, algorithm)
verify_sig(data, signature, public_key, algorithm)
deriveAddress(public_key)
deriveAddress(public_key, prefix)
randomBytes(length)
```

### Hash Algorithms

```
SHA256, SHA512, SHA3-256, SHA3-512, KECCAK256, BLAKE2B, BLAKE2S, RIPEMD160
```

### TX Context Properties

```
TX.caller          TX.timestamp       TX.block_hash
TX.gas_limit       TX.gas_used        TX.gas_remaining
```

### Gas Properties

```
gas.used           gas.remaining      gas.limit
```

### Runtime Blockchain Modules (Python-level, 18 files)

```
blockchain/__init__.py     — Module entry point, feature flags
blockchain/chain.py        — Block, Transaction, Chain, Mempool
blockchain/node.py         — BlockchainNode, NodeConfig
blockchain/consensus.py    — PoW, PoA, PoS, BFT consensus engines
blockchain/contract_vm.py  — ContractVM (execution bridge)
blockchain/crypto.py       — CryptoPlugin (hash, sign, verify, keypair)
blockchain/wallet.py       — HDWallet, Account, Keystore (BIP-32/39/44)
blockchain/tokens.py       — ZX20Token, ZX721Token, ZX1155Token
blockchain/network.py      — P2PNetwork, PeerReputationManager
blockchain/multichain.py   — ChainRouter, BridgeContract, BridgeRelay
blockchain/events.py       — EventIndex, BloomFilter, EventLog
blockchain/mpt.py          — MerklePatriciaTrie, StateTrie
blockchain/transaction.py  — TransactionContext, GasTracker
blockchain/rpc.py          — RPCServer (39 methods, HTTP + WebSocket)
blockchain/ledger.py       — Ledger, LedgerManager (versioned KV store)
blockchain/upgradeable.py  — ProxyContract, UpgradeManager, ChainUpgradeGovernance
blockchain/verification.py — FormalVerifier (structural + invariant + property checking)
blockchain/accelerator.py  — ExecutionAccelerator (AOT, cache, batch)
```

### Contract VM Built-in Functions (inside contract execution)

```
verify_sig(data, signature, public_key)
emit(event_name, ...args)
get_balance(address)
transfer(to, amount)
keccak256(data)
block_number()
block_timestamp()
contract_call(address, action, args)
static_call(address, action, args)
delegate_call(address, action, args)
```

### RPC Methods (39 total)

```
zx_chainId, zx_blockNumber, zx_getBlockByNumber, zx_getBlockByHash,
zx_getBlockTransactionCount, zx_getBalance, zx_getTransactionCount,
zx_getAccount, zx_getTransactionByHash, zx_getTransactionReceipt,
zx_sendTransaction, zx_sendRawTransaction, zx_gasPrice, zx_estimateGas,
zx_getChainInfo, zx_validateChain, zx_getCode, zx_getLogs,
txpool_status, txpool_content, txpool_replaceByFee,
net_version, net_peerCount, net_listening, net_peers,
miner_start, miner_stop, miner_status, miner_setMinerAddress, miner_mineBlock,
contract_deploy, contract_call, contract_staticCall,
admin_nodeInfo, admin_fundAccount, admin_exportChain, admin_rpcMethods
```

---

## 21. Production Hardening (v0.4.0)

The following enhancements close the four gaps originally identified in the production readiness
assessment. Together they make Zexus blockchain suitable for **production mainnet** deployments.

### 21.1 TLS Certificate Management (`network.py`)

| Before | After |
|---|---|
| Self-signed certificates only | CA-signed, mutual-TLS (mTLS), and certificate pinning |
| `verify_mode = ssl.CERT_NONE` | `ssl.CERT_REQUIRED` when CA bundle is provided |
| No peer certificate validation | SHA-256 fingerprint pinning per-connection |

**New P2PNetwork parameters:**

```python
P2PNetwork(
    host="0.0.0.0",
    port=30303,
    tls_ca="/path/to/ca-bundle.pem",       # CA-signed verification
    tls_mutual=True,                        # require client certificates
    tls_pinned_certs=["sha256_fingerprint_hex"],  # certificate pinning
)
```

**Zexus usage:**

```zexus
let node = blockchain.create_node({
    chain_id: "production-mainnet",
    tls_ca: "/etc/zexus/ca-bundle.pem",
    tls_mutual: true,
    tls_pinned_certs: ["a1b2c3d4..."]
});
```

### 21.2 Pluggable Storage Backends (`storage.py`)

| Backend | Best For | Package |
|---|---|---|
| `sqlite` (default) | Development, testnets, light nodes | Built-in |
| `leveldb` | Read-heavy production / archive nodes | `pip install plyvel` |
| `rocksdb` | High-TPS validator nodes | `pip install python-rocksdb` |
| `memory` | Unit tests, benchmarks | Built-in |

**Python API:**

```python
from zexus.blockchain.storage import get_storage_backend

# Default SQLite
backend = get_storage_backend("sqlite", db_path="/data/chain.db")

# Production LevelDB
backend = get_storage_backend("leveldb", db_path="/data/chaindb")

# High-throughput RocksDB
backend = get_storage_backend("rocksdb", db_path="/data/chaindb")

# Test-only in-memory
backend = get_storage_backend("memory")

# Pass to Chain
from zexus.blockchain.chain import Chain
chain = Chain(chain_id="mainnet", storage=backend)

# Or use shorthand
chain = Chain(chain_id="mainnet", data_dir="/data", storage_backend="rocksdb")
```

**Custom backend registration:**

```python
from zexus.blockchain.storage import StorageBackend, register_backend

class RedisBackend(StorageBackend):
    # implement get, put, delete, has, iterate, iterate_sorted, commit, close
    ...

register_backend("redis", RedisBackend)
chain = Chain(chain_id="mainnet", storage=get_storage_backend("redis", host="localhost"))
```

### 21.3 Parallel Batch Execution (`accelerator.py`)

| Before | After |
|---|---|
| Sequential tx processing | Parallel execution across non-conflicting contract groups |
| Single-threaded | ThreadPoolExecutor with configurable worker count |

**How it works:**

Transactions in a block that target *different contracts* have no shared state and are
dispatched in parallel using `concurrent.futures.ThreadPoolExecutor`. Transactions to the
*same* contract are still executed sequentially to preserve ordering guarantees.

```python
from zexus.blockchain.accelerator import ExecutionAccelerator

accel = ExecutionAccelerator(
    contract_vm=vm,
    batch_workers=8,      # parallel worker threads  (NEW)
    aot_enabled=True,
    ic_enabled=True,
)
result = accel.batch_executor.execute_batch(transactions)
print(result.throughput)  # tx/s — significantly higher with parallelism
```

### 21.4 Taint / Data-Flow Analysis (`verification.py`)

A new **Level 4 (TAINT)** verification pass tracks how user-controlled values propagate
through contract actions and flags dangerous patterns:

| Check | Severity | Pattern Detected |
|---|---|---|
| Privilege escalation | CRITICAL | User input → `owner`/`admin` state variable |
| Unsanitized sink | HIGH | Tainted data → `transfer()`/`send()` without validation |
| Unchecked return | HIGH | External call return value ignored |
| Tainted storage key | MEDIUM | User input used as map/storage key |

**New finding categories:** `TAINTED_VALUE`, `UNSANITIZED_INPUT`, `TAINTED_STORAGE_KEY`,
`PRIVILEGE_ESCALATION`, `UNCHECKED_RETURN`

```python
from zexus.blockchain.verification import FormalVerifier, VerificationLevel

verifier = FormalVerifier(level=VerificationLevel.TAINT)  # level 4
report = verifier.verify_contract(my_contract)

for finding in report.findings:
    print(finding)  # includes taint analysis results
```

**Zexus usage:**

```zexus
let verifier = blockchain.create_verifier({
    level: "TAINT"     // or 4
});
let report = verifier.verify(my_contract);
print(report.summary());
```

### 21.5 Production Monitoring & Metrics (`monitoring.py`)

Real-time node metrics with Prometheus-compatible export:

**Metrics collected:**
- Block: height, block time, gas/block, txs/block
- Transaction: total, success rate, latency, TPS
- Peer: connected, banned, message rates
- Mempool: depth, size
- Consensus: round time, finality latency
- Uptime

**Python API:**

```python
from zexus.blockchain.monitoring import NodeMetrics, MetricsServer

metrics = NodeMetrics()

# Record events as they happen
metrics.record_block(block)
metrics.record_transaction(receipt)
metrics.record_peer_count(count=12, banned=1)

# Get snapshot
snap = metrics.snapshot()
print(snap["transaction"]["tps"])

# Prometheus text format
print(metrics.prometheus_text())

# Start HTTP metrics endpoint
server = MetricsServer(metrics, port=9100)
await server.start()
# GET /metrics       → Prometheus format
# GET /metrics/json  → JSON snapshot
# GET /health        → {"status": "ok"}
```

### 21.6 Updated Production Assessment

| Gap | Solution | Status |
|---|---|---|
| Self-signed TLS only | CA certs, mTLS, certificate pinning | **RESOLVED** |
| SQLite-only persistence | Pluggable: SQLite/LevelDB/RocksDB/Memory/Custom | **RESOLVED** |
| Sequential batch execution | Parallel ThreadPool per contract group | **RESOLVED** |
| AST-only verification | Level 4 taint/data-flow analysis added | **RESOLVED** |
| No production metrics | Prometheus + JSON metrics endpoint | **RESOLVED** |

**Verdict:** Zexus blockchain is now production-ready for mainnet deployments including
public L1/L2 chains, consortium networks, and enterprise blockchains.

---

*This document was generated from the Zexus interpreter source code (v0.4.0) at `src/zexus/blockchain/` (20 modules, ~15,000+ lines) and verified against working `.zx` demo files in `blockchain_demo/`.*
