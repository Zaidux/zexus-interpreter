# Blockchain Module

The `blockchain` module provides utilities for blockchain development including address generation, Merkle trees, block creation, and transaction handling.

## Usage

```zexus
use {create_address, validate_address, calculate_merkle_root} from "blockchain"
# or
use {create_address, validate_address, calculate_merkle_root} from "stdlib/blockchain"
```

## Functions

### Address Management

#### `create_address(public_key: string, prefix: string = "0x") -> string`
Create blockchain address from public key (Ethereum-style).

```zexus
let address = create_address("my_public_key")
print(address)  # 0x...
```

#### `validate_address(address: string, prefix: string = "0x") -> bool`
Validate blockchain address format.

```zexus
let is_valid = validate_address("0x1234567890abcdef1234567890abcdef12345678")
print(is_valid)  # true or false
```

### Merkle Trees

#### `calculate_merkle_root(hashes: list) -> string`
Calculate Merkle root from list of hashes.

```zexus
let hashes = ["hash1", "hash2", "hash3", "hash4"]
let merkle_root = calculate_merkle_root(hashes)
print(merkle_root)
```

### Block Management

#### `create_block(index: int, timestamp: float, data: any, previous_hash: string, nonce: int = 0) -> map`
Create a blockchain block.

```zexus
let block = create_block(1, timestamp(), "Block data", "0x0", 0)
print(block.hash)
```

#### `hash_block(block: map) -> string`
Calculate hash of a block.

```zexus
let hash = hash_block(block)
```

#### `validate_block(block: map, previous_block: map = null) -> bool`
Validate a blockchain block.

```zexus
let is_valid = validate_block(block, previous_block)
```

#### `create_genesis_block() -> map`
Create the genesis block (first block in chain).

```zexus
let genesis = create_genesis_block()
print(genesis.index)  # 0
```

### Proof of Work

#### `proof_of_work(block_data: string, difficulty: int = 4) -> (int, string)`
Simple proof-of-work mining (find nonce).

```zexus
let (nonce, hash) = proof_of_work("block_data", 4)
print("Nonce: " + string(nonce))
print("Hash: " + hash)
```

#### `validate_proof_of_work(block_data: string, nonce: int, hash: string, difficulty: int = 4) -> bool`
Validate proof-of-work.

```zexus
let is_valid = validate_proof_of_work("block_data", nonce, hash, 4)
```

### Transaction Management

#### `create_transaction(sender: string, recipient: string, amount: float, timestamp: float = null) -> map`
Create a blockchain transaction.

```zexus
let tx = create_transaction("0xSender", "0xRecipient", 10.5)
print(tx.hash)
```

#### `hash_transaction(tx: map) -> string`
Calculate hash of a transaction.

```zexus
let tx_hash = hash_transaction(tx)
```

### Chain Validation

#### `validate_chain(chain: list) -> bool`
Validate entire blockchain.

```zexus
let chain = [genesis_block, block1, block2]
let is_valid = validate_chain(chain)
```

## Complete Example

```zexus
use {create_address, validate_address, create_genesis_block, 
     create_block, validate_block, calculate_merkle_root, 
     create_transaction, hash_transaction, validate_chain} from "blockchain"

# Create and validate address
let address = create_address("my_public_key_12345")
print("Address: " + address)
print("Valid: " + string(validate_address(address)))

# Create blockchain
let genesis = create_genesis_block()
print("\nGenesis Block:")
print("  Index: " + string(genesis.index))
print("  Hash: " + genesis.hash)

# Create next block
let block1 = create_block(1, timestamp(), "Transaction data", genesis.hash)
print("\nBlock 1:")
print("  Hash: " + block1.hash)
print("  Valid: " + string(validate_block(block1, genesis)))

# Create Merkle tree
let tx_hashes = [
    hash_transaction(create_transaction("0xA", "0xB", 10)),
    hash_transaction(create_transaction("0xB", "0xC", 20)),
    hash_transaction(create_transaction("0xC", "0xA", 5))
]
let merkle_root = calculate_merkle_root(tx_hashes)
print("\nMerkle Root: " + merkle_root)

# Validate chain
let chain = [genesis, block1]
print("\nChain Valid: " + string(validate_chain(chain)))
```

## Block Structure

Blocks created with `create_block()` have the following structure:

```zexus
{
    index: 0,              # Block number
    timestamp: 1234567890, # Unix timestamp
    data: "...",          # Block data
    previous_hash: "...", # Hash of previous block
    nonce: 0,             # Proof-of-work nonce
    hash: "..."           # Block hash
}
```

## Transaction Structure

Transactions created with `create_transaction()` have the following structure:

```zexus
{
    sender: "0x...",      # Sender address
    recipient: "0x...",   # Recipient address
    amount: 10.5,         # Transaction amount
    timestamp: 1234567890, # Unix timestamp
    hash: "..."           # Transaction hash
}
```

## Use Cases

### Creating a Simple Blockchain

```zexus
use {create_genesis_block, create_block, validate_chain} from "blockchain"

let chain = []
chain.push(create_genesis_block())

# Add blocks
for let i = 1; i < 5; i = i + 1 {
    let prev_block = chain[i - 1]
    let new_block = create_block(i, timestamp(), "Block " + string(i), prev_block.hash)
    chain.push(new_block)
}

# Validate
print("Chain valid: " + string(validate_chain(chain)))
```

### Merkle Tree Verification

```zexus
use {calculate_merkle_root} from "blockchain"

# Create transaction hashes
let transactions = [
    "tx1_hash",
    "tx2_hash",
    "tx3_hash",
    "tx4_hash"
]

# Calculate root
let merkle_root = calculate_merkle_root(transactions)
print("Merkle Root: " + merkle_root)

# Can verify individual transactions are in tree
```

### Proof of Work Mining

```zexus
use {proof_of_work, validate_proof_of_work} from "blockchain"

let block_data = "My block data"
let difficulty = 4  # Number of leading zeros

print("Mining...")
let (nonce, hash) = proof_of_work(block_data, difficulty)

print("Found nonce: " + string(nonce))
print("Hash: " + hash)

# Validate
let valid = validate_proof_of_work(block_data, nonce, hash, difficulty)
print("Valid: " + string(valid))
```

## Security Notes

- Always validate blocks before adding to chain
- Use appropriate difficulty for proof-of-work
- Verify Merkle roots when validating transactions
- Validate addresses before accepting transactions
- Store private keys securely (not in blockchain)

## Function Count

**Total: 12 functions**

- Address: 2 functions
- Merkle: 1 function
- Blocks: 4 functions
- Proof of Work: 2 functions
- Transactions: 2 functions
- Chain: 1 function
