"""Blockchain utilities module for Zexus standard library."""

from typing import Any, Dict, List, Optional
import hashlib
import time


# Proof of work maximum iterations constant
MAX_POW_ITERATIONS = 10_000_000


class BlockchainModule:
    """Provides blockchain utility functions."""

    @staticmethod
    def quantum_resistant_hash(data: str, algorithm: str = 'shake256', output_length: int = 64) -> str:
        """Calculate quantum-resistant hash using SHAKE256 or SHA3.
        
        Args:
            data: Data to hash
            algorithm: 'shake256', 'shake128', 'sha3_512', 'sha3_384', 'sha3_256'
            output_length: Output length in bytes for SHAKE (default: 64)
        
        Returns:
            Hexadecimal hash string
        
        Note: SHAKE256 and SHA3 are quantum-resistant hash functions from the SHA-3 family.
        """
        if algorithm == 'shake256':
            return hashlib.shake_256(data.encode()).hexdigest(output_length)
        elif algorithm == 'shake128':
            return hashlib.shake_128(data.encode()).hexdigest(output_length)
        elif algorithm == 'sha3_512':
            return hashlib.sha3_512(data.encode()).hexdigest()
        elif algorithm == 'sha3_384':
            return hashlib.sha3_384(data.encode()).hexdigest()
        elif algorithm == 'sha3_256':
            return hashlib.sha3_256(data.encode()).hexdigest()
        elif algorithm == 'sha3_224':
            return hashlib.sha3_224(data.encode()).hexdigest()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    @staticmethod
    def quantum_safe_address(public_key: str, prefix: str = "0x") -> str:
        """Create quantum-safe address from public key using SHAKE256.
        
        Uses SHAKE256 (SHA-3 family) which is quantum-resistant.
        """
        # Use SHAKE256 for quantum resistance
        shake_hash = hashlib.shake_256(public_key.encode()).hexdigest(20)  # 20 bytes = 40 hex chars
        return f"{prefix}{shake_hash}"

    @staticmethod
    def create_quantum_block(index: int, timestamp: float, data: Any, 
                            previous_hash: str, nonce: int = 0) -> Dict[str, Any]:
        """Create a quantum-resistant blockchain block using SHA3-256."""
        block = {
            'index': index,
            'timestamp': timestamp,
            'data': data,
            'previous_hash': previous_hash,
            'nonce': nonce
        }
        
        # Calculate block hash using SHA3-256 (quantum-resistant)
        block_string = str(block['index']) + str(block['timestamp']) + \
                      str(block['data']) + block['previous_hash'] + str(block['nonce'])
        block['hash'] = hashlib.sha3_256(block_string.encode()).hexdigest()
        
        return block

    @staticmethod
    def quantum_proof_of_work(block_data: str, difficulty: int = 4, 
                             max_iterations: int = MAX_POW_ITERATIONS) -> tuple:
        """Quantum-resistant proof-of-work using SHA3-256.
        
        Args:
            block_data: Data to hash
            difficulty: Number of leading zeros required
            max_iterations: Maximum iterations before giving up
        
        Returns:
            Tuple of (nonce, hash) or (nonce, "") if max iterations reached
        """
        nonce = 0
        prefix = '0' * difficulty
        
        while nonce < max_iterations:
            hash_attempt = hashlib.sha3_256(f"{block_data}{nonce}".encode()).hexdigest()
            
            if hash_attempt.startswith(prefix):
                return nonce, hash_attempt
            
            nonce += 1
        
        return nonce, ""

    @staticmethod
    def validate_quantum_proof_of_work(block_data: str, nonce: int, hash_value: str, 
                                      difficulty: int = 4) -> bool:
        """Validate quantum-resistant proof-of-work."""
        prefix = '0' * difficulty
        calculated_hash = hashlib.sha3_256(f"{block_data}{nonce}".encode()).hexdigest()
        return calculated_hash == hash_value and hash_value.startswith(prefix)

    @staticmethod
    def create_address(public_key: str, prefix: str = "0x") -> str:
        """Create address from public key (Ethereum-style).
        
        Note: Requires pycryptodome for true Keccak-256 hashing.
        Raises RuntimeError if pycryptodome is not available.
        """
        try:
            # Try to use Keccak-256 (Ethereum standard)
            from Crypto.Hash import keccak
            k = keccak.new(digest_bits=256)
            k.update(public_key.encode())
            hash_hex = k.hexdigest()
        except ImportError as exc:
            # Keccak-256 implementation not available; cannot create
            # Ethereum-compatible address without pycryptodome.
            raise RuntimeError(
                "Ethereum-compatible address generation requires the "
                "'pycryptodome' package providing Crypto.Hash.keccak. "
                "Install with: pip install pycryptodome"
            ) from exc
        
        address = hash_hex[-40:]  # Last 20 bytes (40 hex chars)
        return f"{prefix}{address}"

    @staticmethod
    def validate_address(address: str, prefix: str = "0x") -> bool:
        """Validate blockchain address format."""
        if not address.startswith(prefix):
            return False
        
        hex_part = address[len(prefix):]
        
        # Check if it's valid hex
        try:
            int(hex_part, 16)
        except ValueError:
            return False
        
        # Check length (40 hex chars for Ethereum-style)
        return len(hex_part) == 40

    @staticmethod
    def calculate_merkle_root(hashes: List[str]) -> str:
        """Calculate Merkle root from list of hashes."""
        if not hashes:
            return hashlib.sha256(b'').hexdigest()
        
        if len(hashes) == 1:
            return hashes[0]
        
        # Build merkle tree
        current_level = hashes[:]
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Combine two hashes
                    combined = current_level[i] + current_level[i + 1]
                    parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                else:
                    # Odd number, duplicate last hash
                    combined = current_level[i] + current_level[i]
                    parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                
                next_level.append(parent_hash)
            
            current_level = next_level
        
        return current_level[0]

    @staticmethod
    def create_block(index: int, timestamp: float, data: Any, 
                     previous_hash: str, nonce: int = 0) -> Dict[str, Any]:
        """Create a blockchain block."""
        block = {
            'index': index,
            'timestamp': timestamp,
            'data': data,
            'previous_hash': previous_hash,
            'nonce': nonce
        }
        
        # Calculate block hash
        block_string = str(block['index']) + str(block['timestamp']) + \
                      str(block['data']) + block['previous_hash'] + str(block['nonce'])
        block['hash'] = hashlib.sha256(block_string.encode()).hexdigest()
        
        return block

    @staticmethod
    def hash_block(block: Dict[str, Any]) -> str:
        """Calculate hash of a block."""
        block_string = str(block.get('index', 0)) + \
                      str(block.get('timestamp', 0)) + \
                      str(block.get('data', '')) + \
                      str(block.get('previous_hash', '')) + \
                      str(block.get('nonce', 0))
        return hashlib.sha256(block_string.encode()).hexdigest()

    @staticmethod
    def validate_block(block: Dict[str, Any], previous_block: Optional[Dict[str, Any]] = None) -> bool:
        """Validate a blockchain block."""
        # Check if block has required fields
        required_fields = ['index', 'timestamp', 'data', 'previous_hash', 'hash']
        if not all(field in block for field in required_fields):
            return False
        
        # Verify hash is correct
        calculated_hash = BlockchainModule.hash_block(block)
        if calculated_hash != block['hash']:
            return False
        
        # If previous block provided, check linkage
        if previous_block:
            if block['previous_hash'] != previous_block['hash']:
                return False
            if block['index'] != previous_block['index'] + 1:
                return False
        
        return True

    @staticmethod
    def proof_of_work(block_data: str, difficulty: int = 4, max_iterations: int = MAX_POW_ITERATIONS) -> tuple:
        """Simple proof-of-work mining (find nonce).
        
        Args:
            block_data: Data to hash
            difficulty: Number of leading zeros required
            max_iterations: Maximum iterations before giving up (default: 10,000,000)
        
        Returns:
            Tuple of (nonce, hash) or (nonce, "") if max iterations reached
        """
        nonce = 0
        prefix = '0' * difficulty
        
        while nonce < max_iterations:
            hash_attempt = hashlib.sha256(f"{block_data}{nonce}".encode()).hexdigest()
            
            if hash_attempt.startswith(prefix):
                return nonce, hash_attempt
            
            nonce += 1
        
        return nonce, ""

    @staticmethod
    def validate_proof_of_work(block_data: str, nonce: int, hash_value: str, difficulty: int = 4) -> bool:
        """Validate proof-of-work."""
        prefix = '0' * difficulty
        calculated_hash = hashlib.sha256(f"{block_data}{nonce}".encode()).hexdigest()
        return calculated_hash == hash_value and hash_value.startswith(prefix)

    @staticmethod
    def create_transaction(sender: str, recipient: str, amount: float, 
                          timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Create a transaction."""
        if timestamp is None:
            timestamp = time.time()
        
        tx = {
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
            'timestamp': timestamp
        }
        
        # Calculate transaction hash
        tx_string = f"{tx['sender']}{tx['recipient']}{tx['amount']}{tx['timestamp']}"
        tx['hash'] = hashlib.sha256(tx_string.encode()).hexdigest()
        
        return tx

    @staticmethod
    def hash_transaction(tx: Dict[str, Any]) -> str:
        """Calculate hash of a transaction."""
        tx_string = f"{tx.get('sender', '')}{tx.get('recipient', '')}" \
                   f"{tx.get('amount', 0)}{tx.get('timestamp', 0)}"
        return hashlib.sha256(tx_string.encode()).hexdigest()

    @staticmethod
    def create_genesis_block() -> Dict[str, Any]:
        """Create the genesis block (first block in chain)."""
        return BlockchainModule.create_block(
            index=0,
            timestamp=time.time(),
            data="Genesis Block",
            previous_hash="0" * 64,
            nonce=0
        )

    @staticmethod
    def validate_chain(chain: List[Dict[str, Any]]) -> bool:
        """Validate entire blockchain."""
        if not chain:
            return False
        
        # Validate genesis block
        if chain[0]['previous_hash'] != "0" * 64:
            return False
        
        # Validate each block
        for i in range(len(chain)):
            if i == 0:
                if not BlockchainModule.validate_block(chain[i]):
                    return False
            else:
                if not BlockchainModule.validate_block(chain[i], chain[i-1]):
                    return False
        
        return True


# Export functions for easy access
create_address = BlockchainModule.create_address
validate_address = BlockchainModule.validate_address
calculate_merkle_root = BlockchainModule.calculate_merkle_root
create_block = BlockchainModule.create_block
hash_block = BlockchainModule.hash_block
validate_block = BlockchainModule.validate_block
proof_of_work = BlockchainModule.proof_of_work
validate_proof_of_work = BlockchainModule.validate_proof_of_work
create_transaction = BlockchainModule.create_transaction
hash_transaction = BlockchainModule.hash_transaction
create_genesis_block = BlockchainModule.create_genesis_block
validate_chain = BlockchainModule.validate_chain
# Quantum-resistant functions
quantum_resistant_hash = BlockchainModule.quantum_resistant_hash
quantum_safe_address = BlockchainModule.quantum_safe_address
create_quantum_block = BlockchainModule.create_quantum_block
quantum_proof_of_work = BlockchainModule.quantum_proof_of_work
validate_quantum_proof_of_work = BlockchainModule.validate_quantum_proof_of_work
