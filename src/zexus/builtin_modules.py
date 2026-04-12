"""
Builtin Module System for Zexus

This module provides a registry of builtin modules (crypto, datetime, math)
that can be imported using `use "module_name" as alias` syntax.
"""

from .object import Map, String, Integer, Float, Boolean, Builtin, Environment, EvaluationError


def create_builtin_modules(evaluator):
    """
    Create and return a dictionary of builtin modules.
    Each module is an Environment with its functions registered.
    
    Args:
        evaluator: The evaluator instance (needed for accessing CryptoPlugin, etc.)
    
    Returns:
        Dict mapping module names to their Environment objects
    """
    modules = {}
    
    # ===== CRYPTO MODULE =====
    crypto_env = Environment()
    
    # Import CryptoPlugin
    try:
        from .blockchain.crypto import CryptoPlugin
        
        # keccak256(data)
        def _crypto_keccak256(*args):
            if len(args) != 1:
                return EvaluationError("keccak256() expects 1 argument: data")
            data = args[0].value if hasattr(args[0], 'value') else str(args[0])
            try:
                result = CryptoPlugin.keccak256(data)
                return String(result)
            except Exception as e:
                return EvaluationError(f"Keccak256 error: {str(e)}")
        
        # generate_keypair(algorithm?)
        def _crypto_generate_keypair(*args):
            algorithm = args[0].value if len(args) > 0 and hasattr(args[0], 'value') else 'ECDSA'
            try:
                private_key, public_key = CryptoPlugin.generate_keypair(algorithm)
                # Return as a Map with public_key, private_key, and address
                address = CryptoPlugin.derive_address(public_key)
                return Map({
                    String("private_key"): String(private_key),
                    String("public_key"): String(public_key),
                    String("address"): String(address)
                })
            except Exception as e:
                return EvaluationError(f"Keypair generation error: {str(e)}")
        
        # secp256k1_sign(data, private_key)
        def _crypto_secp256k1_sign(*args):
            if len(args) != 2:
                return EvaluationError("secp256k1_sign() expects 2 arguments: data, private_key")
            data = args[0].value if hasattr(args[0], 'value') else str(args[0])
            private_key = args[1].value if hasattr(args[1], 'value') else str(args[1])
            try:
                result = CryptoPlugin.sign_data(data, private_key, 'ECDSA')
                return String(result)
            except Exception as e:
                return EvaluationError(f"Signature error: {str(e)}")
        
        # verify_signature(data, signature, public_key)
        def _crypto_verify_signature(*args):
            if len(args) != 3:
                return EvaluationError("verify_signature() expects 3 arguments: data, signature, public_key")
            data = args[0].value if hasattr(args[0], 'value') else str(args[0])
            signature = args[1].value if hasattr(args[1], 'value') else str(args[1])
            public_key = args[2].value if hasattr(args[2], 'value') else str(args[2])
            try:
                result = CryptoPlugin.verify_signature(data, signature, public_key, 'ECDSA')
                return Boolean(result)
            except Exception as e:
                return EvaluationError(f"Verification error: {str(e)}")
        
        # calculate_merkle_root(hashes)
        def _crypto_calculate_merkle_root(*args):
            if len(args) != 1:
                return EvaluationError("calculate_merkle_root() expects 1 argument: list of hashes")
            
            from .object import List as ListObj
            if not isinstance(args[0], ListObj):
                return EvaluationError("calculate_merkle_root() expects a list")
            
            hashes = [h.value if hasattr(h, 'value') else str(h) for h in args[0].elements]
            
            if len(hashes) == 0:
                return String(CryptoPlugin.keccak256(""))
            
            # Simple merkle root calculation
            while len(hashes) > 1:
                new_level = []
                for i in range(0, len(hashes), 2):
                    if i + 1 < len(hashes):
                        combined = hashes[i] + hashes[i + 1]
                    else:
                        combined = hashes[i] + hashes[i]
                    new_level.append(CryptoPlugin.keccak256(combined))
                hashes = new_level
            
            return String(hashes[0])
        
        # sha256(data)
        def _crypto_sha256(*args):
            if len(args) != 1:
                return EvaluationError("sha256() expects 1 argument: data")
            data = args[0].value if hasattr(args[0], 'value') else str(args[0])
            try:
                result = CryptoPlugin.hash_data(data, 'SHA256')
                return String(result)
            except Exception as e:
                return EvaluationError(f"SHA256 error: {str(e)}")
        
        # aes_encrypt(data, key)
        def _crypto_aes_encrypt(*args):
            if len(args) != 2:
                return EvaluationError("aes_encrypt() expects 2 arguments: data, key")
            data = args[0].value if hasattr(args[0], 'value') else str(args[0])
            key = args[1].value if hasattr(args[1], 'value') else str(args[1])
            try:
                import hashlib, base64, os
                # Derive 256-bit key from user key via SHA-256
                aes_key = hashlib.sha256(key.encode('utf-8')).digest()
                # Use AES-256-GCM for authenticated encryption
                from Crypto.Cipher import AES as _AES
                nonce = os.urandom(12)
                cipher = _AES.new(aes_key, _AES.MODE_GCM, nonce=nonce)
                ciphertext, tag = cipher.encrypt_and_digest(data.encode('utf-8'))
                # Pack: nonce (12) + tag (16) + ciphertext, base64-encoded
                packed = base64.b64encode(nonce + tag + ciphertext).decode('ascii')
                return String(packed)
            except ImportError:
                # Fallback: if pycryptodome not available, use Fernet from cryptography
                try:
                    import hashlib, base64
                    from cryptography.fernet import Fernet as _Fernet
                    fernet_key = base64.urlsafe_b64encode(hashlib.sha256(key.encode('utf-8')).digest())
                    f = _Fernet(fernet_key)
                    encrypted = f.encrypt(data.encode('utf-8'))
                    return String(encrypted.decode('ascii'))
                except ImportError:
                    return EvaluationError("aes_encrypt() requires pycryptodome or cryptography package")
            except Exception as e:
                return EvaluationError(f"aes_encrypt() failed: {str(e)}")
        
        # aes_decrypt(encrypted_data, key)
        def _crypto_aes_decrypt(*args):
            if len(args) != 2:
                return EvaluationError("aes_decrypt() expects 2 arguments: encrypted_data, key")
            encrypted = args[0].value if hasattr(args[0], 'value') else str(args[0])
            key = args[1].value if hasattr(args[1], 'value') else str(args[1])
            try:
                import hashlib, base64
                aes_key = hashlib.sha256(key.encode('utf-8')).digest()
                raw = base64.b64decode(encrypted)
                if len(raw) < 28:  # 12 nonce + 16 tag minimum
                    return EvaluationError("aes_decrypt() invalid ciphertext (too short)")
                nonce = raw[:12]
                tag = raw[12:28]
                ciphertext = raw[28:]
                from Crypto.Cipher import AES as _AES
                cipher = _AES.new(aes_key, _AES.MODE_GCM, nonce=nonce)
                plaintext = cipher.decrypt_and_verify(ciphertext, tag)
                return String(plaintext.decode('utf-8'))
            except ImportError:
                try:
                    import hashlib, base64
                    from cryptography.fernet import Fernet as _Fernet
                    fernet_key = base64.urlsafe_b64encode(hashlib.sha256(key.encode('utf-8')).digest())
                    f = _Fernet(fernet_key)
                    decrypted = f.decrypt(encrypted.encode('ascii'))
                    return String(decrypted.decode('utf-8'))
                except ImportError:
                    return EvaluationError("aes_decrypt() requires pycryptodome or cryptography package")
            except Exception as e:
                return EvaluationError(f"aes_decrypt() failed: {str(e)}")
        
        # Register all crypto functions
        crypto_env.set("keccak256", Builtin(_crypto_keccak256, "keccak256"))
        crypto_env.set("generate_keypair", Builtin(_crypto_generate_keypair, "generate_keypair"))
        crypto_env.set("secp256k1_sign", Builtin(_crypto_secp256k1_sign, "secp256k1_sign"))
        crypto_env.set("verify_signature", Builtin(_crypto_verify_signature, "verify_signature"))
        crypto_env.set("calculate_merkle_root", Builtin(_crypto_calculate_merkle_root, "calculate_merkle_root"))
        crypto_env.set("sha256", Builtin(_crypto_sha256, "sha256"))
        crypto_env.set("aes_encrypt", Builtin(_crypto_aes_encrypt, "aes_encrypt"))
        crypto_env.set("aes_decrypt", Builtin(_crypto_aes_decrypt, "aes_decrypt"))
        
    except ImportError as e:
        # Crypto module not available
        pass
    
    modules["crypto"] = crypto_env
    
    # ===== DATETIME MODULE =====
    datetime_env = Environment()
    
    import time
    import datetime as dt
    
    # now() - returns datetime object-like Map
    def _datetime_now(*args):
        now = dt.datetime.now()
        
        # timestamp() method
        def _timestamp(*a):
            return Integer(int(now.timestamp()))
        
        return Map({
            String("year"): Integer(now.year),
            String("month"): Integer(now.month),
            String("day"): Integer(now.day),
            String("hour"): Integer(now.hour),
            String("minute"): Integer(now.minute),
            String("second"): Integer(now.second),
            String("timestamp"): Builtin(_timestamp, "timestamp")
        })
    
    # timestamp() - returns current unix timestamp
    def _datetime_timestamp(*args):
        return Integer(int(time.time()))
    
    datetime_env.set("now", Builtin(_datetime_now, "now"))
    datetime_env.set("timestamp", Builtin(_datetime_timestamp, "timestamp"))
    
    modules["datetime"] = datetime_env
    
    # ===== MATH MODULE =====
    math_env = Environment()
    
    import math
    import random
    
    # random_int(min, max)
    def _math_random_int(*args):
        if len(args) != 2:
            return EvaluationError("random_int() expects 2 arguments: min, max")
        min_val = args[0].value if hasattr(args[0], 'value') else int(args[0])
        max_val = args[1].value if hasattr(args[1], 'value') else int(args[1])
        return Integer(random.randint(min_val, max_val))
    
    # random() - returns float between 0 and 1
    def _math_random(*args):
        return Float(random.random())
    
    # min(a, b)
    def _math_min(*args):
        if len(args) != 2:
            return EvaluationError("min() expects 2 arguments")
        a = args[0].value if hasattr(args[0], 'value') else args[0]
        b = args[1].value if hasattr(args[1], 'value') else args[1]
        result = min(a, b)
        return Float(result) if isinstance(result, float) else Integer(result)
    
    # max(a, b)
    def _math_max(*args):
        if len(args) != 2:
            return EvaluationError("max() expects 2 arguments")
        a = args[0].value if hasattr(args[0], 'value') else args[0]
        b = args[1].value if hasattr(args[1], 'value') else args[1]
        result = max(a, b)
        return Float(result) if isinstance(result, float) else Integer(result)
    
    # sqrt(n)
    def _math_sqrt(*args):
        if len(args) != 1:
            return EvaluationError("sqrt() expects 1 argument")
        n = args[0].value if hasattr(args[0], 'value') else args[0]
        return Float(math.sqrt(n))
    
    # abs(n)
    def _math_abs(*args):
        if len(args) != 1:
            return EvaluationError("abs() expects 1 argument")
        n = args[0].value if hasattr(args[0], 'value') else args[0]
        result = abs(n)
        return Float(result) if isinstance(result, float) else Integer(result)
    
    math_env.set("random_int", Builtin(_math_random_int, "random_int"))
    math_env.set("random", Builtin(_math_random, "random"))
    math_env.set("min", Builtin(_math_min, "min"))
    math_env.set("max", Builtin(_math_max, "max"))
    math_env.set("sqrt", Builtin(_math_sqrt, "sqrt"))
    math_env.set("abs", Builtin(_math_abs, "abs"))
    
    modules["math"] = math_env
    
    # ===== VALIDATION MODULE =====
    # Functions previously available as global builtins are now in this module.
    # Use:  use "validation"
    #       validation.is_email("user@example.com")
    validation_env = Environment()
    
    import re as _re
    
    def _val_is_email(*args):
        if len(args) != 1:
            return EvaluationError("is_email() takes 1 argument")
        val = args[0]
        email_str = val.value if isinstance(val, String) else str(val)
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return Boolean(bool(_re.match(pattern, email_str)))
    
    def _val_is_url(*args):
        if len(args) != 1:
            return EvaluationError("is_url() takes 1 argument")
        val = args[0]
        url_str = val.value if isinstance(val, String) else str(val)
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return Boolean(bool(_re.match(pattern, url_str)))
    
    def _val_is_phone(*args):
        if len(args) != 1:
            return EvaluationError("is_phone() takes 1 argument")
        val = args[0]
        phone_str = val.value if isinstance(val, String) else str(val)
        clean = _re.sub(r'[\s\-\(\)\.]', '', phone_str)
        return Boolean(clean.isdigit() and 10 <= len(clean) <= 15)
    
    def _val_password_strength(*args):
        if len(args) != 1:
            return EvaluationError("password_strength() takes 1 argument")
        val = args[0]
        password = val.value if isinstance(val, String) else str(val)
        score = 0
        if len(password) >= 8:
            score += 1
        if len(password) >= 12:
            score += 1
        if _re.search(r'[a-z]', password):
            score += 1
        if _re.search(r'[A-Z]', password):
            score += 1
        if _re.search(r'[0-9]', password):
            score += 1
        if _re.search(r'[^a-zA-Z0-9]', password):
            score += 1
        if score <= 2:
            return String("weak")
        elif score <= 4:
            return String("medium")
        else:
            return String("strong")
    
    validation_env.set("is_email", Builtin(_val_is_email, "is_email"))
    validation_env.set("is_url", Builtin(_val_is_url, "is_url"))
    validation_env.set("is_phone", Builtin(_val_is_phone, "is_phone"))
    validation_env.set("password_strength", Builtin(_val_password_strength, "password_strength"))
    
    modules["validation"] = validation_env
    
    # ===== PASSWORD MODULE =====
    # hash_password / verify_password moved here from global builtins.
    # Use:  use "password"
    #       let h = password.hash("secret")
    #       let ok = password.verify("secret", h)
    password_env = Environment()
    
    def _pw_hash(*args):
        if len(args) != 1:
            return EvaluationError("password.hash() takes exactly 1 argument")
        pw = args[0].value if isinstance(args[0], String) else str(args[0])
        try:
            import bcrypt
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(pw.encode('utf-8'), salt)
            return String(hashed.decode('utf-8'), is_trusted=True)
        except ImportError:
            return EvaluationError("password.hash() requires bcrypt library. Install: pip install bcrypt")
        except Exception as e:
            return EvaluationError(f"password.hash() error: {str(e)}")
    
    def _pw_verify(*args):
        if len(args) != 2:
            return EvaluationError("password.verify() takes exactly 2 arguments: password, hash")
        pw = args[0].value if isinstance(args[0], String) else str(args[0])
        pw_hash = args[1].value if isinstance(args[1], String) else str(args[1])
        try:
            import bcrypt
            result = bcrypt.checkpw(pw.encode('utf-8'), pw_hash.encode('utf-8'))
            return Boolean(result)
        except ImportError:
            return EvaluationError("password.verify() requires bcrypt library. Install: pip install bcrypt")
        except Exception as e:
            return EvaluationError(f"password.verify() error: {str(e)}")
    
    password_env.set("hash", Builtin(_pw_hash, "hash"))
    password_env.set("verify", Builtin(_pw_verify, "verify"))
    
    modules["password"] = password_env
    
    return modules


# Global registry of builtin modules
_BUILTIN_MODULES = {}

def get_builtin_module(module_name, evaluator=None):
    """
    Get a builtin module by name.
    
    Args:
        module_name: Name of the module ('crypto', 'datetime', 'math')
        evaluator: Optional evaluator instance for context
    
    Returns:
        Environment object with module functions, or None if not found
    """
    global _BUILTIN_MODULES
    
    # Initialize on first access
    if not _BUILTIN_MODULES and evaluator:
        _BUILTIN_MODULES = create_builtin_modules(evaluator)
    
    return _BUILTIN_MODULES.get(module_name)


def is_builtin_module(module_name):
    """Check if a module name refers to a builtin module"""
    return module_name in ["crypto", "datetime", "math", "validation", "password", "cache"]
