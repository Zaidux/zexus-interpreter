"""
Builtin Module System for Zexus

This module provides a registry of builtin modules that can be imported
using `use "module_name" as alias` syntax.

Modules:
  Core:       crypto, datetime, math
  Validation: validation, password
  Stdlib:     cache, queue, template, testing
  Security:   fuzz, secrets, netsec, payloads, pentest, audit, contract_audit
"""

import json

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
    
    # ===== CACHE MODULE =====
    cache_env = Environment()
    
    from .stdlib.cache import CacheModule
    
    def _cache_create(*args):
        capacity = args[0].value if len(args) > 0 and hasattr(args[0], 'value') else 128
        return Map({String("__cache__"): String(str(id(CacheModule.create(int(capacity)))))})
    
    def _cache_create_ttl(*args):
        capacity = args[0].value if len(args) > 0 and hasattr(args[0], 'value') else 128
        ttl = args[1].value if len(args) > 1 and hasattr(args[1], 'value') else 300
        return Map({String("__cache__"): String(str(id(CacheModule.create_ttl(int(capacity), float(ttl)))))})
    
    cache_env.set("create", Builtin(_cache_create, "create"))
    cache_env.set("create_ttl", Builtin(_cache_create_ttl, "create_ttl"))
    modules["cache"] = cache_env
    
    # ===== QUEUE MODULE =====
    queue_env = Environment()
    
    from .stdlib.queue_module import QueueModule
    
    def _queue_create(*args):
        maxsize = args[0].value if len(args) > 0 and hasattr(args[0], 'value') else 0
        q = QueueModule.create(int(maxsize))
        return Map({String("__type__"): String("queue"), String("__id__"): String(str(id(q)))})
    
    def _queue_create_priority(*args):
        maxsize = args[0].value if len(args) > 0 and hasattr(args[0], 'value') else 0
        q = QueueModule.create_priority(int(maxsize))
        return Map({String("__type__"): String("priority_queue"), String("__id__"): String(str(id(q)))})
    
    def _queue_create_topic(*args):
        t = QueueModule.create_topic()
        return Map({String("__type__"): String("topic"), String("__id__"): String(str(id(t)))})
    
    queue_env.set("create", Builtin(_queue_create, "create"))
    queue_env.set("create_priority", Builtin(_queue_create_priority, "create_priority"))
    queue_env.set("create_topic", Builtin(_queue_create_topic, "create_topic"))
    modules["queue"] = queue_env
    
    # ===== TEMPLATE MODULE =====
    template_env = Environment()
    
    from .stdlib.template import TemplateModule
    
    def _template_render(*args):
        if len(args) < 2:
            return EvaluationError("template.render() expects 2 arguments: template, context")
        tpl = args[0].value if isinstance(args[0], String) else str(args[0])
        ctx = {}
        if isinstance(args[1], Map):
            for k, v in args[1].pairs.items():
                key = k.value if isinstance(k, String) else str(k)
                val = v.value if hasattr(v, 'value') else str(v)
                ctx[key] = val
        result = TemplateModule.render(tpl, ctx)
        return String(result)
    
    def _template_render_safe(*args):
        if len(args) < 2:
            return EvaluationError("template.render_safe() expects 2 arguments: template, context")
        tpl = args[0].value if isinstance(args[0], String) else str(args[0])
        ctx = {}
        if isinstance(args[1], Map):
            for k, v in args[1].pairs.items():
                key = k.value if isinstance(k, String) else str(k)
                val = v.value if hasattr(v, 'value') else str(v)
                ctx[key] = val
        result = TemplateModule.render_safe(tpl, ctx)
        return String(result)
    
    template_env.set("render", Builtin(_template_render, "render"))
    template_env.set("render_safe", Builtin(_template_render_safe, "render_safe"))
    modules["template"] = template_env
    
    # ===== TESTING MODULE =====
    testing_env = Environment()
    
    from .stdlib.testing import TestingModule
    
    def _testing_assert_eq(*args):
        if len(args) < 2:
            return EvaluationError("testing.assert_eq() expects 2+ arguments")
        actual = args[0].value if hasattr(args[0], 'value') else args[0]
        expected = args[1].value if hasattr(args[1], 'value') else args[1]
        msg = args[2].value if len(args) > 2 and hasattr(args[2], 'value') else ""
        try:
            TestingModule.assert_eq(actual, expected, msg)
            return Boolean(True)
        except Exception as e:
            return EvaluationError(str(e))
    
    def _testing_assert_true(*args):
        if len(args) < 1:
            return EvaluationError("testing.assert_true() expects 1+ arguments")
        val = args[0].value if hasattr(args[0], 'value') else args[0]
        msg = args[1].value if len(args) > 1 and hasattr(args[1], 'value') else ""
        try:
            TestingModule.assert_true(val, msg)
            return Boolean(True)
        except Exception as e:
            return EvaluationError(str(e))
    
    def _testing_create_suite(*args):
        name = args[0].value if len(args) > 0 and hasattr(args[0], 'value') else "default"
        suite = TestingModule.create_suite(name)
        return Map({String("__type__"): String("test_suite"), String("name"): String(name)})
    
    testing_env.set("assert_eq", Builtin(_testing_assert_eq, "assert_eq"))
    testing_env.set("assert_true", Builtin(_testing_assert_true, "assert_true"))
    testing_env.set("create_suite", Builtin(_testing_create_suite, "create_suite"))
    modules["testing"] = testing_env
    
    # ===== FUZZ MODULE =====
    fuzz_env = Environment()
    
    from .stdlib.fuzz import FuzzModule
    
    def _fuzz_string(*args):
        min_l = args[0].value if len(args) > 0 and hasattr(args[0], 'value') else 0
        max_l = args[1].value if len(args) > 1 and hasattr(args[1], 'value') else 100
        return String(FuzzModule.fuzz_string(int(min_l), int(max_l)))
    
    def _fuzz_int(*args):
        min_v = args[0].value if len(args) > 0 and hasattr(args[0], 'value') else -(2**31)
        max_v = args[1].value if len(args) > 1 and hasattr(args[1], 'value') else 2**31
        return Integer(FuzzModule.fuzz_int(int(min_v), int(max_v)))
    
    def _fuzz_mutate(*args):
        if len(args) < 1:
            return EvaluationError("fuzz.mutate() expects 1 argument")
        data = args[0].value if isinstance(args[0], String) else str(args[0])
        return String(FuzzModule.mutate(data))
    
    fuzz_env.set("string", Builtin(_fuzz_string, "string"))
    fuzz_env.set("int", Builtin(_fuzz_int, "int"))
    fuzz_env.set("mutate", Builtin(_fuzz_mutate, "mutate"))
    modules["fuzz"] = fuzz_env
    
    # ===== SECRETS MODULE =====
    secrets_env = Environment()
    
    from .stdlib.secrets_module import SecretsModule
    
    def _secrets_generate_token(*args):
        length = args[0].value if len(args) > 0 and hasattr(args[0], 'value') else 32
        return String(SecretsModule.generate_token(int(length)))
    
    def _secrets_from_env(*args):
        if len(args) < 1:
            return EvaluationError("secrets.from_env() expects 1 argument: env var name")
        name = args[0].value if isinstance(args[0], String) else str(args[0])
        required = args[1].value if len(args) > 1 and hasattr(args[1], 'value') else True
        try:
            result = SecretsModule.from_env(name, required)
            return String(result) if result else EvaluationError(f"Required env var '{name}' not set")
        except Exception as e:
            return EvaluationError(str(e))
    
    secrets_env.set("generate_token", Builtin(_secrets_generate_token, "generate_token"))
    secrets_env.set("from_env", Builtin(_secrets_from_env, "from_env"))
    modules["secrets"] = secrets_env
    
    # ===== NETSEC MODULE =====
    netsec_env = Environment()
    
    from .stdlib.netsec import NetsecModule
    
    def _netsec_security_headers(*args):
        if len(args) < 1:
            return EvaluationError("netsec.security_headers() expects 1 argument: url")
        url = args[0].value if isinstance(args[0], String) else str(args[0])
        try:
            result = NetsecModule.security_headers(url)
            # Convert to Map
            return Map({String(k): String(str(v)) for k, v in result.items()})
        except Exception as e:
            return EvaluationError(f"netsec.security_headers() error: {str(e)}")
    
    def _netsec_dns_lookup(*args):
        if len(args) < 1:
            return EvaluationError("netsec.dns_lookup() expects 1-2 arguments: domain, record_type?")
        domain = args[0].value if isinstance(args[0], String) else str(args[0])
        rtype = args[1].value if len(args) > 1 and isinstance(args[1], String) else "A"
        try:
            from .object import List as ListObj
            results = NetsecModule.dns_lookup(domain, rtype)
            return ListObj([String(str(r)) for r in results])
        except Exception as e:
            return EvaluationError(f"netsec.dns_lookup() error: {str(e)}")
    
    def _netsec_tls_check(*args):
        if len(args) < 1:
            return EvaluationError("netsec.tls_check() expects 1-2 arguments: host, port?")
        host = args[0].value if isinstance(args[0], String) else str(args[0])
        port = args[1].value if len(args) > 1 else 443
        try:
            result = NetsecModule.tls_check(host, int(port))
            return Map({String(k): String(str(v)) for k, v in result.items()})
        except Exception as e:
            return EvaluationError(f"netsec.tls_check() error: {str(e)}")

    def _netsec_ssl_cert_info(*args):
        if len(args) < 1:
            return EvaluationError("netsec.ssl_cert_info() expects 1-2 arguments: host, port?")
        host = args[0].value if isinstance(args[0], String) else str(args[0])
        port = args[1].value if len(args) > 1 else 443
        try:
            result = NetsecModule.ssl_cert_info(host, int(port))
            # Boolean 'error' False means the cert lookup errored but the
            # dict carries the reason; convert so .error is truthy in Zexus.
            if result.get("error"):
                return Map({String(k): String(str(v)) for k, v in result.items()})
            # days_remaining can be None pre-expiry parse — keep as string
            return Map({String(k): String(str(v)) for k, v in result.items()})
        except Exception as e:
            return EvaluationError(f"netsec.ssl_cert_info() error: {str(e)}")

    def _netsec_http_methods(*args):
        if len(args) < 1:
            return EvaluationError("netsec.http_methods() expects 1 argument: url")
        url = args[0].value if isinstance(args[0], String) else str(args[0])
        try:
            from .object import List as ListObj
            # NetsecModule.http_methods returns a flat LIST of allowed
            # method names (or ["Error: ..."] entries on failure).
            result = NetsecModule.http_methods(url)
            return ListObj([String(str(m)) for m in result])
        except Exception as e:
            return EvaluationError(f"netsec.http_methods() error: {str(e)}")

    def _netsec_banner_grab(*args):
        if len(args) < 2:
            return EvaluationError("netsec.banner_grab() expects 2-3 arguments: host, port, timeout?")
        host = args[0].value if isinstance(args[0], String) else str(args[0])
        port = args[1].value if len(args) > 1 else 80
        timeout = args[2].value if len(args) > 2 else 3.0
        try:
            result = NetsecModule.banner_grab(host, int(port), float(timeout))
            return Map({String(k): String(str(v)) for k, v in result.items()})
        except Exception as e:
            return EvaluationError(f"netsec.banner_grab() error: {str(e)}")

    def _netsec_port_scan(*args):
        if len(args) < 1:
            return EvaluationError("netsec.port_scan() expects 1-3 arguments: host, ports?, timeout?")
        host = args[0].value if isinstance(args[0], String) else str(args[0])
        try:
            result = NetsecModule.port_scan(host)
            from .object import List as ListObj
            return Map({
                String("open"): ListObj([Integer(int(p)) for p in result.get("open", [])]),
                String("total"): Integer(int(result.get("total", 0))),
            })
        except Exception as e:
            return EvaluationError(f"netsec.port_scan() error: {str(e)}")

    def _netsec_check_open_redirect(*args):
        if len(args) < 1:
            return EvaluationError("netsec.check_open_redirect() expects 1 argument: url")
        url = args[0].value if isinstance(args[0], String) else str(args[0])
        try:
            result = NetsecModule.check_open_redirect(url)
            return Map({String(k): String(str(v)) for k, v in result.items()})
        except Exception as e:
            return EvaluationError(f"netsec.check_open_redirect() error: {str(e)}")

    netsec_env.set("security_headers", Builtin(_netsec_security_headers, "security_headers"))
    netsec_env.set("dns_lookup", Builtin(_netsec_dns_lookup, "dns_lookup"))
    netsec_env.set("tls_check", Builtin(_netsec_tls_check, "tls_check"))
    netsec_env.set("ssl_cert_info", Builtin(_netsec_ssl_cert_info, "ssl_cert_info"))
    netsec_env.set("http_methods", Builtin(_netsec_http_methods, "http_methods"))
    netsec_env.set("banner_grab", Builtin(_netsec_banner_grab, "banner_grab"))
    netsec_env.set("port_scan", Builtin(_netsec_port_scan, "port_scan"))
    netsec_env.set("check_open_redirect", Builtin(_netsec_check_open_redirect, "check_open_redirect"))
    modules["netsec"] = netsec_env
    
    # ===== PAYLOADS MODULE =====
    payloads_env = Environment()
    
    from .stdlib.payloads import PayloadsModule
    
    def _payloads_xss(*args):
        from .object import List as ListObj
        variant = args[0].value if len(args) > 0 and isinstance(args[0], String) else "all"
        return ListObj([String(p) for p in PayloadsModule.xss(variant)])
    
    def _payloads_sqli(*args):
        from .object import List as ListObj
        variant = args[0].value if len(args) > 0 and isinstance(args[0], String) else "all"
        return ListObj([String(p) for p in PayloadsModule.sqli(variant)])
    
    def _payloads_ssrf(*args):
        from .object import List as ListObj
        variant = args[0].value if len(args) > 0 and isinstance(args[0], String) else "all"
        return ListObj([String(p) for p in PayloadsModule.ssrf(variant)])
    
    def _payloads_path_traversal(*args):
        from .object import List as ListObj
        variant = args[0].value if len(args) > 0 and isinstance(args[0], String) else "all"
        return ListObj([String(p) for p in PayloadsModule.path_traversal(variant)])
    
    def _payloads_command_injection(*args):
        from .object import List as ListObj
        variant = args[0].value if len(args) > 0 and isinstance(args[0], String) else "all"
        return ListObj([String(p) for p in PayloadsModule.command_injection(variant)])
    
    def _payloads_encode(*args):
        if len(args) < 1:
            return EvaluationError("payloads.encode() expects 1-2 arguments")
        payload = args[0].value if isinstance(args[0], String) else str(args[0])
        encoding = args[1].value if len(args) > 1 and isinstance(args[1], String) else "url"
        return String(PayloadsModule.encode_payload(payload, encoding))
    
    payloads_env.set("xss", Builtin(_payloads_xss, "xss"))
    payloads_env.set("sqli", Builtin(_payloads_sqli, "sqli"))
    payloads_env.set("ssrf", Builtin(_payloads_ssrf, "ssrf"))
    payloads_env.set("path_traversal", Builtin(_payloads_path_traversal, "path_traversal"))
    payloads_env.set("command_injection", Builtin(_payloads_command_injection, "command_injection"))
    payloads_env.set("encode", Builtin(_payloads_encode, "encode"))
    modules["payloads"] = payloads_env
    
    # ===== PENTEST MODULE =====
    pentest_env = Environment()
    
    from .stdlib.pentest import PentestModule
    
    def _pentest_create_report(*args):
        if len(args) < 2:
            return EvaluationError("pentest.create_report() expects 2 arguments: title, target")
        title = args[0].value if isinstance(args[0], String) else str(args[0])
        target = args[1].value if isinstance(args[1], String) else str(args[1])
        report = PentestModule.create_report(title, target)
        wrapper = Map({String(k): String(str(v)) for k, v in report.items()})
        # Keep the LIVE native dict on the wrapper: add_finding and
        # severity_stats mutate/read it directly — round-tripping the
        # nested findings list through Map pairs loses the structure.
        wrapper._native_report = report
        return wrapper
    
    def _pentest_fingerprint_web(*args):
        if len(args) < 1:
            return EvaluationError("pentest.fingerprint_web() expects 1 argument: url")
        url = args[0].value if isinstance(args[0], String) else str(args[0])
        try:
            result = PentestModule.fingerprint_web(url)
            return Map({String(k): String(str(v)) for k, v in result.items()})
        except Exception as e:
            return EvaluationError(f"pentest.fingerprint_web() error: {str(e)}")
    
    def _pentest_test_headers(*args):
        if len(args) < 1:
            return EvaluationError("pentest.test_headers() expects 1 argument: url")
        url = args[0].value if isinstance(args[0], String) else str(args[0])
        try:
            result = PentestModule.test_headers(url)
            return Map({String(k): String(str(v)) for k, v in result.items()})
        except Exception as e:
            return EvaluationError(f"pentest.test_headers() error: {str(e)}")
    
    def _pentest_discover_subdomains(*args):
        if len(args) < 1:
            return EvaluationError("pentest.discover_subdomains() expects 1-2 arguments: domain, prefixes?")
        domain = args[0].value if isinstance(args[0], String) else str(args[0])
        prefixes = None
        if len(args) > 1:
            from .object import List as ListObj
            list_arg = args[1]
            elements = getattr(list_arg, "elements", None)
            if elements is not None:
                prefixes = [str(getattr(e, "value", e)) for e in elements]
        try:
            from .object import List as ListObj
            result = PentestModule.discover_subdomains(domain, prefixes)
            return ListObj([
                Map({String(k): String(str(v)) for k, v in entry.items()})
                for entry in result
            ])
        except Exception as e:
            return EvaluationError(f"pentest.discover_subdomains() error: {str(e)}")

    def _pentest_severity_stats(*args):
        if len(args) < 1:
            return EvaluationError("pentest.severity_stats() expects 1 argument: report")
        from .object import Map as MapObj
        report_arg = args[0]
        native = getattr(report_arg, "_native_report", None)
        if isinstance(native, dict):
            py_report = native
        else:
            py_report = {}
            for k, v in (getattr(report_arg, "pairs", None) or {}).items():
                py_report[str(k)] = v
        try:
            result = PentestModule.severity_stats(py_report)
            return Map({String(k): String(str(v)) for k, v in result.items()})
        except Exception as e:
            return EvaluationError(f"pentest.severity_stats() error: {str(e)}")

    def _pentest_add_finding(*args):
        # Signature: add_finding(report, severity, title, description, evidence?)
        if len(args) < 4:
            return EvaluationError("pentest.add_finding() expects 4-5 arguments: report, severity, title, description")
        from .object import Map as MapObj
        report_arg = args[0]
        native = getattr(report_arg, "_native_report", None)
        if isinstance(native, dict):
            py_report = native
        else:
            py_report = {}
            for k, v in (getattr(report_arg, "pairs", None) or {}).items():
                py_report[str(k)] = v
        severity = args[1].value if hasattr(args[1], "value") else str(args[1])
        title = args[2].value if hasattr(args[2], "value") else str(args[2])
        desc = args[3].value if hasattr(args[3], "value") else str(args[3])
        evidence = args[4].value if len(args) > 4 and hasattr(args[4], "value") else None
        try:
            result = PentestModule.add_finding(py_report, severity, title, desc, evidence)
            return Map({String(k): (v if hasattr(v, "value") else String(str(v))) for k, v in result.items()})
        except Exception as e:
            return EvaluationError(f"pentest.add_finding() error: {str(e)}")

    pentest_env.set("create_report", Builtin(_pentest_create_report, "create_report"))
    pentest_env.set("fingerprint_web", Builtin(_pentest_fingerprint_web, "fingerprint_web"))
    pentest_env.set("test_headers", Builtin(_pentest_test_headers, "test_headers"))
    pentest_env.set("discover_subdomains", Builtin(_pentest_discover_subdomains, "discover_subdomains"))
    pentest_env.set("severity_stats", Builtin(_pentest_severity_stats, "severity_stats"))
    pentest_env.set("add_finding", Builtin(_pentest_add_finding, "add_finding"))
    modules["pentest"] = pentest_env
    
    # ===== AUDIT MODULE =====
    audit_env = Environment()
    
    from .stdlib.audit import AuditModule
    
    def _audit_scan(*args):
        if len(args) < 1:
            return EvaluationError("audit.scan() expects 1 argument: source_code")
        source = args[0].value if isinstance(args[0], String) else str(args[0])
        filename = args[1].value if len(args) > 1 and isinstance(args[1], String) else "<input>"
        from .object import List as ListObj
        findings = AuditModule.scan(source, filename)
        return ListObj([Map({String(k): String(str(v)) for k, v in f.items()}) for f in findings])
    
    def _audit_scan_file(*args):
        if len(args) < 1:
            return EvaluationError("audit.scan_file() expects 1 argument: filepath")
        fp = args[0].value if isinstance(args[0], String) else str(args[0])
        from .object import List as ListObj
        findings = AuditModule.scan_file(fp)
        return ListObj([Map({String(k): String(str(v)) for k, v in f.items()}) for f in findings])
    
    def _audit_to_sarif(*args):
        if len(args) < 1:
            return EvaluationError("audit.to_sarif() expects 1 argument: findings list")
        from .object import List as ListObj
        findings = []
        if isinstance(args[0], ListObj):
            for item in args[0].elements:
                if isinstance(item, Map):
                    f = {}
                    for k, v in item.pairs.items():
                        f[k.value if isinstance(k, String) else str(k)] = v.value if hasattr(v, 'value') else str(v)
                    findings.append(f)
        sarif = AuditModule.to_sarif(findings)
        return String(json.dumps(sarif))
    
    audit_env.set("scan", Builtin(_audit_scan, "scan"))
    audit_env.set("scan_file", Builtin(_audit_scan_file, "scan_file"))
    audit_env.set("to_sarif", Builtin(_audit_to_sarif, "to_sarif"))
    modules["audit"] = audit_env
    
    # ===== CONTRACT AUDIT MODULE =====
    contract_audit_env = Environment()
    
    from .stdlib.contract_audit import ContractAuditModule
    
    def _contract_audit(*args):
        if len(args) < 1:
            return EvaluationError("contract_audit.audit() expects 1 argument: source_code")
        source = args[0].value if isinstance(args[0], String) else str(args[0])
        filename = args[1].value if len(args) > 1 and isinstance(args[1], String) else "<contract>"
        from .object import List as ListObj
        findings = ContractAuditModule.audit(source, filename)
        return ListObj([Map({String(k): String(str(v)) for k, v in f.items()}) for f in findings])
    
    def _contract_audit_file(*args):
        if len(args) < 1:
            return EvaluationError("contract_audit.audit_file() expects 1 argument: filepath")
        fp = args[0].value if isinstance(args[0], String) else str(args[0])
        from .object import List as ListObj
        findings = ContractAuditModule.audit_file(fp)
        return ListObj([Map({String(k): String(str(v)) for k, v in f.items()}) for f in findings])
    
    contract_audit_env.set("audit", Builtin(_contract_audit, "audit"))
    contract_audit_env.set("audit_file", Builtin(_contract_audit_file, "audit_file"))
    modules["contract_audit"] = contract_audit_env
    
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
    
    # Initialize on first access. create_builtin_modules() never actually
    # consumes the evaluator (docstring aspiration only) — requiring it
    # here meant the VM (which passes evaluator=None) always got an empty
    # registry, so use "crypto" silently shadowed the richer builtin module
    # and sha256/secp256k1_sign were unreachable on the VM path.
    if not _BUILTIN_MODULES:
        _BUILTIN_MODULES = create_builtin_modules(evaluator)
    
    return _BUILTIN_MODULES.get(module_name)


def is_builtin_module(module_name):
    """Check if a module name refers to a builtin module"""
    return module_name in [
        "crypto", "datetime", "math", "validation", "password",
        "cache", "queue", "template", "testing",
        "fuzz", "secrets", "netsec", "payloads", "pentest",
        "audit", "contract_audit",
    ]
