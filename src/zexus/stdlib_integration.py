"""
Standard Library Integration for Zexus
Provides integration between Python stdlib modules and Zexus evaluator.
"""

from .object import Environment, Builtin, String, Integer, Float, Boolean, Map, List as ListObj, EvaluationError


def create_stdlib_module(module_name, evaluator=None):
    """
    Create a Zexus environment for a stdlib module.
    
    Args:
        module_name: Name of the stdlib module (fs, http, json, datetime, crypto, blockchain)
        evaluator: Optional evaluator instance
    
    Returns:
        Environment object with stdlib functions registered
    """
    env = Environment()
    
    if module_name == "fs" or module_name == "stdlib/fs":
        from .stdlib.fs import FileSystemModule
        
        # Register all fs functions
        def _fs_read_file(*args):
            if len(args) < 1:
                return EvaluationError("read_file() requires at least 1 argument: path")
            path = args[0].value if hasattr(args[0], 'value') else str(args[0])
            encoding = args[1].value if len(args) > 1 and hasattr(args[1], 'value') else 'utf-8'
            try:
                result = FileSystemModule.read_file(path, encoding)
                return String(result)
            except Exception as e:
                return EvaluationError(f"read_file error: {str(e)}")
        
        def _fs_write_file(*args):
            if len(args) < 2:
                return EvaluationError("write_file() requires 2 arguments: path, content")
            path = args[0].value if hasattr(args[0], 'value') else str(args[0])
            content = args[1].value if hasattr(args[1], 'value') else str(args[1])
            encoding = args[2].value if len(args) > 2 and hasattr(args[2], 'value') else 'utf-8'
            try:
                FileSystemModule.write_file(path, content, encoding)
                return Boolean(True)
            except Exception as e:
                return EvaluationError(f"write_file error: {str(e)}")
        
        def _fs_exists(*args):
            if len(args) < 1:
                return EvaluationError("exists() requires 1 argument: path")
            path = args[0].value if hasattr(args[0], 'value') else str(args[0])
            result = FileSystemModule.exists(path)
            return Boolean(result)
        
        def _fs_mkdir(*args):
            if len(args) < 1:
                return EvaluationError("mkdir() requires 1 argument: path")
            path = args[0].value if hasattr(args[0], 'value') else str(args[0])
            try:
                FileSystemModule.mkdir(path)
                return Boolean(True)
            except Exception as e:
                return EvaluationError(f"mkdir error: {str(e)}")
        
        def _fs_list_dir(*args):
            path = args[0].value if len(args) > 0 and hasattr(args[0], 'value') else '.'
            try:
                result = FileSystemModule.list_dir(path)
                return ListObj([String(f) for f in result])
            except Exception as e:
                return EvaluationError(f"list_dir error: {str(e)}")
        
        env.set("read_file", Builtin(_fs_read_file))
        env.set("write_file", Builtin(_fs_write_file))
        env.set("exists", Builtin(_fs_exists))
        env.set("mkdir", Builtin(_fs_mkdir))
        env.set("list_dir", Builtin(_fs_list_dir))
        
    elif module_name == "http" or module_name == "stdlib/http":
        from .stdlib.http import HttpModule
        
        def _http_get(*args):
            if len(args) < 1:
                return EvaluationError("get() requires 1 argument: url")
            url = args[0].value if hasattr(args[0], 'value') else str(args[0])
            try:
                result = HttpModule.get(url)
                return Map({
                    String("status"): Integer(result['status']),
                    String("body"): String(result['body']),
                    String("headers"): Map({String(k): String(v) for k, v in result['headers'].items()})
                })
            except Exception as e:
                return EvaluationError(f"get error: {str(e)}")
        
        def _http_post(*args):
            if len(args) < 1:
                return EvaluationError("post() requires at least 1 argument: url")
            url = args[0].value if hasattr(args[0], 'value') else str(args[0])
            data = args[1].value if len(args) > 1 and hasattr(args[1], 'value') else None
            try:
                result = HttpModule.post(url, data)
                return Map({
                    String("status"): Integer(result['status']),
                    String("body"): String(result['body']),
                    String("headers"): Map({String(k): String(v) for k, v in result['headers'].items()})
                })
            except Exception as e:
                return EvaluationError(f"post error: {str(e)}")
        
        env.set("get", Builtin(_http_get))
        env.set("post", Builtin(_http_post))
        
    elif module_name == "json" or module_name == "stdlib/json":
        from .stdlib.json_module import JsonModule
        import json as json_lib
        
        def _json_parse(*args):
            if len(args) < 1:
                return EvaluationError("parse() requires 1 argument: text")
            text = args[0].value if hasattr(args[0], 'value') else str(args[0])
            try:
                result = JsonModule.parse(text)
                return _python_to_zexus(result)
            except Exception as e:
                return EvaluationError(f"parse error: {str(e)}")
        
        def _json_stringify(*args):
            if len(args) < 1:
                return EvaluationError("stringify() requires 1 argument: obj")
            obj = _zexus_to_python(args[0])
            try:
                result = JsonModule.stringify(obj)
                return String(result)
            except Exception as e:
                return EvaluationError(f"stringify error: {str(e)}")
        
        env.set("parse", Builtin(_json_parse))
        env.set("stringify", Builtin(_json_stringify))
        
    elif module_name == "datetime" or module_name == "stdlib/datetime":
        from .stdlib.datetime import DateTimeModule
        from datetime import datetime
        
        def _datetime_now(*args):
            try:
                result = DateTimeModule.now()
                return String(result.isoformat())
            except Exception as e:
                return EvaluationError(f"now error: {str(e)}")
        
        def _datetime_timestamp(*args):
            try:
                result = DateTimeModule.timestamp()
                return Float(result)
            except Exception as e:
                return EvaluationError(f"timestamp error: {str(e)}")
        
        def _datetime_format(*args):
            if len(args) < 1:
                return EvaluationError("format() requires at least 1 argument")
            # For simplicity, accept ISO string and format string
            dt_str = args[0].value if hasattr(args[0], 'value') else str(args[0])
            fmt = args[1].value if len(args) > 1 and hasattr(args[1], 'value') else '%Y-%m-%d %H:%M:%S'
            try:
                dt = datetime.fromisoformat(dt_str)
                result = DateTimeModule.format(dt, fmt)
                return String(result)
            except Exception as e:
                return EvaluationError(f"format error: {str(e)}")
        
        env.set("now", Builtin(_datetime_now))
        env.set("timestamp", Builtin(_datetime_timestamp))
        env.set("format", Builtin(_datetime_format))
        
    elif module_name == "crypto" or module_name == "stdlib/crypto":
        from .stdlib.crypto import CryptoModule
        
        def _crypto_hash_sha256(*args):
            if len(args) < 1:
                return EvaluationError("hash_sha256() requires 1 argument: data")
            data = args[0].value if hasattr(args[0], 'value') else str(args[0])
            try:
                result = CryptoModule.hash_sha256(data)
                return String(result)
            except Exception as e:
                return EvaluationError(f"hash_sha256 error: {str(e)}")
        
        def _crypto_keccak256(*args):
            if len(args) < 1:
                return EvaluationError("keccak256() requires 1 argument: data")
            data = args[0].value if hasattr(args[0], 'value') else str(args[0])
            try:
                result = CryptoModule.keccak256(data)
                return String(result)
            except Exception as e:
                return EvaluationError(f"keccak256 error: {str(e)}")
        
        def _crypto_random_bytes(*args):
            size = 32  # default
            if len(args) > 0:
                if hasattr(args[0], 'value') and isinstance(args[0].value, int):
                    size = args[0].value
                elif isinstance(args[0], int):
                    size = args[0]
                else:
                    return EvaluationError("random_bytes() size argument must be an integer")
            try:
                result = CryptoModule.random_bytes(size)
                return String(result)
            except Exception as e:
                return EvaluationError(f"random_bytes error: {str(e)}")
        
        def _crypto_pbkdf2(*args):
            if len(args) < 2:
                return EvaluationError("pbkdf2() requires at least 2 arguments: password, salt")
            password = args[0].value if hasattr(args[0], 'value') else str(args[0])
            salt = args[1].value if hasattr(args[1], 'value') else str(args[1])
            
            # Validate iterations parameter
            iterations = 100000  # default
            if len(args) > 2:
                if hasattr(args[2], 'value') and isinstance(args[2].value, int):
                    iterations = args[2].value
                elif isinstance(args[2], int):
                    iterations = args[2]
                else:
                    return EvaluationError("pbkdf2() iterations argument must be an integer")
            
            try:
                result = CryptoModule.pbkdf2(password, salt, iterations)
                return String(result)
            except Exception as e:
                return EvaluationError(f"pbkdf2 error: {str(e)}")
        
        env.set("hash_sha256", Builtin(_crypto_hash_sha256))
        env.set("keccak256", Builtin(_crypto_keccak256))
        env.set("random_bytes", Builtin(_crypto_random_bytes))
        env.set("pbkdf2", Builtin(_crypto_pbkdf2))
        
    elif module_name == "blockchain" or module_name == "stdlib/blockchain":
        from .stdlib.blockchain import BlockchainModule
        
        def _blockchain_create_address(*args):
            if len(args) < 1:
                return EvaluationError("create_address() requires 1 argument: public_key")
            public_key = args[0].value if hasattr(args[0], 'value') else str(args[0])
            prefix = args[1].value if len(args) > 1 and hasattr(args[1], 'value') else "0x"
            try:
                result = BlockchainModule.create_address(public_key, prefix)
                return String(result)
            except Exception as e:
                return EvaluationError(f"create_address error: {str(e)}")
        
        def _blockchain_validate_address(*args):
            if len(args) < 1:
                return EvaluationError("validate_address() requires 1 argument: address")
            address = args[0].value if hasattr(args[0], 'value') else str(args[0])
            prefix = args[1].value if len(args) > 1 and hasattr(args[1], 'value') else "0x"
            try:
                result = BlockchainModule.validate_address(address, prefix)
                return Boolean(result)
            except Exception as e:
                return EvaluationError(f"validate_address error: {str(e)}")
        
        def _blockchain_calculate_merkle_root(*args):
            if len(args) < 1:
                return EvaluationError("calculate_merkle_root() requires 1 argument: hashes")
            if not isinstance(args[0], ListObj):
                return EvaluationError("calculate_merkle_root() expects a list")
            hashes = [h.value if hasattr(h, 'value') else str(h) for h in args[0].elements]
            try:
                result = BlockchainModule.calculate_merkle_root(hashes)
                return String(result)
            except Exception as e:
                return EvaluationError(f"calculate_merkle_root error: {str(e)}")
        
        def _blockchain_create_genesis_block(*args):
            try:
                result = BlockchainModule.create_genesis_block()
                return _python_to_zexus(result)
            except Exception as e:
                return EvaluationError(f"create_genesis_block error: {str(e)}")
        
        env.set("create_address", Builtin(_blockchain_create_address))
        env.set("validate_address", Builtin(_blockchain_validate_address))
        env.set("calculate_merkle_root", Builtin(_blockchain_calculate_merkle_root))
        env.set("create_genesis_block", Builtin(_blockchain_create_genesis_block))
    
    return env


def _python_to_zexus(value):
    """Convert Python value to Zexus object."""
    if isinstance(value, bool):
        return Boolean(value)
    elif isinstance(value, int):
        return Integer(value)
    elif isinstance(value, float):
        return Float(value)
    elif isinstance(value, str):
        return String(value)
    elif isinstance(value, list):
        return ListObj([_python_to_zexus(v) for v in value])
    elif isinstance(value, dict):
        return Map({String(k): _python_to_zexus(v) for k, v in value.items()})
    else:
        return String(str(value))


def _zexus_to_python(obj):
    """Convert Zexus object to Python value."""
    if hasattr(obj, 'value'):
        return obj.value
    elif isinstance(obj, ListObj):
        return [_zexus_to_python(e) for e in obj.elements]
    elif isinstance(obj, Map):
        return {_zexus_to_python(k): _zexus_to_python(v) for k, v in obj.pairs.items()}
    else:
        return obj


def is_stdlib_module(module_name):
    """Check if a module name refers to a stdlib module."""
    stdlib_modules = ['fs', 'http', 'json', 'datetime', 'crypto', 'blockchain']
    
    # Handle both "fs" and "stdlib/fs" formats
    if module_name in stdlib_modules:
        return True
    
    if module_name.startswith('stdlib/'):
        module_base = module_name[7:]  # Remove 'stdlib/' prefix
        return module_base in stdlib_modules
    
    return False


def get_stdlib_module(module_name, evaluator=None):
    """Get a stdlib module environment."""
    return create_stdlib_module(module_name, evaluator)
