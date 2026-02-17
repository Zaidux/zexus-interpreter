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

    elif module_name == "perf" or module_name == "stdlib/perf":
        def _perf_vm_stats(*args):
            if evaluator is None or not hasattr(evaluator, "unified_executor") or not evaluator.unified_executor:
                return EvaluationError("perf.vm_stats() requires unified executor")
            stats = evaluator.unified_executor.get_statistics()
            return _python_to_zexus(stats)

        def _perf_set_vm_thresholds(*args):
            if evaluator is None or not hasattr(evaluator, "unified_executor") or not evaluator.unified_executor:
                return EvaluationError("perf.set_vm_thresholds() requires unified executor")
            workload = evaluator.unified_executor.workload
            if len(args) >= 1:
                val = _zexus_to_python(args[0])
                if isinstance(val, int):
                    workload.vm_threshold = val
            if len(args) >= 2:
                val = _zexus_to_python(args[1])
                if isinstance(val, int):
                    workload.jit_threshold = val
            if len(args) >= 3:
                val = _zexus_to_python(args[2])
                if isinstance(val, int):
                    workload.parallel_threshold = val
            return Boolean(True)

        def _perf_enable_vm(*args):
            if evaluator is None:
                return EvaluationError("perf.enable_vm() requires evaluator")
            flag = True
            if len(args) >= 1:
                val = _zexus_to_python(args[0])
                flag = bool(val)
            evaluator.use_vm = flag
            if evaluator.unified_executor:
                evaluator.unified_executor.vm_enabled = flag
            return Boolean(True)

        def _perf_collect_gc(*args):
            if evaluator is None or not hasattr(evaluator, "unified_executor") or not evaluator.unified_executor:
                return EvaluationError("perf.collect_gc() requires unified executor")
            vm = evaluator.unified_executor.vm
            if vm is None:
                return EvaluationError("perf.collect_gc() requires VM initialization")
            force = False
            if len(args) >= 1:
                force = bool(_zexus_to_python(args[0]))
            result = vm.collect_garbage(force=force)
            return _python_to_zexus(result)

        def _perf_memory_stats(*args):
            if evaluator is None or not hasattr(evaluator, "unified_executor") or not evaluator.unified_executor:
                return EvaluationError("perf.memory_stats() requires unified executor")
            vm = evaluator.unified_executor.vm
            if vm is None:
                return EvaluationError("perf.memory_stats() requires VM initialization")
            stats = vm.get_memory_stats()
            return _python_to_zexus(stats)

        def _perf_warmup(*args):
            if evaluator is None or not hasattr(evaluator, "unified_executor") or not evaluator.unified_executor:
                return EvaluationError("perf.warmup() requires unified executor")
            workload = evaluator.unified_executor.workload
            vm_threshold = 1
            jit_threshold = workload.jit_threshold
            if len(args) >= 1:
                val = _zexus_to_python(args[0])
                if isinstance(val, int):
                    vm_threshold = val
            if len(args) >= 2:
                val = _zexus_to_python(args[1])
                if isinstance(val, int):
                    jit_threshold = val
            workload.vm_threshold = vm_threshold
            workload.jit_threshold = jit_threshold
            return Boolean(True)

        def _perf_profile_start(*args):
            if evaluator is None or not hasattr(evaluator, "unified_executor") or not evaluator.unified_executor:
                return EvaluationError("perf.profile_start() requires unified executor")
            evaluator.unified_executor.ensure_vm(profile_active=True)
            vm = evaluator.unified_executor.vm
            if vm is None:
                return EvaluationError("perf.profile_start() requires VM initialization")
            vm.start_profiling()
            return Boolean(True)

        def _perf_profile_stop(*args):
            if evaluator is None or not hasattr(evaluator, "unified_executor") or not evaluator.unified_executor:
                return EvaluationError("perf.profile_stop() requires unified executor")
            vm = evaluator.unified_executor.vm
            if vm is None:
                return EvaluationError("perf.profile_stop() requires VM initialization")
            vm.stop_profiling()
            return Boolean(True)

        def _perf_profile_report(*args):
            if evaluator is None or not hasattr(evaluator, "unified_executor") or not evaluator.unified_executor:
                return EvaluationError("perf.profile_report() requires unified executor")
            evaluator.unified_executor.ensure_vm(profile_active=True)
            vm = evaluator.unified_executor.vm
            if vm is None:
                return EvaluationError("perf.profile_report() requires VM initialization")
            report_format = "text"
            top_n = 20
            if len(args) >= 1:
                report_format = str(_zexus_to_python(args[0]))
            if len(args) >= 2:
                val = _zexus_to_python(args[1])
                if isinstance(val, int):
                    top_n = val
            report = vm.get_profiling_report(format=report_format, top_n=top_n)
            return String(report)

        def _perf_set_vm_config(*args):
            if evaluator is None or not hasattr(evaluator, "unified_executor") or not evaluator.unified_executor:
                return EvaluationError("perf.set_vm_config() requires unified executor")
            if len(args) < 1:
                return EvaluationError("perf.set_vm_config() requires 1 argument: config map")
            config = _zexus_to_python(args[0])
            if not isinstance(config, dict):
                return EvaluationError("perf.set_vm_config() expects a map")
            evaluator.unified_executor.configure_vm(config)
            return Boolean(True)

        def _perf_set_vm_mode(*args):
            if evaluator is None or not hasattr(evaluator, "unified_executor") or not evaluator.unified_executor:
                return EvaluationError("perf.set_vm_mode() requires unified executor")
            if len(args) < 1:
                return EvaluationError("perf.set_vm_mode() requires 1 argument: mode")
            mode_value = _zexus_to_python(args[0])
            evaluator.unified_executor.configure_vm({"mode": mode_value})
            return Boolean(True)

        def _perf_force_vm_loops(*args):
            if evaluator is None or not hasattr(evaluator, "unified_executor") or not evaluator.unified_executor:
                return EvaluationError("perf.force_vm_loops() requires unified executor")
            flag = True
            if len(args) >= 1:
                flag = bool(_zexus_to_python(args[0]))
            evaluator.unified_executor.set_force_vm_loops(flag)
            return Boolean(True)

        def _perf_reset_vm(*args):
            if evaluator is None or not hasattr(evaluator, "unified_executor") or not evaluator.unified_executor:
                return EvaluationError("perf.reset_vm() requires unified executor")
            evaluator.unified_executor.reset_vm()
            return Boolean(True)

        env.set("vm_stats", Builtin(_perf_vm_stats))
        env.set("set_vm_thresholds", Builtin(_perf_set_vm_thresholds))
        env.set("enable_vm", Builtin(_perf_enable_vm))
        env.set("collect_gc", Builtin(_perf_collect_gc))
        env.set("memory_stats", Builtin(_perf_memory_stats))
        env.set("warmup", Builtin(_perf_warmup))
        env.set("profile_start", Builtin(_perf_profile_start))
        env.set("profile_stop", Builtin(_perf_profile_stop))
        env.set("profile_report", Builtin(_perf_profile_report))
        env.set("set_vm_config", Builtin(_perf_set_vm_config))
        env.set("set_vm_mode", Builtin(_perf_set_vm_mode))
        env.set("force_vm_loops", Builtin(_perf_force_vm_loops))
        env.set("reset_vm", Builtin(_perf_reset_vm))
        
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
        
        def _blockchain_create_block(*args):
            if len(args) < 4:
                return EvaluationError("create_block() requires 4 args: index, timestamp, data, previous_hash")
            index = int(args[0].value if hasattr(args[0], 'value') else args[0])
            timestamp = float(args[1].value if hasattr(args[1], 'value') else args[1])
            data = args[2].value if hasattr(args[2], 'value') else str(args[2])
            prev_hash = args[3].value if hasattr(args[3], 'value') else str(args[3])
            nonce = int(args[4].value if hasattr(args[4], 'value') else args[4]) if len(args) > 4 else 0
            try:
                result = BlockchainModule.create_block(index, timestamp, data, prev_hash, nonce)
                return _python_to_zexus(result)
            except Exception as e:
                return EvaluationError(f"create_block error: {str(e)}")

        def _blockchain_hash_block(*args):
            if len(args) < 1:
                return EvaluationError("hash_block() requires 1 argument: block (map)")
            block_obj = args[0]
            block = {}
            if isinstance(block_obj, Map):
                for k, v in block_obj.pairs.items():
                    key = k.value if hasattr(k, 'value') else str(k)
                    val = v.value if hasattr(v, 'value') else str(v)
                    block[key] = val
            try:
                result = BlockchainModule.hash_block(block)
                return String(result)
            except Exception as e:
                return EvaluationError(f"hash_block error: {str(e)}")

        def _blockchain_validate_block(*args):
            if len(args) < 1:
                return EvaluationError("validate_block() requires 1 argument: block")
            block_obj = args[0]
            block = {}
            if isinstance(block_obj, Map):
                for k, v in block_obj.pairs.items():
                    key = k.value if hasattr(k, 'value') else str(k)
                    val = v.value if hasattr(v, 'value') else str(v)
                    block[key] = val
            prev = None
            if len(args) > 1 and isinstance(args[1], Map):
                prev = {}
                for k, v in args[1].pairs.items():
                    key = k.value if hasattr(k, 'value') else str(k)
                    val = v.value if hasattr(v, 'value') else str(v)
                    prev[key] = val
            try:
                result = BlockchainModule.validate_block(block, prev)
                return Boolean(result)
            except Exception as e:
                return EvaluationError(f"validate_block error: {str(e)}")

        def _blockchain_proof_of_work(*args):
            if len(args) < 1:
                return EvaluationError("proof_of_work() requires 1 argument: block_data")
            block_data = args[0].value if hasattr(args[0], 'value') else str(args[0])
            difficulty = int(args[1].value if hasattr(args[1], 'value') else args[1]) if len(args) > 1 else 4
            max_iter = int(args[2].value if hasattr(args[2], 'value') else args[2]) if len(args) > 2 else 1000000
            try:
                nonce, hash_val = BlockchainModule.proof_of_work(block_data, difficulty, max_iter)
                return _python_to_zexus({"nonce": nonce, "hash": hash_val})
            except Exception as e:
                return EvaluationError(f"proof_of_work error: {str(e)}")

        def _blockchain_create_transaction(*args):
            if len(args) < 3:
                return EvaluationError("create_transaction() requires 3 args: sender, recipient, amount")
            sender = args[0].value if hasattr(args[0], 'value') else str(args[0])
            recipient = args[1].value if hasattr(args[1], 'value') else str(args[1])
            amount = float(args[2].value if hasattr(args[2], 'value') else args[2])
            timestamp = float(args[3].value if hasattr(args[3], 'value') else args[3]) if len(args) > 3 else None
            try:
                result = BlockchainModule.create_transaction(sender, recipient, amount, timestamp)
                return _python_to_zexus(result)
            except Exception as e:
                return EvaluationError(f"create_transaction error: {str(e)}")

        def _blockchain_hash_transaction(*args):
            if len(args) < 1:
                return EvaluationError("hash_transaction() requires 1 argument: transaction")
            tx_obj = args[0]
            tx = {}
            if isinstance(tx_obj, Map):
                for k, v in tx_obj.pairs.items():
                    key = k.value if hasattr(k, 'value') else str(k)
                    val = v.value if hasattr(v, 'value') else str(v)
                    tx[key] = val
            try:
                result = BlockchainModule.hash_transaction(tx)
                return String(result)
            except Exception as e:
                return EvaluationError(f"hash_transaction error: {str(e)}")

        def _blockchain_validate_chain(*args):
            if len(args) < 1:
                return EvaluationError("validate_chain() requires 1 argument: chain (list of blocks)")
            if not isinstance(args[0], ListObj):
                return EvaluationError("validate_chain() expects a list of block maps")
            chain = []
            for block_obj in args[0].elements:
                block = {}
                if isinstance(block_obj, Map):
                    for k, v in block_obj.pairs.items():
                        key = k.value if hasattr(k, 'value') else str(k)
                        val = v.value if hasattr(v, 'value') else str(v)
                        block[key] = val
                chain.append(block)
            try:
                result = BlockchainModule.validate_chain(chain)
                return Boolean(result)
            except Exception as e:
                return EvaluationError(f"validate_chain error: {str(e)}")

        env.set("create_address", Builtin(_blockchain_create_address))
        env.set("validate_address", Builtin(_blockchain_validate_address))
        env.set("calculate_merkle_root", Builtin(_blockchain_calculate_merkle_root))
        env.set("create_genesis_block", Builtin(_blockchain_create_genesis_block))
        env.set("create_block", Builtin(_blockchain_create_block))
        env.set("hash_block", Builtin(_blockchain_hash_block))
        env.set("validate_block", Builtin(_blockchain_validate_block))
        env.set("proof_of_work", Builtin(_blockchain_proof_of_work))
        env.set("create_transaction", Builtin(_blockchain_create_transaction))
        env.set("hash_transaction", Builtin(_blockchain_hash_transaction))
        env.set("validate_chain", Builtin(_blockchain_validate_chain))

    elif module_name == "websocket" or module_name == "stdlib/websocket":
        try:
            from .stdlib.websockets import WebSocketModule
        except ImportError:
            return env  # websockets package not installed

        def _ws_create_server(*args):
            if len(args) < 3:
                return EvaluationError("ws_create_server() requires 3 args: host, port, handler")
            host = args[0].value if hasattr(args[0], 'value') else str(args[0])
            port = int(args[1].value if hasattr(args[1], 'value') else args[1])
            handler = args[2]  # Zexus Action/Builtin — caller wraps
            path = args[3].value if len(args) > 3 and hasattr(args[3], 'value') else None
            try:
                server = WebSocketModule.create_server(host, port, handler, path)
                server.start()
                stop_fn = Builtin(lambda *_a: (server.stop(), Boolean(True))[1])
                is_running_fn = Builtin(lambda *_a: Boolean(server.is_running()))
                return Map({
                    String("stop"): stop_fn,
                    String("is_running"): is_running_fn,
                })
            except Exception as e:
                return EvaluationError(f"ws_create_server error: {e}")

        def _ws_connect(*args):
            if len(args) < 1:
                return EvaluationError("ws_connect() requires 1 arg: url")
            url = args[0].value if hasattr(args[0], 'value') else str(args[0])
            timeout = float(args[1].value if len(args) > 1 and hasattr(args[1], 'value') else 10)
            try:
                client = WebSocketModule.connect(url, timeout)
                send_fn = Builtin(lambda *a: (client.send(a[0].value if hasattr(a[0], 'value') else str(a[0])), Boolean(True))[1])
                recv_fn = Builtin(lambda *a: String(client.receive(float(a[0].value) if a else 30)))
                close_fn = Builtin(lambda *_a: (client.close(), Boolean(True))[1])
                connected_fn = Builtin(lambda *_a: Boolean(client.is_connected()))
                return Map({
                    String("send"): send_fn,
                    String("receive"): recv_fn,
                    String("close"): close_fn,
                    String("is_connected"): connected_fn,
                })
            except Exception as e:
                return EvaluationError(f"ws_connect error: {e}")

        env.set("create_server", Builtin(_ws_create_server))
        env.set("connect", Builtin(_ws_connect))

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
    stdlib_modules = ['fs', 'http', 'json', 'datetime', 'crypto', 'blockchain', 'perf', 'websocket']
    
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
