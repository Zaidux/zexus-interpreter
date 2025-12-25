"""Completion provider for Zexus LSP."""

from typing import List, Dict, Any
try:
    from pygls.lsp.types import CompletionItem, CompletionItemKind, Position
    PYGLS_AVAILABLE = True
except ImportError:
    PYGLS_AVAILABLE = False


# Zexus keywords
KEYWORDS = [
    'let', 'const', 'action', 'function', 'lambda', 'return', 'if', 'elif', 'else',
    'while', 'for', 'each', 'in', 'match', 'case', 'default', 'break', 'continue',
    'entity', 'data', 'enum', 'protocol', 'interface', 'type_alias', 'implements',
    'use', 'import', 'export', 'module', 'package', 'from', 'external',
    'protect', 'verify', 'restrict', 'require', 'assert', 'seal', 'sandbox',
    'audit', 'trail', 'capability', 'grant', 'revoke', 'validate', 'sanitize',
    'contract', 'state', 'ledger', 'persistent', 'storage', 'tx', 'gas', 'limit',
    'hash', 'signature', 'verify_sig', 'emit', 'event', 'revert', 'this',
    'public', 'private', 'pure', 'view', 'payable', 'modifier', 'sealed', 'secure',
    'async', 'await', 'channel', 'send', 'receive', 'atomic', 'stream', 'watch',
    'try', 'catch', 'throw', 'finally', 'defer',
    'native', 'inline', 'gc', 'buffer', 'simd', 'pattern', 'exactly', 'embedded', 'using',
    'screen', 'component', 'theme', 'canvas', 'graphics', 'animation', 'clock', 'color',
    'middleware', 'auth', 'throttle', 'cache', 'inject',
    'true', 'false', 'null', 'map', 'TX', 'print', 'debug', 'log', 'immutable'
]

# Built-in functions with their signatures
BUILTINS = {
    # I/O
    'print': 'print(value)',
    'println': 'println(value)',
    'input': 'input(prompt)',
    'read_text': 'read_text(path)',
    'write_text': 'write_text(path, content)',
    
    # Type conversion
    'string': 'string(value)',
    'int': 'int(value)',
    'float': 'float(value)',
    'bool': 'bool(value)',
    
    # Collections
    'len': 'len(collection)',
    'list': 'list(items...)',
    'map': 'map(pairs...)',  # Map data structure
    'set': 'set(items...)',
    'range': 'range(start, end, step)',
    
    # Functional
    'filter': 'filter(collection, predicate)',
    'map_transform': 'map(collection, transform)',  # Functional map
    'reduce': 'reduce(collection, fn, initial)',
    'sort': 'sort(collection, comparator)',
    'reverse': 'reverse(collection)',
    
    # String operations
    'join': 'join(array, separator)',
    'split': 'split(string, delimiter)',
    'replace': 'replace(string, old, new)',
    'uppercase': 'uppercase(string)',
    'lowercase': 'lowercase(string)',
    'trim': 'trim(string)',
    'substring': 'substring(string, start, end)',
    
    # Math
    'abs': 'abs(number)',
    'ceil': 'ceil(number)',
    'floor': 'floor(number)',
    'round': 'round(number, decimals)',
    'min': 'min(numbers...)',
    'max': 'max(numbers...)',
    'sum': 'sum(numbers)',
    'sqrt': 'sqrt(number)',
    'pow': 'pow(base, exponent)',
    'random': 'random() or random(max) or random(min, max)',
    
    # Date/Time
    'now': 'now()',
    'timestamp': 'timestamp()',
    
    # File I/O
    'file_read_text': 'file_read_text(path)',
    'file_write_text': 'file_write_text(path, content)',
    'file_exists': 'file_exists(path)',
    'file_read_json': 'file_read_json(path)',
    'file_write_json': 'file_write_json(path, data)',
    'file_append': 'file_append(path, content)',
    'file_list_dir': 'file_list_dir(path)',
    'read_file': 'read_file(path)',
    'eval_file': 'eval_file(path, language)',
    
    # Persistence
    'persist_set': 'persist_set(key, value)',
    'persist_get': 'persist_get(key)',
    'persist_clear': 'persist_clear(key)',
    'persist_list': 'persist_list()',
    
    # Memory
    'track_memory': 'track_memory()',
    'memory_stats': 'memory_stats()',
    
    # Security
    'protect': 'protect(function, policy, mode)',
    'verify': 'verify(condition)',
    'restrict': 'restrict(value, constraints)',
    'create_policy': 'create_policy(rules)',
    'enforce_policy': 'enforce_policy(policy, value)',
    
    # Dependency Injection
    'register_dependency': 'register_dependency(name, impl)',
    'inject_dependency': 'inject_dependency(name)',
    'mock_dependency': 'mock_dependency(name, mock)',
    'test_mode': 'test_mode(enabled)',
    
    # Channels
    'send': 'send(channel, value)',
    'receive': 'receive(channel)',
    'close_channel': 'close_channel(channel)',
    
    # Blockchain
    'emit': 'emit(event, ...args)',
    'require': 'require(condition, message)',
    'assert': 'assert(condition)',
    'balance_of': 'balance_of(address)',
    'transfer': 'transfer(to, amount)',
    'hash': 'hash(data)',
    'keccak256': 'keccak256(data)',
    'signature': 'signature(data, key)',
    'verify_sig': 'verify_sig(data, sig, key)',
    
    # Renderer
    'define_screen': 'define_screen(name, props)',
    'define_component': 'define_component(name, props)',
    'render_screen': 'render_screen(name)',
    'set_theme': 'set_theme(theme)',
    
    # Debug
    'debug': 'debug(value)',
    'debug_log': 'debug_log(message, context)',
    'debug_trace': 'debug_trace()',
    'is_main': 'is_main()',
    'exit_program': 'exit_program(code)',
    'module_info': 'module_info()',
    
    # Main entry point
    'run': 'run(task_fn)',
    'execute': 'execute(fn)',
    'on_start': 'on_start(fn)',
    'on_exit': 'on_exit(fn)',
    'signal_handler': 'signal_handler(signal, fn)',
    'schedule': 'schedule(fn, delay)',
    'sleep': 'sleep(seconds)',
    
    # Validation
    'is_email': 'is_email(string)',
    'is_url': 'is_url(string)',
    'is_phone': 'is_phone(string)',
    'is_numeric': 'is_numeric(string)',
    'is_alpha': 'is_alpha(string)',
    'is_alphanumeric': 'is_alphanumeric(string)',
    'matches_pattern': 'matches_pattern(str, pattern)',
    'password_strength': 'password_strength(password)',
    'sanitize_input': 'sanitize_input(text, type)',
    'validate_length': 'validate_length(str, min, max)',
    'env_get': 'env_get(name)',
    'env_set': 'env_set(name, value)',
    'env_exists': 'env_exists(name)',
}


class CompletionProvider:
    """Provides completion suggestions for Zexus code."""

    def get_completions(self, text: str, position: Position, doc_info: Dict[str, Any]) -> List:
        """Get completion items for the given position."""
        if not PYGLS_AVAILABLE:
            return []
        
        items = []
        
        # Add keywords
        for keyword in KEYWORDS:
            items.append(CompletionItem(
                label=keyword,
                kind=CompletionItemKind.Keyword,
                detail='Keyword',
                documentation=f'Zexus keyword: {keyword}'
            ))
        
        # Add built-in functions
        for name, signature in BUILTINS.items():
            items.append(CompletionItem(
                label=name,
                kind=CompletionItemKind.Function,
                detail=signature,
                documentation=f'Built-in function: {signature}',
                insert_text=f'{name}($1)',
                insert_text_format=2  # Snippet format
            ))
        
        # TODO: Add user-defined symbols from AST
        # TODO: Add context-aware completions
        
        return items
