"""Compare top-level symbols between the interpreter (`evaluator/`) and
the compiler package (`src/zexus/compiler`) and print a simple report.

This script is intentionally conservative: it only considers top-level
FunctionDef, ClassDef, Assign/AnnAssign targets and simple name exports.
It helps quickly discover API surface differences so we can decide what the
compiler lacks compared to the interpreter.
"""
import ast
import os
import sys


def collect_top_level_symbols(path):
    """Return a set of top-level symbol names defined in the Python file at path."""
    symbols = set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            src = f.read()
    except Exception as e:
        print(f"ERROR: could not read {path}: {e}")
        return symbols

    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        print(f"ERROR: could not parse {path}: {e}")
        return symbols

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            symbols.add(elt.id)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name):
                symbols.add(target.id)
    return symbols


def collect_package_symbols(dirpath):
    """Walk dirpath and collect top-level symbols from all .py files."""
    pkg_symbols = {}
    for root, dirs, files in os.walk(dirpath):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            if fn.startswith('__') and fn.endswith('.py'):
                # still include __init__ possibly
                pass
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, dirpath)
            pkg_symbols[rel] = collect_top_level_symbols(full)
    return pkg_symbols


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # OLD: interp_path = os.path.join(repo_root, 'src', 'zexus', 'evaluator.py')
    # NEW: interp_path is now the directory
    interp_dir = os.path.join(repo_root, 'src', 'zexus', 'evaluator') 
    compiler_dir = os.path.join(repo_root, 'src', 'zexus', 'compiler')

    # Update the check to look for the directory instead of a file
    if not os.path.isdir(interp_dir):
        print('Interpreter directory not found:', interp_dir)
        # Try to look for evaluator modules (Your provided check for directory content)
        evaluator_files = []
        if os.path.isdir(interp_dir):
            evaluator_files = [os.path.join(interp_dir, f) for f in os.listdir(interp_dir) 
                              if f.endswith('.py') and not f.startswith('__')]
            if evaluator_files:
                print('Found evaluator modules:', evaluator_files)
        sys.exit(2)
        
    if not os.path.isdir(compiler_dir):
        print('Compiler directory not found:', compiler_dir)
        sys.exit(2)

    # Collect symbols from the interpreter package
    interp_pkg = collect_package_symbols(interp_dir)

    # Collect symbols from the compiler package
    compiler_pkg = collect_package_symbols(compiler_dir)

    # union all interpreter symbols
    interp_symbols_union = set()
    for fn, syms in interp_pkg.items():
        interp_symbols_union.update(syms)

    # union all compiler symbols
    compiler_symbols_union = set()
    for fn, syms in compiler_pkg.items():
        compiler_symbols_union.update(syms)

    only_in_interp = sorted(interp_symbols_union - compiler_symbols_union)
    only_in_compiler = sorted(compiler_symbols_union - interp_symbols_union)
    common = sorted(interp_symbols_union & compiler_symbols_union)

    print('\nComparison report: interpreter package vs compiler package')
    print('Interpreter package dir:', interp_dir)
    print('Compiler package dir:', compiler_dir)
    print('\nCounts:')
    print('  interpreter union symbols:', len(interp_symbols_union))
    print('  compiler union symbols:', len(compiler_symbols_union))
    print('\nOnly in interpreter ({}):'.format(len(only_in_interp)))
    for name in only_in_interp:
        print('   -', name)

    print('\nOnly in compiler ({}):'.format(len(only_in_compiler)))
    for name in only_in_compiler[:200]:
        print('   -', name)

    print('\nCommon symbols ({}):'.format(len(common)))
    for name in common[:200]:
        print('   -', name)
        
    print('\nPer-file interpreter symbol breakdown:')
    for fn, syms in sorted(interp_pkg.items()):
        print(f'  {fn}: {len(syms)} symbols')

    print('\nPer-file compiler symbol breakdown:')
    for fn, syms in sorted(compiler_pkg.items()):
        print(f'  {fn}: {len(syms)} symbols')

    # Quick hints
    if only_in_interp:
        print("""
HINT: The interpreter package defines top-level symbols that are not present in the compiler package.
Start by inspecting the names listed above to decide which functionality should be implemented
or refactored into the compiler. This tool is conservative (top-level only) and may miss
runtime/exported symbols. For a deeper comparison we can compare call-sites, used imports,
or run tests that exercise both codepaths.
""")


if __name__ == '__main__':
    main()
