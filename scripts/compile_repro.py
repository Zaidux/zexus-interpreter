import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from zexus.compiler import ZexusCompiler

REPRO_DIR = os.path.join(ROOT, 'tests', 'repro')
files = [f for f in os.listdir(REPRO_DIR) if f.endswith('.zx')]

for fn in sorted(files):
    path = os.path.join(REPRO_DIR, fn)
    with open(path, 'r') as fh:
        src = fh.read()
    print('\n=== Reproducer:', fn, '===')
    try:
        compiler = ZexusCompiler(src, enable_optimizations=False)
        bc = compiler.compile()
        if getattr(compiler, 'errors', None):
            print('Compiler.errors:\n', compiler.errors)
        else:
            print('Compiled successfully')
    except Exception as e:
        print('Exception while compiling:', e)
        import traceback
        traceback.print_exc()
