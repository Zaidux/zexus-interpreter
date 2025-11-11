import subprocess
import sys
import os

ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
ZX_CMD = os.path.join(ROOT, 'zx')

def run_zx(script_path):
    proc = subprocess.run([ZX_CMD, script_path], cwd=ROOT, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

def test_simple_use_import():
    code, out, err = run_zx('./tests/fixtures/use_import.zx')
    assert code == 0, f"zx failed: {err}"
    assert '3.14' in out or '3.14' in err

def test_circular_imports():
    # main script that imports module a which imports b which imports a
    main_script = './tests/fixtures/modules_main.zx'
    # create a small main script on-the-fly
    with open(os.path.join(ROOT, main_script), 'w') as f:
        f.write('use "./tests/fixtures/modules/a.zx"\nprint(a)\nprint(b)\n')

    try:
        code, out, err = run_zx(main_script)
        assert code == 0, f"zx failed: {err}"
        # Expect both exported values to be present
        assert '1' in out
        assert '2' in out
    finally:
        try:
            os.remove(os.path.join(ROOT, main_script))
        except Exception:
            pass
