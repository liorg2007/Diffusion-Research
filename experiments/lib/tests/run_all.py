"""Plain assert-based runner (pytest is not installed in env/.venv).
   env/.venv/bin/python -I experiments/lib/tests/run_all.py [name-substring]
Also pytest-compatible: tests only use the `tmp_path` fixture."""
import importlib
import inspect
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _path  # noqa: E402,F401


def main(filt=""):
    passed = failed = 0
    for f in sorted(_path.TESTS.glob("test_*.py")):
        mod = importlib.import_module(f.stem)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__ or filt not in name:
                continue
            tmp = Path(tempfile.mkdtemp(prefix="libtest_"))
            t0 = time.time()
            try:
                fn(tmp) if "tmp_path" in inspect.signature(fn).parameters else fn()
                passed += 1
                print(f"PASS {f.stem}::{name} ({time.time() - t0:.1f}s)")
            except Exception:
                failed += 1
                print(f"FAIL {f.stem}::{name}")
                traceback.print_exc()
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1] if len(sys.argv) > 1 else "") else 0)
