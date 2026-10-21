"""Make lib modules importable even under `python -I` (which drops the script dir from sys.path)."""
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
LIB = TESTS.parent
for p in (str(LIB), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)
