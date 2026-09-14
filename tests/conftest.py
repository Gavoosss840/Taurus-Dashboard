"""Configuration commune des tests.

Le dépôt et le répertoire `tests/` sont ajoutés au chemin d'import : le
premier pour `taurus_core` et `backend`, le second pour que les doublures
définies dans `test_valuation.py` soient réutilisables ailleurs.
"""

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent

for path in (ROOT, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
