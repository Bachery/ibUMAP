"""Put scripts/ on sys.path so the tests import the ``common`` package like the experiment scripts do."""
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
