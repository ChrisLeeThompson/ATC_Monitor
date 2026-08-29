"""
Single source for the application's path roots.

Before v3.3.3 these expressions were duplicated across atc_monitor.py
(MainWindow._get_script_root, _script_logs_dir, _local_logs_dir) and kept
"in sync by convention" -- a silent-divergence hazard. atc_monitor.py now
delegates here; the detailed rationale for which root each consumer uses
(crash-time-local vs. primary-share) stays with the delegates, close to the
logging setup it constrains.

Leaf module on purpose: imports only os/sys/pathlib so it can never pull Qt
or app modules into a path lookup.

Every function evaluates sys.argv / os.environ at call time, never at
import time -- this is load-bearing: tests patch both per-test, and a
module-level constant would freeze the unpatched values.
"""
import os
import sys
from pathlib import Path


def script_root() -> Path:
    """Directory containing the launched script (saved data, cb_plant.json)."""
    return Path(sys.argv[0]).resolve().parent


def script_logs_dir() -> Path:
    """Primary app-log directory: <script dir>/logs (often on the UNC share)."""
    return script_root() / "logs"


def local_app_dir() -> Path:
    """Per-machine app-data root: %LOCALAPPDATA%/ATC_Monitor (Path.home() if unset)."""
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home()
    return root / "ATC_Monitor"


def local_logs_dir() -> Path:
    """Local-disk log directory: crash-time faulthandler home, app-log fallback."""
    return local_app_dir() / "logs"
