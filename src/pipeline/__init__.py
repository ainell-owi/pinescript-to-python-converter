"""
PineScript-to-Python Converter Pipeline

Shared constants, environment, and helpers used across all pipeline modules.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGISTRY_PATH           = Path("strategies_registry.json")
INPUT_DIR               = Path("input")
ARCHIVE_DIR             = Path("archive")
OUTPUT_DIR              = Path("output")
LOGS_ROOT               = Path("logs")
SEEN_URLS_PATH          = Path("seen_urls.json")
ARCHIVE_SCORE_THRESHOLD = 4     # btc + proj < this → archive; >= this → keep
TARGET_STRATEGY_COUNT   = 6     # minimum .pine files to keep in input/
MAX_SEARCH_LOOPS        = 5     # retry cap for auto-selection before giving up
MAX_SKIP_COUNT          = 2     # archive evaluated strategies after this many skips
_EXCLUDED_PINE_FILES    = {"source_strategy.pine"}

# Subprocess environment with CLAUDECODE stripped so nested `claude` calls are allowed.
SUBPROCESS_ENV = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _div(char: str = "─", width: int = 70) -> str:
    return char * width


def _verdict(btc: int, proj: int) -> str:
    total = btc + proj
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    RESET  = '\033[0m'

    if total >= 8: return f"{GREEN}[ RECOMMENDED ]{RESET}"
    if total >= 6: return f"{GREEN}[ GOOD ]{RESET}"
    if total >= 4: return f"{YELLOW}[ OK ]{RESET}"
    if total >= 2: return f"{YELLOW}[ COMPLEX ]{RESET}"
    return f"{RED}[ SKIP ]{RESET}"