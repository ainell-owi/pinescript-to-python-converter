"""
PineScript-to-Python Converter Pipeline

Entry point. Orchestrates: scrape → evaluate → select → convert → archive.

Lifecycle:
  new -> evaluated -> selected -> converted
  new / evaluated (score < threshold or skipped 2x) -> archived
"""

import logging
import shutil
import sys
from datetime import datetime, UTC
from pathlib import Path

from src.pipeline import (
    ARCHIVE_DIR,
    INPUT_DIR,
    LOGS_ROOT,
    MAX_SEARCH_LOOPS,
    OUTPUT_DIR,
    TARGET_STRATEGY_COUNT,
    _div,
    _EXCLUDED_PINE_FILES,
)
from src.pipeline.archiver import archive_remaining
from src.pipeline.evaluator import run_evaluations
from src.pipeline.orchestrator import copy_artifacts, run_orchestrator, verify_artifacts
from src.pipeline.registry import _now_iso, load_registry, save_registry, scan_and_register
from src.pipeline.scraper import run_tv_scraper
from src.pipeline.selector import auto_select_strategy


# ---------------------------------------------------------------------------
# Logging  (file-only; terminal gets clean print() UI)
# ---------------------------------------------------------------------------
def _setup_file_logger() -> logging.Logger:
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOGS_ROOT / f"runner_{ts}.log"

    lg = logging.getLogger("runner")
    lg.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ))
    lg.addHandler(fh)
    lg.propagate = False
    return lg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger = _setup_file_logger()

    print(_div("═"))
    print("  PineScript → Python Converter")
    print(_div("═"))

    # Step 0 — Ensure at least TARGET_STRATEGY_COUNT real .pine files in input/
    INPUT_DIR.mkdir(exist_ok=True)
    existing = [f for f in INPUT_DIR.glob("*.pine") if f.name not in _EXCLUDED_PINE_FILES]
    if len(existing) < TARGET_STRATEGY_COUNT:
        needed = TARGET_STRATEGY_COUNT - len(existing)
        run_tv_scraper(max_results=needed)

    # Step 1 — Scan & Register
    print("\n  Scanning input/ for .pine files...")
    registry = load_registry()
    registry = scan_and_register(registry)
    save_registry(registry)

    # Step 2 — Evaluate new strategies (isolated, one at a time)
    registry = run_evaluations(registry)

    # Step 3 — Auto-select the highest-scoring strategy.
    # If no evaluated strategies exist, fetch a fresh batch and retry.
    chosen_key, chosen_rec = None, None
    for _attempt in range(MAX_SEARCH_LOOPS):
        chosen_key, chosen_rec = auto_select_strategy(registry)
        if chosen_key is not None:
            break
        print(f"\n  No valid strategies found (attempt {_attempt + 1}/{MAX_SEARCH_LOOPS}).")
        print("  Fetching a fresh batch from TradingView...")
        run_tv_scraper(max_results=TARGET_STRATEGY_COUNT)
        registry = scan_and_register(registry)
        registry = run_evaluations(registry)
    else:
        print(f"\n[FAIL] Could not find a valid strategy after {MAX_SEARCH_LOOPS} attempts.")
        sys.exit(1)

    registry[chosen_key]["status"] = "selected"
    save_registry(registry)

    # Step 4 — Transpile (orchestrator → transpiler → validator → test gen → integration)
    meta      = chosen_rec["pine_metadata"]
    ts        = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = meta.get("safe_name") or chosen_key.replace(".pine", "")
    out_dir   = OUTPUT_DIR / safe_name / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    success, run_dir = run_orchestrator(Path(chosen_rec["file_path"]), meta, out_dir)

    # Defense-in-depth: verify artifacts exist even if token was found
    if success and not verify_artifacts(safe_name):
        logger.error("Token indicated success but artifacts are missing.")
        success = False

    if success:
        copy_artifacts(meta, out_dir, run_dir)

        # Post-conversion cleanup: move .pine to archive/
        ARCHIVE_DIR.mkdir(exist_ok=True)
        pine_src = Path(chosen_rec["file_path"])
        new_file_path = str(pine_src)
        if pine_src.exists():
            pine_dest = ARCHIVE_DIR / pine_src.name
            shutil.move(str(pine_src), pine_dest)
            new_file_path = str(pine_dest)

        # Single, atomic registry update
        registry[chosen_key].update({
            "status":       "completed",
            "converted_at": _now_iso(),
            "archived_at":  _now_iso(),
            "output_dir":   str(out_dir),
            "file_path":    new_file_path,
        })
        save_registry(registry)
        print(f"\n  Conversion complete!")
        print(f"    Artifacts → {out_dir}")
    else:
        registry[chosen_key].update({
            "status":    "failed",
            "failed_at": _now_iso(),
        })
        save_registry(registry)
        print(f"\n  Orchestrator failed. See: {run_dir / 'run.log'}")
        sys.exit(1)

    # Step 5 — Smart archive
    print("\n  Archiving low-scoring / stale strategies...")
    registry = archive_remaining(registry, chosen_key)
    save_registry(registry)
    print("\n  Done.\n")


if __name__ == "__main__":
    main()