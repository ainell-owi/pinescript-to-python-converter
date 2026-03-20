"""
Selector — Auto-select the highest-scoring evaluated strategy for conversion.
"""

import logging
from pathlib import Path

from src.pipeline import (
    ARCHIVE_SCORE_THRESHOLD,
    MAX_SKIP_COUNT,
    _EXCLUDED_PINE_FILES,
    _div,
    _verdict,
)

logger = logging.getLogger("runner")


def auto_select_strategy(registry: dict) -> tuple[str | None, dict | None]:
    """
    Select the highest-scoring evaluated strategy for conversion.

    Also increments skip_count for non-selected strategies.
    If no evaluated strategies exist, attempts to recycle from archive.

    Returns (key, record) or (None, None).
    """
    evaluated = {
        k: v for k, v in registry.items()
        if v["status"] in ("evaluated", "failed")
        and k not in _EXCLUDED_PINE_FILES
        and Path(v["file_path"]).exists()
    }

    # Fallback: recycle from archive if no evaluated candidates
    if not evaluated:
        evaluated = _recycle_from_archive(registry)
        if not evaluated:
            return None, None

    ranked = sorted(
        evaluated.items(),
        key=lambda kv: kv[1].get("btc_score", 0) + kv[1].get("project_score", 0),
        reverse=True,
    )

    print(f"\n{'═' * 70}")
    print("  STRATEGY ANALYSIS REPORT")
    print(f"{'═' * 70}")
    print(f"  {'#':<4} {'Strategy':<40} {'BTC':>3} {'Proj':>4}  Verdict")
    print(_div())
    for i, (key, rec) in enumerate(ranked, 1):
        btc  = rec.get("btc_score", 0)
        proj = rec.get("project_score", 0)
        name = key.replace(".pine", "")[:39]
        failed_tag = "  [RETRY]" if rec.get("status") == "failed" else ""
        skip_tag = f"  (skipped {rec.get('skip_count', 0)}x)" if rec.get("skip_count", 0) > 0 else ""
        print(
            f"  [{i}] {name:<40} "
            f"{'*' * btc:>3} {'*' * proj:>4}  {_verdict(btc, proj)}{failed_tag}{skip_tag}"
        )
    print(_div())

    chosen_key, chosen_rec = ranked[0]
    print(f"\n  Auto-selected : {chosen_key}  (highest combined score)")
    print(f"  Reason        : {chosen_rec.get('recommendation_reason', 'N/A')}\n")

    # Increment skip_count for all non-selected evaluated strategies
    for key, rec in registry.items():
        if key != chosen_key and rec["status"] == "evaluated":
            rec["skip_count"] = rec.get("skip_count", 0) + 1

    return chosen_key, chosen_rec


def _recycle_from_archive(registry: dict) -> dict:
    """
    Find recyclable strategies in the archive (previously OK-scored).

    Resets their status to 'evaluated' and skip_count to 0.
    Returns the subset that was recycled (may be empty).
    """
    archived = {
        k: v for k, v in registry.items()
        if v["status"] == "archived"
        and v.get("btc_score", 0) + v.get("project_score", 0) >= ARCHIVE_SCORE_THRESHOLD
        and Path(v["file_path"]).exists()
    }

    if not archived:
        return {}

    recycled_count = 0
    for key, rec in archived.items():
        rec["status"] = "evaluated"
        rec["skip_count"] = 0
        logger.info(f"Recycled from archive: {key}")
        recycled_count += 1

    if recycled_count:
        print(f"  Recycled {recycled_count} strategy(ies) from archive.")

    return archived