"""
Archiver — Move low-scoring or stale strategies to archive/.

Also handles archive recycling when no fresh candidates remain.
"""

import logging
import shutil
from datetime import datetime, UTC
from pathlib import Path

from src.pipeline import ARCHIVE_DIR, ARCHIVE_SCORE_THRESHOLD, MAX_SKIP_COUNT

logger = logging.getLogger("runner")


def archive_remaining(registry: dict, selected_key: str) -> dict:
    """
    Archive strategies that are low-scoring OR have been skipped too many times.

    Strategies meeting the score threshold AND below the skip limit are left
    in input/ with 'evaluated' status for future runs.
    """
    ARCHIVE_DIR.mkdir(exist_ok=True)
    archived = 0

    for key, rec in registry.items():
        if key == selected_key:
            continue
        if rec["status"] not in ("new", "evaluated", "failed"):
            continue

        total = rec.get("btc_score", 0) + rec.get("project_score", 0)
        skip_count = rec.get("skip_count", 0)

        # Keep if score meets threshold AND not skipped too many times
        if total >= ARCHIVE_SCORE_THRESHOLD and skip_count < MAX_SKIP_COUNT:
            logger.info(
                f"Keeping '{key}' in input/ "
                f"(total={total} >= {ARCHIVE_SCORE_THRESHOLD}, skips={skip_count})"
            )
            continue

        reason = (
            f"skip_count={skip_count} >= {MAX_SKIP_COUNT}"
            if skip_count >= MAX_SKIP_COUNT
            else f"total={total} < {ARCHIVE_SCORE_THRESHOLD}"
        )

        src = Path(rec["file_path"])
        if src.exists():
            dest = ARCHIVE_DIR / src.name
            shutil.move(str(src), dest)
            rec["file_path"] = str(dest)
            logger.info(f"Archived: {key} → {dest} ({reason})")

        rec["status"]      = "archived"
        rec["archived_at"] = datetime.now(UTC).isoformat()
        archived += 1

    if archived:
        print(f"  Archived {archived} file(s) → archive/")
    return registry