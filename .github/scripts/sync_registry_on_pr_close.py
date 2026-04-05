"""
GitHub Actions helper: when a PR is closed without merging, mark the matching
registry entry so archive recycling will not resurrect it.

Expects GITHUB_EVENT_PATH (set by Actions) to point at the pull_request webhook JSON.
Integration branches must be feat/<safe_name> (see .claude/agents/integration.md).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, UTC
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = REPO_ROOT / "data" / "strategies_registry.json"


def _load_event() -> dict:
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path or not Path(path).is_file():
        print(
            f"GITHUB_EVENT_PATH missing or not a file (value={path!r}); nothing to do.",
            file=sys.stderr,
        )
        sys.exit(0)
    # utf-8-sig strips BOM if present (Windows editors / some runners).
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main() -> None:
    event = _load_event()
    pr = event.get("pull_request") or {}
    merged = pr.get("merged")
    if merged is True:
        print("PR was merged; registry recycle flags unchanged by this workflow.")
        sys.exit(0)

    head = pr.get("head") or {}
    head_ref = head.get("ref") or ""
    if not head_ref.startswith("feat/"):
        print(f"Head ref {head_ref!r} is not feat/<safe_name>; skipping.")
        sys.exit(0)

    safe_name = head_ref.removeprefix("feat/").strip()
    if not safe_name:
        print("Empty safe_name after feat/; skipping.")
        sys.exit(0)

    if not REGISTRY_PATH.is_file():
        print(f"Registry not found at {REGISTRY_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        # utf-8-sig: same BOM tolerance as for the webhook payload file.
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {REGISTRY_PATH}: {e}", file=sys.stderr)
        sys.exit(1)
    now = datetime.now(UTC).isoformat()
    pr_number = pr.get("number")
    matched_key: str | None = None

    for key, rec in registry.items():
        meta = rec.get("pine_metadata") or {}
        if str(meta.get("safe_name") or "") == safe_name:
            matched_key = key
            break

    if not matched_key:
        print(f"No registry entry with pine_metadata.safe_name == {safe_name!r}; skipping.")
        sys.exit(0)

    rec = registry[matched_key]
    changed = False

    if rec.get("recycle_eligible") is not False:
        rec["recycle_eligible"] = False
        changed = True
    if pr_number is not None and rec.get("github_pr_number") != pr_number:
        rec["github_pr_number"] = pr_number
        changed = True
    if rec.get("github_pr_state") != "CLOSED":
        rec["github_pr_state"] = "CLOSED"
        changed = True
    if not rec.get("github_pr_closed_without_merge_at"):
        rec["github_pr_closed_without_merge_at"] = now
        changed = True
    note = "PR closed without merge; excluded from archive recycling (via GitHub Actions)."
    if rec.get("github_pr_rejection_note") != note:
        rec["github_pr_rejection_note"] = note
        changed = True

    if not changed:
        print(f"Registry already up to date for {matched_key}.")
        sys.exit(0)

    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Updated registry entry {matched_key!r} for closed PR #{pr_number} (feat/{safe_name}).")


if __name__ == "__main__":
    main()
