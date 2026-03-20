"""
Evaluator — Spawns isolated AI agent processes to score PineScript strategies.
"""

import json
import logging
import subprocess
import threading
from pathlib import Path

from src.pipeline import SUBPROCESS_ENV, _div, _verdict
from src.pipeline.registry import _now_iso, save_registry

logger = logging.getLogger("runner")


def _parse_json_from_output(raw: str) -> dict:
    """
    Defensively parse a JSON object from raw LLM text output.

    Strips Markdown fences and slices from first '{' to last '}'.
    """
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    start   = cleaned.find("{")
    end     = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in agent output")
    return json.loads(cleaned[start:end])


def evaluate_strategy(pine_file: Path) -> dict | None:
    """
    Spawn an isolated strategy_selector agent to evaluate a single .pine file.

    Returns the parsed JSON evaluation dict, or None on failure/timeout.
    """
    try:
        raw = pine_file.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning(f"Cannot read {pine_file}: {e}")
        return None

    logger.info(f"Evaluating: {pine_file.name} ({len(raw)} chars)")
    logger.info(f"CLAUDE input (file path sent to agent):\n{pine_file.resolve()}")

    prompt = (
        f"Evaluate this PineScript strategy. File: {pine_file.name}\n\n"
        f"{raw}\n\n"
        "Output ONLY a raw JSON object matching this exact schema — no markdown, no extra fields:\n"
        '{"pine_metadata": {"name": "...", "safe_name": "...", "timeframe": "...", "lookback_bars": 0}, '
        '"btc_score": 0, "project_score": 0, "recommendation_reason": "..."}'
    )
    command = [
        "claude", "-p",
        "--agent", "strategy_selector",
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        prompt,
    ]

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=SUBPROCESS_ENV,
        )
    except FileNotFoundError:
        logger.error("'claude' command not found for selector.")
        print("'claude' CLI not found.", flush=True)
        return None

    collected: list[str] = []
    killed_by_watchdog = threading.Event()

    def _kill_on_timeout():
        killed_by_watchdog.set()
        try:
            process.kill()
        except OSError:
            pass

    watchdog = threading.Timer(180, _kill_on_timeout)
    try:
        watchdog.start()
        for line in process.stdout:
            stripped = line.rstrip()
            if stripped:
                print(f"    {stripped}", flush=True)
                logger.info(f"CLAUDE [SELECTOR]: {stripped}")
            collected.append(line)
        process.wait()
    except KeyboardInterrupt:
        process.kill()
        raise
    except Exception:
        try:
            process.kill()
        except OSError:
            pass
        process.wait()
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.terminate()

    if killed_by_watchdog.is_set():
        logger.warning(f"Selector timed out (180s) for {pine_file.name}")
        print("TIMED OUT", flush=True)
        return None

    full_output = "".join(collected)
    if process.returncode != 0:
        logger.warning(f"Selector non-zero exit for {pine_file.name}")

    try:
        return _parse_json_from_output(full_output)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"JSON parse error for {pine_file.name}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error evaluating {pine_file.name}: {e}")

    return None


def run_evaluations(registry: dict) -> dict:
    """Evaluate all pending (new or zero-scored) strategies."""
    new_entries = [(k, v) for k, v in registry.items() if v["status"] == "new"]
    retry_entries = [
        (k, v) for k, v in registry.items()
        if v["status"] == "evaluated"
        and v.get("btc_score", 0) == 0
        and v.get("project_score", 0) == 0
    ]
    to_evaluate = new_entries + retry_entries
    if not to_evaluate:
        return registry

    print(f"\n Evaluating {len(to_evaluate)} strategy file(s)...")
    print(_div())

    for key, rec in to_evaluate:
        print(f"    {key} ... ", end="", flush=True)
        result = evaluate_strategy(Path(rec["file_path"]))

        required = {"pine_metadata", "btc_score", "project_score"}
        if result and required.issubset(result):
            btc  = result["btc_score"]
            proj = result["project_score"]
            meta = result["pine_metadata"]
            if not meta.get("safe_name"):
                raw_name = meta.get("name", key.replace(".pine", ""))
                meta["safe_name"] = "".join(
                    c if c.isalnum() else "_" for c in raw_name
                ).strip("_")
            if not meta.get("timeframe"):
                meta["timeframe"] = "1h"
            if not meta.get("lookback_bars"):
                meta["lookback_bars"] = 100
            registry[key].update({
                "status":                "evaluated",
                "evaluated_at":          _now_iso(),
                "pine_metadata":         meta,
                "btc_score":             btc,
                "project_score":         proj,
                "recommendation_reason": result.get("recommendation_reason", ""),
            })
            print(f"BTC: {'*' * btc}  Proj: {'*' * proj}  {_verdict(btc, proj)}")
        else:
            registry[key].update({
                "status":                "evaluated",
                "evaluated_at":          _now_iso(),
                "pine_metadata":         {
                    "name": key, "safe_name": "", "timeframe": "?", "lookback_bars": 0
                },
                "btc_score":             0,
                "project_score":         0,
                "recommendation_reason": "Evaluation failed — scored 0.",
            })
            print("   FAILED (scored 0)")
            logger.warning(f"Evaluation failed for {key}")

        save_registry(registry)   # crash-safe: save after each file

    print(_div())
    return registry