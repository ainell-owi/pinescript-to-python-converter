"""
Evaluator — Spawns isolated AI agent processes to score PineScript strategies.
"""

import json
import logging
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.pipeline import SUBPROCESS_ENV, _div, _verdict
from src.pipeline.registry import _now_iso, save_registry

logger = logging.getLogger("runner")


# ---------------------------------------------------------------------------
# Sidecar metadata schema (type-safe dataclass to prevent KeyError/TypeError
# on malformed or partially-scraped JSON sidecars)
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    total_trades:     Optional[int]   = None
    profit_factor:    Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    sharpe_ratio:     Optional[float] = None


@dataclass
class StrategyMetadata:
    url:              str = ""
    description:      Optional[str] = None
    backtest_metrics: BacktestMetrics = field(default_factory=BacktestMetrics)


def _load_strategy_metadata(pine_file: Path) -> Optional[StrategyMetadata]:
    """
    Load and validate the .meta.json sidecar that the scraper writes alongside
    each .pine file.  Returns None (and logs a warning) if the file is absent,
    unreadable, or structurally invalid.  Individual missing inner fields are
    tolerated — they default to None in the dataclass.
    """
    sidecar = pine_file.with_suffix(".meta.json")
    if not sidecar.exists():
        return None
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Sidecar root is not a JSON object.")

        bm_raw = raw.get("backtest_metrics") or {}
        if not isinstance(bm_raw, dict):
            bm_raw = {}

        def _int_or_none(v) -> Optional[int]:
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def _float_or_none(v) -> Optional[float]:
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        bm = BacktestMetrics(
            total_trades=_int_or_none(bm_raw.get("total_trades")),
            profit_factor=_float_or_none(bm_raw.get("profit_factor")),
            max_drawdown_pct=_float_or_none(bm_raw.get("max_drawdown_pct")),
            sharpe_ratio=_float_or_none(bm_raw.get("sharpe_ratio")),
        )
        desc = raw.get("description")
        url  = raw.get("url", "")
        return StrategyMetadata(
            url=str(url) if url else "",
            description=str(desc).strip() if desc else None,
            backtest_metrics=bm,
        )
    except Exception as exc:
        logger.warning(f"Could not load metadata sidecar {sidecar.name}: {exc}")
        return None


def _format_metadata_block(meta: StrategyMetadata) -> str:
    """
    Render a StrategyMetadata object as a plain-text BACKTEST_METADATA block
    suitable for injection into the selector agent prompt.
    """
    bm = meta.backtest_metrics
    lines = [
        "--- BACKTEST_METADATA (scraped from TradingView) ---",
        f"total_trades:     {bm.total_trades if bm.total_trades is not None else 'N/A'}",
        f"profit_factor:    {bm.profit_factor if bm.profit_factor is not None else 'N/A'}",
        f"max_drawdown_pct: {bm.max_drawdown_pct if bm.max_drawdown_pct is not None else 'N/A'}",
        f"sharpe_ratio:     {bm.sharpe_ratio if bm.sharpe_ratio is not None else 'N/A'}",
    ]
    if meta.description:
        # Truncate very long descriptions to avoid bloating the context window.
        desc = meta.description[:1500]
        if len(meta.description) > 1500:
            desc += " [truncated]"
        lines.append(f"author_description: |\n  {desc.replace(chr(10), chr(10) + '  ')}")
    lines.append("--- END BACKTEST_METADATA ---")
    return "\n".join(lines)


def _normalize_timeframe(value: str | None) -> str:
    """Normalize selector timeframe output to the lowercase forms used downstream."""
    if not value:
        return "1h"

    cleaned = str(value).strip()
    mapping = {
        "1": "1m",
        "3": "3m",
        "5": "5m",
        "15": "15m",
        "30": "30m",
        "60": "1h",
        "120": "2h",
        "240": "4h",
        "D": "1d",
        "1D": "1d",
        "d": "1d",
        "1d": "1d",
        "H": "1h",
        "1H": "1h",
        "h": "1h",
        "1h": "1h",
    }
    return mapping.get(cleaned, cleaned.lower())


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

    meta_block = ""
    strategy_meta = _load_strategy_metadata(pine_file)
    if strategy_meta is not None:
        meta_block = _format_metadata_block(strategy_meta) + "\n\n"
        logger.info(
            f"Metadata sidecar loaded for {pine_file.name}: "
            f"trades={strategy_meta.backtest_metrics.total_trades} "
            f"pf={strategy_meta.backtest_metrics.profit_factor} "
            f"dd={strategy_meta.backtest_metrics.max_drawdown_pct}% "
            f"sharpe={strategy_meta.backtest_metrics.sharpe_ratio}"
        )

    prompt = (
        f"Evaluate this PineScript strategy. File: {pine_file.name}\n\n"
        f"{meta_block}"
        f"{raw}\n\n"
        "Output ONLY a raw JSON object matching this exact schema — no markdown, no extra fields:\n"
        '{"pine_metadata": {"name": "...", "safe_name": "...", "timeframe": "...", "lookback_bars": 0}, '
        '"category": "Trend", "btc_score": 0, "project_score": 0, '
        '"recommendation_reason": "..."}'
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

        required = {"pine_metadata", "category", "btc_score", "project_score"}
        if result and required.issubset(result):
            btc  = result["btc_score"]
            proj = result["project_score"]
            meta = result["pine_metadata"]
            category = result["category"]
            if not meta.get("safe_name"):
                raw_name = meta.get("name", key.replace(".pine", ""))
                meta["safe_name"] = "".join(
                    c if c.isalnum() else "_" for c in raw_name
                ).strip("_")
            meta["timeframe"] = _normalize_timeframe(meta.get("timeframe"))
            if not meta.get("lookback_bars"):
                meta["lookback_bars"] = 100
            registry[key].update({
                "status":                "evaluated",
                "evaluated_at":          _now_iso(),
                "pine_metadata":         meta,
                "category":              category,
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
                "category":              "Other",
                "btc_score":             0,
                "project_score":         0,
                "recommendation_reason": "Evaluation failed — scored 0.",
            })
            print("   FAILED (scored 0)")
            logger.warning(f"Evaluation failed for {key}")

        save_registry(registry)   # crash-safe: save after each file

    print(_div())
    return registry