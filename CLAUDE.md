# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

An AI-driven pipeline that converts TradingView Pine Script v5 strategies into vectorized Python strategies. The pipeline uses a multi-agent architecture (via Claude subagents) to transpile, validate, test, and submit a GitHub PR for human review.

## Commands

```bash
# Run the full pipeline (requires Claude CLI in PATH)
python main.py

# Run tests (Linux/macOS)
pytest tests/strategies/ -v

# Run tests (Windows — use the venv interpreter)
.venv/Scripts/python.exe -m pytest tests/strategies/ -v

# Run a single test file
pytest tests/strategies/test_<name>.py -v

# Run integration smoke tests (importability)
pytest tests/integrations/ -v
```

**Dependencies:** TA-Lib requires the C library to be installed separately before `pip install -r requirements.txt`. On Windows use a pre-built wheel; on Linux build from source (see `ci.yml` for the build steps). `ccxt` is also required — it is used in `src/utils/timeframes.py` for candle boundary alignment (`ccxt.Exchange.round_timeframe`).

## Pipeline Flow

`main.py` is the single entry point. Pipeline logic is split into `src/pipeline/` modules. It orchestrates these phases:

1. **Scrape** — If `input/` has fewer than 6 `.pine` files, `TradingViewScraper` (Selenium) auto-downloads public strategies from TradingView.
2. **Evaluate** — Spawns `claude -p --agent strategy_selector` as a subprocess for each new `.pine` file. Returns JSON metadata including `category`, `btc_score`, and `project_score` (0–5 each score). Results are persisted to `data/strategies_registry.json`, and category saturation is tracked in `data/category_counts.json`.
3. **Select** — Auto-selects the highest-scoring strategy. Increments `skip_count` for non-selected strategies (archived after 2 skips). Recycles from archive when no candidates remain.
4. **Convert** — Spawns `claude -p --agent orchestrator` which delegates sequentially to four sub-agents:
   - **Transpiler** → writes `src/strategies/<name>_strategy.py`
   - **Validator** → static analysis (lookahead bias, forbidden functions, class contract)
   - **Test Generator** → writes `tests/strategies/test_<name>.py` and runs pytest
   - **Integration** → git branch + GitHub MCP PR
5. **Archive** — Low-scoring strategies (combined score < 4) or stale strategies (skipped 2+ times) are moved to `archive/`.

**Success detection:** `main.py` requires `INTEGRATION_PASS` or `INTEGRATION_FALLBACK` tokens in orchestrator stdout AND verifies artifact files exist on disk. Exit code 0 alone is not sufficient.

Agent subprocesses use `--dangerously-skip-permissions` so tool-approval prompts don't block them. The `CLAUDECODE` env var is stripped so nested `claude` calls are allowed.

## Agent System

All agent definitions live in `.claude/agents/`. The orchestrator reads `.claude/skills/CONVERSION_FLOW/SKILL.md` as its master playbook.

**Required logging protocol:** When the Orchestrator hands off to a sub-agent it must print `[SYSTEM] Handing over to: <AgentName>`. On return: `[SYSTEM] Control returned to: Orchestrator`. `main.py` parses these strings to prefix log lines with the active agent name.

**Orchestrator error handling:** The orchestrator runs in non-interactive mode (`-p`). It auto-fixes structural issues (warmup guards, imports, naming) with bounded retries (max 2 cycles). Trading logic failures abort immediately — no auto-fix.

**Strategy Selector output contract:** Must be raw JSON only (no markdown fences). Schema: `{ pine_metadata, category, btc_score, project_score, recommendation_reason }`.
The selector must reject long-only/short-only strategies, excessive warmups above 1000 bars, trade-management-driven systems without real entry logic, heavy `O(N^2)` scans, and oversaturated categories.

## Strategy Contract (`BaseStrategy`)

All generated strategies must inherit from `src/base_strategy.py`:

```python
class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="...", description="...", timeframe="15m", lookback_hours=48)

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        # Must return StrategyRecommendation(signal=SignalType.LONG/SHORT/FLAT/HOLD, timestamp=timestamp)
```

- Strategy receives a full OHLCV DataFrame + a UTC `timestamp`. It returns only a signal — no position sizing, fees, or broker logic.
- All columns are lowercase: `open`, `high`, `low`, `close`, `volume`, `date` (UTC-aware).

## Anti-Lookahead Bias Rules

1. All `shift()` operations must use positive integers (backward-looking only).
2. Multi-timeframe logic **must** use `src/utils/resampling.py` (import as `from src.utils.resampling import ...`):
   - `resample_to_interval(df, interval)` — upsample base data to higher TF
   - `resampled_merge(original, resampled)` — merge back with `ffill`, shifting resampled timestamps to prevent future-peeking
3. **Forbidden:** `df.resample()` directly inside a strategy, `future_shift`, `barstate.isrealtime`.

## Allowed Libraries in Generated Strategies

Strategies may only import from: `pandas`, `numpy`, `talib`, and `src.*`. No other third-party libraries. If a TA-Lib indicator is missing, implement it in pure Pandas/NumPy inside the strategy file.

## Missing TA-Lib Indicators

Implement in pure Pandas when TA-Lib lacks them:
- **RMA:** `df.ewm(alpha=1/length, adjust=False).mean()`
- **Supertrend:** Custom ATR band logic
- **VWAP:** `(cumulative price×volume) / cumulative volume`

## Strategy Naming Convention

The pipeline's `safe_name()` converts the Pine filename to snake_case for the Python module name (e.g., `My-Strategy.pine` → `my_strategy`). Generated files follow this pattern:
- `src/strategies/<safe_name>_strategy.py`
- `tests/strategies/test_<safe_name>.py`

## Registry State Machine

`data/strategies_registry.json` tracks each `.pine` file through these lifecycle states:

```
new → evaluated → selected → converted → archived
                           ↘ failed (retry via menu)
```

Each entry stores the file path, category, scores (`btc_score`, `project_score`), `recommendation_reason`, `skip_count`, and current `status`. Strategies are archived after 2 skips regardless of score.

## Test Fixture

`tests/conftest.py` provides a single shared fixture `sample_ohlcv_data`:
- 1,100 candles at 15m intervals (seed=42, fully deterministic)
- Phase 0 (0–600): Warmup — flat at 10,000 for indicator convergence (EMA-200 etc. stabilise here)
- Phase 1 (600–700): Sideways / Accumulation (low volatility)
- Phase 2 (700–900): Bull Run (10,000 → 12,000)
- Phase 3 (900–1,100): Bear Crash (12,000 → 9,000)

All generated strategy tests must use this fixture. The warmup phase ensures `min_bars` guards are exercised; the 3 market-regime phases ensure signal logic is tested across varied conditions.

## Key Files

| File | Purpose |
|---|---|
| `main.py` | Pipeline entry point (replaces old `runner.py`) |
| `src/pipeline/` | Pipeline modules: `registry.py`, `evaluator.py`, `selector.py`, `orchestrator.py`, `archiver.py`, `scraper.py` |
| `data/strategies_registry.json` | State tracker for all `.pine` files (new → evaluated → selected → converted → archived) |
| `data/category_counts.json` | Running counts of accepted strategy categories used to penalize overrepresented categories |
| `src/base_strategy.py` | Abstract base class all strategies must inherit |
| `src/utils/resampling.py` | MTF utilities (`resample_to_interval`, `resampled_merge`) |
| `src/utils/timeframes.py` | Timeframe helpers: `timeframe_to_minutes`, `timeframe_to_cron`, candle arithmetic |
| `src/utils/tv_scraper.py` | Selenium scraper for TradingView public strategies |
| `tests/conftest.py` | `sample_ohlcv_data` fixture (1,100 candles: warmup + sideways/bull/bear phases) |
| `tests/integrations/` | Importability smoke tests for all generated strategies |
| `.claude/agents/` | Agent persona definitions |
| `.github/workflows/ci.yml` | CI pipeline definition — fully commented out (disabled), runs `pytest tests/strategies/` |

## Runtime Directories

| Directory | Purpose |
|---|---|
| `output/<safe_name>/<timestamp>/` | Per-run conversion snapshot: `strategy.py`, `test_strategy.py`, `run.log`, agent decision logs |
| `logs/<strategy_name>/<timestamp>/` | Orchestrator process logs: `run.log` (DEBUG) and `errors.log` (ERROR only) |
| `archive/` | Low-scoring `.pine` files moved here after a run (combined score < 4) |
| `data/seen_urls.json` | Persisted set of scraped TradingView URLs — prevents re-downloading across runs |

## Agent Decision Logs (Output Snapshot)

Each conversion run writes agent decision logs to `output/<safe_name>/<timestamp>/`:

- `agent_transpiler.md` — mapping table, warnings, files written
- `agent_validator.md` — checklist results
- `agent_test_generator.md` — test design decisions
- `agent_integration.md` — branch name, PR URL, pass/fallback status

The Integration Agent must output exactly `INTEGRATION_PASS` or `INTEGRATION_FALLBACK` so `main.py` can parse the result. Without one of these tokens, the pipeline treats the run as a failure regardless of exit code.

## `/convert` Slash Command

Drop a `.pine` file in `input/` then run:
```
/convert input/MyStrategy.pine
```
This invokes the orchestrator sub-agent directly, skipping the evaluation/selection phases of `main.py`.
