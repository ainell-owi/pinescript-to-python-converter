# Testing Standards & Fixtures

This rule applies when generating or modifying pytest files for strategies.

## 1. Naming & Execution
- **File Naming (CI/CD Constraint):** Test files MUST be named `tests/strategies/test_<safe_name>_strategy.py` (e.g., `test_kama_trend_strategy.py`). Do NOT use the suffix-only format `<safe_name>_strategy_test.py` — the `test_` prefix is mandatory for pytest discovery.
- **Running Tests:** `pytest tests/strategies/test_<safe_name>_strategy.py -v`

## 2. The `sample_ohlcv_data` Fixture
You MUST use the shared fixture defined in `tests/conftest.py`. DO NOT mock your own OHLCV data. 
The fixture contains 1,100 candles at 15m intervals with 4 distinct phases:
- **Phase 0 (0–600):** Warmup phase. Flat at 10,000 for indicator convergence (tests the `MIN_CANDLES_REQUIRED` guard).
- **Phase 1 (600–700):** Sideways / Accumulation (low volatility).
- **Phase 2 (700–900):** Bull Run (10,000 → 12,000).
- **Phase 3 (900–1,100):** Bear Crash (12,000 → 9,000).

## 3. Test Coverage Requirements
Your tests MUST verify:
1. The strategy returns `HOLD` during the warmup phase (Phase 0).
2. The strategy correctly identifies signals during the volatile phases (Phases 2 & 3).
3. The strategy handles edge cases (e.g., empty dataframe, all NaNs) gracefully without throwing raw exceptions.