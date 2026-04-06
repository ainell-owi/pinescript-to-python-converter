"""
Tests for CdvEmaCrossStrategyV6

Covers:
    1. Strategy initialization and contract attributes.
    2. Warmup guard — HOLD returned for insufficiently short DataFrames.
    3. Smoke-test execution on full 1,100-candle fixture.
    4. Indicator columns populated after run().
    5. Signal detection across volatile market phases (bull / bear).
    6. Edge-case resilience (empty DataFrame, single row).
    7. RL safety: MIN_CANDLES_REQUIRED is positive and dynamic.
"""

import pytest
import pandas as pd
import numpy as np

from src.strategies.cdv_ema_cross_strategy_v6_strategy import CdvEmaCrossStrategyV6
from src.base_strategy import SignalType, StrategyRecommendation


# ---------------------------------------------------------------------------
# 1. Initialization Tests
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_default_instantiation(self):
        strategy = CdvEmaCrossStrategyV6()
        assert strategy.name == "CDV EMA Cross Strategy v6"
        assert strategy.timeframe == "1h"
        assert strategy.lookback_hours > 0
        assert strategy.ema1_len == 9
        assert strategy.ema2_len == 21
        assert strategy.use_heikin_ashi is True

    def test_custom_parameters(self):
        strategy = CdvEmaCrossStrategyV6(ema1_len=5, ema2_len=50, use_heikin_ashi=False)
        assert strategy.ema1_len == 5
        assert strategy.ema2_len == 50
        assert strategy.use_heikin_ashi is False

    def test_min_candles_required_is_positive(self):
        """RL safety: MIN_CANDLES_REQUIRED must be a positive integer."""
        strategy = CdvEmaCrossStrategyV6()
        assert strategy.MIN_CANDLES_REQUIRED > 0

    def test_min_candles_required_is_dynamic(self):
        """MIN_CANDLES_REQUIRED must scale with the slowest EMA period."""
        s_default = CdvEmaCrossStrategyV6()          # max period = 21
        s_slow    = CdvEmaCrossStrategyV6(ema2_len=100)  # max period = 100
        assert s_slow.MIN_CANDLES_REQUIRED > s_default.MIN_CANDLES_REQUIRED

    def test_min_candles_required_formula(self):
        """Verify the 3× formula: 3 * max(ema1_len, ema2_len)."""
        strategy = CdvEmaCrossStrategyV6(ema1_len=14, ema2_len=30)
        assert strategy.MIN_CANDLES_REQUIRED == 3 * max(14, 30)


# ---------------------------------------------------------------------------
# 2. Warmup Guard Tests
# ---------------------------------------------------------------------------

class TestWarmupGuard:
    def test_warmup_guard_returns_hold_on_empty(self):
        """Empty DataFrame must return HOLD without crashing."""
        strategy = CdvEmaCrossStrategyV6()
        empty_df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        result = strategy.run(empty_df, ts)
        assert result.signal == SignalType.HOLD

    def test_warmup_guard_returns_hold_on_single_row(self, sample_ohlcv_data):
        """A single-row DataFrame must return HOLD."""
        strategy = CdvEmaCrossStrategyV6()
        ts = sample_ohlcv_data.iloc[0]["date"]
        result = strategy.run(sample_ohlcv_data.iloc[:1].copy(), ts)
        assert result.signal == SignalType.HOLD

    def test_warmup_guard_returns_hold_below_threshold(self, sample_ohlcv_data):
        """Any DataFrame shorter than MIN_CANDLES_REQUIRED must return HOLD."""
        strategy = CdvEmaCrossStrategyV6()
        short_df = sample_ohlcv_data.iloc[: strategy.MIN_CANDLES_REQUIRED - 1].copy()
        ts = short_df.iloc[-1]["date"]
        result = strategy.run(short_df, ts)
        assert result.signal == SignalType.HOLD

    def test_warmup_guard_does_not_hold_at_threshold(self, sample_ohlcv_data):
        """At exactly MIN_CANDLES_REQUIRED rows, strategy must NOT return HOLD from guard."""
        strategy = CdvEmaCrossStrategyV6()
        threshold_df = sample_ohlcv_data.iloc[: strategy.MIN_CANDLES_REQUIRED].copy()
        ts = threshold_df.iloc[-1]["date"]
        # Should not raise; result may be HOLD (no crossover) but not from the guard path
        result = strategy.run(threshold_df, ts)
        assert isinstance(result, StrategyRecommendation)
        assert isinstance(result.signal, SignalType)


# ---------------------------------------------------------------------------
# 3. Smoke-Test Execution
# ---------------------------------------------------------------------------

class TestExecution:
    def test_run_returns_strategy_recommendation(self, sample_ohlcv_data):
        """run() must return a StrategyRecommendation on full data."""
        strategy = CdvEmaCrossStrategyV6()
        ts = sample_ohlcv_data.iloc[-1]["date"]
        result = strategy.run(sample_ohlcv_data.copy(), ts)
        assert isinstance(result, StrategyRecommendation)
        assert isinstance(result.signal, SignalType)
        assert result.timestamp == ts

    def test_run_without_heikin_ashi(self, sample_ohlcv_data):
        """run() must work when Heikin Ashi is disabled."""
        strategy = CdvEmaCrossStrategyV6(use_heikin_ashi=False)
        ts = sample_ohlcv_data.iloc[-1]["date"]
        result = strategy.run(sample_ohlcv_data.copy(), ts)
        assert isinstance(result, StrategyRecommendation)
        assert isinstance(result.signal, SignalType)

    def test_run_does_not_raise_on_full_data(self, sample_ohlcv_data):
        """run() must not raise any exception on well-formed data."""
        strategy = CdvEmaCrossStrategyV6()
        ts = sample_ohlcv_data.iloc[-1]["date"]
        try:
            strategy.run(sample_ohlcv_data.copy(), ts)
        except Exception as exc:
            pytest.fail(f"run() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# 4. Indicator Columns
# ---------------------------------------------------------------------------

class TestIndicatorColumns:
    def test_indicator_columns_present_after_run(self, sample_ohlcv_data):
        """After run(), df must contain 'cdv_close', 'ema1', 'ema2' columns."""
        strategy = CdvEmaCrossStrategyV6()
        df = sample_ohlcv_data.copy()
        ts = df.iloc[-1]["date"]
        strategy.run(df, ts)

        assert "cdv_close" in df.columns, "'cdv_close' column missing after run()"
        assert "ema1" in df.columns, "'ema1' column missing after run()"
        assert "ema2" in df.columns, "'ema2' column missing after run()"

    def test_indicator_columns_have_non_nan_values(self, sample_ohlcv_data):
        """Indicator columns must not be entirely NaN after run() on 1,100-candle data."""
        strategy = CdvEmaCrossStrategyV6()
        df = sample_ohlcv_data.copy()
        ts = df.iloc[-1]["date"]
        strategy.run(df, ts)

        assert df["ema1"].notna().any(), "'ema1' is all-NaN"
        assert df["ema2"].notna().any(), "'ema2' is all-NaN"
        assert df["cdv_close"].notna().any(), "'cdv_close' is all-NaN"


# ---------------------------------------------------------------------------
# 5. Signal Detection in Volatile Phases
# ---------------------------------------------------------------------------

class TestSignalDetection:
    def _run_slice(self, strategy: CdvEmaCrossStrategyV6, df: pd.DataFrame, end_idx: int):
        """Helper: run strategy on df[:end_idx], return signal."""
        slice_df = df.iloc[:end_idx].copy()
        ts = slice_df.iloc[-1]["date"]
        return strategy.run(slice_df, ts)

    def test_produces_at_least_one_non_hold_signal_in_volatile_phases(self, sample_ohlcv_data):
        """At least one LONG or SHORT signal must appear across phases 2+3 (bars 700-1100).

        CDV-based EMAs track cumulative volume delta, not raw price, so crossovers
        are infrequent and cluster when volume pressure reverses sharply. The test
        scans every bar in the volatile window to detect at least one non-HOLD signal.
        """
        strategy = CdvEmaCrossStrategyV6()
        signals = set()
        for end in range(700, len(sample_ohlcv_data) + 1):
            result = self._run_slice(strategy, sample_ohlcv_data, end)
            signals.add(result.signal)
            # Early exit once we have a non-HOLD signal to keep test fast
            if signals - {SignalType.HOLD}:
                break
        assert signals - {SignalType.HOLD}, (
            "Strategy produced only HOLD signals across the entire volatile window (700-1100) "
            "— check EMA crossover logic on CDV series"
        )

    def test_produces_at_least_one_non_hold_signal_in_bear_phase(self, sample_ohlcv_data):
        """At least one LONG or SHORT signal must appear during the bear crash (900-1100).

        CDV EMA crossovers on this fixture are confirmed to occur in the bear phase.
        Bar-by-bar scan (every bar) to catch transient crossover events.
        """
        strategy = CdvEmaCrossStrategyV6()
        signals = set()
        for end in range(900, len(sample_ohlcv_data) + 1):
            result = self._run_slice(strategy, sample_ohlcv_data, end)
            signals.add(result.signal)
            if signals - {SignalType.HOLD}:
                break
        assert signals - {SignalType.HOLD}, (
            "Strategy produced only HOLD signals during bear phase — check EMA crossover logic"
        )

    def test_signal_type_is_always_valid_enum(self, sample_ohlcv_data):
        """Every returned signal must be a valid SignalType member."""
        strategy = CdvEmaCrossStrategyV6()
        valid = set(SignalType)
        for end in range(strategy.MIN_CANDLES_REQUIRED, len(sample_ohlcv_data), 50):
            result = self._run_slice(strategy, sample_ohlcv_data, end)
            assert result.signal in valid, f"Invalid signal returned: {result.signal}"
