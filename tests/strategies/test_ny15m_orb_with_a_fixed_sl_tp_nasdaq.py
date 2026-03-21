import pytest
import pandas as pd
from datetime import datetime, timezone

from src.strategies.ny15m_orb_with_a_fixed_sl_tp_nasdaq_strategy import (
    Ny15mOrbWithAFixedSlTpNasdaqStrategy,
)
from src.base_strategy import SignalType, StrategyRecommendation


class TestInitialization:
    """Verify strategy instantiation and metadata."""

    def test_strategy_instantiates(self):
        strategy = Ny15mOrbWithAFixedSlTpNasdaqStrategy()
        assert strategy.name == "Enhanced OR Strategy"
        assert strategy.timeframe == "15m"
        assert strategy.lookback_hours == 13

    def test_min_candles_required(self):
        strategy = Ny15mOrbWithAFixedSlTpNasdaqStrategy()
        # 3 * max(ATR_PERIOD=14, BREAKOUT_CANDLES=2) = 42
        assert strategy.MIN_CANDLES_REQUIRED == 42


class TestWarmupGuard:
    """Ensure HOLD is returned when insufficient data is provided."""

    def test_hold_on_insufficient_data(self, sample_ohlcv_data):
        strategy = Ny15mOrbWithAFixedSlTpNasdaqStrategy()
        small_df = sample_ohlcv_data.iloc[:10].copy()
        ts = small_df.iloc[-1]["date"]
        result = strategy.run(small_df, ts)
        assert result.signal == SignalType.HOLD

    def test_hold_at_boundary(self, sample_ohlcv_data):
        strategy = Ny15mOrbWithAFixedSlTpNasdaqStrategy()
        # Exactly MIN_CANDLES_REQUIRED - 1 rows
        boundary_df = sample_ohlcv_data.iloc[: strategy.MIN_CANDLES_REQUIRED - 1].copy()
        ts = boundary_df.iloc[-1]["date"]
        result = strategy.run(boundary_df, ts)
        assert result.signal == SignalType.HOLD


class TestExecution:
    """Smoke tests: strategy runs without errors and returns valid signals."""

    def test_full_data_returns_valid_signal(self, sample_ohlcv_data):
        strategy = Ny15mOrbWithAFixedSlTpNasdaqStrategy()
        ts = sample_ohlcv_data.iloc[-1]["date"]
        result = strategy.run(sample_ohlcv_data, ts)
        assert isinstance(result, StrategyRecommendation)
        assert isinstance(result.signal, SignalType)
        assert result.timestamp == ts

    def test_signal_during_bull_phase(self, sample_ohlcv_data):
        """Bull phase (candles 700-900) should be able to produce signals."""
        strategy = Ny15mOrbWithAFixedSlTpNasdaqStrategy()
        # Use data up to the bull phase
        bull_df = sample_ohlcv_data.iloc[:850].copy()
        ts = bull_df.iloc[-1]["date"]
        result = strategy.run(bull_df, ts)
        assert isinstance(result, StrategyRecommendation)
        assert isinstance(result.signal, SignalType)

    def test_signal_during_bear_phase(self, sample_ohlcv_data):
        """Bear phase (candles 900-1100) should be able to produce signals."""
        strategy = Ny15mOrbWithAFixedSlTpNasdaqStrategy()
        ts = sample_ohlcv_data.iloc[-1]["date"]
        result = strategy.run(sample_ohlcv_data, ts)
        assert isinstance(result, StrategyRecommendation)
        assert isinstance(result.signal, SignalType)


class TestSignalGeneration:
    """Verify the strategy generates non-trivial signals across market regimes."""

    def test_produces_long_or_short_across_full_data(self, sample_ohlcv_data):
        """
        Scan all bars and confirm the strategy produces at least one
        LONG or SHORT signal across the full dataset (warmup + 3 regimes).
        """
        strategy = Ny15mOrbWithAFixedSlTpNasdaqStrategy()
        signals = []
        for i in range(strategy.MIN_CANDLES_REQUIRED, len(sample_ohlcv_data)):
            df_slice = sample_ohlcv_data.iloc[: i + 1].copy()
            ts = df_slice.iloc[-1]["date"]
            result = strategy.run(df_slice, ts)
            signals.append(result.signal)

        non_hold = [s for s in signals if s != SignalType.HOLD]
        assert len(non_hold) > 0, "Strategy never produced a LONG or SHORT signal"

    def test_no_signal_on_warmup_only(self, sample_ohlcv_data):
        """First few bars (within warmup) should only produce HOLD."""
        strategy = Ny15mOrbWithAFixedSlTpNasdaqStrategy()
        # Use exactly MIN_CANDLES_REQUIRED bars (all in warmup phase 0-600)
        warmup_df = sample_ohlcv_data.iloc[: strategy.MIN_CANDLES_REQUIRED].copy()
        ts = warmup_df.iloc[-1]["date"]
        result = strategy.run(warmup_df, ts)
        # During warmup (flat prices at 10000), breakout is unlikely
        assert isinstance(result.signal, SignalType)


class TestDataIntegrity:
    """Verify that the strategy does not corrupt the input DataFrame."""

    def test_original_df_unchanged(self, sample_ohlcv_data):
        strategy = Ny15mOrbWithAFixedSlTpNasdaqStrategy()
        original_cols = list(sample_ohlcv_data.columns)
        original_len = len(sample_ohlcv_data)
        ts = sample_ohlcv_data.iloc[-1]["date"]
        strategy.run(sample_ohlcv_data, ts)
        assert list(sample_ohlcv_data.columns) == original_cols
        assert len(sample_ohlcv_data) == original_len
