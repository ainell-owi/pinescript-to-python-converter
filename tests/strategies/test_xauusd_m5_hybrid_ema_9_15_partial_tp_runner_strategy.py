"""
Tests for XauusdM5HybridEma915PartialTpRunnerStrategy.

Coverage:
1. Warmup guard — HOLD returned when len(df) < MIN_CANDLES_REQUIRED.
   Uses sample_ohlcv_data fixture; guard fires before any resampling so the
   15m fixture interval is irrelevant for these tests.
2. Signal detection — uses a _make_5m_df helper that produces 5m-interval data
   so resample_to_interval("15m") is a valid downsampling operation.
3. Edge cases — empty DataFrame, all-NaN close handled gracefully.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.strategies.xauusd_m5_hybrid_ema_9_15_partial_tp_runner_strategy import (
    XauusdM5HybridEma915PartialTpRunnerStrategy,
)
from src.base_strategy import SignalType, StrategyRecommendation


VALID_SIGNALS = frozenset(SignalType)
_TS = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helper: craft 5m-interval data with trend + rejection candle on final bar
# ---------------------------------------------------------------------------

def _make_5m_df(n: int = 300, direction: str = "up") -> pd.DataFrame:
    """
    Build a synthetic 5m-interval OHLCV DataFrame designed to produce a clear
    EMA trend on both M5 and M15 timeframes, ending with a rejection candle.

    Structure
    ---------
    - Bars 0 … n-2 : linear trend (up 10,000→10,500 or down 10,500→10,000)
                     so that EMA9 > EMA15 (bull) or EMA9 < EMA15 (bear) on
                     both the M5 frame and the M15 resampled frame.
    - Bar n-1      : rejection candle that satisfies entry conditions:
                       Bull: long lower wick (wick > 1.5 × body), bullish close
                       Bear: long upper wick (wick > 1.5 × body), bearish close

    The helper mirrors the fixture's regime philosophy but at 5m granularity,
    which is required so resample_to_interval("15m") is a valid downsampling.
    """
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    dates = [start + timedelta(minutes=5 * i) for i in range(n)]

    if direction == "up":
        close = np.linspace(10_000.0, 10_500.0, n)
        base = close[-2]
        # Bullish rejection candle: long lower wick, close > open
        last_open  = base + 2.0
        last_close = base + 12.0   # close > open (bullish body = 10)
        last_high  = base + 15.0
        last_low   = base - 18.0   # lower_wick = open - low = 2 - (-18) = 20 > body*1.5=15
    else:
        close = np.linspace(10_500.0, 10_000.0, n)
        base = close[-2]
        # Bearish rejection candle: long upper wick, close < open
        last_open  = base - 2.0
        last_close = base - 12.0   # close < open (bearish body = 10)
        last_high  = base + 18.0   # upper_wick = high - open = 18+2 = 20 > body*1.5=15
        last_low   = base - 15.0

    close[-1] = last_close

    open_ = np.roll(close, 1)
    open_[0] = close[0]
    open_[-1] = last_open

    high = np.maximum(open_, close) + 2.0
    low  = np.minimum(open_, close) - 2.0
    high[-1] = last_high
    low[-1]  = last_low

    volume = np.full(n, 100.0)

    df = pd.DataFrame(
        {
            "date":   dates,
            "open":   open_,
            "high":   high,
            "low":    low,
            "close":  close,
            "volume": volume,
        }
    )
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"]  = df[["open", "close", "low"]].min(axis=1)
    return df


# ---------------------------------------------------------------------------
# 1. Warmup guard (uses sample_ohlcv_data fixture — guard fires before resampling)
# ---------------------------------------------------------------------------

class TestWarmupGuard:
    def test_empty_dataframe_returns_hold(self):
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        result = strategy.run(empty, _TS)
        assert result.signal == SignalType.HOLD

    def test_below_min_candles_returns_hold(self, sample_ohlcv_data):
        """One bar below MIN_CANDLES_REQUIRED must trigger the RL guard."""
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        cutoff = strategy.MIN_CANDLES_REQUIRED - 1
        result = strategy.run(sample_ohlcv_data.iloc[:cutoff].copy(), _TS)
        assert result.signal == SignalType.HOLD

    def test_warmup_phase_slices_return_hold(self, sample_ohlcv_data):
        """Multiple sub-threshold slices must all return HOLD."""
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        for length in [1, 10, 20, strategy.MIN_CANDLES_REQUIRED - 1]:
            result = strategy.run(sample_ohlcv_data.iloc[:length].copy(), _TS)
            assert result.signal == SignalType.HOLD, (
                f"Expected HOLD for {length} bars, got {result.signal}"
            )

    def test_min_candles_required_is_dynamic(self):
        """MIN_CANDLES_REQUIRED must be 3 × max(ema_fast_len, ema_slow_len)."""
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        expected = 3 * max(strategy.ema_fast_len, strategy.ema_slow_len)
        assert strategy.MIN_CANDLES_REQUIRED == expected

    def test_timestamp_preserved_on_hold(self, sample_ohlcv_data):
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        result = strategy.run(sample_ohlcv_data.iloc[:1].copy(), _TS)
        assert result.timestamp == _TS


# ---------------------------------------------------------------------------
# 2. Signal detection on structured 5m data
# ---------------------------------------------------------------------------

class TestSignalDetection:
    def test_bull_trend_with_rejection_candle_produces_long(self):
        """Sustained uptrend + bullish rejection candle on last bar → LONG."""
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        df = _make_5m_df(n=300, direction="up")
        result = strategy.run(df, _TS)
        assert result.signal == SignalType.LONG

    def test_bear_trend_with_rejection_candle_produces_short(self):
        """Sustained downtrend + bearish rejection candle on last bar → SHORT."""
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        df = _make_5m_df(n=300, direction="down")
        result = strategy.run(df, _TS)
        assert result.signal == SignalType.SHORT

    def test_bull_and_bear_return_different_signals(self):
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        long_r  = strategy.run(_make_5m_df(300, "up"),   _TS)
        short_r = strategy.run(_make_5m_df(300, "down"), _TS)
        assert long_r.signal != short_r.signal

    def test_result_is_strategy_recommendation(self):
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        result = strategy.run(_make_5m_df(300, "up"), _TS)
        assert isinstance(result, StrategyRecommendation)
        assert isinstance(result.signal, SignalType)
        assert result.timestamp == _TS

    def test_exactly_min_candles_does_not_raise(self):
        """Exactly MIN_CANDLES_REQUIRED rows must not raise."""
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        df = _make_5m_df(n=strategy.MIN_CANDLES_REQUIRED, direction="up")
        try:
            result = strategy.run(df, _TS)
            assert result.signal in VALID_SIGNALS
        except Exception as exc:
            pytest.fail(f"Strategy raised at MIN_CANDLES_REQUIRED: {exc}")

    def test_signal_is_valid_enum(self):
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        result = strategy.run(_make_5m_df(300, "up"), _TS)
        assert result.signal in VALID_SIGNALS

    def test_repeated_calls_are_stable(self):
        """Calling run() twice on identical data must return the same signal."""
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        df1 = _make_5m_df(300, "up")
        df2 = _make_5m_df(300, "up")
        assert strategy.run(df1, _TS).signal == strategy.run(df2, _TS).signal


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_nan_close_guard_fires_before_compute(self, sample_ohlcv_data):
        """All-NaN OHLCV on a short slice: RL guard fires, no exception raised."""
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        df = sample_ohlcv_data.iloc[:10].copy()
        df["close"] = np.nan
        df["open"]  = np.nan
        df["high"]  = np.nan
        df["low"]   = np.nan
        try:
            result = strategy.run(df, _TS)
            assert result.signal in VALID_SIGNALS
        except Exception as exc:
            pytest.fail(f"Strategy raised on all-NaN input: {exc}")

    def test_timeframe_is_lowercase(self):
        strategy = XauusdM5HybridEma915PartialTpRunnerStrategy()
        assert strategy.timeframe == strategy.timeframe.lower()

    def test_timeframe_is_5m(self):
        assert XauusdM5HybridEma915PartialTpRunnerStrategy().timeframe == "5m"

    def test_lookback_hours_is_positive(self):
        assert XauusdM5HybridEma915PartialTpRunnerStrategy().lookback_hours > 0

    def test_strategy_name(self):
        assert (
            XauusdM5HybridEma915PartialTpRunnerStrategy().name
            == "XauusdM5HybridEma915PartialTpRunnerStrategy"
        )
