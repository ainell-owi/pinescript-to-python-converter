"""
Tests for KineticInertiaShort1hStrategy.

Coverage:
  1. Warmup guard: returns HOLD when df has fewer than MIN_CANDLES_REQUIRED rows.
  2. Signal validity: run() always returns a well-formed StrategyRecommendation.
  3. Short signal detection:
       a. At least one SHORT fires when scanning the full fixture across all market phases.
       b. Only HOLD emitted on a perfectly flat (zero-noise) price series (ROC=0 → velocity=0).
       c. Consistently falling prices (negative ROC) never produce a SHORT.
       d. Bespoke bull-then-deceleration DataFrame guarantees a SHORT fires.
  4. Edge cases: empty DataFrame, all-NaN, zero-volume, constant price.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.base_strategy import SignalType
from src.strategies.kinetic_inertia_short_1h_strategy import KineticInertiaShort1hStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_ohlcv(
    close: np.ndarray,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    interval_minutes: int = 60,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from arrays."""
    n = len(close)
    dates = [
        datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=interval_minutes * i)
        for i in range(n)
    ]
    if open_ is None:
        open_ = np.roll(close, 1)
        open_[0] = close[0]
    if volume is None:
        volume = np.full(n, 100.0)
    rng = np.random.default_rng(42)
    if high is None:
        high = np.maximum(open_, close) + np.abs(rng.normal(0, 5, n))
    if low is None:
        low = np.minimum(open_, close) - np.abs(rng.normal(0, 5, n))
    df = pd.DataFrame({
        "date":   pd.to_datetime(dates, utc=True),
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    })
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"]  = df[["open", "close", "low"]].min(axis=1)
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy():
    return KineticInertiaShort1hStrategy()


# ---------------------------------------------------------------------------
# 1. Warmup guard
# ---------------------------------------------------------------------------

class TestWarmupGuard:

    def test_returns_hold_when_below_min_candles(self, strategy, sample_ohlcv_data):
        """Strategy must return HOLD when fewer than MIN_CANDLES_REQUIRED rows are provided."""
        short_df = sample_ohlcv_data.iloc[: strategy.MIN_CANDLES_REQUIRED - 1]
        rec = strategy.run(short_df, _ts())
        assert rec.signal == SignalType.HOLD

    def test_returns_hold_on_single_row(self, strategy, sample_ohlcv_data):
        single = sample_ohlcv_data.iloc[:1]
        rec = strategy.run(single, _ts())
        assert rec.signal == SignalType.HOLD

    def test_min_candles_required_is_positive(self, strategy):
        assert strategy.MIN_CANDLES_REQUIRED > 0

    def test_min_candles_required_is_dynamic(self, strategy):
        """MIN_CANDLES_REQUIRED must equal 3 * max(length, smooth, atr_length)."""
        expected = 3 * max(strategy.length, strategy.smooth, strategy.atr_length)
        assert strategy.MIN_CANDLES_REQUIRED == expected


# ---------------------------------------------------------------------------
# 2. Signal validity — run() always returns a well-formed recommendation
# ---------------------------------------------------------------------------

class TestSignalValidity:

    def test_full_dataset_returns_valid_signal_type(self, strategy, sample_ohlcv_data):
        rec = strategy.run(sample_ohlcv_data, _ts())
        assert isinstance(rec.signal, SignalType)

    def test_run_returns_strategy_recommendation_namedtuple(self, strategy, sample_ohlcv_data):
        rec = strategy.run(sample_ohlcv_data, _ts())
        assert hasattr(rec, "signal")
        assert hasattr(rec, "timestamp")

    def test_timestamp_is_preserved(self, strategy, sample_ohlcv_data):
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        rec = strategy.run(sample_ohlcv_data, ts)
        assert rec.timestamp == ts

    def test_signal_not_none(self, strategy, sample_ohlcv_data):
        rec = strategy.run(sample_ohlcv_data, _ts())
        assert rec.signal is not None


# ---------------------------------------------------------------------------
# 3. Short signal detection
# ---------------------------------------------------------------------------

class TestShortSignals:
    """
    KIO fires on the FIRST bar a "setup condition" becomes True AND the candle is bearish.
    The setup requires: velocity (ROC) > 0, smoothed acceleration < 0, kinetic energy < SMA(KE).

    The fixture's Phase-0 (flat at 10,000) keeps velocity exactly zero, so setup is always
    False there.  The bull→bear transition around bar 900 is the prime territory for SHORT
    signals: price is still above its level from 10 bars prior (positive ROC) while
    decelerating hard (negative smooth acceleration) and losing kinetic energy.
    """

    def test_short_signal_fires_in_fixture_data(self, strategy, sample_ohlcv_data):
        """At least one SHORT must appear when scanning the fixture across all market phases."""
        min_bars = strategy.MIN_CANDLES_REQUIRED
        found = any(
            strategy.run(sample_ohlcv_data.iloc[:end], _ts()).signal == SignalType.SHORT
            for end in range(min_bars + 1, len(sample_ohlcv_data) + 1, 5)
        )
        assert found, "Expected at least one SHORT signal scanning fixture data (none found)"

    def test_hold_only_during_synthetic_flat_warmup(self, strategy):
        """
        On a genuinely flat (zero-noise) price series, ROC = 0 throughout, so
        velocity > 0 (condition 1) is never satisfied → setup always False →
        enter_short never fires → only HOLD expected.

        Note: the shared fixture's Phase-0 contains Gaussian noise, which can produce
        non-zero ROC and occasionally trigger a SHORT even on the flat phase.  This test
        instead constructs a perfect constant-price dataset so the assertion is reliable.
        """
        n = strategy.MIN_CANDLES_REQUIRED + 100
        close = np.full(n, 10_000.0)
        open_ = close.copy()
        high = close + 1.0
        low = close - 1.0
        df = _make_ohlcv(close, open_=open_, high=high, low=low)
        non_hold = [
            end
            for end in range(strategy.MIN_CANDLES_REQUIRED, n, 5)
            if strategy.run(df.iloc[:end], _ts()).signal != SignalType.HOLD
        ]
        assert not non_hold, (
            f"Unexpected non-HOLD signals on constant-price data at window ends: {non_hold}"
        )

    def test_no_short_on_monotone_falling_prices(self, strategy):
        """
        Consistently falling prices yield negative ROC → velocity ≤ 0 → condition 1 fails
        for every bar → enter_short can never fire.
        """
        n = strategy.MIN_CANDLES_REQUIRED + 30
        close = np.linspace(15000.0, 9000.0, n)   # strict monotone decline
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        open_[1:] = close[:-1] + 10.0              # open always above close → irrelevant here
        df = _make_ohlcv(close, open_=open_, interval_minutes=60)
        rec = strategy.run(df, _ts())
        assert rec.signal != SignalType.SHORT, (
            f"Unexpected SHORT on monotone falling prices: {rec.signal}"
        )

    def test_short_signal_in_bespoke_bull_then_deceleration(self, strategy):
        """
        Craft a bespoke DataFrame that guarantees at least one SHORT signal.

        Construction:
          Phase A (0 – n_fast-1):   Strong linear growth 10,000 → 11,980.  Produces
              large positive velocity and stable positive kinetic energy.
          Phase B (n_fast onwards): Near-flat with alternating small up/down moves.
              After 10 bars ROC collapses (velocity stays positive but tiny),
              smooth_accel turns negative, and KE < SMA(KE).
              Alternating open/close ensures bearish candles appear throughout.

        The alternating close pattern guarantees bearish candles (close < open)
        coincide with the setup window, making enter_short fire.
        """
        n_fast = 100
        n_slow = 150
        n = n_fast + n_slow

        close = np.empty(n)
        # Phase A: strong rally
        close[:n_fast] = 10_000.0 + np.arange(n_fast) * 20.0

        # Phase B: alternating zigzag around a very slow drift
        base = close[n_fast - 1]
        for i in range(n_slow):
            # slow upward drift + alternating ±30 so every odd bar is bearish
            drift = i * 0.5
            close[n_fast + i] = base + drift + (30.0 if i % 2 == 0 else -30.0)

        # Build explicit open array: each open is the previous bar's close.
        # In the zigzag phase, even bars (close high) have open = previous (low) close
        # → bullish.  Odd bars (close low) have open = previous (high) close → bearish.
        open_ = np.empty(n)
        open_[0] = close[0]
        open_[1:] = close[:-1]

        df = _make_ohlcv(close, open_=open_, interval_minutes=60)

        min_bars = strategy.MIN_CANDLES_REQUIRED
        found_short = any(
            strategy.run(df.iloc[:end], _ts()).signal == SignalType.SHORT
            for end in range(min_bars + 1, n + 1)
        )
        assert found_short, (
            "Expected at least one SHORT in bull-then-deceleration bespoke DataFrame"
        )


# ---------------------------------------------------------------------------
# 4. Edge cases — no exception should be raised
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_dataframe_returns_hold(self, strategy):
        empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        rec = strategy.run(empty, _ts())
        assert rec.signal == SignalType.HOLD

    def test_all_nan_does_not_raise(self, strategy, sample_ohlcv_data):
        """All-NaN OHLCV must not raise — strategy should handle gracefully."""
        nan_df = sample_ohlcv_data.copy()
        for col in ("open", "high", "low", "close", "volume"):
            nan_df[col] = np.nan
        try:
            rec = strategy.run(nan_df, _ts())
            assert isinstance(rec.signal, SignalType)
        except Exception as exc:
            pytest.fail(f"Strategy raised an exception on all-NaN data: {exc}")

    def test_zero_volume_does_not_raise(self, strategy, sample_ohlcv_data):
        """Zero volume zeroes kinetic energy; strategy must not divide by zero or raise."""
        zero_vol_df = sample_ohlcv_data.copy()
        zero_vol_df["volume"] = 0.0
        try:
            rec = strategy.run(zero_vol_df, _ts())
            assert isinstance(rec.signal, SignalType)
        except Exception as exc:
            pytest.fail(f"Strategy raised an exception on zero-volume data: {exc}")

    def test_constant_price_returns_hold(self, strategy, sample_ohlcv_data):
        """
        Constant price → ROC = 0 → velocity = 0 → condition 1 never satisfied
        → setup always False → enter_short never fires → HOLD expected.
        """
        flat_df = sample_ohlcv_data.copy()
        flat_df["close"] = 10000.0
        flat_df["open"]  = 10000.0
        flat_df["high"]  = 10001.0
        flat_df["low"]   = 9999.0
        rec = strategy.run(flat_df, _ts())
        assert rec.signal == SignalType.HOLD
