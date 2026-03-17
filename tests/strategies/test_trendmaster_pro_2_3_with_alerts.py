"""Tests for TrendMaster Pro 2.3 with Alerts Strategy.

Uses the shared `sample_ohlcv_data` fixture (1,100 candles at 15m intervals):
  Phase 0 (0–600):   Warmup — flat at 10,000 (indicators converge)
  Phase 1 (600–700): Sideways / Accumulation (low volatility)
  Phase 2 (700–900): Bull Run (10,000 → 12,000)
  Phase 3 (900–1100): Bear Crash (12,000 → 9,000)
"""
import pytest
import pandas as pd

from src.strategies.trendmaster_pro_2_3_with_alerts_strategy import (
    TrendmasterPro23WithAlertsStrategy,
)
from src.base_strategy import SignalType, StrategyRecommendation


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

def test_strategy_initialization():
    """Strategy instantiates with correct metadata."""
    strategy = TrendmasterPro23WithAlertsStrategy()
    assert strategy.name == "TrendMaster Pro 2.3 with Alerts"
    assert strategy.timeframe == "15m"
    assert strategy.lookback_hours > 0
    assert strategy.lookback_hours == 50


def test_strategy_min_candles_required_is_dynamic():
    """MIN_CANDLES_REQUIRED is computed dynamically and scales with parameters."""
    strategy_default = TrendmasterPro23WithAlertsStrategy()
    # Default: 3 * max(21, 20, 14, 26+9, 14+3, 14) = 3 * 35 = 105
    assert strategy_default.MIN_CANDLES_REQUIRED == 105

    # With a larger slow MACD period, warmup should be larger
    strategy_large = TrendmasterPro23WithAlertsStrategy(macd_slow=50)
    assert strategy_large.MIN_CANDLES_REQUIRED > strategy_default.MIN_CANDLES_REQUIRED


def test_strategy_default_parameters():
    """Default parameter values match Pine Script defaults."""
    s = TrendmasterPro23WithAlertsStrategy()
    assert s.ma_type == "SMA"
    assert s.short_ma_length == 9
    assert s.long_ma_length == 21
    assert s.bb_length == 20
    assert s.bb_multiplier == 2.0
    assert s.rsi_length == 14
    assert s.rsi_long_thresh == 55.0
    assert s.rsi_short_thresh == 45.0
    assert s.macd_fast == 12
    assert s.macd_slow == 26
    assert s.macd_signal == 9
    assert s.stoch_length == 14
    assert s.stoch_smoothing == 3
    assert s.adx_length == 14
    assert s.adx_threshold == 25.0


# ---------------------------------------------------------------------------
# Warmup / insufficient data guard
# ---------------------------------------------------------------------------

def test_hold_on_insufficient_data(sample_ohlcv_data):
    """Strategy returns HOLD when data is shorter than MIN_CANDLES_REQUIRED."""
    strategy = TrendmasterPro23WithAlertsStrategy()
    short_df = sample_ohlcv_data.iloc[:10].copy()
    timestamp = short_df.iloc[-1]["date"]
    result = strategy.run(short_df, timestamp)
    assert result.signal == SignalType.HOLD


def test_hold_just_below_min_candles(sample_ohlcv_data):
    """Strategy returns HOLD when data is exactly one row below MIN_CANDLES_REQUIRED."""
    strategy = TrendmasterPro23WithAlertsStrategy()
    threshold = strategy.MIN_CANDLES_REQUIRED - 1
    short_df = sample_ohlcv_data.iloc[:threshold].copy()
    timestamp = short_df.iloc[-1]["date"]
    result = strategy.run(short_df, timestamp)
    assert result.signal == SignalType.HOLD


def test_warmup_phase_returns_hold(sample_ohlcv_data):
    """During the warmup phase (first 600 candles), strategy returns HOLD
    for any slice shorter than MIN_CANDLES_REQUIRED."""
    strategy = TrendmasterPro23WithAlertsStrategy()
    # Use just warmup data (600 candles) — all flat at 10,000
    warmup_df = sample_ohlcv_data.iloc[:100].copy()
    timestamp = warmup_df.iloc[-1]["date"]
    result = strategy.run(warmup_df, timestamp)
    # 100 < 105 (MIN_CANDLES_REQUIRED), so must return HOLD
    assert result.signal == SignalType.HOLD


# ---------------------------------------------------------------------------
# Execution (smoke) tests
# ---------------------------------------------------------------------------

def test_strategy_execution_returns_valid_recommendation(sample_ohlcv_data):
    """run() on full dataset returns a valid StrategyRecommendation."""
    strategy = TrendmasterPro23WithAlertsStrategy()
    timestamp = sample_ohlcv_data.iloc[-1]["date"]
    result = strategy.run(sample_ohlcv_data, timestamp)
    assert isinstance(result, StrategyRecommendation)
    assert isinstance(result.signal, SignalType)
    assert result.signal in (SignalType.LONG, SignalType.SHORT, SignalType.HOLD)
    assert result.timestamp == timestamp


def test_strategy_execution_bull_phase(sample_ohlcv_data):
    """Strategy runs without error on bull market phase data (candles 700–900)."""
    strategy = TrendmasterPro23WithAlertsStrategy()
    # Include enough leading context for indicator warmup
    bull_df = sample_ohlcv_data.iloc[500:900].copy()
    timestamp = bull_df.iloc[-1]["date"]
    result = strategy.run(bull_df, timestamp)
    assert isinstance(result, StrategyRecommendation)
    assert isinstance(result.signal, SignalType)


def test_strategy_execution_bear_phase(sample_ohlcv_data):
    """Strategy runs without error on bear market phase data (candles 900–1100)."""
    strategy = TrendmasterPro23WithAlertsStrategy()
    # Include enough leading context for indicator warmup
    bear_df = sample_ohlcv_data.iloc[700:1100].copy()
    timestamp = bear_df.iloc[-1]["date"]
    result = strategy.run(bear_df, timestamp)
    assert isinstance(result, StrategyRecommendation)
    assert isinstance(result.signal, SignalType)


# ---------------------------------------------------------------------------
# Data integrity (indicator column) tests
# ---------------------------------------------------------------------------

def test_indicators_generated(sample_ohlcv_data):
    """After run(), indicator columns are present in the DataFrame."""
    strategy = TrendmasterPro23WithAlertsStrategy()
    timestamp = sample_ohlcv_data.iloc[-1]["date"]
    strategy.run(sample_ohlcv_data, timestamp)

    expected_columns = [
        "short_ma",
        "long_ma",
        "bb_basis",
        "bb_upper",
        "bb_lower",
        "rsi",
        "macd_line",
        "macd_signal",
        "stoch_k",
        "stoch_d",
        "adx",
    ]
    for col in expected_columns:
        assert col in sample_ohlcv_data.columns, f"Expected column '{col}' not found"


def test_indicator_values_are_numeric(sample_ohlcv_data):
    """Indicator columns contain numeric (non-object) data after run()."""
    strategy = TrendmasterPro23WithAlertsStrategy()
    timestamp = sample_ohlcv_data.iloc[-1]["date"]
    strategy.run(sample_ohlcv_data, timestamp)

    numeric_cols = ["short_ma", "long_ma", "bb_basis", "rsi", "macd_line", "stoch_k", "adx"]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data[col]), \
            f"Column '{col}' is not numeric"


def test_rsi_values_in_valid_range(sample_ohlcv_data):
    """RSI values (where not NaN) must be between 0 and 100."""
    strategy = TrendmasterPro23WithAlertsStrategy()
    timestamp = sample_ohlcv_data.iloc[-1]["date"]
    strategy.run(sample_ohlcv_data, timestamp)

    rsi_valid = sample_ohlcv_data["rsi"].dropna()
    assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all(), \
        "RSI values outside [0, 100] range"


def test_stochastic_values_in_valid_range(sample_ohlcv_data):
    """Stochastic %K values (where not NaN) must be between 0 and 100."""
    strategy = TrendmasterPro23WithAlertsStrategy()
    timestamp = sample_ohlcv_data.iloc[-1]["date"]
    strategy.run(sample_ohlcv_data, timestamp)

    k_valid = sample_ohlcv_data["stoch_k"].dropna()
    assert (k_valid >= 0).all() and (k_valid <= 100).all(), \
        "Stoch %K values outside [0, 100] range"


def test_bb_bands_ordering(sample_ohlcv_data):
    """BB upper band must be >= BB basis >= BB lower band (where not NaN)."""
    strategy = TrendmasterPro23WithAlertsStrategy()
    timestamp = sample_ohlcv_data.iloc[-1]["date"]
    strategy.run(sample_ohlcv_data, timestamp)

    valid_mask = (
        sample_ohlcv_data["bb_upper"].notna()
        & sample_ohlcv_data["bb_basis"].notna()
        & sample_ohlcv_data["bb_lower"].notna()
    )
    upper = sample_ohlcv_data.loc[valid_mask, "bb_upper"]
    basis = sample_ohlcv_data.loc[valid_mask, "bb_basis"]
    lower = sample_ohlcv_data.loc[valid_mask, "bb_lower"]
    assert (upper >= basis).all(), "BB upper must be >= BB basis"
    assert (basis >= lower).all(), "BB basis must be >= BB lower"


# ---------------------------------------------------------------------------
# MA type variant tests
# ---------------------------------------------------------------------------

def test_ema_variant_runs(sample_ohlcv_data):
    """Strategy runs correctly with EMA moving average type."""
    strategy = TrendmasterPro23WithAlertsStrategy(ma_type="EMA")
    timestamp = sample_ohlcv_data.iloc[-1]["date"]
    result = strategy.run(sample_ohlcv_data.copy(), timestamp)
    assert isinstance(result, StrategyRecommendation)
    assert isinstance(result.signal, SignalType)


def test_smma_variant_runs(sample_ohlcv_data):
    """Strategy runs correctly with SMMA moving average type."""
    strategy = TrendmasterPro23WithAlertsStrategy(ma_type="SMMA")
    timestamp = sample_ohlcv_data.iloc[-1]["date"]
    result = strategy.run(sample_ohlcv_data.copy(), timestamp)
    assert isinstance(result, StrategyRecommendation)
    assert isinstance(result.signal, SignalType)


def test_ma_types_produce_consistent_signal_types(sample_ohlcv_data):
    """All three MA types return valid signals (not necessarily the same value)."""
    timestamp = sample_ohlcv_data.iloc[-1]["date"]
    for ma_type in ("SMA", "EMA", "SMMA"):
        strategy = TrendmasterPro23WithAlertsStrategy(ma_type=ma_type)
        result = strategy.run(sample_ohlcv_data.copy(), timestamp)
        assert result.signal in (SignalType.LONG, SignalType.SHORT, SignalType.HOLD), \
            f"Unexpected signal for ma_type={ma_type}: {result.signal}"


# ---------------------------------------------------------------------------
# No lookahead bias check
# ---------------------------------------------------------------------------

def test_no_future_data_dependency(sample_ohlcv_data):
    """Adding a single future candle must not change the signal for the current bar.

    This verifies there is no lookahead bias: the signal for bar N should be
    identical whether computed on df[:N] or df[:N+1].
    """
    strategy = TrendmasterPro23WithAlertsStrategy()
    n = len(sample_ohlcv_data)
    cutoff = n - 5  # use 5 bars before the end to allow signal comparison

    df_now = sample_ohlcv_data.iloc[:cutoff].copy()
    df_plus_one = sample_ohlcv_data.iloc[: cutoff + 1].copy()

    ts_now = df_now.iloc[-1]["date"]

    result_now = strategy.run(df_now, ts_now)
    strategy2 = TrendmasterPro23WithAlertsStrategy()
    result_plus = strategy2.run(df_plus_one, ts_now)

    # Both calls use the same timestamp (last bar of df_now), so signals must match.
    # A mismatch would indicate the extra future candle changed the evaluation.
    assert result_now.signal == result_plus.signal, (
        f"Lookahead bias detected: signal changed from {result_now.signal} "
        f"to {result_plus.signal} when a future bar was added."
    )
