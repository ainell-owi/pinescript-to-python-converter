import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """
    Generates a synthetic DataFrame with structured market regimes for testing.

    Structure:
    - Phase 1 (0-100 candles): Sideways / Accumulation (Low volatility).
    - Phase 2 (100-300 candles): Bull Run (Strong Uptrend).
    - Phase 3 (300-500 candles): Bear Crash (Strong Downtrend).

    Total: 500 candles (15m interval).
    Data is strictly UTC-aware to simulate ISO format compliance.
    """
    periods = 500
    start_date = datetime(2024, 1, 1, 0, 0, 0)

    # 1. Generate UTC Timestamps
    dates = [start_date + timedelta(minutes=15 * i) for i in range(periods)]

    # 2. Generate Structured Price Movements (Regimes)
    # We use sine waves + linear trends to ensure signals trigger
    x = np.linspace(0, 50, periods) # we split to 500 equal intervals to create different regimes

    # Base pattern: Sine wave (volatility) + Linear Trend
    trend = np.zeros(periods)

    # Phase 1: Sideways (Flat)
    trend[:100] = 10000 + np.sin(x[:100]) * 50 # Low volatility around 10,000

    # Phase 2: Bull Market (Linear Increase)
    # Ramps up from 10,000 to ~12,000
    trend[100:300] = np.linspace(10000, 12000, 200) + (np.sin(x[100:300]) * 100)

    # Phase 3: Bear Market (Linear Decrease)
    # Crashes down from 12,000 to ~9,000
    trend[300:] = np.linspace(12000, 9000, 200) + (np.sin(x[300:]) * 150)

    # 3. Add Noise (Randomness)
    np.random.seed(42)  # Deterministic for consistent tests
    noise = np.random.normal(0, 20, periods)
    close_price = trend + noise

    # 4. Construct OHLC
    # Open is roughly the previous close
    open_price = np.roll(close_price, 1)
    open_price[0] = close_price[0]  # Fix first value because it rolled from the end

    # High/Low derived from Open/Close with some expansion
    high_price = np.maximum(open_price, close_price) + np.abs(np.random.normal(0, 10, periods))
    low_price = np.minimum(open_price, close_price) - np.abs(np.random.normal(0, 10, periods))

    # Volume: Higher volume on trend changes
    volume = np.abs(np.random.normal(100, 50, periods)) * (1 + np.abs(np.gradient(close_price)) / 10) #in breaking regimes, volume spikes

    data = {
        'date': dates,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': volume
    }

    df = pd.DataFrame(data)

    # 5. Enforce Data Integrity
    # High must be strictly >= Low, Open, Close
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    # 6. Ensure Date is UTC (ISO-8601 compatible internally)
    df['date'] = pd.to_datetime(df['date'], utc=True)

    return df