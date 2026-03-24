---
name: utils-reference
description: Provides mandatory utility functions for PineScript transpilation, specifically for Multi-Timeframe (MTF) resampling. Automatically use this skill WHENEVER the PineScript contains `request.security` or requires higher timeframe data. It ensures strictly causal merges to prevent lookahead bias and enforces the correct lowercase timeframe formatting required by the RL Engine.
---

# Utils Reference

When translating PineScript to Python, you must use the following pre-built utility functions from the `src.utils` package. 

## Multi-Timeframe Resampling (`src.utils.resampling`)
If the PineScript code uses `request.security` to fetch data from a higher timeframe, you MUST use these exact functions to avoid lookahead bias. The RL Agent will exploit any future-leaking data, rendering the training useless.

**CRITICAL RULE - Timeframe Formatting:** ALL timeframe strings passed to `resample_to_interval` MUST be strictly lowercase (e.g., `"1d"`, `"4h"`, `"15m"`, `"1w"`).

**Imports:**
```python
import talib
from src.utils.resampling import resample_to_interval, resampled_merge
```

**Implementation Pattern:**
```python
# 1. Resample the base dataframe to a higher timeframe (e.g., 4 hours)
# REMEMBER: Must be strictly lowercase!
resampled_df = resample_to_interval(df, "4h")

# 2. Compute higher-timeframe indicators on the resampled dataframe
# STRICT PIPELINE RULE: Always use vectorized talib functions, NOT iterative loops.
resampled_df['sma'] = talib.SMA(resampled_df['close'].values, timeperiod=14)

# 3. Merge the higher-timeframe data back to the base dataframe
# This utility inherently shifts the data to prevent lookahead bias.
merged_df = resampled_merge(original=df, resampled=resampled_df, fill_na=True)

# The higher timeframe columns will be prefixed automatically.
# The naming convention is: resample_{pine_timeframe_minutes}_{original_column_name}
# e.g., for a 4h (240m) timeframe: 'resample_240_close', 'resample_240_sma_14'
# The suffix always includes the exact indicator/column name as computed on the resampled df.
# You can now use merged_df['resample_240_sma'] in your strategy logic.
```

