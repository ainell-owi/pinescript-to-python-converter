# Validation Checklist

The Validator Agent MUST verify the generated Python strategy against this exact checklist. If ANY check fails, the validation process is considered FAILED.

## 1. Syntax & Imports Validation
- [ ] Code is valid Python 3.11.
- [ ] No syntax errors or unresolved references.
- [ ] All necessary imports are present (e.g., `pandas`, `datetime`).
- [ ] Third-party technical indicator libraries (e.g., `ta-lib`) are imported correctly if used.

## 2. Contract Compliance
- [ ] The class inherits correctly from `BaseStrategy`.
- [ ] The `__init__` method explicitly calls `super().__init__(...)` with `name`, `description`, `timeframe`, and `lookback_hours`.
- [ ] The `run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation` method is implemented with the exact signature.
- [ ] The return value of `run` is strictly a `StrategyRecommendation` object using a valid `SignalType` enum.

## 3. Semantic & Trading Logic Validation
- [ ] **No Lookahead Bias:** The strategy must NEVER access future data. When processing the current row `i` or timestamp `t`, the strategy must only use data from index `<= i`. Shifts or rolling windows must be strictly backward-looking.
- [ ] Logic precisely matches the original PineScript intent (e.g., condition thresholds, crossover directions).
- [ ] **Multi-Timeframe Handling:** If the strategy fetches data from a higher timeframe (like `request.security` in PineScript), it MUST import and use `resample_to_interval` and `resampled_merge` from `src.utils.resample`. Custom or manual pandas resampling logic (`df.resample()`) inside the strategy class is strictly FORBIDDEN to prevent lookahead bias.