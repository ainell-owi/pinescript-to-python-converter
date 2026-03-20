# PineScript to Python Conversion Workflow

This document defines the step-by-step execution flow for the Orchestrator Agent.

## Phase 0: Strategy Selection (Pre-flight — handled by main.py)
- `main.py` scans `input/` and registers all .pine files in
  `strategies_registry.json`.
- Each new file is evaluated **in isolation** by the `strategy_selector` agent,
  which returns a JSON object with BTC + project scores.
- The pipeline auto-selects the highest-scoring strategy.
- Strategies skipped 2+ times are archived regardless of score.
- When no evaluated candidates remain, the pipeline recycles from archive.

The Orchestrator is only invoked AFTER this step and starts at Phase 1.

## Phase 1: Ingestion & Transpilation
1. **Input:** Receive PineScript code and Metadata.
2. **Action:** Call **Transpiler Agent**.
   - Task: Convert PineScript to `src/strategies/<name>_strategy.py`.
   - Reference: `.claude/skills/PINESCRIPT_REFERENCE`.

## Phase 2: Validation (Gatekeeper)
3. **Action:** Call **Validator Agent**.
   - Task: Review code against `.claude/skills/VALIDATION_CHECKLIST`.
   - **On STRUCTURAL failure:** Auto-fix loop — re-invoke Transpiler with fix instructions,
     re-validate. Max 2 retry cycles.
   - **On TRADING LOGIC failure:** Abort immediately (`CONVERSION_FAILED`).
   - **On PASS:** Proceed.

## Phase 3: Test Generation & Execution (QA)
4. **Action:** Call **Test Generator Agent**.
   - Task: Create `tests/strategies/test_<name>.py` using `sample_ohlcv_data`.
   - **MANDATORY:** Test Generator MUST run pytest after writing tests.
   - **On test failure (test bug):** Test Generator fixes the test (max 2 attempts).
   - **On test failure (strategy bug):** Test Generator reports `TEST_VALID_STRATEGY_BROKEN`.
     Orchestrator routes back to Transpiler → Validator → Test Generator (max 1 full loop).

## Phase 4: Delivery (GitHub Integration)
5. **Action:** Call **Integration Agent**.
   - Task: Push code and **Open Pull Request** using GitHub MCP.
   - **Goal:** Prepare the code for human review.
   - **Must emit:** `INTEGRATION_PASS` or `INTEGRATION_FALLBACK` token.

## Phase 5: Human Code Review
6. **Output:** Provide user with:
   - **PR Link** (Created by Integration Agent).
   - Status: **WAITING FOR CODE REVIEW**.
   - Instruction: "Please review the PR on GitHub. Once approved and merged, the CI pipeline will run the tests."

## Success Detection
`main.py` verifies orchestrator success by:
1. Scanning stdout for `INTEGRATION_PASS` or `INTEGRATION_FALLBACK` token
2. Verifying strategy file (`src/strategies/<name>_strategy.py`) exists on disk
3. Verifying test file (`tests/strategies/test_<name>.py`) exists on disk

Exit code 0 alone is NOT sufficient — all three checks must pass.