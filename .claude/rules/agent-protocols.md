# Agent Communication & Output Protocols

This rule applies to orchestrating the multi-agent transpilation pipeline.

## 1. Handoff Logging Protocol
To maintain state synchronization with the external Python `runner.py`, you MUST print exact transition markers:
- On delegation: `[SYSTEM] Handing over to: <AgentName>`
- On return: `[SYSTEM] Control returned to: ORCHESTRATOR`

## 2. Execution & Error Handling
- The Orchestrator runs in non-interactive mode (`-p`).
- **Auto-Fixes:** Structural issues (warmup guards, imports, naming) trigger a bounded retry (max 2 cycles). 
- **Immediate Abort:** Trading logic failures (e.g., unresolvable lookahead bias) MUST abort immediately. No auto-fix.

## 3. Output Contracts (Strict Strings)
- **Strategy Selector:** Output MUST be raw JSON only (no markdown code blocks). Schema: `{ "pine_metadata": {}, "category": "", "btc_score": 0, "project_score": 0, "recommendation_reason": "" }`.
- **Integration Agent:** MUST output exactly `INTEGRATION_PASS` or `INTEGRATION_FALLBACK` to `stdout`. Without these tokens, `main.py` fails the run regardless of the exit code.