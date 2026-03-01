"""
Conversion Pipeline Runner with Structured Logging

This script invokes the Claude Code CLI with the orchestrator agent,
embedding PineScript file content directly into the prompt string.
All output and errors are logged to a structured directory tree organized
by strategy name and run timestamp.
"""

import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime


def setup_logger(strategy_name: str, logs_root: str = "logs") -> tuple[logging.Logger, Path]:
    """
    Creates a structured log directory and configures a logger for a single run.

    Directory structure:
        logs/
        └── MovingAverageCross/
            └── 2025-02-23_14-30-00/
                ├── run.log       Human-readable log (INFO+)
                └── errors.log    Errors only (ERROR+)

    :param strategy_name: Used as the top-level folder name under logs/.
    :param logs_root: Root directory for all logs. Defaults to 'logs/'.
    :return: Tuple of (configured logger, run directory path).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Sanitize strategy name for use as a directory name
    safe_name = strategy_name.replace(" ", "_").replace("/", "-")

    run_dir = Path(logs_root) / safe_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"{safe_name}.{timestamp}")
    logger.setLevel(logging.DEBUG)

    # Shared formatter for all handlers
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler 1: run.log — captures everything (DEBUG and above)
    run_log_path = run_dir / "run.log"
    run_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    run_handler.setLevel(logging.DEBUG)
    run_handler.setFormatter(formatter)

    # Handler 2: errors.log — captures only ERROR and CRITICAL
    error_log_path = run_dir / "errors.log"
    error_handler = logging.FileHandler(error_log_path, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # Handler 3: Console — mirrors INFO+ to the terminal in real time
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(run_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger, run_dir


def extract_strategy_name(metadata: str) -> str:
    """
    Parses the strategy name from a metadata string.
    Expects format: 'Name: StrategyName, Timeframe: '

    Falls back to 'UnknownStrategy' if the name cannot be extracted.

    :param metadata: Raw metadata string.
    :return: Strategy name string.
    """
    for part in metadata.split(","):
        part = part.strip()
        if part.lower().startswith("name:"):
            return part.split(":", 1)[1].strip()
    return "UnknownStrategy"


def run_claude_orchestrator(pine_script_path: str, metadata: str, logs_root: str = "logs"):
    """
    Executes the Claude Code CLI using the orchestrator sub-agent.
    Streams output to the terminal and writes structured logs to disk.

    :param pine_script_path: Path to the raw PineScript file.
    :param metadata: Strategy metadata string (name, timeframe, lookback).
    :param logs_root: Root directory where logs will be written.
    """
    strategy_name = extract_strategy_name(metadata)
    logger, run_dir = setup_logger(strategy_name, logs_root)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Metadata: {metadata}")

    # --- Validate input file ---
    script_file = Path(pine_script_path)
    if not script_file.exists():
        logger.error(f"PineScript file not found: {pine_script_path}")
        sys.exit(1)

    pine_script_content = script_file.read_text(encoding="utf-8")
    logger.info(f"Loaded PineScript file: {pine_script_path} ({len(pine_script_content)} chars)")

    # --- Build prompt ---
    prompt = f"""Start the conversion workflow.

Metadata: {metadata}

PineScript Source:
{pine_script_content}
"""

    # --- Build CLI command ---
    command = [
        "claude",
        "--agent", "orchestrator",
        "--print",
        "--verbose",
        prompt
    ]

    logger.info("Launching Claude Code orchestrator...")
    logger.debug(f"Command: {' '.join(command[:4])} [prompt omitted]")

    # --- Execute and stream output ---
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8"
        )

        # Stream each line to both terminal (via logger) and run.log
        for line in process.stdout:
            stripped = line.rstrip()
            if stripped:
                logger.debug(f"[claude] {stripped}")

        process.wait()

        if process.returncode == 0:
            logger.info("Workflow completed successfully.")
        else:
            logger.error(f"Workflow failed with exit code: {process.returncode}")
            sys.exit(process.returncode)

    except FileNotFoundError:
        logger.error("'claude' command not found. Ensure Claude Code CLI is installed and in your PATH.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error during workflow execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    INPUT_FILE = "input/source_strategy.pine"
    STRATEGY_METADATA = "Name: MovingAverageCross, Timeframe: 15m, Lookback: 24h"

    run_claude_orchestrator(INPUT_FILE, STRATEGY_METADATA)