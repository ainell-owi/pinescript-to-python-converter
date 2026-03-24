"""
Scraper bridge — Wraps TradingViewScraper for pipeline use.
"""

import json
import logging
import sys
from pathlib import Path

from src.pipeline import INPUT_DIR, SEEN_URLS_PATH, TARGET_STRATEGY_COUNT, _div

logger = logging.getLogger("runner")


def _load_seen_urls() -> set[str]:
    """Load the persisted global URL dedup store (O(1) lookup set)."""
    if SEEN_URLS_PATH.exists():
        try:
            return set(json.loads(SEEN_URLS_PATH.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError):
            logger.warning("seen_urls.json is corrupt — starting fresh.")
    return set()


def _save_seen_urls(seen_urls: set[str]) -> None:
    """Persist the global URL dedup store back to disk."""
    SEEN_URLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_URLS_PATH.write_text(json.dumps(sorted(seen_urls), indent=2), encoding="utf-8")


def run_tv_scraper(max_results: int = 6) -> None:
    """
    Populate input/ by scraping public TradingView strategies.

    Fetches from Popular + Editor's Picks, using data/seen_urls.json for dedup.
    """
    print(f"\n[SCRAPER] input/ has fewer than {TARGET_STRATEGY_COUNT} strategies.")
    print(f"          Fetching {max_results} more strategy file(s) from TradingView...")
    print(f"          Sources: Popular (x{max_results // 2}) + Editor's Picks (x{max_results // 2})")
    print(_div())

    # Block tv_scraper's logging.basicConfig from adding a root StreamHandler.
    _root_log = logging.getLogger()
    if not _root_log.handlers:
        _root_log.addHandler(logging.NullHandler())

    try:
        from src.utils.tv_scraper import TradingViewScraper
    except ImportError as exc:
        print(f"\n[!] Cannot import TradingViewScraper: {exc}")
        print("    Install missing deps: pip install selenium webdriver-manager")
        sys.exit(1)

    # Redirect scraper / driver logs to our file handler — off the terminal.
    for _lgr_name in ("TV_Scraper", "WDM", "selenium", "urllib3"):
        _lgr = logging.getLogger(_lgr_name)
        _lgr.handlers.clear()
        for _h in logger.handlers:
            _lgr.addHandler(_h)
        _lgr.propagate = False

    seen_urls = _load_seen_urls()
    logger.info(f"Loaded {len(seen_urls)} previously-seen URL(s) from {SEEN_URLS_PATH}")

    n_per_source = max(1, max_results // 2)
    saved = 0
    failed = 0

    try:
        with TradingViewScraper(headless=False) as scraper:
            urls = scraper.fetch_from_two_sources(
                n_per_source=n_per_source,
                seen_urls=seen_urls,
            )
            logger.info(f"TV scraper found {len(urls)} new strategy URL(s) across both sources")

            for url, scrape_source in urls:
                if saved >= max_results:
                    break

                slug = TradingViewScraper._extract_strategy_slug(url)
                dest = INPUT_DIR / f"{slug}.pine"

                if dest.exists():
                    logger.info(f"Skipping already-downloaded: {slug}")
                    seen_urls.add(url)
                    continue

                print(f"  [{saved + 1}/{max_results}] {slug} [{scrape_source}] ... ", end="", flush=True)

                try:
                    pine = scraper.fetch_pinescript(url)
                    meta = scraper.fetch_strategy_metadata(url)
                    scraper.save_to_input(pine, url, source=scrape_source, metadata=meta)
                    metrics_summary = ""
                    if meta and meta.get("backtest_metrics"):
                        bm = meta["backtest_metrics"]
                        metrics_summary = (
                            f" | trades={bm.get('total_trades')} "
                            f"pf={bm.get('profit_factor')} "
                            f"dd={bm.get('max_drawdown_pct')}%"
                        )
                    print(f"[OK]  ({len(pine):,} chars{metrics_summary})")
                    logger.info(f"Scraped: {slug} [{scrape_source}] ({len(pine)} chars{metrics_summary})")
                    seen_urls.add(url)
                    saved += 1
                except NotImplementedError as exc:
                    first_line = str(exc).splitlines()[0]
                    print(f"[SKIP]  {first_line}")
                    logger.warning(f"Skipped {slug}: {first_line}")
                    failed += 1
                except Exception as exc:
                    print(f"[FAIL]  {exc}")
                    logger.exception(f"Error scraping {slug}: {exc}")
                    failed += 1

    except RuntimeError as exc:
        print(f"\n[FATAL] Scraper runtime error: {exc}")
        logger.error(f"TV scraper runtime error: {exc}")
        sys.exit(1)
    finally:
        _save_seen_urls(seen_urls)
        logger.info(f"Saved {len(seen_urls)} URL(s) to {SEEN_URLS_PATH}")

    print(_div())
    print(f"  Scraped {saved} strategy file(s) -> input/")
    if failed:
        print(f"  Skipped {failed} file(s) (private or unsupported)")

    if saved == 0:
        print("\n[FAIL] No strategies could be scraped.")
        print("       Manual fallback: paste PineScript into input/source_strategy.pine")
        sys.exit(1)