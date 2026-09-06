#!/usr/bin/env python
"""Phase 8 -- Screenshot script for fig_4_5_prototype.png (build spec
section 10).

Launches app/app.py's Gradio demo in-process (no separate server needed),
drives it with a real sample question through Gradio's Playwright-based
screenshot support, and saves a 300-dpi-equivalent PNG to reports/.

Requires `playwright` (`pip install playwright && playwright install
chromium`) in addition to requirements.txt -- this is a one-off dev-time
tool, not part of the reproducible pipeline, so it is deliberately not added
to requirements.txt.

Usage:
    python app/screenshot.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.app import build_demo
from src.utils.logging import get_logger

logger = get_logger("app.screenshot")

SAMPLE_QUESTION = "Kí ni àwọn àmì àrùn ibà?"


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "playwright is required for screenshots but is not installed.\n"
            "Run: pip install playwright && playwright install chromium"
        )

    import gradio as gr

    demo = build_demo()
    _, local_url, _ = demo.launch(prevent_thread_lock=True, quiet=True, theme=gr.themes.Soft(),
                                   css=".gradio-container {max-width: 640px !important}")
    time.sleep(2)  # let the local server finish starting

    out_path = Path("reports/fig_4_5_prototype.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 480, "height": 800})  # mobile-ish, per NFR7
            page.goto(local_url)
            page.wait_for_selector("textarea")
            page.fill("textarea", SAMPLE_QUESTION)
            page.keyboard.press("Enter")
            page.wait_for_timeout(3000)
            page.screenshot(path=str(out_path))
            browser.close()
        logger.info(f"saved screenshot -> {out_path}")
    finally:
        demo.close()


if __name__ == "__main__":
    main()
