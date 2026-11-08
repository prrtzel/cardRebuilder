#!/usr/bin/env python3
"""Headless smoke test for the Symbols tab workflow.

Uses Microsoft Edge through Playwright (no browser download needed).  This is a
developer-only tool, not a runtime dependency:

    venv\\Scripts\\python.exe -m pip install playwright

Start the backend on port 8018 first:

    venv\\Scripts\\python.exe python-backend\\server.py --port 8018

Then run:

    venv\\Scripts\\python.exe scripts\\ui-symbol-smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGE = ROOT / "output" / "original.png"
UI_URL = (ROOT / "ui" / "index.html").as_uri()
SCREENSHOT = ROOT / "output" / "symbols_ui_test.png"


def main() -> int:
    errors: list[str] = []
    console: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on(
            "console",
            lambda msg: console.append(f"[{msg.type}] {msg.text}")
            if msg.type in ("error", "warning")
            else None,
        )

        page.add_init_script(
            f"""
            window.__TAURI__ = {{
              core: {{ invoke: async () => 8018 }},
              dialog: {{ open: async () => {json.dumps(str(TEST_IMAGE))} }},
            }};
            """
        )

        page.goto(UI_URL, wait_until="domcontentloaded")
        page.wait_for_function(
            "document.querySelector('#model-label').textContent.includes('Active Model')",
            timeout=30000,
        )

        # --- Load the test card through the real button -------------------
        page.click("#btn-load")
        page.wait_for_function(
            "!document.querySelector('#btn-detect-symbols').disabled",
            timeout=30000,
        )
        assert page.evaluate("state.currentImage") == str(TEST_IMAGE), "image path not stored"

        # --- Detect with the template matcher -----------------------------
        page.click('#tabs .tab[data-tab="symbols"]')
        page.select_option("#symbol-matcher", "template")
        page.click("#btn-detect-symbols")
        page.wait_for_function(
            "document.querySelectorAll('#symbol-table-body tr[data-row-id]').length === 3",
            timeout=120000,
        )
        rows = page.eval_on_selector_all(
            "#symbol-table-body tr[data-row-id]",
            """rows => rows.map(row => ({
                id: row.dataset.rowId,
                symbol: row.querySelector('.symbol-match-select').value,
                meta: row.querySelector('.symbol-meta').textContent.trim(),
            }))""",
        )
        print("template rows:", rows)
        assert [row["symbol"] for row in rows] == [
            "aggro_grey",
            "offensive",
            "stamina",
        ], f"unexpected template matches: {rows}"
        assert page.evaluate(
            "[...document.querySelectorAll('.symbol-thumb')].every("
            "img => img.complete && img.naturalWidth > 0)"
        ), "golden thumbnails did not load"
        assert page.inner_text("#symbol-status").startswith("3 boxes"), page.inner_text(
            "#symbol-status"
        )

        # --- Manual override then restore ---------------------------------
        second = page.locator("#symbol-table-body tr[data-row-id]").nth(1)
        second.locator(".symbol-match-select").select_option("defensive")
        assert second.locator(".symbol-thumb").get_attribute("src").endswith("/defensive")
        second.locator(".symbol-match-select").select_option("offensive")

        # --- Disagreement warning + one-click "Use template" ---------------
        page.evaluate(
            """() => {
                const row = state.symbols[0];
                row.symbol_id = 'aggro_red';
                row.template_id = 'aggro_grey';
                row.template_score = 0.88;
                row.method = 'ai';
                row.confidence = 0.99;
                row.warning = 'AI matched Aggro (red), template suggests Aggro (grey) (0.88).';
                renderSymbolTable();
                updateSymbolStatus();
            }"""
        )
        assert page.locator(".symbol-use-template").count() == 1
        assert page.eval_on_selector(
            "#tabs .tab[data-tab=\"symbols\"]",
            "el => el.classList.contains('symbol-warn')",
        )
        page.locator(".symbol-use-template").click()
        assert page.eval_on_selector(
            "#symbol-table-body tr[data-row-id]:first-child .symbol-match-select",
            "el => el.value",
        ) == "aggro_grey"
        assert page.locator(".symbol-use-template").count() == 0
        assert not page.eval_on_selector(
            "#tabs .tab[data-tab=\"symbols\"]",
            "el => el.classList.contains('symbol-warn')",
        )

        # --- Draw a manual box around the Stamina drop --------------------
        page.click("#btn-draw-symbol")
        rect = page.evaluate(
            """() => {
                const r = document.querySelector('#canvas-img').getBoundingClientRect();
                return {left: r.left, top: r.top, width: r.width, height: r.height};
            }"""
        )
        natural = page.evaluate(
            """() => {
                const img = document.querySelector('#canvas-img');
                return {w: img.naturalWidth, h: img.naturalHeight};
            }"""
        )
        assert (natural["w"], natural["h"]) == (240, 336), natural

        def client(x: float, y: float) -> tuple[float, float]:
            return (
                rect["left"] + x / natural["w"] * rect["width"],
                rect["top"] + y / natural["h"] * rect["height"],
            )

        start = client(14, 291)
        end = client(36, 326)
        page.mouse.move(*start)
        page.mouse.down()
        page.mouse.move(*end, steps=8)
        page.mouse.up()
        page.wait_for_function(
            """() => [...document.querySelectorAll('#symbol-table-body tr[data-row-id]')]
                .some(row => row.querySelector('.symbol-meta').textContent.includes('Manual box')
                     && row.querySelector('.symbol-match-select').value === 'stamina')""",
            timeout=30000,
        )
        assert page.evaluate(
            "document.querySelectorAll('#symbol-table-body tr[data-row-id]').length"
        ) == 3, "manual box should replace the overlapping Stamina auto row"
        page.click("#btn-draw-symbol")  # leave draw mode before rendering

        # --- Render at 4x and verify the result appears -------------------
        page.click("#btn-render")
        page.wait_for_function(
            "document.querySelector('#final-size').textContent.includes('Final:')",
            timeout=300000,
        )
        page.wait_for_function(
            "document.querySelector('#image-toggle button[data-name=\"final\"]').classList.contains('active')",
            timeout=30000,
        )
        assert page.evaluate("state.activeImage") == "final"
        final_text = page.inner_text("#final-size")
        print("render:", final_text)
        assert "960 × 1344" in final_text, final_text
        assert not page.evaluate("state.symbolWarnings.length"), page.evaluate(
            "state.symbolWarnings"
        )

        # --- Delete the manual row, then re-detect ------------------------
        manual_row = page.locator(
            "#symbol-table-body tr[data-row-id]",
        ).filter(has_text="Manual box")
        manual_row.locator(".symbol-delete").click()
        assert page.evaluate(
            "document.querySelectorAll('#symbol-table-body tr[data-row-id]').length"
        ) == 2, "manual row was not deleted"
        page.click("#btn-detect-symbols")
        page.wait_for_function(
            "document.querySelectorAll('#symbol-table-body tr[data-row-id]').length === 3",
            timeout=120000,
        )

        # --- Right-click delete of a canvas-drawn manual box --------------
        page.click("#btn-draw-symbol")
        page.wait_for_function("state.activeImage === 'original'")
        page.wait_for_function("document.querySelector('#canvas-img').complete")
        start = client(95, 8)
        end = client(142, 56)
        page.mouse.move(*start)
        page.mouse.down()
        page.mouse.move(*end, steps=8)
        page.mouse.up()
        page.wait_for_function(
            """() => [...document.querySelectorAll('#symbol-table-body tr[data-row-id]')]
                .some(row => row.querySelector('.symbol-meta').textContent.includes('Manual box'))""",
            timeout=30000,
        )
        page.mouse.click(*client(118, 32), button="right")
        page.wait_for_function(
            """() => ![...document.querySelectorAll('#symbol-table-body tr[data-row-id]')]
                .some(row => row.querySelector('.symbol-meta').textContent.includes('Manual box'))""",
            timeout=10000,
        )
        assert page.evaluate(
            "document.querySelectorAll('#symbol-table-body tr[data-row-id]').length"
        ) == 2, "right-click should delete the manual box and its overlapped auto row"
        page.click("#btn-draw-symbol")
        page.click("#btn-detect-symbols")
        page.wait_for_function(
            "document.querySelectorAll('#symbol-table-body tr[data-row-id]').length === 3",
            timeout=120000,
        )

        page.screenshot(path=str(SCREENSHOT), full_page=False)
        print("screenshot:", SCREENSHOT)
        browser.close()

    for line in console:
        print(line)
    if errors:
        print("PAGE ERRORS:")
        for error in errors:
            print(" ", error)
        return 1
    print("UI symbol smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
