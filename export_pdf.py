"""
Export the Streamlit dashboard to a single PDF.
Usage:
    1. Start the app:  streamlit run app.py
    2. Run this script: python export_pdf.py [--url http://localhost:8501]
"""

import asyncio
import argparse
import sys
import time
from pathlib import Path

import urllib.request
import urllib.error

import pypdf
from playwright.async_api import async_playwright

PAGES = [
    "Pipeline Health",
    "Data Load",
    "PCA & Regime",
    "VaR Engine",
    "Alert History",
    "Daily Briefings",
]

OUT = Path("data/output")
TEMP_DIR = OUT / "_pdf_tmp"


def wait_for_server(url: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


async def capture_pages(url: str) -> list[Path]:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    captured: list[Path] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": 1400, "height": 900},
        )
        page = await ctx.new_page()

        print(f"  Navigating to {url} …")
        await page.goto(url, wait_until="networkidle", timeout=30_000)
        await asyncio.sleep(3)  # allow Streamlit's React tree to settle

        for page_name in PAGES:
            print(f"  Capturing: {page_name}")

            # Streamlit hides the <input> visually; click the wrapping <label> instead
            label = page.locator("[data-testid='stSidebar'] label").filter(has_text=page_name).first
            if await label.count():
                await label.click()
                await asyncio.sleep(2)  # wait for content to render
            else:
                print(f"    ⚠  Sidebar label not found for '{page_name}', skipping.")
                continue

            pdf_path = TEMP_DIR / f"{page_name.replace(' ', '_')}.pdf"
            await page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                margin={"top": "15mm", "bottom": "15mm", "left": "10mm", "right": "10mm"},
            )
            captured.append(pdf_path)

        await browser.close()

    return captured


def merge_pdfs(parts: list[Path], output: Path) -> None:
    writer = pypdf.PdfWriter()
    for p in parts:
        writer.append(str(p))
    with open(output, "wb") as f:
        writer.write(f)


def cleanup(parts: list[Path]) -> None:
    for p in parts:
        p.unlink(missing_ok=True)
    try:
        TEMP_DIR.rmdir()
    except OSError:
        pass


async def main(url: str) -> None:
    print(f"Checking that the app is reachable at {url} …")
    if not wait_for_server(url):
        print(f"ERROR: Could not reach {url}. Make sure `streamlit run app.py` is running.")
        sys.exit(1)
    print("  App is up.\n")

    parts = await capture_pages(url)
    if not parts:
        print("No pages were captured. Aborting.")
        sys.exit(1)

    output = OUT / "dashboard_export.pdf"
    print(f"\nMerging {len(parts)} pages -> {output}")
    merge_pdfs(parts, output)
    cleanup(parts)

    print(f"\nDone! PDF saved to: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501",
                        help="Streamlit app URL (default: http://localhost:8501)")
    args = parser.parse_args()
    asyncio.run(main(args.url))
