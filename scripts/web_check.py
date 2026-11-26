"""Browser click-through check of the GRACE web page (app/server.py must be running; Playwright drives the installed
Microsoft Edge). Same checks as scripts/ui_check.py for the Gradio app, plus progress, cancel and error messages.

python scripts/web_check.py <video> <out_prefix> [quick] [--url http://127.0.0.1:8000] [--no-face <image>]
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("quick", nargs="?")
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--no-face")
a = ap.parse_args()
results = []


def check(name, ok, extra=""):
    results.append((name, bool(ok), extra))


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1280, "height": 900}, accept_downloads=True)
    errors, console = [], []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
    pg.goto(a.url, wait_until="load")
    pg.wait_for_selector("#modes input", timeout=30000)
    pg.screenshot(path=f"{a.out}_0_empty.png", full_page=True)
    check("header + nav links", pg.locator(".g-header nav a").count() == 2)
    check("check button disabled until a file is chosen", pg.locator("#go").is_disabled())

    # wrong file type -> friendly message
    bad = Path(a.out + "_bad.txt")
    bad.write_text("not a video")
    pg.locator("#file").set_input_files(str(bad))
    pg.locator("#go").click()
    pg.wait_for_selector("#message:not([hidden])", timeout=15000)
    check("wrong file type: friendly message", "file type" in pg.locator("#message-title").inner_text().lower(),
          pg.locator("#message-title").inner_text())
    bad.unlink()

    # the real check, with progress
    pg.locator("#file").set_input_files(a.video)
    if a.quick:
        pg.locator('input[name="mode"][value="Quick"]').check()
    pg.locator("#go").click()
    seen = set()
    for _ in range(600):
        if not pg.locator("#status").is_hidden():
            seen.add(pg.locator("#status-title").inner_text().split("…")[0])
        if pg.locator("#result").is_visible() or pg.locator("#message").is_visible():
            break
        pg.wait_for_timeout(200)
    check("progress shown while working", len(seen) >= 2, " / ".join(sorted(seen)))
    pg.wait_for_timeout(800)
    head = pg.locator("#r-headline").inner_text() if pg.locator("#result").is_visible() else pg.locator("#message-title").inner_text()
    check("verdict card", pg.locator("#result").is_visible(), head)
    check("3 clue cards", pg.locator(".g-reason").count() == 3, str(pg.locator(".g-reason").count()))
    check("next-steps card visible", pg.get_by_role("button", name="Check another file").is_visible())
    pg.screenshot(path=f"{a.out}_1_result.png", full_page=True)

    with pg.expect_download(timeout=30000) as dl:
        pg.locator("#r-save").click()
    saved = Path(f"{a.out}_report.html")
    dl.value.save_as(saved)
    check("Save this result downloads report", saved.stat().st_size > 10000 and "SHA-256" in saved.read_text(encoding="utf-8"),
          f"{saved.stat().st_size} bytes")

    pg.get_by_text("Show technical details").click()
    pg.wait_for_timeout(500)
    check("tech: summary text", "probability" in pg.locator("#t-summary").inner_text().lower(), pg.locator("#t-summary").inner_text()[:120])
    n_gal = pg.locator("#t-gallery img").count()
    check("tech: gallery filled", n_gal >= 4, str(n_gal))
    before = pg.locator("#t-gallery img").first.get_attribute("src")
    pg.locator("#t-concept").select_option(label="Mouth region edited")
    changed = False
    for _ in range(50):
        pg.wait_for_timeout(200)
        if pg.locator("#t-gallery img").first.get_attribute("src") != before:
            changed = True
            break
    check("tech: concept overlay updates", changed)
    pg.locator("#t-off label", has_text="Blend strength").locator("input").check()
    pg.locator("#t-rescore").click()
    txt = ""
    for _ in range(100):
        pg.wait_for_timeout(200)
        txt = pg.locator("#t-rescore-out").inner_text()
        if "switched off" in txt:
            break
    check("tech: re-score", "switched off" in txt, txt[:160])
    pg.screenshot(path=f"{a.out}_2_tech.png", full_page=True)

    pg.evaluate("window.scrollTo(0,0)")
    pg.locator(".g-header nav a", has_text="What it can't do").click()
    pg.wait_for_timeout(900)
    check("nav: What it can't do scrolls", pg.evaluate("window.scrollY") > 300, str(pg.evaluate("window.scrollY")))
    pg.evaluate("window.scrollTo(0,0)")
    pg.locator(".g-header nav a", has_text="How it works").click()
    pg.wait_for_timeout(900)
    check("nav: How it works scrolls", pg.evaluate("window.scrollY") > 300, str(pg.evaluate("window.scrollY")))

    pg.set_viewport_size({"width": 390, "height": 844})
    pg.wait_for_timeout(500)
    check("phone (result): no horizontal scroll", pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"),
          str(pg.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")))
    pg.screenshot(path=f"{a.out}_4_phone_result.png", full_page=True)
    pg.set_viewport_size({"width": 1280, "height": 900})

    pg.get_by_role("button", name="Check another file").click()
    pg.wait_for_timeout(800)
    check("Check another file clears", pg.locator("#result").is_hidden() and pg.locator("#go").is_disabled())

    # cancel while working
    pg.locator("#file").set_input_files(a.video)
    pg.locator("#go").click()
    pg.wait_for_selector("#status:not([hidden])", timeout=10000)
    pg.wait_for_timeout(700)
    pg.locator("#cancel").click()
    pg.wait_for_timeout(1500)
    check("cancel stops the check", pg.locator("#status").is_hidden() and pg.locator("#result").is_hidden())

    if a.no_face:
        pg.wait_for_timeout(2500)  # the cancelled job has to end before this client may start another
        pg.locator("#file").set_input_files(a.no_face)
        pg.locator("#go").click()
        pg.wait_for_selector("#message:not([hidden])", timeout=60000)
        check("no face: friendly message", "face" in pg.locator("#message-title").inner_text().lower(),
              pg.locator("#message-title").inner_text())

    pg.set_viewport_size({"width": 390, "height": 844})
    pg.wait_for_timeout(500)
    check("phone (start): no horizontal scroll", pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"))
    pg.screenshot(path=f"{a.out}_3_phone.png", full_page=True)
    check("no page errors", not errors, "; ".join(errors)[:200])
    check("no console errors (CSP etc.)", not [c for c in console if "status of 4" not in c], "; ".join(console)[:200])
    b.close()

for n, ok, extra in results:
    print(("PASS " if ok else "FAIL ") + n + (f"  [{extra}]" if extra else ""))
print(f"{sum(ok for _, ok, _ in results)}/{len(results)} passed")
