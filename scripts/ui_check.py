"""Browser click-through check of the GRACE app (Playwright driving the installed Microsoft Edge; app must be running).

python scripts/ui_check.py cache/app_test_videos/fake_ffpp_Deepfakes_233_995.mp4 <out_prefix> [quick]
Checks: header links, verdict card, 3 clue cards, Save this result download, technical panel (summary, galleries,
concept overlay, switch-off re-score), nav scrolling, Check another file, no horizontal scroll at phone width, no page errors.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

video, out = sys.argv[1], sys.argv[2]
quick = len(sys.argv) > 3
results = []


def check(name, ok, extra=""):
    results.append((name, bool(ok), extra))


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1280, "height": 900}, accept_downloads=True)
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto("http://127.0.0.1:7860/", wait_until="load"); pg.wait_for_selector(".g-header", timeout=60000); pg.wait_for_timeout(2000)
    pg.screenshot(path=f"{out}_0_empty.png", full_page=True)
    check("header + nav links", pg.locator(".g-header nav a").count() == 2)

    pg.locator("input[type=file]").first.set_input_files(video)
    pg.wait_for_timeout(1500)
    if quick:
        pg.get_by_label("Quick check").check()
    pg.get_by_role("button", name="Check this file").click()
    pg.wait_for_selector(".g-hero h1, .g-msg h2", timeout=120000)
    pg.wait_for_timeout(1500)
    head = pg.locator(".g-hero h1, .g-msg h2").first.inner_text()
    check("verdict card", True, head)
    check("3 clue cards", pg.locator(".g-reason").count() == 3, str(pg.locator(".g-reason").count()))
    check("next-steps card visible", pg.get_by_role("button", name="Check another file").is_visible())
    pg.screenshot(path=f"{out}_1_result.png", full_page=True)

    with pg.expect_download(timeout=30000) as dl:
        pg.get_by_role("button", name="Save this result").click()
    d = dl.value
    saved = Path(f"{out}_report.html")
    d.save_as(saved)
    check("Save this result downloads report", saved.stat().st_size > 10000, f"{saved.stat().st_size} bytes")

    pg.get_by_text("Show technical details (for experts)").click()
    pg.wait_for_timeout(1500)
    check("tech: summary text", "probability" in pg.locator(".g-tech").inner_text().lower())
    n_gal = pg.locator(".g-tech .gallery img, .g-tech button.thumbnail-item img").count()
    check("tech: galleries filled", n_gal >= 4, str(n_gal))
    # concept overlay dropdown -> gallery updates (overlay endpoint with state)
    gimg = pg.locator("#g-gallery img").first
    before = gimg.get_attribute("src")
    pg.get_by_label("Concept map to overlay").click()
    pg.get_by_role("option", name="Mouth region edited").click()
    changed = False
    for _ in range(30):
        pg.wait_for_timeout(1000)
        if pg.locator("#g-gallery img").first.get_attribute("src") != before:
            changed = True
            break
    check("tech: concept overlay updates", changed)
    # switch-off test
    pg.get_by_label("Blend strength", exact=True).last.check()
    pg.get_by_role("button", name="Re-score").click()
    txt = ""
    for _ in range(60):
        pg.wait_for_timeout(1000)
        txt = pg.locator("#g-rescore").inner_text()
        if "switched off" in txt:
            break
    check("tech: re-score", "switched off" in txt, txt[:160])
    pg.screenshot(path=f"{out}_2_tech.png", full_page=True)

    # nav links scroll
    pg.evaluate("window.scrollTo(0,0)")
    pg.locator(".g-header nav a", has_text="What it can't do").click()
    pg.wait_for_timeout(800)
    check("nav: What it can't do scrolls", pg.evaluate("window.scrollY") > 300, str(pg.evaluate("window.scrollY")))
    pg.evaluate("window.scrollTo(0,0)")
    pg.locator(".g-header nav a", has_text="How it works").click()
    pg.wait_for_timeout(800)
    check("nav: How it works scrolls", pg.evaluate("window.scrollY") > 300, str(pg.evaluate("window.scrollY")))

    pg.get_by_role("button", name="Check another file").click()
    pg.wait_for_timeout(2000)
    check("Check another file clears", pg.locator(".g-hero").count() == 0 and not pg.get_by_role("button", name="Check another file").is_visible())
    pg.screenshot(path=f"{out}_3_cleared.png", full_page=True)

    pg.set_viewport_size({"width": 390, "height": 844})
    pg.wait_for_timeout(800)
    check("phone: no horizontal scroll", pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"),
          str(pg.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")))
    check("no page errors", not errors, "; ".join(errors)[:200])
    b.close()

for n, ok, extra in results:
    print(("PASS " if ok else "FAIL ") + n + (f"  [{extra}]" if extra else ""))
