import sys
from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8770"
out = sys.argv[2] if len(sys.argv) > 2 else "out_ui.png"

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1600, "height": 1000})
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append("PAGEERROR: " + str(e)))
    pg.goto(url, wait_until="networkidle")
    pg.wait_for_timeout(1200)
    items = pg.query_selector_all(".sample-item")
    print("样本项:", len(items))
    if items:
        items[0].click()
        pg.wait_for_timeout(2600)
    pg.screenshot(path=out)
    print("截图:", out)
    if errs:
        print("控制台错误:")
        for e in errs[:8]:
            print("  ", e[:200])
    else:
        print("无控制台错误")
    b.close()
