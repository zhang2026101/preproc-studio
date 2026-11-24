from playwright.sync_api import sync_playwright
errs=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8770", wait_until="networkidle"); pg.wait_for_timeout(1000)
    pg.query_selector_all(".sample-item")[0].click(); pg.wait_for_timeout(3200)
    print("诊断面板:", pg.inner_text("#diagBody").replace("\n"," | ")[:130])
    print("轮次:")
    for r in pg.query_selector_all(".round-row"):
        print("  ", r.inner_text().replace("\n"," | ")[:78])
    notes = [n.inner_text()[:56] for n in pg.query_selector_all(".round-note")]
    for n in notes[:4]: print("   注:", n)
    pg.screenshot(path="docs/screenshot_phase.png")
    print("错误:", errs[:2])
    b.close()
