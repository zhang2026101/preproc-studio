from playwright.sync_api import sync_playwright
errs=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8770", wait_until="networkidle"); pg.wait_for_timeout(1200)
    pg.query_selector_all(".sample-item")[0].click(); pg.wait_for_timeout(3400)
    print("链来源:", pg.inner_text("#chainHint"))
    print("轮次:")
    for r in pg.query_selector_all(".round-row"):
        print("  ", r.inner_text().replace("\n"," | ")[:76])
    for n in pg.query_selector_all(".round-note")[:3]:
        print("   注:", n.inner_text()[:62])
    pg.screenshot(path="docs/screenshot_phase.png")
    # 顺便验证批量弹窗
    pg.click("#runAllBtn"); pg.wait_for_timeout(700)
    opts = [o.inner_text() for o in pg.query_selector_all("#runMode option")]
    print("批量模式:", opts)
    pg.screenshot(path="docs/screenshot_batch.png")
    print("错误:", errs[:2])
    b.close()
