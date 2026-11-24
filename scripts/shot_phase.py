from playwright.sync_api import sync_playwright
errs=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8770", wait_until="networkidle"); pg.wait_for_timeout(1200)
    items = pg.query_selector_all(".sample-item")
    items[0].click(); pg.wait_for_timeout(3000)
    print("点 AI 重出方案（真实模型，约 8 秒）")
    pg.click("#aiAgainBtn"); pg.wait_for_timeout(14000)
    rows = pg.query_selector_all(".round-row")
    print("轮次行数:", len(rows))
    for r in rows:
        print("  ", r.inner_text().replace("\n"," | ")[:80])
    note = pg.query_selector(".round-note")
    print("理由行:", note.inner_text()[:70] if note else "无")
    print("链来源:", pg.inner_text("#chainHint"))
    print("诊断摘要:", pg.inner_text("#diagBody").replace("\n"," | ")[:160])
    pg.screenshot(path="shot_three_phase.png")
    print("错误:", errs[:3])
    b.close()
