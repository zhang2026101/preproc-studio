import sys
from playwright.sync_api import sync_playwright
errs=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8770", wait_until="networkidle"); pg.wait_for_timeout(1000)
    pg.query_selector_all(".sample-item")[6].click(); pg.wait_for_timeout(2600)
    before = len(pg.query_selector_all(".op-card"))
    print("点击 AI 重出方案…")
    pg.click("#aiAgainBtn")
    pg.wait_for_timeout(900)
    pg.screenshot(path="shot_ai_loading.png")
    print("  载入中状态已截图")
    pg.wait_for_timeout(6000)
    after = len(pg.query_selector_all(".op-card"))
    diag = pg.inner_text("#diagBody").replace("\n"," | ")[:120]
    rounds = len(pg.query_selector_all(".round-row"))
    print(f"  算子 {before} -> {after}, 轮次行 {rounds}")
    print(f"  诊断: {diag}")
    print(f"  状态栏: {pg.inner_text('#statusText')} / {pg.inner_text('#statusMeta')}")
    pg.screenshot(path="shot_ai_done.png")
    print("错误:", len(errs), errs[:3])
    b.close()
