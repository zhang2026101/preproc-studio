from playwright.sync_api import sync_playwright
errs=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8770", wait_until="networkidle"); pg.wait_for_timeout(1500)
    items = pg.query_selector_all(".sample-item")
    print("样本数:", len(items))
    # 检查缩略图是否真的加载成功
    ok = pg.evaluate("""() => {
        const imgs = [...document.querySelectorAll('.sample-item .thumb')];
        return imgs.map(i => ({src: i.currentSrc.slice(-40), w: i.naturalWidth, h: i.naturalHeight}));
    }""")
    for o in ok[:5]:
        print(f"  缩略图 {o['w']}x{o['h']}  {o['src']}")
    # 检查顶部状态
    print("health:", pg.inner_text("#health"))
    if items: items[0].click(); pg.wait_for_timeout(3000)
    pg.screenshot(path="shot_thumb_fixed.png")
    print("错误:", errs[:3])
    b.close()
