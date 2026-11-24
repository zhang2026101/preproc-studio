from playwright.sync_api import sync_playwright
errs=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8770", wait_until="networkidle"); pg.wait_for_timeout(900)
    pg.click("#settingsBtn"); pg.wait_for_timeout(800)
    pg.screenshot(path="shot_settings_banner.png")
    print("黄条存在:", pg.query_selector("#cfgBanner") is not None)
    # 点「改为线上模型」再测
    btn = pg.query_selector("#cfgBanner button")
    if btn:
        btn.click(); pg.wait_for_timeout(300)
        print("provider 现在:", pg.input_value("#cfgProvider"))
        pg.click("#modalRoot .modal-foot button"); pg.wait_for_timeout(1800)
        res = pg.query_selector("#testResult")
        print("测试结果:", (res.inner_text()[:110].replace("\n"," ") if res else "无"))
    pg.screenshot(path="shot_settings_after.png")
    print("错误:", errs[:3])
    b.close()
