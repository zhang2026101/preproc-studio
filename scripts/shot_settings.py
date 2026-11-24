from playwright.sync_api import sync_playwright
errs=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8770", wait_until="networkidle"); pg.wait_for_timeout(900)
    pg.click("#settingsBtn"); pg.wait_for_timeout(900)
    pg.screenshot(path="shot_settings_deepseek.png")
    print("预设下拉选项:", [o.inner_text() for o in pg.query_selector_all("#cfgPreset option")])
    print("当前 base:", pg.input_value("#cfgBase"))
    print("当前 model:", pg.input_value("#cfgModel"))
    print("detail:", pg.input_value("#cfgDetail"), "| style:", pg.input_value("#cfgStyle"))
    print("错误:", errs[:3])
    b.close()
