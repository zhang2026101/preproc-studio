from playwright.sync_api import sync_playwright
errs=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://127.0.0.1:8770", wait_until="networkidle"); pg.wait_for_timeout(1000)
    pg.click("#settingsBtn"); pg.wait_for_timeout(900)
    pg.screenshot(path="docs/ui_modal.png")
    print("弹窗截图完成 | 错误:", errs[:2])
    b.close()
