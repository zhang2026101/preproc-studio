from playwright.sync_api import sync_playwright
import cv2
errs=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1600,"height":1000})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.goto("http://127.0.0.1:8770", wait_until="networkidle"); pg.wait_for_timeout(1200)
    pg.query_selector_all(".sample-item")[0].click(); pg.wait_for_timeout(6000)
    n = pg.evaluate("""() => [...document.querySelectorAll('.op-card .op-thumb')].map(e => e.tagName + ':' + (e.naturalWidth||0))""")
    print("算子缩略图:", n)
    pg.screenshot(path="docs/ui_light.png")
    import numpy as np
    from PIL import Image
    im = Image.open("docs/ui_light.png").convert("L")
    im.save("docs/ui_light_gray.png")
    print("截图完成")
    print("错误:", errs[:3])
    b.close()
