"""界面交互验收：逐项点开，截图，收集控制台错误"""
import sys
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8770"
errs = []

def snap(pg, name):
    pg.screenshot(path=f"shot_{name}.png")
    print("  截图 shot_%s.png" % name)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1600, "height": 1000})
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append("PAGEERROR: " + str(e)))

    pg.goto(URL, wait_until="networkidle")
    pg.wait_for_timeout(1000)
    pg.query_selector_all(".sample-item")[0].click()
    pg.wait_for_timeout(2600)
    print("1. 载入 + 选中样本")

    pg.click('.mode-btn[data-mode="compare"]')
    pg.wait_for_timeout(700)
    snap(pg, "compare")
    pg.click('.mode-btn[data-mode="final"]')

    print("2. 添加算子")
    pg.click("#addOpBtn")
    pg.wait_for_timeout(600)
    snap(pg, "addop")
    before = len(pg.query_selector_all(".op-card"))
    pg.query_selector_all(".op-choice")[3].click()
    pg.wait_for_timeout(2600)
    after = len(pg.query_selector_all(".op-card"))
    print(f"  算子数 {before} -> {after}")
    snap(pg, "after_add")

    print("3. 改参数（拖滑杆）")
    sl = pg.query_selector_all('.op-card input[type="range"]')
    if sl:
        box = sl[0].bounding_box()
        pg.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] / 2)
        pg.mouse.down()
        pg.mouse.move(box["x"] + box["width"] * 0.82, box["y"] + box["height"] / 2, steps=8)
        pg.mouse.up()
        pg.wait_for_timeout(2200)
        print("  已拖动滑杆")
        snap(pg, "after_param")

    print("4. 删除算子")
    pg.hover(".op-card")
    del_btn = pg.query_selector('.op-card [data-act="del"]')
    if del_btn:
        del_btn.click()
        pg.wait_for_timeout(2200)
        print("  算子数 ->", len(pg.query_selector_all(".op-card")))

    print("5. 准入表")
    pg.click("#preflightBtn")
    pg.wait_for_timeout(6000)
    snap(pg, "preflight")
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(400)

    print("6. 改动流水")
    pg.click("#editsBtn")
    pg.wait_for_timeout(1500)
    snap(pg, "edits")
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(400)

    print("7. 设置")
    pg.click("#settingsBtn")
    pg.wait_for_timeout(1200)
    snap(pg, "settings")
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(300)

    print("8. 换一条 + 通过核验")
    pg.click("#rejectBtn")
    pg.wait_for_timeout(2600)
    snap(pg, "reject")
    pg.click("#acceptBtn")
    pg.wait_for_timeout(1800)
    flags = len(pg.query_selector_all(".flag.accepted"))
    print(f"  已通过核验标记数 {flags}")

    print("9. 聚合 + 导出")
    pg.click("#aggregateBtn")
    pg.wait_for_timeout(2500)
    snap(pg, "aggregate")
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(300)
    pg.click("#exportBtn")
    pg.wait_for_timeout(9000)
    snap(pg, "export")

    print("\n控制台错误:", len(errs))
    for e in errs[:10]:
        print("  -", e[:180])
    b.close()
