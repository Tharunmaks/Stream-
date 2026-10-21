from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":390,"height":844},device_scale_factor=2); pg.goto("http://127.0.0.1:8765/"); pg.wait_for_timeout(1200)
    pg.select_option("#data",label="qa_sample.jsonl"); pg.fill("#run","ui_demo"); pg.evaluate("scrollTo(0,0)"); pg.wait_for_timeout(700)
    pg.screenshot(path="screenshots/1_setup.png"); b.close()
