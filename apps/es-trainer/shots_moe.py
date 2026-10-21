from playwright.sync_api import sync_playwright
U="http://127.0.0.1:8765/"; D="screenshots/"
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":390,"height":844},device_scale_factor=2); pg.goto(U); pg.wait_for_timeout(1200)
    pg.click("text=MoE from scratch"); pg.select_option("#data",label="qa_sample.jsonl"); pg.fill("#run","moe_ui")
    pg.evaluate("scrollTo(0,0)"); pg.wait_for_timeout(500); pg.screenshot(path=D+"6_moe_setup.png")
    pg.evaluate("document.querySelector('.card.moe').scrollIntoView()"); pg.wait_for_timeout(400); pg.screenshot(path=D+"6b_moe_options.png")
    pg.click("#go"); pg.wait_for_function("document.getElementById('pilltxt').textContent=='Training'",timeout=20000); pg.wait_for_timeout(15000); pg.screenshot(path=D+"7_moe_training.png"); pg.wait_for_function("document.getElementById('pilltxt').textContent=='Done'",timeout=120000); pg.wait_for_timeout(1500)
    pg.screenshot(path=D+"7b_moe_done.png"); b.close()
