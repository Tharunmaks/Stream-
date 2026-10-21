import hmac, hashlib, json
from playwright.sync_api import sync_playwright
S, T, D = "http://127.0.0.1:8780", "http://127.0.0.1:8781", "/workspace/es-apps/screenshots/"
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1280, "height": 860})
    pg.goto(S + "/login"); pg.fill("input[name=email]", "tharun.test@example.com"); pg.fill("input[name=password]", "testpass123"); pg.click("form button")
    pg.wait_for_url(S + "/")
    pg.goto(T + "/"); pg.fill("#name", "nb-tiny"); pg.click("#run")
    pg.wait_for_function("document.getElementById('out').textContent.includes('[exit code')", timeout=300000)
    pg.screenshot(path=D + "06_trainer_trained.png", full_page=True); print("trainer:", pg.inner_text("#out")[-400:])
    pg.goto(S + "/run"); pg.click("#go"); pg.wait_for_function("document.getElementById('o').textContent.includes('tok/s')", timeout=60000)
    pg.screenshot(path=D + "07_run.png", full_page=True); print("run:", pg.inner_text("#o"))
    pg.goto(S + "/mcp"); pg.screenshot(path=D + "08_mcp.png", full_page=True)
