"""Teste ponta a ponta da interface (navegador real). Opcional; não roda no pytest padrão.

Pré-requisitos: servidor ativo (uvicorn backend.main:app --port 8000) e
`pip install -r requirements-dev.txt`. Usa o Microsoft Edge instalado.

Cobre a interface de 3 abas: Simular um imóvel (abre primeiro), Imóveis reais e
Como o modelo funciona. Salva capturas de cada tela.

Uso: python tests/e2e_ui.py [pasta_para_screenshots]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE = "http://localhost:8000"


def price(page) -> int:
    return int(re.sub(r"\D", "", page.inner_text("#result-value")) or 0)


def wait_price_change(page, old: int, timeout: int = 8000) -> int:
    page.wait_for_function(f"Number(document.querySelector('#result-value').textContent.replace(/\\D/g,'')) !== {old}", timeout=timeout)
    page.wait_for_timeout(450)
    return price(page)


def run(shots: Path) -> None:
    shots.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)

        # 1. Simular um imóvel (aba inicial)
        page.goto(BASE + "/")
        expect(page.get_by_role("tab", name=re.compile("Simular"))).to_have_attribute("aria-selected", "true", timeout=20000)
        page.wait_for_function("/\\d/.test(document.querySelector('#result-value').textContent)", timeout=20000)
        p0 = price(page)
        expect(page.locator("#result-range")).to_contain_text("US$")
        expect(page.locator("#similar-list tbody tr")).to_have_count(4, timeout=10000)
        expect(page.locator("#cmpx-table")).to_contain_text("não foi vendida")
        page.screenshot(path=str(shots / "01_simular.png"))
        page.locator('[data-field="overall_qual"] input[value="9"]').check(force=True)
        p1 = wait_price_change(page, p0)
        assert p1 > p0, (p0, p1)  # restrição monotônica: mais qualidade nunca baixa o preço
        num = page.locator("#in-gr_liv_area_m2")
        num.fill("99999")
        num.press("Enter")
        expect(page.locator('[data-field="gr_liv_area_m2"] .field__err')).to_contain_text("m²")
        page.click("#btn-typical")
        page.wait_for_timeout(900)
        page.locator("#cmp-similar").screenshot(path=str(shots / "02_sua_casa_x_vendidas.png"))
        page.wait_for_selector("#chart-sens .main-svg", timeout=10000)

        # 2. Imóveis reais
        page.get_by_role("tab", name=re.compile("Imóveis reais")).click()
        expect(page.locator("#real-body")).to_be_visible(timeout=20000)
        expect(page.locator("#real-stats")).to_contain_text("292")
        assert page.locator("#real-table tbody tr").count() == 292
        expect(page.locator("#detail .pair__tile--real")).to_contain_text("US$")
        expect(page.locator("#detail .pair__tile--model")).to_contain_text("US$")
        page.screenshot(path=str(shots / "03_imoveis_reais.png"))
        page.select_option("#f-result", "good")
        expect(page.locator("#real-table tbody tr").first).to_contain_text("Perto do real")
        page.select_option("#f-result", "bad")
        expect(page.locator("#real-table tbody tr").first).to_contain_text("Erro grande")
        page.select_option("#f-result", "all")
        page.click("#btn-open-sim")
        expect(page.locator("#result-compare .pair__tile--real")).to_contain_text("US$", timeout=10000)
        expect(page.locator("#origin")).to_contain_text("Voltar para a casa que você montou")

        # 3. Como o modelo funciona
        page.get_by_role("tab", name=re.compile("Como o modelo funciona")).click()
        expect(page.locator("#story li")).to_have_count(5, timeout=10000)
        assert "undefined" not in page.inner_text("#story")
        page.wait_for_selector("#chart-perm .main-svg", timeout=10000)
        page.screenshot(path=str(shots / "04_como_funciona.png"))
        browser.close()

    bad = [e for e in errors if "favicon" not in e]
    assert not bad, bad
    print("E2E OK — screenshots em", shots)


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/reports/ui_screenshots"))
