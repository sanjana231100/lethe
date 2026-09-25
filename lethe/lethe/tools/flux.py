"""Black Forest Labs FLUX (stretch): a 'price drop' hero graphic for repriced units.

Verified shape (docs.bfl.ai, Sept 2026): POST https://api.bfl.ai/v1/flux-2-pro  header x-key,
body {prompt, width, height, output_format, safety_tolerance} -> {id, polling_url}; poll polling_url
until status == "Ready", image at result.sample (URL expires in 10 minutes -> download immediately).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from ..config import ROOT, Settings, settings as default_settings


class FluxClient:
    name = "flux"

    async def price_drop_graphic(self, label: str, old_price: float, new_price: float, out_dir: Path) -> Path | None:
        raise NotImplementedError


class MockFlux(FluxClient):
    name = "flux-mock"

    async def price_drop_graphic(self, label: str, old_price: float, new_price: float, out_dir: Path) -> Path | None:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{label.replace(' ', '_')}.svg"
        p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="512"><rect width="100%" height="100%" fill="#0b0f14"/>'
                     f'<text x="50%" y="40%" fill="#e6edf3" font-size="44" text-anchor="middle">{label}</text>'
                     f'<text x="50%" y="62%" fill="#3ddc97" font-size="64" text-anchor="middle">${new_price:,.0f}</text>'
                     f'<text x="50%" y="78%" fill="#8b98a8" font-size="28" text-anchor="middle">was ${old_price:,.0f} (mock FLUX)</text></svg>')
        return p


class FluxAPI(FluxClient):
    name = "flux-live"
    BASE = "https://api.bfl.ai"

    def __init__(self, api_key: str, model: str = "flux-2-pro"):
        self.api_key = api_key
        self.model = model

    async def price_drop_graphic(self, label: str, old_price: float, new_price: float, out_dir: Path) -> Path | None:
        prompt = (f"Clean dealership marketing hero image of a {label}, studio lighting, wide shot, no text overlays, "
                  f"photorealistic, neutral background")
        headers = {"x-key": self.api_key, "accept": "application/json"}
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{self.BASE}/v1/{self.model}", headers=headers,
                             json={"prompt": prompt, "width": 1024, "height": 576, "output_format": "jpeg", "safety_tolerance": 2})
            r.raise_for_status()
            sub = r.json()
            polling_url, task_id = sub["polling_url"], sub["id"]
            delay = 0.5
            for _ in range(240):
                await asyncio.sleep(delay)
                p = await c.get(polling_url, headers=headers, params={"id": task_id})
                if p.status_code == 429:
                    delay = min(delay * 2, 5)
                    continue
                p.raise_for_status()
                res = p.json()
                st = res.get("status")
                if st == "Ready":
                    img = await c.get(res["result"]["sample"], timeout=60)
                    img.raise_for_status()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out = out_dir / f"{label.replace(' ', '_')}.jpg"
                    out.write_bytes(img.content)
                    return out
                if st in ("Error", "Failed", "Content Moderated", "Request Moderated", "Task not found"):
                    raise RuntimeError(f"flux {st}: {res.get('details')}")
                delay = 0.5
        raise TimeoutError("flux generation timed out")


def make_flux(cfg: Settings | None = None) -> FluxClient:
    cfg = cfg or default_settings
    if cfg.mock or not cfg.bfl_api_key:
        return MockFlux()
    return FluxAPI(cfg.bfl_api_key)
