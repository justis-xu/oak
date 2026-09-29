"""模型探测：验证双档端点连通性（普通 + json_mode），结果落 runs/model_probe.json。

用法：uv run python -m locomo.probe_models
"""
from __future__ import annotations

import asyncio
import json
import time

from .config import NS_PROBE, load_locomo_config
from oak.llm.client import LLMClient


async def _probe(client: LLMClient, role: str, json_mode: bool) -> dict:
    t0 = time.time()
    try:
        r = await client.chat(
            role=role, use_cache=False, namespace=NS_PROBE, json_mode=json_mode,
            temperature=0.0, max_tokens=512,
            messages=[
                {"role": "system", "content": "你是连通性探测助手。"},
                {"role": "user", "content": (
                    '回复 JSON：{"ok": true, "model": "你的模型名"}'
                    if json_mode else "请原样回复：PONG")},
            ],
        )
        return {"ok": bool(r.content.strip()), "model": r.model,
                "latency_s": round(time.time() - t0, 1),
                "reply": r.content.strip()[:120]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)[:300],
                "latency_s": round(time.time() - t0, 1)}


async def main() -> None:
    lc = load_locomo_config()
    client = LLMClient(lc.cfg)
    print(f"strong 端点: {lc.cfg.api_base_url}  模型: {lc.cfg.model_strong}")
    print(f"fast   端点: {lc.cfg.fast_base_url}  模型: {lc.cfg.model_fast}")
    strong = await _probe(client, "locomo_judge", json_mode=True)
    fast_plain = await _probe(client, "locomo_extract", json_mode=False)
    fast_json = await _probe(client, "locomo_extract", json_mode=True)
    result = {
        "ts": time.time(),
        "strong": strong, "fast_plain": fast_plain, "fast_json": fast_json,
        "fast_ok": fast_plain.get("ok") and fast_json.get("ok"),
        "fast_model": lc.cfg.model_fast,
        "fallback_model": "glm-5.3-flash",
    }
    lc.probe_path().write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not strong.get("ok"):
        raise SystemExit("strong 档（glm-5.3/智谱）不可用——检查 ZHIPU_API_KEY")
    if not result["fast_ok"]:
        print(f"!! fast 档不可用，后续 load_locomo_config 将回退 {result['fallback_model']}")


if __name__ == "__main__":
    asyncio.run(main())
