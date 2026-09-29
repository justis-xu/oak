"""阶段0 验收：Java / owlready2+HermiT / 双模型调用+缓存 / 官方 eval.py 连通。"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from datasets.travelplanner.pipeline.config_task import load_config                     # noqa: E402
from oak.llm.client import LLMClient                   # noqa: E402


def check_java() -> None:
    r = subprocess.run(["java", "-version"], capture_output=True, text=True)
    assert r.returncode == 0, f"java not available:\n{r.stderr}"
    print("[1/5] java OK:", (r.stderr or r.stdout).splitlines()[0])


def check_hermit() -> None:
    r = subprocess.run(
        [sys.executable, "-c", """
import owlready2 as owl
with owl.default_world:
    class A(owl.Thing): pass
    class B(owl.Thing): pass
    owl.AllDisjoint([A, B])
    class C(A): pass
owl.sync_reasoner_hermit(owl.default_world)
inc = list(owl.default_world.inconsistent_classes())
print("hermit-ran-ok", "inconsistent:", inc)
"""], capture_output=True, text=True, cwd=PROJECT)
    assert "hermit-ran-ok" in r.stdout, f"HermiT smoke failed:\n{r.stdout}\n{r.stderr}"
    print("[2/5] owlready2 + HermiT OK:", r.stdout.strip())


async def check_llm() -> None:
    cfg = load_config()
    client = LLMClient(cfg)
    for role, model in (("schema", cfg.model_strong), ("react", cfg.model_fast)):
        res = await client.chat(role=role, namespace="smoke",
                                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                                temperature=0.0, max_tokens=64, use_cache=False)
        assert res.content.strip().lower().startswith("ok"), f"{role} unexpected: {res.content!r}"
        print(f"      {role} -> {model}: {res.content.strip()[:20]!r} "
              f"({res.usage.get('prompt_tokens')}+{res.usage.get('completion_tokens')} tok, {res.elapsed_s}s)")
    # 缓存命中
    res2 = await client.chat(role="react", namespace="smoke",
                             messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                             temperature=0.0, max_tokens=64)
    assert res2.cache_hit, "cache miss on identical request"
    print("[3/5] LLM dual-model + cache OK")


def check_eval_official() -> None:
    cfg = load_config()
    ev = cfg.tp_root / "evaluation"
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "empty_plans.jsonl"
        f.write_text("\n".join(json.dumps({"query_idx": i, "plan": []})
                               for i in range(45)))
        r = subprocess.run(
            [sys.executable, "eval.py", "--set_type", "train",
             "--evaluation_file_path", str(f)],
            cwd=ev, capture_output=True, text=True, timeout=1800,
        )
        assert r.returncode == 0, f"official eval failed:\n{r.stderr[-1500:]}"
        rates = [ln for ln in r.stdout.splitlines() if "%" in ln]
        assert rates, f"no rates in output:\n{r.stdout[:500]}"
        print("[4/5] official eval.py (train, 45 empty plans) OK:")
        for ln in rates:
            print("      ", ln)


def check_database() -> None:
    cfg = load_config()
    files = [
        "database/flights/clean_Flights_2022.csv",
        "database/accommodations/clean_accommodations_2022.csv",
        "database/restaurants/clean_restaurant_2022.csv",
        "database/attractions/attractions.csv",
        "database/googleDistanceMatrix/distance.csv",
        "database/background/citySet_with_states.txt",
    ]
    for f in files:
        p = cfg.tp_root / f
        assert p.exists() and p.stat().st_size > 0, f"missing {f}"
    print("[5/5] database files OK")


if __name__ == "__main__":
    import asyncio
    load_config()                     # 注入 JAVA_HOME→PATH
    check_java()
    check_hermit()
    asyncio.run(check_llm())
    check_eval_official()
    check_database()
    print("\nALL ENV CHECKS PASSED")
