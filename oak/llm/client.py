"""统一 LLM 客户端：模型路由、并发限制、退避重试、磁盘缓存、成本台账。

全项目所有 LLM 调用的唯一通道——任何模块不得直接实例化 OpenAI client。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError
import httpx

from ..config import Config, MODEL_ROLES


class BudgetExceeded(RuntimeError):
    pass


# 思考型模型的 reasoning token 余量（正文预算之外追加的请求侧 max_tokens）
REASONING_BUFFER = 3072

# 关闭深度思考的角色（机械执行类任务：抽取/ReAct/槽位/格式修复走 flash 且无需长思考）；
# 核心推理步骤（schema 草拟、函数编译、评判器）保留思考
THINKING_OFF_ROLES = {"kg", "react", "slots", "plan_repair"}


@dataclass
class LLMResult:
    content: str
    usage: dict
    cache_hit: bool
    elapsed_s: float
    model: str
    role: str


class LLMClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # 直连不走系统代理（本地代理会吞掉国内 API 的长请求）
        def _mk(base_url: str, api_key: str) -> AsyncOpenAI:
            return AsyncOpenAI(
                base_url=base_url, api_key=api_key, timeout=300.0,
                http_client=httpx.AsyncClient(trust_env=False, timeout=300.0),
            )
        self._client = _mk(cfg.api_base_url, cfg.api_key)             # strong 档
        self._client_fast = (_mk(cfg.fast_base_url, cfg.fast_api_key)
                             if cfg.fast_base_url != cfg.api_base_url
                             or cfg.fast_api_key != cfg.api_key
                             else self._client)                        # fast 档（可异站）
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        self._ledger_lock = asyncio.Lock()
        self._retry_log = cfg.work_dir / "logs" / "retry.jsonl"
        self._retry_log.parent.mkdir(parents=True, exist_ok=True)
        # 命名空间调用计数（内存 + 台账聚合），供限额检查
        self._ns_calls: dict[str, int] = {}

    # ---------- 对外主入口 ----------
    async def chat(
        self,
        *,
        role: str,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = False,
        namespace: str = "default",
        use_cache: bool = True,
    ) -> LLMResult:
        model = self.cfg.model_for(role)
        key = self._cache_key(model, messages, temperature, max_tokens, json_mode)
        cache_file = self.cfg.cache_dir / namespace / f"{key}.json"

        if use_cache and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                self._count(namespace)
                return LLMResult(
                    content=data["content"], usage=data.get("usage", {}),
                    cache_hit=True, elapsed_s=0.0, model=model, role=role,
                )
            except Exception:
                pass  # 缓存损坏则重打

        self._check_budget(namespace)

        last_err: Exception | None = None
        use_client = (self._client_fast if MODEL_ROLES.get(role) == "fast"
                      else self._client)
        is_glm_endpoint = "bigmodel" in str(getattr(use_client, "base_url", "") or "")
        for attempt in range(self.cfg.max_retries):
            try:
                async with self._sem:
                    t0 = time.time()
                    # glm-5.3 系列为思考型模型：reasoning_content 消耗 completion 预算，
                    # 请求侧加 reasoning 余量，保证正文拿满 max_tokens
                    kwargs: dict[str, Any] = dict(
                        model=model, messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens + REASONING_BUFFER,
                        stream=True,                          # 流式：防 TUN 代理掐长连接
                        stream_options={"include_usage": True},
                        timeout=300.0,
                    )
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    if role in THINKING_OFF_ROLES and is_glm_endpoint:
                        # thinking 开关仅智谱端点支持；第三方网关忽略
                        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                    # 流式聚合（本地 TUN 代理会挂起非流式长请求）
                    parts: list[str] = []
                    usage: dict = {}
                    async for chunk in await use_client.chat.completions.create(**kwargs):
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                parts.append(delta.content)
                        if getattr(chunk, "usage", None):
                            usage = {
                                "prompt_tokens": chunk.usage.prompt_tokens or 0,
                                "completion_tokens": chunk.usage.completion_tokens or 0,
                            }
                    content = "".join(parts)
                result = LLMResult(
                    content=content, usage=usage, cache_hit=False,
                    elapsed_s=round(time.time() - t0, 2), model=model, role=role,
                )
                # 空正文（思考吃光预算）不入缓存，下次调用自动重试
                if content.strip():
                    self._save_cache(cache_file, result)
                else:
                    result.usage["empty_content"] = True
                await self._append_ledger(namespace, role, model, usage, cache_hit=False)
                self._count(namespace)
                return result
            except (APIConnectionError, APITimeoutError) as e:
                last_err = e
            except APIStatusError as e:
                last_err = e
                if e.status_code not in (429, 500, 502, 503, 504):
                    raise                      # 4xx（除 429）不重试
                if e.status_code == 429 and json_mode:
                    # 有些网关不支持 json_object 参数 → 降级纯 prompt 约束后重试一次
                    json_mode = False
            # 指数退避 + 抖动（429 首次退避从 8s 起，避免反复撞限流）
            base = 8.0 if (isinstance(last_err, APIStatusError)
                           and getattr(last_err, "status_code", None) == 429) else 1.0
            backoff = min(60.0, base * (2 ** attempt)) + random.uniform(0, 2)
            self._log_retry(namespace, role, attempt, repr(last_err), backoff)
            await asyncio.sleep(backoff)

        raise RuntimeError(
            f"LLM 调用在 {self.cfg.max_retries} 次重试后仍失败: {role=} {last_err=!r}"
        )

    def chat_sync(self, **kwargs) -> LLMResult:
        return asyncio.run(self.chat(**kwargs))

    # ---------- 限额与台账 ----------
    def _count(self, namespace: str) -> None:
        self._ns_calls[namespace] = self._ns_calls.get(namespace, 0) + 1

    def _check_budget(self, namespace: str) -> None:
        # 命名空间 r{n}_* 受单轮上限约束；inf_q* 受单题上限约束
        n = self._ns_calls.get(namespace, 0)
        if namespace.startswith("r") and "_inf" not in namespace:
            limit = self.cfg.limits.get("build_round_calls")
            if limit and n >= limit:
                raise BudgetExceeded(f"{namespace}: {n} >= build_round_calls={limit}")
        elif namespace.startswith("inf_q"):
            limit = self.cfg.limits.get("inference_calls_per_q")
            if limit and n >= limit:
                raise BudgetExceeded(f"{namespace}: {n} >= inference_calls_per_q={limit}")

    async def _append_ledger(self, namespace, role, model, usage, cache_hit: bool) -> None:
        async with self._ledger_lock:
            with self.cfg.ledger_path.open("a") as f:
                f.write(json.dumps({
                    "ts": round(time.time(), 1), "namespace": namespace,
                    "role": role, "model": model, "cache_hit": cache_hit,
                    **usage,
                }, ensure_ascii=False) + "\n")

    def ledger_summary(self) -> dict:
        """汇总台账：按模型/命名空间的调用数与 token。"""
        summary: dict[str, Any] = {"total_calls": 0, "cache_hits": 0,
                                   "prompt_tokens": 0, "completion_tokens": 0,
                                   "by_model": {}, "by_ns": {}}
        if not self.cfg.ledger_path.exists():
            return summary
        for line in self.cfg.ledger_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            summary["total_calls"] += 1
            if rec.get("cache_hit"):
                summary["cache_hits"] += 0  # 命中不写台账；此处恒 0，留字段
            summary["prompt_tokens"] += rec.get("prompt_tokens", 0)
            summary["completion_tokens"] += rec.get("completion_tokens", 0)
            m = summary["by_model"].setdefault(rec["model"], {"calls": 0})
            m["calls"] += 1
            ns = summary["by_ns"].setdefault(rec["namespace"], {"calls": 0})
            ns["calls"] += 1
        return summary

    # ---------- 缓存 ----------
    @staticmethod
    def _cache_key(model, messages, temperature, max_tokens, json_mode) -> str:
        blob = json.dumps({
            "m": model, "msg": messages, "t": temperature,
            "mt": max_tokens, "j": json_mode,
        }, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def _save_cache(path: Path, result: LLMResult) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "content": result.content, "usage": result.usage,
            "model": result.model,
        }, ensure_ascii=False))
        tmp.rename(path)                      # 原子替换，并发安全

    def _log_retry(self, namespace, role, attempt, err, backoff) -> None:
        try:
            with self._retry_log.open("a") as f:
                f.write(json.dumps({
                    "ts": round(time.time(), 1), "ns": namespace, "role": role,
                    "attempt": attempt, "err": err, "backoff_s": round(backoff, 1),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
