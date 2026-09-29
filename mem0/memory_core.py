"""mem0 只-ADD 记忆核心（V3 加法管线的最小蒸馏，中文场景）。

代码流程逐阶段对应 vendored mem0 2.1.0 `mem0/memory/main.py` 的
`_add_to_vector_store`（V3 PHASED BATCH PIPELINE，main.py:916-1075）：

  Phase 0  上下文收集：最近 10 条历史消息          main.py:919-921
  Phase 1  已有记忆检索（向量 top_k）→ id 映射       main.py:923-938
  Phase 2  LLM 加法提取（ADDITIVE 提示 + 自定义段）  main.py:940-984
  Phase 3  批量嵌入                                  main.py:991-1002
  Phase 4/5 逐条加工 + MD5 去重（对旧集与批内）      main.py:1005-1039
  Phase 6  批量持久化 + ADD 历史                      main.py:1045-1075

刻意裁剪（与原版的差异）：
- 无 UPDATE/DELETE/图/procedural/平台客户端/遥测——只保留 ADD 路径；
- 向量库用内置 TinyVectorStore（JSONL + 余弦线性扫描）替代 Chroma，接口
  （insert/search）与 mem0 的 vector_store 抽象同形，量级（百~千条记忆）无压力；
- LLM 提取直连 oak.llm.LLMClient（复用框架的路由/缓存/重试），role=mem0_extract；
- 嵌入走 OpenAI 兼容 /embeddings（env：MEM0_EMBED_BASE_URL/KEY/MODEL）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from oak.llm.client import LLMClient

from .prompts import (ADDITIVE_EXTRACTION_PROMPT, CUSTOM_INSTRUCTIONS_ZH,
                      generate_additive_user_prompt, parse_extraction)

LAST_K = 10          # Phase 0 的最近消息窗口（main.py:920 limit=10）
EXISTING_TOP_K = 10  # Phase 1 的已有记忆检索数（main.py:929 top_k=10）


# ---------------------------------------------------------------- 嵌入
class Embedder:
    """OpenAI 兼容 /embeddings 薄封装（同步；每批一次请求）。"""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None):
        self.base_url = (base_url or os.environ.get("MEM0_EMBED_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("MEM0_EMBED_KEY", "")
        self.model = model or os.environ.get("MEM0_EMBED_MODEL", "")

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.available:
            raise RuntimeError("嵌入端点未配置（.env: MEM0_EMBED_BASE_URL/KEY/MODEL）")
        r = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60.0, trust_env=False)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


# ---------------------------------------------------------------- 向量库（Chroma 同形的最小实现）
@dataclass
class _Record:
    id: str
    text: str
    vector: list[float]
    payload: dict


class TinyVectorStore:
    """JSONL 持久化 + 余弦线性扫描；接口与 mem0 vector_store 抽象同形。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[_Record] = []
        self.messages_log: dict[str, list[dict]] = {}   # user_id -> 已存消息（Phase 0 源）
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("_kind") == "msg":
                    self.messages_log[o["user"]] = o["msgs"]
                else:
                    self.records.append(_Record(o["id"], o["text"], o["vector"], o["payload"]))

    # ---- 记忆 ----
    def insert(self, vectors, ids, payloads) -> None:
        for vec, mid, pay in zip(vectors, ids, payloads):
            self.records.append(_Record(mid, pay["data"], vec, pay))
        self._flush()

    def search(self, vector: list[float], user_id: str, top_k: int = 10) -> list[_Record]:
        cands = [r for r in self.records if r.payload.get("user_id") == user_id]
        if not cands:
            return []

        def _cos(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1e-9
            nb = math.sqrt(sum(y * y for y in b)) or 1e-9
            return dot / (na * nb)

        ranked = sorted(cands, key=lambda r: -_cos(vector, r.vector))
        return ranked[:top_k]

    # ---- 消息日志（Phase 0 的 get_last_messages 等价物）----
    def get_last_messages(self, user_id: str, limit: int = LAST_K) -> list[dict]:
        return self.messages_log.get(user_id, [])[-limit:]

    def save_messages(self, user_id: str, messages: list[dict]) -> None:
        log = self.messages_log.setdefault(user_id, [])
        log.extend(messages)
        self._flush()

    def _flush(self) -> None:
        lines = [json.dumps({"_kind": "msg", "user": u, "msgs": m}, ensure_ascii=False)
                 for u, m in self.messages_log.items()]
        lines += [json.dumps({"id": r.id, "text": r.text, "vector": r.vector,
                              "payload": r.payload}, ensure_ascii=False)
                  for r in self.records]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines))
        tmp.replace(self.path)


# ---------------------------------------------------------------- 只-ADD 核心
@dataclass
class AddResult:
    id: str
    memory: str
    event: str = "ADD"          # 本管线只有 ADD（main.py 历史记录全部 event=ADD）


class Mem0AdditiveCore:
    """对话 → 中文事实记忆（只 ADD）。用法见 README。"""

    def __init__(self, client: LLMClient, store: TinyVectorStore,
                 embedder: Embedder | None = None, namespace: str = "mem0_add",
                 custom_instructions: str | None = None):
        self.client = client
        self.store = store
        self.embedder = embedder or Embedder()
        self.namespace = namespace
        self.custom_instructions = custom_instructions or CUSTOM_INSTRUCTIONS_ZH

    def add(self, messages: list[dict], user_id: str,
            metadata: dict | None = None) -> list[AddResult]:
        # Phase 0：最近 10 条历史（main.py:919-921）
        last_messages = self.store.get_last_messages(user_id)

        # Phase 1：已有记忆检索 + id 映射（防幻觉编号，main.py:923-938）
        parsed = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)
        q_vec = self.embedder.embed([parsed])[0]
        existing = self.store.search(q_vec, user_id, top_k=EXISTING_TOP_K)
        existing_memories = [{"id": str(i), "text": r.text} for i, r in enumerate(existing)]
        existing_hashes = {r.payload.get("hash") for r in existing}

        # Phase 2：LLM 加法提取（main.py:940-984）
        res = self.client.chat_sync(
            role="mem0_extract",
            messages=[
                {"role": "system", "content": ADDITIVE_EXTRACTION_PROMPT},
                {"role": "user", "content": generate_additive_user_prompt(
                    existing_memories=existing_memories,
                    new_messages=messages,
                    last_k_messages=last_messages,
                    custom_instructions=self.custom_instructions)},
            ],
            temperature=0.0, max_tokens=2000, json_mode=True, namespace=self.namespace)
        extracted = parse_extraction(res.content)

        # Phase 3：批量嵌入（main.py:991-1002）
        mem_texts = [m["text"] for m in extracted]
        embed_map: dict[str, list[float]] = {}
        if mem_texts:
            try:
                for text, vec in zip(mem_texts, self.embedder.embed(mem_texts)):
                    embed_map[text] = vec
            except Exception as e:  # noqa: BLE001 —— 单条兜底
                for text in mem_texts:
                    try:
                        embed_map[text] = self.embedder.embed([text])[0]
                    except Exception:
                        pass

        # Phase 4/5：逐条加工 + MD5 去重（main.py:1005-1039）
        base_meta = dict(metadata or {})
        base_meta["user_id"] = user_id
        now = datetime.now(timezone.utc).isoformat()
        records, seen_hashes, results = [], set(), []
        for mem in extracted:
            text = mem.get("text")
            if not text or text not in embed_map:
                continue
            mem_hash = hashlib.md5(text.encode()).hexdigest()      # main.py:1020
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                continue                                           # main.py:1021-1023
            seen_hashes.add(mem_hash)
            pay = dict(base_meta)
            pay.update({"data": text, "hash": mem_hash,
                        "created_at": now, "updated_at": now})
            if mem.get("attributed_to"):
                pay["attributed_to"] = mem["attributed_to"]
            mid = str(uuid.uuid4())
            records.append((mid, embed_map[text], pay))
            results.append(AddResult(id=mid, memory=text))

        # Phase 6：批量持久化 + 消息日志（main.py:1045-1075 / 988）
        if records:
            self.store.insert(vectors=[r[1] for r in records],
                              ids=[r[0] for r in records],
                              payloads=[r[2] for r in records])
        self.store.save_messages(user_id, messages)
        return results

    def search(self, query: str, user_id: str, top_k: int = 10) -> list[dict]:
        """top-k 记忆检索（对应 Memory.search 的向量路径）。"""
        vec = self.embedder.embed([query])[0]
        return [{"id": r.id, "memory": r.text, **{
            k: v for k, v in r.payload.items() if k not in ("data", "hash")}}
                for r in self.store.search(vec, user_id, top_k=top_k)]
