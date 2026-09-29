# mem0 只-ADD 基线核心（中文）

> 记忆系统基线：对话 → 中文事实记忆，**只有 ADD 操作**（无 UPDATE/DELETE/图），
> 与 locomo 的本体（oak）路线形成对照。数据集、记忆、本体全中文。

## 来源与裁剪

- **代码流程**：vendored mem0 2.1.0（`/Users/xu/git/memory-schema-rsi/third_party/mem0-src`）
  `_add_to_vector_store` 的 **V3 PHASED BATCH PIPELINE**（main.py:916-1075）逐阶段蒸馏：
  最近消息窗口 → 已有记忆检索 → LLM 加法提取 → 批量嵌入 → MD5 去重 → 批量插入。
  `memory_core.py` 内注释标注了每个 Phase 对应的 mem0 源码行号。
- **提示词**：`prompts.py` = mem0 `ADDITIVE_EXTRACTION_PROMPT`（prompts.py:468，只 ADD 契约）
  + memory-schema-rsi `config/locomo_zh.yaml` 的 `fact_extraction_prompt` 中文强制段
  （经 `build_mem0_config` 的 custom_instructions 插口注入）。
- **刻意删掉**：UPDATE/DELETE 路径、图记忆、procedural、平台客户端、遥测、代理。
- **存储**：内置 `TinyVectorStore`（JSONL + 余弦扫描）替代 Chroma——接口同形
  （insert/search），百~千条记忆量级无压力；需要时可换回 Chroma。
- **LLM**：直连 `oak.llm.LLMClient`（role=`mem0_extract`，走 fast 档）；
  **嵌入**：OpenAI 兼容 `/embeddings`（.env：`MEM0_EMBED_BASE_URL/KEY/MODEL`）。

## 用法

```python
from oak.config import Config
from oak.llm.client import LLMClient
from mem0.memory_core import Mem0AdditiveCore, TinyVectorStore

cfg = Config()                    # 按任务补 api 端点字段
client = LLMClient(cfg)
store = TinyVectorStore("mem0_store.jsonl")
mem = Mem0AdditiveCore(client, store)

mem.add([{"role": "user", "content": "梅拉妮上周六参加了心理健康公益跑。"}], user_id="conv-26")
hits = mem.search("梅拉妮参加过什么活动？", user_id="conv-26", top_k=5)
```

## 与 oak 本体路线的关系

| | mem0 基线（本目录） | locomo 本体（datasets/locomo） |
|---|---|---|
| 记忆形态 | 平面事实条目 + 向量检索 | 原子事实节点 + 实体枢纽图（零向量） |
| 更新语义 | 只 ADD（MD5 去重） | 主体归因 + 派生边 + 确定性日期 |
| 用途 | 对照基线 | 主线 |
