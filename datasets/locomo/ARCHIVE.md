# locomo 产物归档清单（本地 / git / HF 三处对照）

复现产生的全部文件，按"在哪有"分类。**HF 数据集仓库：https://huggingface.co/datasets/justis-xu/oak-locomo（公开）**

| 路径 | 体积 | git 仓库 | HF 数据集 | 说明 |
|---|---|---|---|---|
| `pipeline/`（源码+中文提示词+本体骨架） | ~100 KB | ✅ | ✅ | datasets/locomo/pipeline |
| `pipeline/README.md` / `OPTIMIZATION_LOG.md` | ~30 KB | ✅ | ✅ | 文档（12 轮逐轮+复盘） |
| `data/locomo10_zh.json` + `locomo10.json` | 4.7 MB | ✅ | ✅ | 数据集本体（中文版+英文原版） |
| `runs/<conv>/graph_*/`（图+事实+统计） | ~15 MB | ✅ | ✅ | 每轮建图产物（含 observation 语料版） |
| `runs/<conv>/iter*/`（answers/report/failures） | ~8 MB | ✅ | ✅ | 每轮作答、判分、失败归因 |
| `runs/conv-26/ceiling_audit.json` | 小 | ✅ | ✅ | 天花板审计（逐题 gold 可达性） |
| `runs/schema_cache.json` | 小 | ✅ | ✅ | P1/P2 起草缓存（主题词表） |
| `runs/anchor_set.json` | 小 | ✅ | ✅ | 固定锚点集（37 题） |
| **`runs/cache/llm/`** | ~57 MB | ❌ | ✅ | LLM 请求缓存（可零 API 费用复现全部轨迹） |
| `runs/cost_ledger.jsonl` / `runs/logs/` | ~2 MB | ❌ | ✅ | 调用台账 / 重试日志 |
| `runs/anchor_*.log` | 小 | ✅ | ✅ | 每轮运行日志 |
| `mem0/`（只-ADD 基线核心） | 小 | ✅ | ✅ | 上传时一并归档 |
| 根 `README.md` / `pyproject.toml` / `uv.lock` / `.env.example` | 小 | ✅ | ✅ | 工程总览与配置 |

## 明确不上传的（含原因）

| 路径 | 原因 |
|---|---|
| `.env` | **API 密钥**，绝不外传（含智谱/commandcode/嵌入端点三类 key） |
| `.venv/` `__pycache__/` | `uv sync` 可重建 |
| `datasets/travelplanner/` | 另有独立 HF 仓库 justis-xu/oak-travelplanner |
| `third_party/` | 官方评测器，原始出处另有分发 |
| `runs/frozen/` | 尚未冻结（迭代进行中，冻结后补传） |

## 用归档缓存复现（不调 API）

`runs/cache/llm/` 是请求级磁盘缓存（键=请求内容哈希）。放回同路径后重跑
`python -m datasets.locomo.pipeline.run_anchor`，命中缓存的调用零费用——
图构建/审计/作答轨迹全部可复现；判分缓存同理（lc26_judge/lc26_lenient）。

## 注意：HF 仓库的 `.gitignore` 与项目 git 仓库不同

HF 侧只排除 `.env`/`.venv`/`__pycache__`/`third_party`，**不排除**重产物
（缓存/台账按上表全量归档）；项目 git 为控体积排除了 cache/ledger。
两处刻意不同，勿同步回去（同 justis-xu/oak-travelplanner 既有原则）。
