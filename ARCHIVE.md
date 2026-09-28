# 产物归档清单（本地 / git / HF 三处对照）

复现产生的全部文件，按"在哪有"分类。**HF 数据集仓库：https://huggingface.co/datasets/justis-xu/oak-travelplanner（公开）**

## 三处对照

| 路径 | 体积 | git 仓库 | HF 数据集 | 说明 |
|---|---|---|---|---|
| `oak/`、`scripts/`、`tests/` | ~1 MB | ✅ | ✅ | 源码 |
| `README.md` / `REPRODUCTION.md` / `OPTIMIZATION_LOG.md` | ~40 KB | ✅ | ✅ | 文档（README 兼作 HF dataset card） |
| `pyproject.toml` / `uv.lock` / `.gitignore` / `.env.example` | 小 | ✅ | ✅ | 工程配置 |
| `runs/final/` | 100 KB | ✅ | ✅ | 冻结内核 K=(S,F) |
| `runs/data/` | 5.8 MB | ✅ | ✅ | queries 落盘 + 固定抽样索引 |
| `runs/replay/` | 824 KB | ✅ | ✅ | 离线重放 v1~v8 + 第三轮官方实测成绩 |
| `runs/inference/{scores,plans,official_eval_output}.json` | 180 KB | ✅ | ✅ | 最终成绩（50 题） |
| **`runs/inference/q*/`** | **456 MB** | ❌ | ✅ | 每题图 + ReAct 全轨迹 + 抽取统计 |
| **`runs/inference.round1.bak/`** | **454 MB** | ❌ | ✅ | 第一轮（Final 8/50）快照 |
| **`runs/build/round_1~5/`** | **52 MB** | ❌ | ✅ | 5 轮构建全轨迹（OaK 自举证据） |
| **`runs/anchor/` + `anchor.v1~v4.bak.*`** | **82 MB × 6** | ❌ | ✅ | anchor 迭代过程 |
| **`runs/cache/llm/`** | **61 MB** | ❌ | ✅ | LLM 请求缓存（可离线重放） |
| **`runs/cost_ledger.jsonl`** | **2 MB** | ❌ | ✅ | 12,945 次调用台账 |
| `runs/logs/` | 32 KB | ❌ | ✅ | 运行日志 |

git 排除重产物的原因：单题 `graph.json` 可达 9 MB，全量 1.5 GB 不适合放进 git 历史。

## 明确不上传的（含原因）

| 路径 | 原因 |
|---|---|
| `.env` | **API 密钥**，绝不外传 |
| `.venv/` | 虚拟环境，`uv sync` 可重建 |
| `.jdk/` | Adoptium JDK 17，HermiT 依赖，可重装 |
| `third_party/TravelPlanner/` | OSU-NLP-Group 官方仓库 clone + 官方 database；原始出处另有分发（见下） |
| `**/__pycache__/`、`*.pyc` | 编译缓存 |

## 复现时需自备的前置（未随归档分发）

```bash
# 1. 官方评测器与环境数据库
git clone https://github.com/OSU-NLP-Group/TravelPlanner third_party/TravelPlanner
# database 从 HF Space 下载：
#   https://huggingface.co/spaces/osunlp/TravelPlannerEnvironment
#   → third_party/TravelPlanner/database/

# 2. Adoptium JDK 17（HermiT 本体推理子进程用）
#    → .jdk/，并在 .env 设 JAVA_HOME

# 3. API 密钥
cp .env.example .env   # 填 ZHIPU_API_KEY

# 4. Python 依赖
uv sync
```

## 用归档里的缓存复现（不调 API）

`runs/cache/llm/` 是请求级磁盘缓存（键=请求内容哈希）。把它放回 `runs/cache/llm/`
后重跑构建/推理，命中缓存的调用**不产生 API 费用**——可复现全部轨迹但不再花钱。
未命中的部分才需真实 key。
