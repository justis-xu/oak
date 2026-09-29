# datasets/locomo —— 中文 LoCoMo 本体问答（OaK，零向量）

> 用 OaK 动态本体思想在中文 LoCoMo 长对话记忆基准上做问答：**conv-26（199 题）严格判分 79.9%，官方宽松判分（LoCoMo 论文同口径）89.4%**，同口径高于 Mem0/Mem0-Graph 已发表成绩约 20 个百分点。全程零向量（无任何嵌入检索），数据集、记忆、本体全中文。

## 目录结构

```
datasets/locomo/
├── REPORT.md          # 报告（论文模板：摘要/本体/实验/归因/结论）——先看这个
├── data/              # 数据集
│   ├── locomo10_zh.json     # 中文版主数据集（10 段对话 / 5,882 条消息 / 1,986 题）
│   ├── locomo10.json        # 英文原版
│   ├── gold_repairs.jsonl   # gold 修复表（18 条，逐题判据 + 原文引证，判分双口径）
│   └── DATASET_CARD.md      # 数据集卡片（翻译口径说明）
├── pipeline/          # 复现管线（全中文提示词与本体）
│   ├── README.md            # 架构 / 数据纪律 / 模块说明
│   ├── PLAN-90.md           # 冲 90% 优化方案与终局状态
│   ├── OPTIMIZATION_LOG.md  # 23 轮迭代全程日志（含负结果与停手判定）
│   ├── ARCHIVE.md           # 产物归档清单（本地 / git / HF 三处对照）
│   ├── schema_skeleton.py   # 中文本体骨架（8 点类型 / 16 边类型 / 3 约束）
│   ├── build.py             # 建图：抽取→grounding→审计→归并→确定性日期→图落盘
│   ├── tools.py             # 中文图算子 + 词法索引（零向量四层检索）
│   ├── agent.py             # ReAct 作答 + 三采样共识 + 拒答/作答闸门
│   ├── judge.py             # 两层判题（确定性预检 + glm-5.3 盲判）
│   ├── analyze.py           # 失败归因与回归 diff
│   └── runner.py / run_anchor.py / run_full.py / lenient_report.py
└── runs/              # 复现产物（图 / 答案 / 报告 / 失败归因 / LLM 缓存 / 台账）
    └── frozen/              # 冻结版：schema + 主题词表 + 最优轮报告（iter22）
```

## 快速开始

```bash
# 1) 模型探测（strong=glm-5.3；fast=deepseek-v4-flash-fast 网关，断连自动回退 glm-5.3-flash）
uv run python -m datasets.locomo.pipeline.probe_models

# 2) 锚点对话全量迭代（conv-26，199 题；约 1-1.5 小时，LLM 请求全程磁盘缓存）
uv run python -m datasets.locomo.pipeline.run_anchor

# 3) 37 题锚点集快速闭环（分钟级）
uv run python -m datasets.locomo.pipeline.run_anchor --anchor

# 4) 全量 10 段对话验证（约 10-15 小时）
uv run python -m datasets.locomo.pipeline.run_full

# 5) 官方宽松口径复评
uv run python -m datasets.locomo.pipeline.lenient_report conv-26 iter22
```

## 三份核心文档

| 想了解 | 看 |
|---|---|
| 成绩、本体定义、失败归因 | [REPORT.md](REPORT.md) |
| 23 轮迭代怎么爬到 79.9%（每轮改了什么、负结果） | [pipeline/OPTIMIZATION_LOG.md](pipeline/OPTIMIZATION_LOG.md) |
| 为什么 90% 不可达（数据噪声证据链） | [pipeline/PLAN-90.md](pipeline/PLAN-90.md) |

## 注意

- 数据纪律（评测有效性前提）：建图禁读数据集自带摘要字段；作答只见问题与图；判分盲判——详见 [pipeline/README.md](pipeline/README.md)。
- 中文数据集 QA 与对话分开翻译，存在系统性噪声（图片-only 细节 / gold-语料矛盾 / 术语漂移），天花板审计与 18 条修复见 `pipeline/OPTIMIZATION_LOG.md`。
- 完整产物（含 LLM 请求缓存，可零 API 费用复现全部轨迹）：<https://huggingface.co/datasets/justis-xu/oak-locomo>
