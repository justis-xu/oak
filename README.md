---
license: apache-2.0
language:
- en
tags:
- travelplanner
- llm-agent
- ontology
- reproduction
task_categories:
- text-generation
- question-answering
pretty_name: OaK × TravelPlanner Reproduction
---

# OaK 复现（TravelPlanner）

复现 *Toward Effective and Reliable LLM Agents via Dynamic Ontology*（arXiv:2608.22974，无官方代码）在 TravelPlanner 上的效果。论文翻译与精读笔记见 notebook 仓库 `货拉拉记忆系统/论文/oak-dynamic-ontology/`。

**怎么复现的（从零到成绩的每一步）→ `REPRODUCTION.md`；优化全程留痕 → `OPTIMIZATION_LOG.md`。**

## 数据集内容（HF Hub 上的完整运行产物）

本仓库同时作为复现结果的**公开档案**发布。除源码外，包含：

| 路径 | 内容 |
|---|---|
| `runs/final/` | 冻结内核 K=(S,F)：`schema.yaml` + 9 个编译好的领域函数 |
| `runs/build/round_1~5/` | **5 轮 OaK 构建循环全轨迹**：每轮 schema 进程、知识图谱、函数目录、ReAct 轨迹、评分、评判器反馈、函数补丁重测 |
| `runs/inference/` | 最终 50 题推理产物：每题图（`graph.json`）+ ReAct 全轨迹（`trajectory.json`）+ 抽取统计，以及 `scores.json` / `plans.validation.jsonl` |
| `runs/inference.round1.bak/` | 第一轮快照（Final 8/50），供前后对比 |
| `runs/anchor/` + `anchor.v1~v4.bak.*` | anchor 9 题快速闭环 + 四轮迭代过程 |
| `runs/replay/v1~v8/` | 离线重放各版本（确定性终态层演进 + 第三轮官方评测器实测成绩） |
| `runs/cache/llm/` | LLM 请求级磁盘缓存（断点续跑不重复计费） |
| `runs/cost_ledger.jsonl` | 全部 12,945 次 LLM 调用台账（模型/角色/token） |
| `runs/data/` | 落盘 queries + 固定抽样索引（`sample_indices.json`，seed=42） |

> 不包含：`.env`（API 密钥）、`third_party/TravelPlanner`（OSU-NLP-Group 官方仓库 clone，见下"前置"）、
> `.venv/`、`.jdk/`。复现时需自备这些。

## 当前结果（2026-09-27，第二轮优化后）

**validation 固定 50 题（官方评测器 + 官方固定分母）**：

| 指标 | 第一轮 | 第二轮 | 第三轮 | 论文（validation 180 题，DeepSeek 版） |
|---|---|---|---|---|
| Delivery | 44/50 = 88% | 46/50 = 92% | 46/50 = 92% | — |
| CS Micro | 74.75% (299/400) | 91.00% (364/400) | **91.00% (364/400)** | 86.1% |
| CS Macro | 34% | 84% | **84%** | — |
| HC Micro | 24.35% (28/115) | 82.61% (95/115) | **83.48% (96/115)** | 59.3% |
| HC Macro | 20% | 74% | **76%** | — |
| Final | 16% (8/50) | 70% (35/50) | **72% (36/50)** | 55.9% |

> 第三轮 = 第二轮最终计划 + 新增确定性层（官方库权威属性覆盖、降级去早退），
> 由官方评测器离线实测（`runs/replay/v8/scores_official.json`）。**第三轮第三次全量
> 复跑因智谱周限额未完成**（2026-09-28 21:18 重置），额度恢复后应重跑核实。

> 第一轮的 CS Micro 84.94%/HC Micro 75.68% 是聚合器分母错误（排除了未运行项），
> 已按官方固定分母口径修正（详见 OPTIMIZATION_LOG.md 第〇轮）。

**第二轮把 Final 从 16% 提到 70%**（超论文 55.9%），其中离线确定性层（零 LLM 调用）
贡献 8→21，P5/repair/salvage 与抽取保真贡献 21→35。优化全程留痕见
`OPTIMIZATION_LOG.md`（现象→诊断→根因→修复→效果对比，含可复用方法论）。

**残余 15 题失败**：valid_cost 5（预算本质紧张）、未交付 4（salvage 组不出路线）、
reasonable_visiting_city 2、is_not_absent(gated) 2、room rule/type 2。

5 轮构建轨迹（micro_cs）：73.6 → 69.4 → 70.8 → 73.6 → 79.2%（训练集上限）。

## 用法

```bash
cp .env.example .env        # 填 ZHIPU_API_KEY（JAVA_HOME 已配置则保留）
uv sync
bash scripts/setup_env.sh   # HF 数据落盘 + 分组抽样（third_party 见下）
uv run python -m oak.pipeline.build_loop --rounds 5     # 5 轮构建（断点续跑）
uv run python -m oak.pipeline.inference --limit 50      # 推理 + 评测
uv run python -m unittest tests.test_gate_a             # 终态层单测（离线）
uv run python scripts/replay_plans.py                   # 旧计划离线重放（不调 LLM）
uv run python -m oak.pipeline.inference --anchor        # anchor 9 题快速闭环
```

前置：`third_party/TravelPlanner/`（clone 自 OSU-NLP-Group/TravelPlanner）+ `database/`（HF Space osunlp/TravelPlannerEnvironment 下载）+ `.jdk/`（Adoptium 17，HermiT 依赖，见 .env JAVA_HOME）。

## 架构

```
oak/
├── llm/client.py        # 唯一 LLM 出口：role→模型路由、Semaphore(4)、退避重试、
│                        #   流式（防 TUN 代理挂长连接）、思考开关（机械任务关闭）、
│                        #   磁盘缓存（空输出不缓存）、成本台账、限额
├── schema/              # 步骤①：model（YAML 严格解析）→ builder（P1 需求分析→P2 草拟，
│                        #   字段白名单核对）→ owlcheck（静态检查+HermiT 子进程+最小冲突定位）
├── kg/                  # 步骤②：extract（P3 紧凑 JSON 契约+1 次 repair）→ graph（键签名
│                        #   合并+重连去重+derive_relations 属性派生边+距离矩阵程序化建图）
├── operators/           # 九件套算子 + sandbox（AST 白名单含安全方法、受限 exec、双图试跑）
├── funcs/               # 步骤③-3a：能力规划（P4，强制覆盖航班等七类）→逐函数生成→试跑
├── agent/               # ③-3b ReAct（P5：covered 城市权威块/闭环行程/保真规则）
│                        #   + graph_index（官方语义解析+官方库保真门+成本公式）
│                        #   + validator（全量结构化 PlanIssue，镜像官方 CS/HC）
│                        #   + planner（finalize 单一入口：normalize→校验→确定性修复
│                        #   →LLM 修复→预算降级，候选支配性回滚，no-plan salvage）
├── eval/                # 官方评测器子进程对接 + 官方固定分母聚合（未运行计失败）
│                        #   + subset padded 复跑 + scripts/reaggregate 离线重聚合
├── judge/               # 步骤④：P6 五输入→σ 四元组；快慢两回路路由；补丁应用
└── pipeline/            # build_loop（5 轮编排+断点+修补重测取优+冻结）/ inference
```

模型路由：模式构建/函数编译/评判器 → `glm-5.3`（保留深度思考）；抽取/ReAct/槽位/修复 → `glm-5.3-flash`（关思考提速 5 倍）。

## 复现过程中修复的关键问题（按发现顺序）

1. schema 编造字段名（PropertyName 等）→ 语料字段白名单硬约束 + 兜底核对
2. 思考型模型 reasoning 吃光 max_tokens 致空输出 → 请求侧 +3072 思考余量 + 空输出不入缓存
3. 8 并发触发 429 限流 → 降 4 并发 + 429 长退避
4. TUN 代理挂起非流式长请求（进程假死）→ 全部调用改流式
5. 抽取输出过大触发截断-修复循环 → 紧凑 JSON 格式 + 可派生关系改代码生成（LLM 只抽实体）
6. adapter 硬编码 visiting_city_number=1 → 多城市题全误判（前几轮分数失真）
7. 任务理解偏差：TravelPlanner 是**闭环行程**（末日回 org，州内选城）→ P5 重写行程结构
8. 模型编造航班号（函数目录缺航班查询）→ P4 能力清单强制覆盖 + 沙箱放行字符串方法
   （`.lower()` 等被 AST 白名单误杀导致 search_flights 等生成失败）
9. 距离矩阵大块 LLM 抽取反复截断 → 程序化直建（csv → 实体，derive 补边）

## 断点续跑

LLM 请求级磁盘缓存（runs/cache/llm/<ns>/）+ stage 级 state.json + react 按题 + inference 每 5 题 checkpoint；崩溃重跑不重复计费。

## 已知差距（下一步）

- valid_cost 残余 5 题：餐/住/航班已全是最低价合规项仍超预算，需要换城市/换路线级决策
- 未交付 4 题（q43/q47/q56/q116）：no-plan salvage 组不出合法路线（语料无航班等）
- 评判器补丁的归因质量未人工抽检（论文也没有，建议复现者自加）
