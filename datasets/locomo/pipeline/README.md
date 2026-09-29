# locomo：OaK 本体思想 × 中文 LoCoMo 长对话记忆基准

> 目标：零向量检索——只用中文本体图（原子事实 + 实体 + 关系）回答长对话记忆问题，
> 严格判题（exact 口径），锚点对话 ≥90% 后全量 10 段验证。

## 数据与纪律

- 数据集：`/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json`
  （10 段对话 / 5,882 条消息 / 1,986 题；类别映射实测为 **1=多跳 2=时间 3=开放域 4=单跳 5=对抗**）
- **数据纪律**（评测有效性前提，违反即成绩作废）：
  - 建图输入 = session 原文 + session 日期；禁读 `event_summary/observation/session_summary`（答案泄漏）。
  - 作答输入 = 问题 + 图；禁 gold/evidence/类别/原文全文。
  - 判题 = 盲判（题目/类别/gold/预测）。
  - 失败归因可读 evidence/答案，但结论只能改提示词/schema/工具，不得写死题目。
- 已知数据集噪声（不可修复，影响所有系统）：QA 翻译与对话翻译独立进行，
  存在术语漂移（"心理学"vs"心理咨询"）、细节丢失（海报文字/棕榈树/狗狗脸）、
  说话人错位（手绘碗/吉他）、数量错位（两只猫vs一只猫一只狗）。
  锚点对话 conv-26 实测约 17-19 题不可达，上限 ≈91%。

## 架构（OaK 四步循环的 LoCoMo 映射）

```
session 原文（注入日期锚）→ fast 模型抽「原子事实+实体+人-人关系」紧凑 JSON
  → 逐字 grounding（数字/实体名/出处 dia_id 必须在原文）→ 二道审计补抽
  → 实体归并（别名→规范名，带说话人隔离消毒）
  → 确定性日期覆写（代码按会话锚重算全部相对日期，粒度感知）
  → networkx 图（原子事实=一等节点，实体=枢纽）+ unigram/bigram 混合词法索引
作答：中文 ReAct（步骤=fast ≤10 步）→ 三采样终答（strong）+ 无 gold 共识择优
判题：确定性预检（数字/日期/列表集合等价 + 对抗陷阱短路）→ glm-5.3 盲判 exact/partial/wrong
闭环：失败归因（抽取缺失/检索miss/误拒答/推理表述错/对抗失效）→ 修最集中层 → 迭代
```

## 模型路由

| 档 | 模型 | 用途 |
|---|---|---|
| strong | glm-5.3（智谱） | P1/P2 本体起草、终答合成、严格判分 |
| fast | deepseek-v4-flash-fast（commandcode 网关，`LOCOMO_FAST_*` 独立变量） | 事实抽取、ReAct 步骤、审计/归并/共识 |
| fast 兜底 | glm-5.3-flash（智谱，探测自动切换） | 网关不可用时 |

与 TravelPlanner 工作流共享 .env 但使用独立 `LOCOMO_FAST_*` 变量（曾因共享 FAST_* 被并行会话改写导致一次运行事故——方法论#13 环境漂移）。

## 命令

```bash
uv run python -m datasets.locomo.pipeline.probe_models              # 双档连通性探测
uv run python -m datasets.locomo.pipeline.run_anchor                # conv-26 全量迭代
uv run python -m datasets.locomo.pipeline.run_anchor --anchor       # 37 题固定锚点集（快速闭环）
uv run python -m datasets.locomo.pipeline.run_anchor --idx 0,5,9    # 指定题冒烟
uv run python -m datasets.locomo.pipeline.run_full                  # 全量 10 段（冻结后）
```

产物：`datasets/locomo/runs/<conv>/graph_<fp8>/`（图+事实+统计）、`<conv>/iter<k>/`（答案/报告/失败归因）、
`runs/frozen/`（冻结 schema/主题词表）、`runs/cost_ledger.jsonl`（调用台账）。

## 模块

| 文件 | 职责 |
|---|---|
| `config.py` | 独立 Config（work_dir=datasets/locomo/runs）、网关探测回退 |
| `data.py` | 数据加载 + 数据纪律（QA 的 gold/evidence 只暴露给 judge/analyze） |
| `dates.py` | 确定性日期内核：英文会话日期解析、相对日期推算（gold 语义）、答案等价 |
| `schema_skeleton.py` | 手工中文本体骨架（八实体+十六关系）+ P2 合并 |
| `build.py` | 建图管线（抽取/grounding/审计/归并/日期覆写/派生边/图指纹缓存） |
| `tools.py` | 中文九件套图算子 + 混合词法索引（零向量） |
| `agent.py` | 中文 ReAct + 三采样终答 + 共识择优 + 证据强制 |
| `judge.py` | 两层判题（确定性预检 + glm-5.3 盲判）+ F1 对照 |
| `analyze.py` | 失败归因（五类）+ 回归 diff + 争议抽样 |
| `runner.py` | eval_conversation 全流程（断点续跑） |

优化全程逐轮记录见 `OPTIMIZATION_LOG.md`。
