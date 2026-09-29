# OaK × TravelPlanner 完整复现文档

> 一句话结论：在 validation 固定 50 题上 **Final 35/50 = 70%（论文 55.9%）、CS Micro 91.0%（论文 86.1%）**，复现达成并反超；全程优化过程见 `OPTIMIZATION_LOG.md`，架构见 `README.md`，本文回答"怎么复现的"——从零到成绩的每一步、每条命令、每个产物。

---

## 1. 复现什么

论文 *Toward Effective and Reliable LLM Agents via Dynamic Ontology*（arXiv:2608.22974，OaK）没有官方代码。本仓库从零实现了 OaK 的四步构建循环（模式 → KG 实例化 → 函数编译 + ReAct 执行 → Eval+Judge），并在 TravelPlanner validation 上复现其效果。对照数字（论文为 DeepSeek 版、180 题全量；本文为 glm-5.3 系列、固定 50 题分层子集，仅作方向对照）：

| 指标 | 论文 | 本复现（50 题，第五轮最终实测） |
|---|---|---|
| Delivery | — | **94% (47/50)** |
| CS Micro | 86.1% | **92.75%** (371/400) |
| CS Macro | — | **88%** |
| HC Micro | 59.3% | **90.43%** (104/115) |
| HC Macro | — | **82%** |
| **Final** | **55.9%** | **78% (39/50)** |

> 迭代轨迹：0% → 16%（第一轮）→ 70%（第二轮）→ 78%（第五轮，经三重根因连环修：
> 官方库 dropna 僵尸行、fast 档模型被切污染审计、空计划逃生门拦截）。全过程含负结果
> 见 `OPTIMIZATION_LOG.md`。

50 题子集：`stratified_test_subset(50, seed=42)`（easy/medium/hard 分层，indices 固定落在 `datasets/travelplanner/data/sample_indices.json`，可完全复现）。

## 2. 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| 机器 | macOS / Linux，可联网 | 本机 darwin 25.6 |
| Python | 3.14（uv 管理） | `uv sync` 装依赖（networkx、python-dotenv、datasets 等） |
| Java | Adoptium 17 | HermiT 本体推理子进程用；`.env` 里 `JAVA_HOME` 指向 `.jdk/` |
| third_party | `git clone OSU-NLP-Group/TravelPlanner` | 官方评测器 + 环境 database |
| database | HF Space `osunlp/TravelPlannerEnvironment` 下载到 `third_party/TravelPlanner/database/` | 航班/住宿/餐馆/景点/距离矩阵官方 CSV |
| API key | `.env` 填 `ZHIPU_API_KEY` | 核心档 glm-5.3；fast 档默认 glm-5.3-flash（可用 `FAST_API_BASE/FAST_MODEL/FAST_API_KEY` 切网关） |

模型路由（`oak/config.py::MODEL_ROLES`）：

| 角色 | 档位 | 模型 | 深度思考 |
|---|---|---|---|
| schema（P1/P2）、func_gen（P4）、judicator（P6） | strong | glm-5.3 | 开 |
| kg（P3）、react（P5）、plan_repair | fast | glm-5.3-flash | 关 |

总并发 Semaphore(4)（8 并发实测触发 429 限流）。

## 3. 从零复现的完整步骤

```bash
# ① 环境
cp .env.example .env          # 填 ZHIPU_API_KEY（JAVA_HOME 已配置则保留）
uv sync

# ② 数据落盘 + 抽样（HF 拉 queries → datasets/travelplanner/data/*.queries.jsonl，
#    固定 train 45 题分 5 组、validation 分层抽 50 题，seed=42）
bash datasets/travelplanner/scripts/setup_env.sh

# ③ 5 轮构建循环（OaK 内核 K=(S,F) 的自举）
uv run python -m datasets.travelplanner.pipeline.pipeline.build_loop --rounds 5
#    每轮四步（对 9 题/轮）：
#    步骤① 模式：P1 需求分析 → P2 YAML 草拟（语料字段白名单硬约束）
#                → 静态检查 + HermiT 子进程一致性验证
#    步骤② KG：每题语料分块（表头复制进每块）→ P3 紧凑 JSON 抽取（精确源
#                grounding + 空 chunk 重试）→ 键签名合并 → derive 派生边
#                + 距离矩阵程序化建图 + City enrich（州/covered）
#    步骤③ 函数：P4 能力规划（8 类能力强制覆盖）→ 逐函数生成 → AST 沙箱
#                → 双图试跑（search 类必须非空）→ ReAct 执行 9 题
#    步骤④ 评判：P6 五输入 → σ 反馈四元组 → 快回路函数补丁重测（严格改善才采纳）
#    断点：round 级 state.json + LLM 请求级磁盘缓存，崩溃重跑不重复计费
#    产物：runs/build/round_*/、冻结内核 runs/final/（schema.yaml + functions）

# ④ 推理 + 评测（冻结内核 + 每题现建图）
uv run python -m datasets.travelplanner.pipeline.pipeline.inference --limit 50
#    每题：建图 → ReAct ≤20 步（P5：covered 城市权威块 + observation 信封）
#    → finalize_plan：非破坏 normalize → 全量 validator（PlanIssue）
#      → 确定性修复（同城回填/住宿块/交通重建/跨列去重/菜系覆盖）
#      → ≤2 轮 LLM 修复（issue-scoped CANDIDATES_JSON）→ 预算降级
#      → 候选支配性回滚，no-plan 走结构化 salvage
#    → 官方评测器子进程（CS 8 项/题 + 门控 HC 适用项，未运行计失败）
#    产物：datasets/travelplanner/runs/inference/{plans.validation.jsonl, scores.json, q*/（图+轨迹+抽取统计）,
#          plans.validation.official.jsonl + official_eval_output.json（padded 复跑留档）}
```

辅助验证命令（优化过程中用的快速闸门，复现时可选）：

```bash
uv run python -m unittest tests.test_gate_a   # 终态层 28 项单测（离线，不调 LLM）
uv run python datasets/travelplanner/scripts/replay_plans.py         # 对已有 plans 重放新终态层（不调 LLM）
uv run python -m datasets.travelplanner.pipeline.pipeline.inference --anchor   # anchor 9 题快速闭环（train 题分层锚点）
uv run python datasets/travelplanner/scripts/reaggregate.py          # 对已有 scores.json 按官方分母离线重聚合
```

## 4. 评测口径（怎么算的分）

1. **subset 主口径**：50 题直接送官方 `evaluation/commonsense_constraint.py::evaluation` 与 `hard_constraint.py::evaluation`（子进程 cwd=evaluation，官方模块 import 时自行加载数据库）。HC 只在 `is_not_absent` 与 `is_valid_information_in_sandbox` 双过后运行（官方门控）。
2. **聚合分母镜像官方 eval.py**：CS 每题恒 8 项；HC 每题 = `applicable_hc_keys(q)`（valid_cost 恒 1 + medium 不计 transportation + hard 按约束存在计）。**未交付、门控未运行、worker error 一律计失败**（进分母不进分子）——第一轮曾因分母排除未运行项虚高（CS 84.9%→实际 74.75%），已修正并回归测试。
3. **对拍闭环**：官方 padded 口径（180 行稀释分母）`official_eval_output.json` 的 Final 19.44% × 180 = 35 题 = subset 口径 35/50，两个口径互相印证。

## 5. 产物档案索引（runs/）

| 路径 | 内容 |
|---|---|
| `runs/final/` | 冻结内核：schema.yaml + functions.{py,json}（9 个 published 函数） |
| `datasets/travelplanner/runs/inference/` | 第二轮全量产物：scores.json（summary+逐题）、plans.validation.jsonl、q*/（graph.json、trajectory.json、extraction_stats.json）、official_eval_output.json |
| `datasets/travelplanner/runs/inference.round1.bak/`、`round2.bak`（见日志） | 各轮快照，供前后对比 |
| `runs/replay/v8/scores_official.json` | **第三轮成绩（官方评测器离线实测 Final 36/50）** |
| `runs/replay/v1~v8/` | 旧 50 题离线重放各版本（确定性层演进留档） |
| `runs/anchor/`、`anchor.v1~v4.bak.*` | 最终 anchor 9 题 + 四轮迭代过程留档 |
| `runs/build/round_1~5/` | 5 轮构建全轨迹：schema 进程、图、函数目录、react 轨迹、scores、judge 反馈、补丁重测 |
| `runs/cache/llm/` | LLM 请求级磁盘缓存（断点续跑不重复计费） |
| `runs/cost_ledger.jsonl` | 全部 12,945 次调用台账（模型/角色/token） |
| `datasets/travelplanner/data/` | queries 落盘 + anchor.json + sample_indices.json（固定抽样） |
| `tests/test_gate_a.py` | 终态层 28 项单测 |
| `scripts/` | setup_env / reaggregate / replay_plans / smoke |

## 6. 成本与耗时

- LLM 调用 **12,945 次**（react 7,929 / kg 4,308 / func_gen 290 / plan_repair 321 / schema 60 / judicator 37）
- token：prompt **40.45M** + completion **17.26M**（含中途网关 deepseek-v4.1-flash 4,621 次）
- 5 轮构建约 1.5h/轮（首轮），50 题推理约 25min（抽取缓存命中后），anchor 9 题约 6min/轮
- 优化迭代（第二轮）主要是代码工作 + 缓存命中的复跑，新增 LLM 开销集中在 4 轮 anchor 与最终 50 题

## 7. 复现过程中踩过的坑（防再踩）

1. TUN 代理（198.19 fake-ip）挂死非流式长请求 → 全部调用改流式 + `trust_env=False`
2. 8 并发触发 429 → 降 4 并发 + 指数退避
3. 思考型模型 reasoning 吃光 max_tokens 致空输出 → 请求侧 +3072 思考余量、空输出不入缓存
4. 函数沙箱与正式运行 namespace 不一致（`ceil` 试跑过、正式 NameError）→ 共享 `safe_exec_namespace()`
5. 主键存于 `__key__` 元数据，检索层各自半套解析会丢 city/name → 全仓统一 `node_view()`
6. reference 语料是 DataFrame repr，个别行损坏（名称跨行粘连）→ 官方库保真门过滤
7. 自建聚合器分母排除未运行项会虚高 → 先镜像官方固定分母再谈优化
8. 日期字段字符串化列表被 `list(str)` 拆成字符数组 → `parse_dates` 统一解析

## 8. 残余失败（11/50）与后续方向

| 类别 | 题数 | 说明 |
|---|---|---|
| valid_cost | 5（q1/56/99/125/143） | 餐/住/航班已全是最低价合规项仍超预算；q125 属**题目硬约束无解**（entire room + 3 天，但 Albany 唯一合规住宿 `minimum nights=30`） |
| 未交付 | 3（q17/q43/q74） | salvage 仍组不出路线（语料无航班/预算预判放弃） |
| current_city / sandbox | 4（q34/q116/q179） | 城市对齐长尾 |

### 重要设计教训（第五轮，三条新增）

1. **官方库的加载语义也是口径**：`pd.read_csv().dropna()` 会把任一列为空的行整行丢弃——CSV 里"存在"的名字评测数据里未必存在。复刻官方判据要连加载行为一起复刻。
2. **审计前先查环境漂移**：`.env` 的模型配置被其它会话改动，让连续四轮代码归因全部失真（真凶是思考型 deepseek 挤占输出）。先查 `cost_ledger.jsonl` 的 model 列再做代码归因。
3. **别给模型放弃的逃生门**："证据不足可交空计划"被 flash 滥用为 0 次调查交白卷；必须在代码层拦截（证据调用 <3 时拒绝空 Final Plan）。

## 9. 相关文档

- `OPTIMIZATION_LOG.md` — 两轮优化全程留痕（现象→诊断→根因→修复→实测对比 + 8 条可复用方法论 + 负结果）
- `README.md` — 架构总览与模块说明
- 论文精读笔记 — notebook 仓库 `货拉拉记忆系统/论文/oak-dynamic-ontology/`（翻译/思考/解读三篇）
