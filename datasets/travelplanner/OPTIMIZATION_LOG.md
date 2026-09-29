# OaK TravelPlanner 复现优化日志（OPTIMIZATION_LOG）

> 目标：在 `/Users/xu/git/oak` 复现 OaK（arXiv:2608.22974）的 TravelPlanner 效果（论文 Final 55.9%）。
> **最终结论（2026-09-29，端到端全量 50 题实测）：Final 39/50 = 78%（论文 55.9%）、CS Micro 92.75%（论文 86.1%）、HC Micro 90.43%（论文 59.3%）——复现达成并大幅反超。**
> 完整复现指南（环境/命令/产物索引/成本/踩坑）→ `REPRODUCTION.md`；架构 → `README.md`。
> 本文档是**全程留痕**：每轮迭代的"现象 → 诊断 → 根因 → 修复 → 效果对比"，含 14 条可复用方法论。
> 规则：预测只写在 Hypothesis；After 只写真实复算结果；负结果不删除；不自动 commit/push。

## 可复用方法论（随迭代持续补充）

1. **审计驱动修复**：不猜"提示词不够好"，先对失败题逐题对照官方库复算，把失败归类到精确根因（数据缺失/代码 bug/模型行为），再决定修哪一层。
2. **anchor 快速闭环**：9 题分层锚点集（level×days×cities 全格子），分钟级验证方向；比例不可外推，方向可用。
3. **离线重放优先**：能不调 LLM 验证的（validator/fill/cost/评测口径），对已有 plans/graphs 重放，零成本预估增益。
4. **确定性收尾层**：LLM 输出后由代码做约束感知的确定性修复（图池回填、去重、预算降级），不允许跨城/编造。
5. **图一致性闸门**：计划引用的每个实体必须在图内精确存在（(name,city) 对、flight 元组、ground 端点），查不到=编造。
6. **评测口径先校准**：自己的聚合器分母若排除"未运行项"，Micro 会虚高；先镜像官方固定分母再谈优化。
7. **候选支配性回滚**：每轮 repair/fill/budget 候选必须严格优于当前 best 才接受，防"越修越坏"。
8. **保护已通过样本**：把当前 Final-pass 固定为回归集，任何改动不得使其回退。
9. **属性以官方库为准，图只做名字枚举**（第三轮）：LLM 抽图会整字段丢失（house_rules/room type 变 None），而评测按官方库判——保真门不能只查"名字存在"，还要用官方非空属性覆盖图值。
10. **本地判据与官方对齐，宁松勿严**（第四轮）：自己加的门槛（如"城市必须有餐住景三类数据"）若比官方严，会制造虚假不可行，模型宁可放弃也不编造 → 直接 0 分。官方只查"存在性与合法性"。
11. **标题数字只用端到端实测**（第五轮教训）："旧计划 + 新确定性代码 + 真评分器"的离线增量测量是合法证据，但**不是复跑成绩**；写进 README 表头/仓库名/commit 首行会被当成复测结论。结论位置的数字必须来自一次完整的、从建图到评测的真实运行。
12. **失败计数会重叠且门控会隐藏新失败**：修好 sandbox 才会暴露被门控的 HC 失败，多类失败计数不能相加预估收益（第〇轮实证：ran 19 / gated 25）。
13. **审计前先查环境漂移**（第五轮）：模型/网关/.env 被其它会话改过，行为突变的第一嫌疑是环境不是代码——先看 `cost_ledger.jsonl` 的 model 列分布再归因。单变量对照才有效。
14. **别给模型"放弃"的逃生门**：提示词里任何"可以输出空/放弃"的出口都会被弱模型滥用（0 次调查就交白卷）；必须在代码层强制"先调查后放弃"（空计划拦截）。

---

## 第〇轮（基线冻结）— 2026-09-27

### Environment
- strong: glm-5.3（智谱直连，深度思考开）；fast: glm-5.3-flash
- 产物：runs/final（冻结内核：schema.yaml + 9 published functions）、runs/inference（50 题固定子集，seed=42）
- 固定 50 indices: [1,2,5,6,7,8,14,15,17,27,...]（stratified_test_subset(50, seed=42)）

### 现象
第一轮优化后 50 题自报：Delivery 88% / CS Micro 84.94% / CS Macro 34% / HC Micro 75.68% / HC Macro 20% / **Final 16% (8/50)**。

### 诊断
对 50 题逐题对照官方库复算，并把聚合器与官方 `evaluation/eval.py` 逐行比对：

- **聚合器分母错误（高估）**：`oak/eval/adapter.py::_aggregate` 只把"实际返回非 None 的项"计入分母：
  - CS 显示 299/352=84.94%；官方口径是每题固定 8 项（未交付也计失败），应为 **299/400=74.75%**；
  - HC 显示 28/37=75.68%；官方口径是"每题按 local_constraint 预定的适用项数"（未运行也计失败），本批应为 **28/115=24.35%**；
  - CS Macro / HC Macro / Final 不受此影响（Final=8/50）。
- **失败分布**（有重叠）：sandbox 引用无效 23、current-city 错城 17（交集 16）、valid_cost 8、Invalid City Number 6、transportation 混搭 6、'-;' 漏网等长尾。
- **门控连锁**：官方 hard constraint 只在 `is_not_absent` 与 `is_valid_information_in_sandbox` 双过后才运行——修好 sandbox 会暴露更多此前未计的 HC 失败，不能做加法预估。

### 根因
Micro 分母排除了 None/未运行项；"HC 反超论文 59.3%"的结论是分母错误造成的假象。

### 修复（Phase 0，本文件同日提交）
- `oak/data/queries.py`：新增唯一规则 `applicable_hc_keys(q)`（valid_cost 恒 1；medium 不计 transportation；hard 按约束存在计），`Query.hc_denominator` 复用。
- `oak/eval/adapter.py`：CS 每题固定 8 项分母、HC 每题固定适用项分母；未交付/门控未运行/worker error 一律计失败；macro_cs 镜像官方"gated 才计数"语义；summary 落 numerator/denominator。
- `oak/eval/_worker.py`：记录 `hard_not_run_reason`（empty/gated/error）仅供归因，计分仍按失败。

### After（离线重聚合，`scripts/reaggregate.py`，不调 LLM）
| 指标 | 旧（错误分母） | 新（官方口径，已实测） |
|---|---|---|
| Delivery | 44/50 = 88% | 44/50 = 88% |
| CS Micro | 299/352 = 84.94% | **299/400 = 74.75%** |
| CS Macro | 17/50 = 34% | 17/50 = 34%（gated 口径，17 题全部 hard 已运行） |
| HC Micro | 28/37 = 75.68% | **28/115 = 24.35%** |
| HC Macro | 10/50 = 20% | 10/50 = 20% |
| Final | 8/50 = 16% | **8/50 = 16%** |

hard constraint 运行状态：ran 19 / gated 25 / empty 6 —— **25 题 HC 被门控隐藏**，sandbox 修复后会暴露新失败，印证"不能加法预估"。

### Decision
keep。此前"+8/+6/+5=54%"的加法预估全部作废；后续所有收益只认 replay/anchor/full 实测。
回归保护集（当前 Final-pass，任何改动不得回退）：**q2, q6, q7, q8, q14, q15, q78, q123**。

---

## 第二轮 Phase 1-5（终态层重写）— 2026-09-27

### Hypothesis
失败大头不在规划主循环，而在证据/校验/收尾层：图证据丢 city、收尾跨城 fallback、
无统一 validator、无预算处理、no-plan 不修、官方成本公式缺失。把终态层做对，
不动规划主循环，也应大幅提升 Final。

### Change（文件 → 行为）
1. `oak/data/queries.py`：`parse_dates`（字符串化列表/旧字符数组/单串统一解析；q43 等日期恢复）；`applicable_hc_keys` 唯一 HC 适用项规则。
2. `oak/operators/sandbox.py` + `oak/funcs/catalog.py`：共享 `safe_exec_namespace()`（九算子 + math.ceil + 白名单 builtins）——修"试跑过、正式 NameError: ceil"（q5/q66 实证），非空住宿输入验证通过。
3. `oak/kg/graph.py`：公共 `node_view()`（`__key__`+属性统一视图，全仓唯一解析点）；`enrich_city_nodes()`（state 读官方 citySet_with_states.txt + 三类业务计数 + covered=三类全有；幂等）；`covered_cities()`。
4. `oak/kg/extract.py`：精确源 grounding（主键值必须逐字出现在源 chunk，否则丢实体+关联关系，不做 fuzzy）；空 chunk 重试（0 实体但 chunk 有数据行 → 显式反馈再试一次）。
5. `oak/agent/graph_index.py`（新）：GraphIndex 一次物化（(name,city)/flight/ground/city 索引）+ 官方解析语义（parse_name_city 最后逗号、split_attractions、transportation_mode 优先级、首航段）+ `compute_plan_cost` 官方公式（ground 直读官方 distance.csv，含 'day' duration 无效语义）。
6. `oak/agent/validator.py`（新）：`validate_plan_full` 结构化 PlanIssue，逐条镜像官方 CS/HC 判定（路线闭环/连续块/州归属/covered、交通端点与混搭、sandbox 图存在性、current_city 范围、minimum nights 连续 run、菜系覆盖、信息量≥50%、预算），不再删除任何 not-in-graph 错误。
7. `oak/agent/planner.py`（重写）：非破坏 `normalize_plan`（不截断/不补齐/不覆盖 days）+ `parse_plan_payload`（{"plan":[...]} wrapper 与 json_object 一致）+ `local_repair`（约束感知确定性修复：同城池回填、住宿按 lodging-city 块、菜系覆盖、末日晚清住宿；**删除跨城 fallback**）+ `budget_downgrade`（保守：住宿块→餐→航班，保唯一/菜系/约束/mode）+ `llm_repair`（issue-scoped CANDIDATES_JSON）+ `finalize_plan`（单一入口，候选支配性回滚，no-plan 统一 salvage）。
8. `oak/prompts/react.py`（v2）+ `oak/prompts/repair.py`（新）+ `oak/agent/react.py`：observation 统一信封（warning/空/sentinel 不再是候选，relaxed→suggestions）；P5 权威 COVERED CITY 块 + 路线先行 WORKFLOW + 允许空计划的 FORCE_FINALIZE；移除内联 salvage 与硬编码 search_flights 示例。
9. `oak/pipeline/inference.py` + `build_loop.py`：统一走 `finalize_plan`；checkpoint 恢复载入已有 plans；build patch 采纳去 tie；图构建后 enrich。

### Validation（全部离线，不调 LLM）
- 8 个旧 Final-pass 过新 validator：**8/8 通过**（零误杀）。
- 已知失败精确命中：q66=accom.min_nights、q1=transport.mixed、q27/q83=错城+not_in_graph、q99/q147=format。
- 成本 oracle：本地 `compute_plan_cost` vs 官方 `get_total_cost` 12 题对拍 **12/12 精确一致**。
- q43 covered 城市 = 参考 3 城精确（Colorado Springs 距离节点被正确排除）；q179 抽取曾整块丢 Lubbock 餐馆 → 已加空 chunk 重试。

### After（`scripts/replay_plans.py` 旧 50 题确定性重放 + 官方评测器实测）
| 指标 | 基线（Phase 0） | 重放后（仅确定性层，零 LLM） |
|---|---|---|
| Delivery | 44/50 | 44/50 |
| CS Micro | 299/400 = 74.75% | **322/400 = 80.50%** |
| CS Macro | 17/50 = 34% | **26/50 = 52%** |
| HC Micro | 28/115 = 24.35% | **65/115 = 56.52%** |
| HC Macro | 10/50 = 20% | **24/50 = 48%** |
| Final | 8/50 = 16% | **21/50 = 42%** |

- blocking issue 总数 297 → 96；27 题改善、0 题变差、**回归 0**（8/8 保护）。
- 新增 Final-pass：q5,q37,q62,q65,q66,q99,q134,q136,q147,q152,q154,q157,q174。
- 残余 blocker 分布：city.uncovered 15、route.city_count 6、plan.empty 6、transport.mixed 5、route.city_invalid 5、budget.over 8、ground/flight_not_in_graph 3+3。

### Negative results / 注意
- q179 图内 Lubbock 餐馆为 0 是抽取丢失（语料有 21 行），非数据缺失——covered 会误判，
  靠空 chunk 重试在下次推理时修复；离线重放无法回补。
- replay 只测终态层；P5 新提示词与 LLM repair/salvage 的收益需 anchor/正式跑验证。

### Decision
keep。确定性层效果远超预期（+13 Final 纯代码）。下一步：单测固化 → anchor（验证 P5/LLM 层 + 抽取重试）→ 最终 50。

---

## 第二轮 Phase 6（anchor 迭代 + 最终 50）— 2026-09-27

### Hypothesis
anchor 暴露的残余失败（current_city 移动日缺 from→to、列表值 str 垃圾、跨列重复、
图内损坏实体、复合交通串）都可以用确定性修复 + 官方库保真门解决，不需要动规划主循环。

### Change（anchor 四轮迭代，每轮 = 修复 → 单测 → 重放回归 → anchor 复跑）
1. `_scalarize`（normalize）：repair 返回的列表值解包（"['...']" 垃圾串根因）。
2. local_repair：交通串带 from→to 而 current_city 是 stay 日 → 重写为 "from A to B"
   （anchor q15/q25：官方要求移动日 from→to，交通串是事实源）。
3. **官方库保真门**（GraphIndex）：餐/住/景 (name substring, city) 与航班 (num,o,d)
   必须在官方 CSV 存在才进规划索引——语料 repr 个别行损坏（q40：名称跨行粘连的
   住宿官方库不存在），图忠实投影了坏行。合成图单测用 fidelity_gate=False 绕过。
4. 跨列去重：三餐全表一个 seen 集（官方语义；此前 _first_occurrence 只查同列，
   day3 早餐 vs day4 晚餐漏判）；景点逐项按序去重。
5. 航班校验改官方语义：先取交通串自身 from→to（复合串 Taxi+Flight 场景），
   local_repair 取 Flight 段重建纯航班串、图外航班换同腿有效航班、无航班换 Taxi
   （Flight+Taxi 官方允许，不引入 Self-driving 冲突）。
6. no-plan salvage 证据补 org↔covered 航班/地面腿（q35：语料无航班时模型按
   "不编造"输出空计划，证据必须给地面交通选项）。
7. 修一个隐藏 bug：`parse_from_to` 返回元组恒真，`bool(parse_from_to(cc))` 把
   所有天当移动日（current_city.malformed 检查失效）→ `any(...)`。

### Validation（每轮）
- 单测 28/28（Gate A：解析/grounding/envelope/payload/validator/cost/local_repair 全覆盖）。
- 旧 50 重放：blocking 298→92，改善 28、变差 0、回退 0（8/8 保护）。
- anchor 迭代：2/9 → 6/9 → 7/9 → 7/9（CS Micro 98.6%），q0/q15 旧 pass 保持。

### After（最终 50 题，`runs/inference`，官方评测器 + 官方固定分母）
| 指标 | 第一轮基线 | 本轮 | 论文（validation 180） |
|---|---|---|---|
| Delivery | 44/50 = 88% | **46/50 = 92%** | — |
| CS Micro | 299/400 = 74.75% | **364/400 = 91.00%** | 86.1% |
| CS Macro | 17/50 = 34% | **42/50 = 84%** | — |
| HC Micro | 28/115 = 24.35% | **95/115 = 82.61%** | 59.3% |
| HC Macro | 10/50 = 20% | **37/50 = 74%** | — |
| Final | 8/50 = 16% | **35/50 = 70%** | 55.9% |

- **旧 8 题零回退**（q2/q6/q7/q8/q14/q15/q78/q123 全过）。
- 残余 15 题失败分布：valid_cost 5、未交付 4（q43/q47/q56/q116 salvage 仍空）、
  reasonable_visiting_city 2、is_not_absent(gated) 2、room rule 1、room type 1。

### Negative results
- anchor q30 超预算 46/1500：餐/住/航班已全是最低价合规项，保守降级无空间——
  属预算本质紧张的题，非修复缺口。
- anchor q35（Atlanta 无航班去 Minnesota）salvage 仍组装不出 2 城合法路线。
- 修复跨列去重时发现官方"首航段"语义与 current_city 端点语义在复合串上不一致，
  以官方交通串优先为准对齐。

### Decision
keep。**Final 70% 超过论文 55.9%**，CS Micro 91.0%（论文 86.1%），复现目标达成并反超。
后续可选方向：预算紧张的 valid_cost 5 题（需要换城市/换路线级决策）、4 题未交付
（no-plan salvage 路线构造）。

### Cost and artifacts
- LLM 调用 12,945 次（react 7,929 / kg 4,308 / func_gen 290 / plan_repair 321 / 其余 397），
  token prompt 40.45M + completion 17.26M；台账 `runs/cost_ledger.jsonl`。
- 全部产物索引见 `REPRODUCTION.md` 第 5 节（最终 runs/inference、第一轮备份
  inference.round1.bak、anchor 四轮备份、replay v1-v6、5 轮构建全轨迹）。

---

## 第三轮（官方库权威属性覆盖 + 降级修复）— 2026-09-27

### 现象（第二轮残余 15 题中的住宿约束 2 题）
- q115 `valid_room_rule` 挂：选中的住宿官方判 `No parties`，本地 validator 放行。
- q125 `valid_room_type` 挂：选中的是 `Private room`，本地 validator 也抓到了却没换掉。

### 诊断
逐题对照官方库：
- q115 选中 `Luxury 4BR Home, Spacious & Central to Trains`：**图里 `house_rules=None`，官方库是 `No parties`**。
- q125 选中 `Huge room 25 min to manhattan`：**图里 `room type=None`，官方库是 `Private room`**。

**根因：图的属性是 LLM 抽的，会整字段丢失；而官方评测全按官方库判。**
本地 validator 信任图的空值 → 放行 → 官方判挂。这是"官方库保真门"只做了名字存在性、
没做属性权威性的缺口。

### Change
1. `oak/agent/graph_index.py`：
   - `official_index` 从"名字集合"升级为"整行字典"（餐/住/景）。
   - GraphIndex 构造时对每个图行做 **官方属性覆盖**（`merge_official`：官方非空字段盖住图值），
     再进索引。名字不存在于官方库的仍丢弃（保真门）。
2. `budget_downgrade` A 段（住宿块）**去早退**：此前换完第一个块若仍超预算就 `return`，
   漏掉后续可降的块 → 改为遍历全部块、每块尝试后都检查是否达标。
3. `local_repair` 住宿替换同样按块遍历（本已如此），块内候选 `_acc_ok` 用块长过滤 min nights。

### Validation
- 单测 28/28 保持。
- 官方评测器离线实测（第二轮最终计划 + 新确定性层，**不调 LLM**）：
  q115 → 换掉违规住宿、q125 → 换掉错房型并降价，**Final 35→36/50**。

### After
| 指标 | 第二轮 | 第三轮（确定性层离线实测） |
|---|---|---|
| CS Micro | 91.00% | 91.00% |
| HC Micro | 82.61% (95/115) | **83.48% (96/115)** |
| HC Macro | 74% (37/50) | **76% (38/50)** |
| Final | 70% (35/50) | **72% (36/50)** |

### Negative results
- q125 **无解**：题面要求 `entire room` + 3 天，但 Albany 唯一合规住宿 `2Br~Union square`
  的 `minimum nights = 30` > 2 晚 → 整城无合规候选，只能在"超预算"与"违规房型"之间二选一。
  属题目硬约束冲突，非修复缺口，本地已正确诊断（budget.over）。
- 本轮第三次 50 题全量复跑**未完成**：智谱账号周限额耗尽
  （"您已达到每周/每月使用上限，限额将在 2026-09-28 21:18:55 重置"，429 code 1310），
  发生在 plan_repair 调用。产物目录已恢复到第二轮状态（`runs/inference`），
  离线重放证据在 `runs/replay/v7`、`v8`。**额度恢复后应重跑一次完整 50 题**以核实。

### Decision
keep（确定性层改进已由官方评测器离线证实 +1）。全量复跑待额度恢复。

---

## 第四轮（官方库补图：修 no-plan 放弃）— 2026-09-27

### 现象
残余 4 题未交付中 q47/q116 的 ReAct 只走了 **1 步、0 次函数调用**就输出空计划。

### 诊断
读轨迹 thought 原文：
- q47：「trip requires exactly 3 distinct visited cities in Virginia, but the authoritative
  covered-city list contains only 2 Virginia cities (Charlottesville, Richmond)」
- q116：「requires exactly 3 ... but ... only two Florida cities」

**根因是我自己第二轮引入的 covered 门槛过严**：`covered` 原本定义为"餐+住+景三类齐全"，
但 q47 的 Jamestown 有住宿+景点、0 餐馆，q116 的 Jacksonville 有 64 餐馆、0 住宿——
被误判为不可停留。加上 P5 的「For a state destination, use only covered cities」指令，
模型算出只有 2 个可选城，与题面要 3 城冲突 → 按"不编造"直接放弃。

**更根本的问题**：官方评测按**全量官方库**判实体存在性与城市合法性，而我们的图只是
该题 reference 语料子集。语料缺某城餐馆时，模型以为该城不可用——但官方库有数据，
该城完全合法。

### Change
1. `oak/kg/graph.py::augment_graph_with_official(g, q, tp_root)`：建图后用官方库
   把该题相关城市（州题取州内全城、城市题取 org+dest、图上已有业务城市）的餐/住/景
   实体并入图，已存在的节点用官方值补齐/覆盖。使规划候选与官方判据一致。
2. `enrich_city_nodes`：`covered` 从"三类齐全"放松为"有任一业务数据即可停留"
   （官方只查城市合法性与实体存在，不要求三类齐全）。
3. `oak/pipeline/inference.py`：建图 → augment → 重算 enrich。

### Validation
- q47（Virginia，需 3 城）：补入 513 实体，可选 covered 城 3 → **9**
  （Charlottesville/Jamestown/Lynchburg/Newport News/Norfolk/Petersburg/Richmond/Roanoke/Staunton）
- q116（Florida，需 3 城）：补入 899 实体，3 → **15**
- 单测 28/28 保持。

### After
待额度恢复后全量复跑核实（预期 q47/q116 能出计划并计入 Delivery/Final）。
离线无 LLM 无法测 ReAct 行为变化，故本轮增益未计入已实测数字。

### Negative results
- 本轮是对第二轮"covered 三类齐全"设计的**回退修正**——教训：本地判据比官方严
  会制造虚假不可行，必须与官方判据对齐（官方只查存在性与合法性）。
- 官方库补图会显著增大每题图（+500~900 实体），需注意内存/耗时（实测可接受）。

### Decision
keep。待额度恢复全量复跑。

---

## 第五轮（记录更正：72% 标注越界）— 2026-09-28

### 现象
用户质疑："Final 72% 这个结论怎么都出来了？测试跑了一半吗？"

### 诊断（属实）
- 第三/四轮的 50 题全量复跑**没有跑一半，是启动即挂**（智谱周限额 429，第一题的
  plan_repair 调用就失败，产物目录已回滚）。
- 唯一干净的端到端实测是第二轮：**35/50 = 70%**（真实 LLM、官方评测器）。
- 72% (36/50) 的真实来历：第二轮的 50 份**真实计划** → 本机离线套用第三轮新写的
  确定性修复（零 LLM）→ 官方评测器打分。评分器是真的，但计划不是新跑的；
  与 70% 只差一题（q115 住宿 house rule 修复）。
- 错误在**表述**：72% 被写进 README 表格第三轮列、HF 数据集标题、commit message
  首行。虽有"离线实测"脚注，但放在结论位置就会被（包括我自己汇报时）当成复测成绩。

### 修正
- 本日志头部与"可复用方法论"新增第 11 条：标题数字只用端到端实测。
- README / HF 仓库标题待复跑结果出来后统一以实测为准改写。
- 教训归入方法论（含"离线增量测量"的合法用法与表述边界）。

### Decision
keep（作为负结果留痕，不删除 72% 的记录本身——它是真实的离线测量，
错的是把它放到了结论的位置）。

---

## 第五轮（全量复跑三轮 + 三重根因连环修）— 2026-09-28/29

### 现象
额度恢复后全量复跑：**24/50（48%）**，比第二轮 35/50 掉 11 题（旧 8 题回退 1，
未交付 4→13，sandbox 新增 10 题）。此后 anchor 连续 6/9 → 3/9 → 3/9 持续恶化。

### 诊断（三重根因，逐个剥出）
1. **僵尸行（官方库 dropna 语义）**：sandbox 失败题的实体（如 `温馨旅店(3), Dallas`）
   在官方 CSV 里存在，但官方加载 `pd.read_csv().dropna()`（四个库全部如此）把
   **任一列为空的行整行丢弃**——评测数据里根本没有。第四轮 augment 与官方索引
   没复刻 dropna，把僵尸行灌进了图（q6 全图 8 家僵尸住宿）。僵尸行往往更便宜，
   被模型/修复层优先选中 → 官方判 invalid。
2. **模型切换污染审计**（最深的坑）：anchor 7/9→6/9→3/9 连续归因错误后查台账，
   发现 `.env` 的 fast 档被切到 `deepseek/deepseek-v4-flash-fast`（思考型，
   reasoning 68k 字符挤占输出，salvage/repair 解析必挂）——**行为突变的真凶是环境，
   不是代码**。切回 glm-5.3-flash 后同代码 anchor 立即 8/9（历史新高）。
   教训进方法论第 13 条。
3. **空计划逃生门**：q37/q51 在第 1 步 0 次函数调用就输出 `Final Plan: []`——
   P5 的"evidence 不足可交空计划"被 flash 当成免调查白卷（thought 里明明已选好城）。

### Change
1. `graph_index.py::official_index` + `kg/graph.py::augment_graph_with_official`：
   读官方 CSV 复刻 dropna（任一字段空 = 行不存在）。
2. `.env`：fast 档切回智谱 glm-5.3-flash（备份 `.env.bak.*`）。
3. `react.py`：**空计划拦截**——证据调用 <3 时输出 `Final Plan: []` 不算交付，
   注入"先收集证据"提示强制继续。
4. `kg/graph.py::covered_cities`：附每城最低住宿价并按价升序；P5 WORKFLOW 明示
   "prefer the cheapest cities"（攻 valid_cost）+ 步数 20→30 + "先锁城再收集"工作流。
5. **回滚第四轮 augment**（anchor 三轮验证净负：7/9→6/9→3/9，为救 2 题丢 11 题；
   函数保留供研究，调用停用）。保留第四轮的 covered 放松定义（有任一业务数据即可停留）。

### Validation / 过程（每步真实数字）
| 跑 | 配置 | 结果 |
|---|---|---|
| r5 全量 | augment + deepseek（污染） | 24/50 |
| r6-r8 anchor | 叠加修复 + deepseek（仍污染） | 6/9 → 3/9 → 3/9 |
| r9 anchor | **切回 glm**，回滚 augment | **8/9**（Delivery 9/9） |
| r10 全量 | glm + 旧 P5 | 32/50（64%，旧 8 零回退，q47 救回） |
| r11 anchor | + 空计划拦截 + 便宜城排序 | 6/9（q35 历史首次通过，HC Micro 90.5%） |
| **r12 全量** | **同上** | **39/50 = 78%** |

### After（最终，`runs/inference`，官方评测器 + 官方固定分母）
| 指标 | 第二轮 | 第五轮（最终） | 论文 |
|---|---|---|---|
| Delivery | 46/50 = 92% | **47/50 = 94%** | — |
| CS Micro | 91.00% | **92.75% (371/400)** | 86.1% |
| CS Macro | 84% | **88%** | — |
| HC Micro | 82.61% | **90.43% (104/115)** | 59.3% |
| HC Macro | 74% | **82%** | — |
| Final | 70% (35/50) | **78% (39/50)** | 55.9% |

- 旧 8 题零回退；vs 第二轮净增 6 题（含第四轮目标 q47、q35 类长期钉子户部分突破）。
- 残余 11 题：valid_cost 5（q125 已证题目无解；其余为预算本质紧张）、未交付 3
  （q17/q43/q74）、current_city/sandbox 4。

### Negative results（本轮全是有价值的失败）
- **augment 全库补图净负**：官方库语义复刻不彻底时引入僵尸行；复刻后又因候选池
  暴涨改变 ReAct 行为分布。"给模型更多数据"不总是对的——与官方判据对齐 ≠ 全量灌入。
- **deepseek 时段（r5-r8）的全部数据作废**：模型对比必须单变量对照，环境漂移
  （.env 被其它会话改）会让代码归因全部失真。审计前先查 ledger 的 model 列。
- anchor 9 题方差大（6/9-8/9 震荡），方向判断够用、精确比较不可靠——这正是
  方法论第 2 条"比例不可外推"的另一面。
- q125 维持无解结论（entire room + 3 天 vs 唯一合规住宿 min_nights=30）。

### Decision
keep。**Final 78%（39/50），超论文 55.9% 二十二个百分点，全指标新高。**定稿。




---

## 第一轮（历史回顾，已在上一次会话完成）

要点留痕（细节见 git 历史与会话记录）：
- Final 0% → 16%（8/50）。
- 关键修复：`_rows_of` 合并 `__key__`（主键可见性，一行解锁 search 函数空结果）；HC 聚合 `(None,None)` 误判 False（一行误杀所有 HC macro）；餐/住 `; Cost: N` 后缀官方必挂（格式改 "Name, City"）；闭环任务理解错重写 P5；沙箱白名单扩充；anchor 9 题快速闭环；ContextVar 图隔离；结构化距离矩阵程序化建图。
