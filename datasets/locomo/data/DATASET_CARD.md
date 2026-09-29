---
license: mit
language:
  - zh
  - en
task_categories:
  - question-answering
tags:
  - locomo
  - long-term-memory
  - evaluation
  - memory-benchmark
  - chinese
---

# LoCoMo 中文版（LoCoMo Chinese）

[LoCoMo](https://github.com/snap-research/locomo)（ACL 2024，*Evaluating Very Long-Term Conversational Memory of LLM Agents*）基准 `locomo10.json` 的全量中文化版本。源数据采用 MIT 许可（HF: adymaharana/locomo），本译本同样 MIT。

- 翻译模型：智谱 glm-5.3-flash，284 条全局术语表统一人名/地名译法
- 结构与源数据完全一致（对话数、每 session 消息数、dia_id、QA 类别/evidence、date_time 逐条对齐）
- 100% 全量校验：5,882 条对话消息 + 1,986 条 QA，结构 0 错、漏译 0、人名拉丁残留 0

## 文件

| 文件 | 说明 |
|---|---|
| `locomo10_zh.json` | **中文版主数据集**（2.0MB，10 段对话 × ~35 session，1,986 QA） |
| `locomo10.json` | 英文原始数据集（2.7MB，未改动） |

## 内容与翻译口径

- 10 段超长对话（每段两位说话人、~20-35 个 session），全部发言文本、说话人姓名、QA 题目与答案均已译为简体中文
- QA 五类（单跳/多跳/时间推理/开放域/对抗）：1,542 条文本答案已译；6 条整数答案（2022、2、3）保留数字；444 条对抗题保留 `adversarial_answer` 键（同样已译）
- 标注字段（event_summary / observation / session_summary）一并翻译，dia_id 引用保留
- 原样保留：dia_id、category、evidence、`*_date_time` 元数据、sample_id；品牌/产品名（Nintendo、H&M）、代码/链接按规范保留原文
- 人名按术语表统一（Caroline→卡罗琳、Melanie→梅拉妮、Andrew→安德鲁），全量扫描无中英混杂

## 加载示例

```python
import json
data = json.load(open("locomo10_zh.json"))
c = data[0]
# c["conversation"]["session_1"] = [{"speaker": "卡罗琳", "dia_id": "D1:1", "text": "..."}]
# c["qa"] = [{"question": "...", "answer": "...", "evidence": ["D1:3"], "category": 2}]
```

## 引用

- 源数据：[snap-research/locomo](https://github.com/snap-research/locomo)（MIT）
- 论文：Maharana et al., *Evaluating Very Long-Term Conversational Memory of LLM Agents*, ACL 2024
