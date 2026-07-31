---
name: 《早期结直肠癌全程管理指南(2025版)》解读
description: "类别：肠的其他疾病。用于早期结直肠癌（early colorectal cancer，CRC）的筛查、诊断、治疗和随访全程管理，仅在病例候选诊断包含早期结直肠癌或疑似早期结直肠癌时使用。"
---

# 早期结直肠癌全程管理指南(2025版)

## 工作流程

使用本 skill 回答与《早期结直肠癌全程管理指南(2025版)》相关的问题时，以 `references/guideline-full-text.md` 为原文依据。

1. 先读取 `references/recommendations-index.md`，定位相关推荐意见、诊断标准、鉴别诊断、检查、治疗、监测、随访等重要信息。
2. 再读取 `references/guideline-full-text.md` 中的相关内容，补充适用人群、限制条件、解释依据和上下文。
3. 如果问题没有明显对应推荐意见，使用 `scripts/search_guideline.py` 进行关键词搜索。
4. 如用户询问该文件之外的最新证据、药品获批状态、医保或现实可及性，应使用当前权威来源另行核实。

## 回答规则

- 明确说明回答依据《早期结直肠癌全程管理指南(2025版)》。
- 有推荐意见编号时，列出对应编号。
- 有证据等级和推荐强度时，按索引或原文原样列出。
- 区分“指南/共识推荐、建议、可考虑、不推荐”和 Codex 自己的解释性总结。
- 不要编造原文没有给出的剂量、疗程、监测阈值、禁忌证或随访间隔。
- 对患者个体化决策，说明指南或共识不能替代临床医生评估；诊疗选择需结合疾病分期、活动度、并发症、既往治疗反应、感染风险、合并症和药物可及性。
- 如果原文和索引不一致，以 `guideline-full-text.md` 原文为准。

## 资源

- `references/recommendations-index.md`：LLM 根据全文自动生成的重要信息索引，用于定位推荐意见、诊断标准、鉴别诊断、检查、治疗、监测和随访等关键内容。
- `references/guideline-full-text.md`：MinerU 解析得到的指南 Markdown 全文。
- `scripts/search_guideline.py`：关键词/正则搜索脚本。

## 常用缩写

- CRC：colorectal cancer
