---
name: 中国淋巴瘤诊疗指南（2026版）
description: "主要疾病及适用范围：用于淋巴瘤（Lymphoma）及其常见亚型霍奇金淋巴瘤（HL）、弥漫大B细胞淋巴瘤（DLBCL）、滤泡淋巴瘤（FL）、边缘区淋巴瘤（MZL）、慢性淋巴细胞白血病/小淋巴细胞淋巴瘤（CLL/SLL）、套细胞淋巴瘤（MCL）、伯基特淋巴瘤（BL）、外周T细胞淋巴瘤非特指型（PTCL-NOS）、结外NK/T细胞淋巴瘤（ENKTL）、蕈样真菌病（MF）和Sézary综合征（SS）的诊断、鉴别诊断、分期、治疗和随访。明确鉴别疾病：间变性大细胞淋巴瘤（Anaplastic large cell lymphoma，ALCL）、弥漫大B细胞淋巴瘤（Diffuse large B-cell lymphoma，DLBCL）、伴MYC和BCL2重排的高级别B细胞淋巴瘤（High-grade B-cell lymphoma with MYC and BCL2 rearrangements）、高级别B细胞淋巴瘤伴11q异常（High-grade B-cell lymphoma with 11q anomaly）、外周T细胞淋巴瘤非特指型（Peripheral T-cell lymphoma, not otherwise specified，PTCL-NOS）、未分化癌（Undifferentiated carcinoma）。"
---

# 中国淋巴瘤诊疗指南（2026版）

## 工作流程

使用本 skill 回答与《中国淋巴瘤诊疗指南（2026版）》相关的问题时，以 `references/guideline-full-text.md` 为原文依据。

1. 完整读取 `references/recommendations-index.md`，根据问题或病例阳性特征语义匹配相关推荐意见、诊断标准、鉴别诊断、检查、治疗、监测、随访等重要信息。
2. 按索引条目的“原文位置”直接读取 `references/guideline-full-text.md` 对应行，核实适用人群、限制条件、解释依据和上下文。
3. 如用户询问该文件之外的最新证据、药品获批状态、医保或现实可及性，应使用当前权威来源另行核实。

## 回答规则

- 明确说明回答依据《中国淋巴瘤诊疗指南（2026版）》。
- 有推荐意见编号时，列出对应编号。
- 有证据等级和推荐强度时，按索引或原文原样列出。
- 区分“指南/共识推荐、建议、可考虑、不推荐”和 Codex 自己的解释性总结。
- 不要编造原文没有给出的剂量、疗程、监测阈值、禁忌证或随访间隔。
- 对患者个体化决策，说明指南或共识不能替代临床医生评估；诊疗选择需结合疾病分期、活动度、并发症、既往治疗反应、感染风险、合并症和药物可及性。
- 如果原文和索引不一致，以 `guideline-full-text.md` 原文为准。

## 资源

- `references/recommendations-index.md`：LLM 根据全文自动生成的重要信息索引；每个条目带有确定性的全文行号范围，用于直接定位原文。
- `references/guideline-full-text.md`：MinerU 解析得到的指南 Markdown 全文。

## 常用缩写

- HL：Hodgkin lymphoma
- NHL：non-Hodgkin lymphoma
- cHL：classic Hodgkin lymphoma
- NLPHL：nodular lymphocyte predominant Hodgkin lymphoma
- DLBCL：diffuse large B-cell lymphoma
- FL：follicular lymphoma
- MZL：marginal zone lymphoma
- CLL/SLL：chronic lymphocytic leukemia/small lymphocytic lymphoma
- MCL：mantle cell lymphoma
- BL：Burkitt lymphoma
- PTCL-NOS：peripheral T-cell lymphoma, not otherwise specified
- ENKTL：extranodal NK/T-cell lymphoma
- MF：mycosis fungoides
- SS：Sézary syndrome
- PET-CT：positron emission tomography-computed tomography
- AHSCT：autologous hematopoietic stem cell transplantation
- CAR-T：chimeric antigen receptor T cell
- ISRT：involved site radiotherapy
