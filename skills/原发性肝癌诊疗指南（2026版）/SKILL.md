---
name: 原发性肝癌诊疗指南（2026版）
description: "主要疾病及适用范围：用于肝细胞癌（hepatocellular carcinoma，HCC）的预防、筛查、诊断、鉴别诊断、分期、治疗和随访，涵盖外科手术、消融、经动脉介入、放射治疗、系统治疗和中医药等诊疗建议。明确鉴别疾病：肝内胆管癌（Intrahepatic cholangiocarcinoma，ICC）、混合型肝细胞癌-胆管癌（Combined hepatocellular-cholangiocarcinoma，cHCC-CCA）、肝转移癌（Liver metastasis）、肝血管瘤（Hepatic hemangioma）、高度异型增生结节（High-grade dysplastic nodule）、肝局灶性结节性增生（Focal nodular hyperplasia，FNH）、肝细胞腺瘤（Hepatocellular adenoma）。"
---

# 原发性肝癌诊疗指南（2026年版）

## 工作流程

使用本 skill 回答与《原发性肝癌诊疗指南（2026年版）》相关的问题时，以 `references/guideline-full-text.md` 为原文依据。

1. 完整读取 `references/recommendations-index.md`，根据问题或病例阳性特征语义匹配相关推荐意见、诊断标准、鉴别诊断、检查、治疗、监测、随访等重要信息。
2. 按索引条目的“原文位置”直接读取 `references/guideline-full-text.md` 对应行，核实适用人群、限制条件、解释依据和上下文。
3. 如用户询问该文件之外的最新证据、药品获批状态、医保或现实可及性，应使用当前权威来源另行核实。

## 回答规则

- 明确说明回答依据《原发性肝癌诊疗指南（2026年版）》。
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

- HCC：hepatocellular carcinoma
- ICC：intrahepatic cholangiocarcinoma
- cHCC-CCA：combined hepatocellular-cholangiocarcinoma
- AFP：alpha-fetoprotein
- PIVKA-II：protein induced by vitamin K absence/antagonist-II
- CNLC：China Liver Cancer Staging
- TACE：transarterial chemoembolization
- HAIC：hepatic arterial infusion chemotherapy
- SBRT：stereotactic body radiotherapy
