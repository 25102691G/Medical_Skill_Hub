---
name: 原发性硬化性胆管炎诊疗指南(2025年版)
description: "主要疾病及适用范围：用于原发性硬化性胆管炎（Primary sclerosing cholangitis，PSC）的概述、病因、流行病学、临床表现、辅助检查、诊断、鉴别诊断、治疗及并发症监测。明确鉴别疾病：继发性硬化性胆管炎（Secondary sclerosing cholangitis）、IgG4相关硬化性胆管炎（IgG4-related sclerosing cholangitis）、胆管细胞癌（Cholangiocarcinoma）、原发性胆汁性胆管炎（Primary biliary cholangitis）、药物性肝损伤（Drug-induced liver injury）、自身免疫性肝炎（Autoimmune hepatitis）、慢性病毒性肝炎（Chronic viral hepatitis）、酒精性肝病（Alcoholic liver disease）。"
---

# 原发性硬化性胆管炎诊疗指南（2025年版）

## 工作流程

使用本 skill 回答与《原发性硬化性胆管炎诊疗指南（2025年版）》相关的问题时，以 `references/guideline-full-text.md` 为原文依据。

1. 根据问题或病例阳性特征明确需要检索的诊断概念；不得把候选诊断当作患者已观察到的事实。
2. 在本 Skill 目录中调用一次 `python3 scripts/search_guideline.py catalog`，读取 `references/recommendations-index.md` 的完整章节目录。根据输入与章节含义进行 LLM 语义匹配，选择所有可能相关的章节 ID；脚本不判断医学相关性。
3. 调用一次 `python3 scripts/search_guideline.py entries --heading-id H0001`；多个章节使用多个 `--heading-id`。语义比较返回的完整索引条目与病例阳性特征，选择所有直接支持判断的原文块 ID。
4. 调用一次 `python3 scripts/search_guideline.py sources --source-id L000001-L000003`；多个原文块使用多个 `--source-id`。根据返回的 `guideline-full-text.md` 原文核实适用人群、限制条件和上下文。
5. 原文核实完成后立即作答，不要继续执行探索性文件读取。如用户询问该文件之外的最新证据、药品获批状态、医保或现实可及性，应使用当前权威来源另行核实。

## 回答规则

- 明确说明回答依据《原发性硬化性胆管炎诊疗指南（2025年版）》。
- 有推荐意见编号时，列出对应编号。
- 有证据等级和推荐强度时，按索引或原文原样列出。
- 区分“指南/共识推荐、建议、可考虑、不推荐”和 Codex 自己的解释性总结。
- 不要编造原文没有给出的剂量、疗程、监测阈值、禁忌证或随访间隔。
- 对患者个体化决策，说明指南或共识不能替代临床医生评估；诊疗选择需结合疾病分期、活动度、并发症、既往治疗反应、感染风险、合并症和药物可及性。
- 如果原文和索引不一致，以 `guideline-full-text.md` 原文为准。

## 资源

- `references/recommendations-index.md`：LLM 根据全文自动生成的重要信息索引；每个条目带有确定性的全文行号范围，用于直接定位原文。
- `references/guideline-full-text.md`：MinerU 解析得到的指南 Markdown 全文。
- `scripts/search_guideline.py`：确定性输出索引章节、完整条目和原文块；医学相关性由 LLM 语义判断。

## 常用缩写

- PSC：primary sclerosing cholangitis
- IBD：inflammatory bowel disease
- MRCP：magnetic resonance cholangiopancreatography
- ERCP：endoscopic retrograde cholangiopancreatography
- ALP：alkaline phosphatase
- GGT：gamma-glutamyl transferase
- AIH：autoimmune hepatitis
- CCA：cholangiocellular carcinoma
- IgG4-SC：IgG4-related sclerosing cholangitis
- UDCA：ursodeoxycholic acid
- ANA：antinuclear antibody
- SMA：smooth muscle antibody
- pANCA：perinuclear anti-neutrophil cytoplasmic antibody
- cANCA：cytoplasmic anti-neutrophil cytoplasmic antibody
- IgG4：immunoglobulin G4
- HLA：human leukocyte antigen
- GWAS：genome-wide association study
- FXR：farnesoid X receptor
- OCA：obeticholic acid
