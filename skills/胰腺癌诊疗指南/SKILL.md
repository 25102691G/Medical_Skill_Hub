---
name: 胰腺癌诊疗指南
description: "主要疾病及适用范围：用于胰腺导管腺癌（Pancreatic ductal adenocarcinoma，PDAC）的诊断、鉴别诊断、分期、治疗和随访。涵盖影像学、内镜、病理学、外科治疗、内科治疗、放射治疗、介入治疗、中医药及支持治疗等临床建议。明确鉴别疾病：慢性胰腺炎（Chronic pancreatitis）、壶腹癌（Ampullary carcinoma）、胰腺囊腺瘤（Pancreatic cystadenoma）、胰腺囊腺癌（Pancreatic cystadenocarcinoma）、胆总管结石（Choledocholithiasis）、胰腺假性囊肿（Pancreatic pseudocyst）、胰岛素瘤（Insulinoma）、实性假乳头状瘤（Solid pseudopapillary neoplasm）。"
---

# 胰腺癌诊疗指南（2022年版）

## 工作流程

使用本 skill 回答与《胰腺癌诊疗指南（2022年版）》相关的问题时，以 `references/guideline-full-text.md` 为原文依据。

1. 完整读取 `references/recommendations-index.md`，根据问题或病例阳性特征语义匹配相关推荐意见、诊断标准、鉴别诊断、检查、治疗、监测、随访等重要信息。
2. 按索引条目的“原文位置”直接读取 `references/guideline-full-text.md` 对应行，核实适用人群、限制条件、解释依据和上下文。
3. 如用户询问该文件之外的最新证据、药品获批状态、医保或现实可及性，应使用当前权威来源另行核实。

## 回答规则

- 明确说明回答依据《胰腺癌诊疗指南（2022年版）》。
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

- PDAC：Pancreatic ductal adenocarcinoma
- CA19-9：Carbohydrate antigen 19-9
- EUS：Endoscopic ultrasonography
- ERCP：Endoscopic retrograde cholangiopancreatography
- PTCD：Percutaneous transhepatic cholangiodrainage
