---
name: 慢性乙型肝炎防治指南（2022年版）
description: "主要疾病及适用范围：用于慢性乙型肝炎（Chronic hepatitis B，CHB）的预防、筛查、诊断、抗病毒治疗和随访管理，涵盖HBV感染的自然史与临床分期、实验室检查、治疗适应证、核苷（酸）类似物和干扰素治疗，以及特殊人群（妊娠、儿童、肾功能不全、合并HCV/HIV感染等）的处理建议。明确鉴别疾病：吉尔伯特综合征（Gilbert syndrome）、Dubin-Johnson综合征（Dubin-Johnson syndrome）、肾病综合征（nephrotic syndrome）、蛋白丢失性肠病（protein-losing enteropathy）。"
---

# 慢性乙型肝炎防治指南（2022年版）

## 工作流程

使用本 skill 回答与《慢性乙型肝炎防治指南（2022年版）》相关的问题时，以 `references/guideline-full-text.md` 为原文依据。

1. 根据问题或病例阳性特征明确需要检索的诊断概念；不得把候选诊断当作患者已观察到的事实。
2. 在本 Skill 目录中调用一次 `python3 scripts/search_guideline.py catalog`，读取 `references/recommendations-index.md` 的完整章节目录。根据输入与章节含义进行 LLM 语义匹配，选择所有可能相关的章节 ID；脚本不判断医学相关性。
3. 调用一次 `python3 scripts/search_guideline.py entries --heading-id H0001`；多个章节使用多个 `--heading-id`。语义比较返回的完整索引条目与病例阳性特征，选择所有直接支持判断的原文块 ID。
4. 调用一次 `python3 scripts/search_guideline.py sources --source-id L000001-L000003`；多个原文块使用多个 `--source-id`。根据返回的 `guideline-full-text.md` 原文核实适用人群、限制条件和上下文。
5. 原文核实完成后立即作答，不要继续执行探索性文件读取。如用户询问该文件之外的最新证据、药品获批状态、医保或现实可及性，应使用当前权威来源另行核实。

## 回答规则

- 明确说明回答依据《慢性乙型肝炎防治指南（2022年版）》。
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

- CHB：chronic hepatitis B
- HBV：hepatitis B virus
- HBsAg：hepatitis B surface antigen
- HBeAg：hepatitis B e antigen
- HBV DNA：hepatitis B virus deoxyribonucleic acid
- ALT：alanine aminotransferase
- HCC：hepatocellular carcinoma
- NAs：nucleoside/nucleotide analogues
- Peg-IFN-α：pegylated interferon α
- TDF：tenofovir disoproxil fumarate
- TAF：tenofovir alafenamide fumarate
