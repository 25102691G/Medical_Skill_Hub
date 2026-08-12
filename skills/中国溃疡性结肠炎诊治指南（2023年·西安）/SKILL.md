---
name: 中国溃疡性结肠炎诊治指南（2023年·西安）
description: "主要疾病及适用范围：用于溃疡性结肠炎（Ulcerative colitis，UC）的诊断、鉴别诊断、治疗和随访，涵盖疾病评估（蒙特利尔分型、Mayo评分、UCEIS）、轻中度/中重度/急性重度UC的治疗、维持治疗及癌变监测，适用于中国成人UC患者的临床诊疗。明确鉴别疾病：感染性肠炎（Infectious enteritis）、阿米巴肠病（Amebiasis）、肠道血吸虫病（Intestinal schistosomiasis）、药物性肠病（Drug-induced enteropathy）、结肠克罗恩病（Colonic Crohn's disease）、肠结核（Intestinal tuberculosis）、真菌性肠炎（Fungal enteritis）、缺血性肠炎（Ischemic colitis）、放射性肠炎（Radiation enteritis）、嗜酸粒细胞性肠炎（Eosinophilic enteritis）、过敏性紫癜（Henoch-Schönlein purpura）、胶原性结肠炎（Collagenous colitis）、肠白塞病（Intestinal Behçet's disease）、结肠息肉病（Colonic polyposis）、结肠憩室炎（Colonic diverticulitis）、HIV感染合并的结肠病变（HIV-associated colonic lesions）。"
---

# 中国溃疡性结肠炎诊治指南（2023年·西安）

## 工作流程

使用本 skill 回答与《中国溃疡性结肠炎诊治指南（2023年·西安）》相关的问题时，以 `references/guideline-full-text.md` 为原文依据。

1. 根据问题或病例阳性特征明确需要检索的诊断概念；不得把候选诊断当作患者已观察到的事实。
2. 在本 Skill 目录中调用一次 `python3 scripts/search_guideline.py catalog`，读取 `references/recommendations-index.md` 的完整章节目录。根据输入与章节含义进行 LLM 语义匹配，选择所有可能相关的章节 ID；脚本不判断医学相关性。
3. 调用一次 `python3 scripts/search_guideline.py entries --heading-id H0001`；多个章节使用多个 `--heading-id`。语义比较返回的完整索引条目与病例阳性特征，选择所有直接支持判断的原文块 ID。
4. 调用一次 `python3 scripts/search_guideline.py sources --source-id L000001-L000003`；多个原文块使用多个 `--source-id`。根据返回的 `guideline-full-text.md` 原文核实适用人群、限制条件和上下文。
5. 原文核实完成后立即作答，不要继续执行探索性文件读取。如用户询问该文件之外的最新证据、药品获批状态、医保或现实可及性，应使用当前权威来源另行核实。

## 回答规则

- 明确说明回答依据《中国溃疡性结肠炎诊治指南（2023年·西安）》。
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

- UC：Ulcerative colitis
- IBD：Inflammatory bowel disease
- CD：Crohn's disease
- 5-ASA：5-aminosalicylic acid
- IFX：Infliximab
- VDZ：Vedolizumab
- ASUC：Acute severe ulcerative colitis
- UCEIS：Ulcerative colitis endoscopic index of severity
