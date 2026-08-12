---
name: 中国成人慢性便秘评估与外科处理临床实践指南(2022版)
description: "主要疾病及适用范围：用于中国成人慢性便秘（chronic constipation，CC）的临床评估、非手术治疗和外科处理，涵盖慢传输型便秘（STC）、出口梗阻型便秘（OOC）、盆底功能障碍相关便秘及成人巨结肠的术前评估、术式选择和术后管理。明确鉴别疾病：成人先天性巨结肠（Hirschsprung disease，HD）、成人特发性巨结肠（idiopathic megacolon，IMC）。"
---

# 中国成人慢性便秘评估与外科处理临床实践指南(2022版)

## 工作流程

使用本 skill 回答与《中国成人慢性便秘评估与外科处理临床实践指南(2022版)》相关的问题时，以 `references/guideline-full-text.md` 为原文依据。

1. 根据问题或病例阳性特征明确需要检索的诊断概念；不得把候选诊断当作患者已观察到的事实。
2. 在本 Skill 目录中调用一次 `python3 scripts/search_guideline.py catalog`，读取 `references/recommendations-index.md` 的完整章节目录。根据输入与章节含义进行 LLM 语义匹配，选择所有可能相关的章节 ID；脚本不判断医学相关性。
3. 调用一次 `python3 scripts/search_guideline.py entries --heading-id H0001`；多个章节使用多个 `--heading-id`。语义比较返回的完整索引条目与病例阳性特征，选择所有直接支持判断的原文块 ID。
4. 调用一次 `python3 scripts/search_guideline.py sources --source-id L000001-L000003`；多个原文块使用多个 `--source-id`。根据返回的 `guideline-full-text.md` 原文核实适用人群、限制条件和上下文。
5. 原文核实完成后立即作答，不要继续执行探索性文件读取。如用户询问该文件之外的最新证据、药品获批状态、医保或现实可及性，应使用当前权威来源另行核实。

## 回答规则

- 明确说明回答依据《中国成人慢性便秘评估与外科处理临床实践指南(2022版)》。
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

- CC：chronic constipation（慢性便秘）
- STC：slow transit constipation（慢传输型便秘）
- OOC：outlet obstruction constipation（出口梗阻型便秘）
- HD：Hirschsprung disease（先天性巨结肠）
- IMC：idiopathic megacolon（特发性巨结肠）
- DD：dyssynergic defaecation（不协调性排粪障碍）
- TC-IRA：total colectomy with ileorectal anastomosis（全结肠切除回肠直肠吻合术）
- SNM：sacral neuromodulation（骶神经调节术）
- LVMR：laparoscopic ventral mesh rectopexy（腹腔镜腹侧补片直肠固定术）
- STARR：stapled transanal rectal resection（经肛门直肠切除钉合术）
