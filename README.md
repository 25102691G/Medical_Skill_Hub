# 消化内科医疗诊断 Agent Demo

这是一个基于 OpenAI Agents SDK 的消化内科医疗诊断 demo。程序先根据病例生成临床问题、候选诊断和医学文献检索查询，再以 `search_queries` 为核心输入调用 Knowledge Searcher Agent，并在最终诊断前调用 Guideline Searcher Agent 检索本地 sandbox guideline skills。最终诊断输出 `DiagnosisResult.topk_diagnoses` 后，会由 Diagnostic Judgement Agent 比较 `topk_diagnoses` 与 `hypotheses` 哪个更贴近病人信息；如果 `topk_diagnoses` 更贴近则结束，否则重新生成 `search_queries` 并再执行一轮诊断，最多执行 2 轮。

核心代码结构：

```text
main.py
config.py
schemas.py
diagnosis/agents/phenotype_extraction_agent.py
diagnosis/agents/digestive_diagnosis_agent.py
diagnosis/agents/guideline_searcher_agent.py
diagnosis/agents/knowledge_searcher_agent.py
diagnosis/agents/search_planning_agent.py
diagnosis/agents/similar_case_retrieval_agent.py
diagnosis/agents/diagnostic_judgement_agent.py
diagnosis/tools/disease_normalization_tool.py
skills/china-crohns-guideline-2023/
skills/china-ulcerative-colitis-guideline-2023/
skills/intestinal-behcet-consensus-2022/
skills/intestinal-tuberculosis-diagnosis-treatment/
skills/china-lymphoma-guideline-2022/
```

## 环境准备

优先使用项目内虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

配置 OpenAI API Key：

```bash
export OPENAI_API_KEY="你的 API Key"
```

可选环境变量：

```bash
export OPENAI_MODEL="gpt-4.1-mini"
export DIAGNOSIS_TOPK="5"
```

## 运行方式

直接传入病例文本：

```bash
python main.py --case "患者男，28岁，反复腹痛腹泻半年，体重下降，外院初步诊断疑似克罗恩病，肠镜提示回末多发溃疡..."
```

或从标准输入读取：

```bash
python main.py
```

查看中间过程：

```bash
python main.py --debug --case "患者男，28岁，反复腹痛腹泻半年，体重下降，外院初步诊断疑似克罗恩病，肠镜提示回末多发溃疡..."
```

## Skill

主流程分为五步：

1. 简要诊断与检索规划：从病例文本中提取当前临床问题、主要候选诊断和医学文献检索查询。
2. 知识检索：将 `search_queries` 传入 Knowledge Searcher Agent，检索相关医学文献和知识。
3. 指南检索：Guideline Searcher Agent 检查 `skills/` 中是否存在与检索规划、知识检索或相似病例结果相关的疾病指南 skill，并通过 sandbox capability 挂载到 `.agents` 自动发现目录后按需调用 `load_skill`。
4. 最终诊断：Digestive Diagnosis Agent 结合病例文本、知识检索结果和指南检索结果输出 topk 疑似诊断。
5. 诊断判断：Diagnostic Judgement Agent 比较 `topk_diagnoses` 和 `hypotheses` 哪个更贴近病人信息；如果 `topk_diagnoses` 更贴近则结束，否则重新生成 `search_queries` 并再执行一轮诊断，诊断阶段最多执行 2 轮。

`Similar Case Retrieval Agent` 的函数仍保留在代码中，但当前 `main.py` 主流程里对应调用处于注释状态。

当前已接入：

- `skills/china-crohns-guideline-2023`：《中国克罗恩病诊治指南（2023年·广州）》
- `skills/china-ulcerative-colitis-guideline-2023`：《中国溃疡性结肠炎诊治指南（2023年·西安）》
- `skills/intestinal-behcet-consensus-2022`：《肠型贝赫切特综合征（肠白塞病）诊断和治疗共识意见》
- `skills/intestinal-tuberculosis-diagnosis-treatment`：《肠结核的诊断与治疗》
- `skills/china-lymphoma-guideline-2022`：《淋巴瘤诊疗指南（2022年版）》

Guideline Searcher Agent 会在最终诊断之前检查 `skills/` 中是否存在与检索规划、知识检索或相似病例结果相关的疾病指南 skill。存在对应 skill 时，通过 OpenAI Agents SDK sandbox capability 挂载到 sandbox 内的 `.agents` 自动发现目录，并按需调用 `load_skill` 加载。Digestive Diagnosis Agent 不再直接加载本地 skills，而是使用 Guideline Searcher Agent 输出的指南证据。

## Knowledge Searcher Agent

`diagnosis/agents/knowledge_searcher_agent.py` 是文献检索 Agent，在当前 `main.py` 诊断主流程中运行于检索规划之后。它以 `SearchPlanningResult.search_queries` 为主要输入，输出会传入最终诊断阶段作为辅助上下文。

它内置两个 function tool：

- `arxiv_search`：基于 `langchain_community.retrievers.ArxivRetriever` 检索 Arxiv。
- `pubmed_search`：基于 `langchain_community.retrievers.PubMedRetriever` 检索 PubMed。

示例：

```python
from agents import Runner
from diagnosis.agents.knowledge_searcher_agent import build_knowledge_searcher_agent

agent = build_knowledge_searcher_agent()
result = Runner.run_sync(agent, "病例信息：反复腹痛腹泻，疑似炎症性肠病，请检索相关文献。")
print(result.final_output)
```

## Disease Normalization Tool

`diagnosis/tools/disease_normalization_tool.py` 是疾病名标准化工具，已接入最终诊断阶段。`final_diagnosis` agent 可以调用 `normalize_disease_name` FunctionTool；同时 `main.py` 会在最终诊断输出后，对 `topk_diagnoses[*].disease` 逐个调用 `normalize_disease_name_text` 做强制标准化。它输入一个诊断疾病名，使用 `FremyCompany/BioLORD-2023-C` 与 ICD10 诊断名称做语义匹配，并输出最接近的标准诊断疾病名。

该 tool 依赖 `database/icd10_id2diagnose.json`。首次运行时会生成 ICD10 诊断名称 embedding 缓存：

```text
database/icd10_diagnose_embeddings.pt
```

示例：

```python
from diagnosis.tools.disease_normalization_tool import normalize_disease_name

tools = [normalize_disease_name]
```

主流程后处理示例：

```python
from diagnosis.tools.disease_normalization_tool import normalize_disease_name_text

standard_name = normalize_disease_name_text("克罗恩病")
```

## Similar Case Retrieval Agent

`diagnosis/agents/similar_case_retrieval_agent.py` 是相似病例诊断归纳 Agent，当前已接入 `main.py` 主流程。它以 `SearchPlanningResult.search_queries` 作为输入，并输出每条查询对应的相似病例诊断结果。

当前仓库尚未接入真实相似病例库，因此该 Agent 不会声称检索到了真实病例；输出会标明结果仅基于 query 语义归纳。后续接入病例库后，可以在该模块内增加检索 tool。

示例：

```python
from agents import Runner
from diagnosis.agents.similar_case_retrieval_agent import (
    build_similar_case_retrieval_agent,
    build_similar_case_retrieval_prompt,
)
search_queries = ["Crohn disease ileal ulcers chronic diarrhea weight loss diagnosis"]

agent = build_similar_case_retrieval_agent()
prompt = build_similar_case_retrieval_prompt(search_queries)
result = Runner.run_sync(agent, prompt)
print(result.final_output)
```

## 输出

程序输出 JSON，包含：

- 是否使用 skill
- 使用的 skill 名称列表
- topk 疑似诊断
- 每个诊断的支持证据、缺失信息、下一步建议
- 如使用指南 skill，会补充指南依据
- 医疗安全提示

默认情况下，标准输出只打印最终诊断 JSON。使用 `--debug` 时，会额外在标准错误中打印每轮检索规划结果、知识检索结果、相似病例结果、最终诊断结果和 Diagnostic Judgement Result。

## 医疗声明

本 demo 仅用于技术演示和辅助分析，不能替代临床医生诊断、治疗建议或线下医疗评估。
