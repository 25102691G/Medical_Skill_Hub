# 消化内科医疗诊断 Agent Demo

这是一个基于 OpenAI Agents SDK 的消化内科医疗诊断 demo。程序先从病例文本中提取英文表型关键词，再调用 Knowledge Searcher Agent 检索相关医学知识，然后对病例文本做初筛，只检索文本中是否明确提到“疑似、考虑、倾向、初步诊断、待排”等疾病方向；如果提到的疑似疾病有对应本地指南 skill，则通过 sandbox skills 懒加载对应指南辅助诊断；如果没有明确疑似疾病，或疑似疾病没有对应 skill，则不使用 skill，直接输出 topk 疑似诊断。

核心代码结构：

```text
main.py
config.py
schemas.py
diagnosis_agents/phenotype_extraction_agent.py
diagnosis_agents/digestive_diagnosis_agent.py
diagnosis_agents/knowledge_searcher_agent.py
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

主流程分为四步：

1. 表型提取：从病例文本中提取英文 phenotype，尽量补充可信的 HPO 编号。
2. 知识检索：调用 Knowledge Searcher Agent 检索相关医学文献和知识，并把检索结果传入初筛阶段。
3. 疾病初筛：只根据病例文本中明确写出的疑似疾病方向判断是否需要启用 skill，表型关键词不作为 skill 触发依据。
4. 最终诊断：结合病例文本、表型关键词、初筛结果和可用 skill 输出 topk 疑似诊断。

当前已接入：

- `skills/china-crohns-guideline-2023`：《中国克罗恩病诊治指南（2023年·广州）》
- `skills/china-ulcerative-colitis-guideline-2023`：《中国溃疡性结肠炎诊治指南（2023年·西安）》
- `skills/intestinal-behcet-consensus-2022`：《肠型贝赫切特综合征（肠白塞病）诊断和治疗共识意见》
- `skills/intestinal-tuberculosis-diagnosis-treatment`：《肠结核的诊断与治疗》
- `skills/china-lymphoma-guideline-2022`：《淋巴瘤诊疗指南（2022年版）》

Agent 只有在病例文本明确提到对应疑似疾病时，才会检查 `skills/` 中是否有对应 skill；如果同时检索到多个疑似疾病，则只启用其中存在本地 skill 的疾病指南。
这些 skill 通过 OpenAI Agents SDK sandbox capability 挂载到 sandbox 内的 `.agents` 自动发现目录，并在最终诊断阶段按需调用 `load_skill` 加载。

## Knowledge Searcher Agent

`diagnosis_agents/knowledge_searcher_agent.py` 是文献检索 Agent，在当前 `main.py` 诊断主流程中运行于表型提取之后、疾病初筛之前。它的输出会传入初筛阶段，作为初筛分析的辅助上下文。

它内置两个 function tool：

- `arxiv_search`：基于 `langchain_community.retrievers.ArxivRetriever` 检索 Arxiv。
- `pubmed_search`：基于 `langchain_community.retrievers.PubMedRetriever` 检索 PubMed。

示例：

```python
from agents import Runner
from diagnosis_agents.knowledge_searcher_agent import build_knowledge_searcher_agent

agent = build_knowledge_searcher_agent()
result = Runner.run_sync(agent, "病例信息：反复腹痛腹泻，疑似炎症性肠病，请检索相关文献。")
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

默认情况下，标准输出只打印最终诊断 JSON。使用 `--debug` 时，会额外在标准错误中打印表型提取结果、知识检索结果、初筛结果、实际可用 skill 和最终诊断结果。

## 医疗声明

本 demo 仅用于技术演示和辅助分析，不能替代临床医生诊断、治疗建议或线下医疗评估。
