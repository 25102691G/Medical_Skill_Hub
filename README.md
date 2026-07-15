# 消化内科医疗诊断 Agent Demo

这是一个基于 OpenAI Agents SDK 的消化内科医疗诊断 demo。程序先根据病例生成候选诊断和医学文献检索查询，再以 `search_queries` 为核心输入调用 Knowledge Searcher Agent，并在最终诊断前调用 Guideline Searcher Agent 检索本地 sandbox guideline skills。最终诊断输出 `DiagnosisResult.topk_diagnoses` 后，会由 Diagnostic Judgement Agent 比较 `topk_diagnoses` 与 `hypotheses` 哪个更贴近病人信息；如果 `topk_diagnoses` 更贴近则结束，否则重新生成 `search_queries` 并再执行一轮诊断，最多执行 2 轮。

核心代码结构：

```text
main.py
run_main.sh
run_compile_skill.sh
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

也可以在 `run_main.sh` 顶部配置病例文本、模型和诊断数量后运行：

```bash
./run_main.sh
```

## Skill

主流程分为五步：

1. 简要诊断与检索规划：从病例文本中生成主要候选诊断和医学文献检索查询。
2. 知识检索：将 `search_queries` 传入 Knowledge Searcher Agent，检索相关医学文献和知识。
3. 指南检索：将 `search_queries` 传入 Guideline Searcher Agent；Agent 检查 `skills/` 中是否存在相关疾病指南 skill，并通过 sandbox capability 挂载到 `.agents` 自动发现目录后按需调用 `load_skill`。
4. 最终诊断：Digestive Diagnosis Agent 结合病例文本、知识检索结果和指南检索结果输出 topk 疑似诊断。
5. 诊断判断：Diagnostic Judgement Agent 比较 `topk_diagnoses` 和 `hypotheses` 哪个更贴近病人信息；如果 `topk_diagnoses` 更贴近则结束，否则重新生成 `search_queries` 并再执行一轮诊断，诊断阶段最多执行 2 轮。

`Similar Case Retrieval Agent` 的函数仍保留在代码中，但当前 `main.py` 主流程里对应调用处于注释状态。

当前已接入：

- `skills/china-crohns-guideline-2023`：《中国克罗恩病诊治指南（2023年·广州）》
- `skills/china-ulcerative-colitis-guideline-2023`：《中国溃疡性结肠炎诊治指南（2023年·西安）》
- `skills/intestinal-behcet-consensus-2022`：《肠型贝赫切特综合征（肠白塞病）诊断和治疗共识意见》
- `skills/intestinal-tuberculosis-diagnosis-treatment`：《肠结核的诊断与治疗》
- `skills/china-lymphoma-guideline-2022`：《淋巴瘤诊疗指南（2022年版）》

Guideline Searcher Agent 只使用 `SearchPlanningResult.search_queries` 作为检索输入，并在最终诊断之前检查 `skills/` 中是否存在与查询相关的疾病指南 skill。存在对应 skill 时，通过 OpenAI Agents SDK sandbox capability 挂载到 sandbox 内的 `.agents` 自动发现目录，并按需调用 `load_skill` 加载。Digestive Diagnosis Agent 不再直接加载本地 skills，而是使用 Guideline Searcher Agent 输出的指南证据。

## Knowledge Searcher Agent

`diagnosis/agents/knowledge_searcher_agent.py` 是文献检索 Agent，在当前 `main.py` 诊断主流程中运行于检索规划之后。它只使用 `SearchPlanningResult.search_queries` 作为输入，输出会传入最终诊断阶段作为辅助上下文。

它内置一个 function tool：

- `pubmed_search`：基于 `langchain_community.retrievers.PubMedRetriever` 检索 PubMed。

示例：

```python
from agents import Runner
from diagnosis.agents.knowledge_searcher_agent import build_knowledge_searcher_agent

agent = build_knowledge_searcher_agent()
result = Runner.run_sync(agent, "病例信息：反复腹痛腹泻，疑似炎症性肠病，请检索相关文献。")
print(result.final_output)
```

## Skill Compiler Agent

`compile_skill.py` 是独立的 guideline skill 编译入口，不接入 `main.py` 诊断主流程。它将 PDF 指南编译成 `skills/<输入文件名>/` 目录，输出结构与现有 guideline skills 保持一致：

```text
skills/<输入文件名>/SKILL.md
skills/<输入文件名>/agents/openai.yaml
skills/<输入文件名>/references/guideline-full-text.md
skills/<输入文件名>/references/guideline-page-map.json  # MinerU 提供页码元数据时生成
skills/<输入文件名>/references/recommendations-index.md
skills/<输入文件名>/scripts/search_guideline.py
```

默认流程会调用 MinerU 解析 PDF，读取 Markdown 和 `*_content_list.json`，在 `guideline-full-text.md` 中加入 PDF 物理页码/印刷页码边界标记，并生成 `guideline-page-map.json`。随后 Skill Compiler Agent 从带页码的全文中自动生成 `recommendations-index.md` 重要信息索引和 skill 元数据。索引内容由 LLM 根据全文决定，不限于固定推荐意见表格，可包含诊断标准、鉴别诊断、检查、治疗、监测和随访等关键内容，并尽可能同时标注 PDF 物理页码、印刷页码和原文行号：

```bash
python compile_skill.py --pdf path/to/guideline.pdf
python compile_skill.py --pdfs path/to/guidelines
```

也可以在 `run_compile_skill.sh` 顶部配置输入、输出目录、模型供应商和 MinerU 命令后运行：

```bash
./run_compile_skill.sh
```

`compile_skill.py` 的 `--pdf` 用于单文件编译，`--pdfs` 用于批量编译指定目录第一层的所有 PDF。`run_compile_skill.sh` 默认使用批量目录模式：

```bash
INPUT_PDFS="./guidelines"
```

批量目录不存在时程序会报错；目录中没有 PDF 时会打印提示并正常退出。PDF 会按文件名排序后逐份编译，其中一份编译失败时批处理会停止。脚本传入 `--force`，因此会覆盖已存在 Skill 中的同名生成文件。

该脚本固定使用项目 `.venv` 中的 Python 和 MinerU，并默认设置 `MINERU_DEVICE_MODE=cpu`，避免共享 GPU 显存不足导致 MinerU 解析失败。确认 GPU 有足够空闲显存后，可临时切换为 CUDA：

```bash
MINERU_DEVICE_MODE=cuda ./run_compile_skill.sh
```

Skill 名称固定使用输入文件名（不含扩展名）。例如输入 `path/to/guideline.pdf`，生成目录为 `skills/guideline/`，`SKILL.md` 中的 `name` 也为 `guideline`。

如果已经有 MinerU 解析后的 Markdown，可跳过 PDF 解析。Markdown 同目录存在同名的 `*_content_list.json` 时，也会自动读取页码和文本块坐标；不存在时保持原来的纯 Markdown 编译方式，不生成 `guideline-page-map.json`：

```bash
python compile_skill.py --full-text-md path/to/guideline-full-text.md
```

默认 MinerU 命令模板为：

```bash
mineru -p {input} -o {output} -b pipeline -m auto -l ch
```

`guideline-page-map.json` 保留每一页的零基 `page_idx`、一基 PDF 物理页码、识别到的印刷页码、Markdown 行号范围，以及各文本块的类型、边界框和行号映射。`page_idx` 与印刷页码不是同一概念，例如 `page_idx=0` 表示 PDF 第 1 页，印刷页码可能是 177。

每次通过 PDF 编译时，MinerU 的输出根路径固定为项目根目录下的 `mineru/`。Compiler 会按 PDF 文件名检查 `mineru/<PDF 文件名>/`：当前 PDF 的目录和 Markdown 已存在时直接复用（包括对应的 `*_content_list.json`）；当前 PDF 的目录不存在时，即使 `mineru/` 根目录已存在，也会运行 MinerU。当前 PDF 的目录存在但没有 Markdown 时会直接报错。MinerU 自身会在该路径下生成 `<PDF 文件名>/auto/` 目录，其中包含 Markdown、`content_list.json`、`middle.json`、`model.json`、`layout.pdf`、`span.pdf` 等产物；这些大体积调试文件不会复制到 skill 中。

如本地 MinerU 命令不同，可用环境变量或参数覆盖：

```bash
export MINERU_COMMAND="mineru -p {input} -o {output} -b pipeline -m auto -l ch"
python compile_skill.py --pdf path/to/guideline.pdf
```

MinerU 也支持通过 `MINERU_DEVICE_MODE` 选择运行设备；未使用上述脚本时，可以显式指定 `cpu` 或 `cuda`：

```bash
export MINERU_DEVICE_MODE="cpu"
python compile_skill.py --pdf path/to/guideline.pdf
```

默认情况下，Skill Compiler Agent 使用 `OPENAI_MODEL` 和 `OPENAI_API_KEY`，并通过 OpenAI Agents SDK 的结构化输出生成索引。如需让 compiler 单独使用 DeepSeek API，可配置：

```bash
export SKILL_COMPILER_PROVIDER="deepseek"
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
export DEEPSEEK_MODEL="deepseek-chat"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

然后照常运行：

```bash
python compile_skill.py --pdf path/to/guideline.pdf
```

DeepSeek 分支会先使用普通 Chat Completions 文本 JSON 输出简短的 skill 元数据，并在本地用 Pydantic 解析，不使用 OpenAI structured `response_format`。指南重要信息索引会按带行号的原文分块生成 Markdown 片段，再按原文顺序合并为 `recommendations-index.md`，避免将超长索引放入单个 JSON 字符串而被模型截断。如果任一请求仍因输出长度截断，compiler 会明确报告对应的元数据或索引分块。

如需切回 OpenAI：

```bash
export SKILL_COMPILER_PROVIDER="openai"
export OPENAI_API_KEY="你的 OpenAI API Key"
```

目标目录已存在且未传入 `--force` 时，程序会打印跳过信息并正常退出，不读取 MinerU 产物，也不调用 Skill Compiler Agent；确认需要覆盖时再加 `--force`。

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

`diagnosis/agents/similar_case_retrieval_agent.py` 是相似病例诊断归纳 Agent。它只使用 `SearchPlanningResult.search_queries` 作为输入，并输出每条查询对应的相似病例诊断结果；当前 `main.py` 主流程中的调用处于注释状态。

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
