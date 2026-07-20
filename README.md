# 消化内科医疗诊断 Agent Demo

## 环境准备

优先使用项目内虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行方式

在 `run_main.sh` 顶部配置病例文本、模型和诊断数量后运行：

```bash
./run_main.sh
```

## 批量运行

`batch_main.py` 默认读取 `database/mimic_iv_test_case.csv`，逐行使用
`discharge_text_before_disposition` 作为 `case_text` 运行完整诊断流水线。使用
`--limit` 控制本次处理的病例数量：

```bash
.venv/bin/python batch_main.py --limit 10
```

如需读取其他结构相同的 CSV，可通过 `--csv` 指定：

```bash
.venv/bin/python batch_main.py --csv database/mimic_iv_test_case.csv --limit 10
```

结果逐条写入 `output/mimiv_iv_diagnosis_results_<时间戳>.jsonl`。每行对应一个成功完成
的病例，仅包含 `subject_id`、`hadm_id`、`long_title` 和结构化
`diagnosis_result`。单个病例失败时，错误会输出到终端，脚本继续处理下一条病例。

## 纯 LLM Baseline

`llm_baseline.py` 与 `batch_main.py` 使用相同的输入 CSV、病例文本列和外层 JSONL
字段，但每个病例仅执行直接的 LLM 诊断调用，不执行指南检索、PubMed
检索、相似病例检索或疾病名称标准化。`diagnosis_result` 只保存评估所需的
`topk_diagnoses`，每项包含 `rank` 和 `disease`。模型调用统一通过
`interface.py` 中的 `LLM_handler` 完成，目前支持 OpenAI、DeepSeek、Gemini 和
Claude。

```bash
./run_llm_baseline.sh
```

在项目根目录的 `.env` 中配置供应商、API Key 和模型。例如：

```dotenv
LLM_BASELINE_PROVIDER="deepseek"
DEEPSEEK_API_KEY="your_deepseek_api_key"
DEEPSEEK_MODEL="deepseek-chat"
```

四种供应商对应的配置分别为 `OPENAI_API_KEY` / `OPENAI_MODEL`、
`DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL`、`GEMINI_API_KEY` / `GEMINI_MODEL`
和 `CLAUDE_API_KEY` / `CLAUDE_MODEL`。`run_llm_baseline.sh` 会加载 `.env`，
再通过命令行参数传给 `llm_baseline.py`。也可以直接运行 Python 入口：

```bash
.venv/bin/python llm_baseline.py \
  --model deepseek \
  --csv database/mimic_iv_test_case.csv \
  --limit 10
```

API Key 和模型参数均为可选参数；未显式传入时，`LLM_handler` 会读取 `.env`
中的对应配置。Gemini 和 Claude 分别需要可选依赖 `google-generativeai` 和
`anthropic`，只有选择相应供应商时才会导入。

结果逐条写入 `output/mimiv_iv_llm_baseline_results_<时间戳>.jsonl`，可以直接作为
`evaluate_diagnosis_results.py` 的输入。

## 诊断结果评估

`evaluate_diagnosis_results.py` 使用 DeepSeek 对批量诊断结果进行评估。脚本逐行读取
`long_title` 作为标准诊断，并提取 `diagnosis_result.topk_diagnoses` 中前五项的
`disease`，让模型判断标准诊断在预测疾病中的排名。运行前需要在 `.env` 中配置
`DEEPSEEK_API_KEY`；模型、接口地址和重试次数分别由 `DEEPSEEK_MODEL`、
`DEEPSEEK_BASE_URL` 和 `DEEPSEEK_EVALUATION_RETRIES` 控制。

评估默认的诊断结果文件：

```bash
.venv/bin/python evaluate_diagnosis_results.py
```

也可以指定输入和输出 JSONL：

```bash
.venv/bin/python evaluate_diagnosis_results.py \
  --input output/mimiv_iv_diagnosis_results_<时间戳>.jsonl \
  --output output/diagnosis_evaluation.jsonl
```

每条评估结果会实时写入输出文件。程序结束时在标准输出中给出 `recall1`、`recall3`
和 `recall5`；模型返回 `No` 时，该病例在三个指标中都记为未命中。

## ChatKit 聊天界面

项目提供基于 ChatKit 的自托管聊天界面。现有 `make_diagnosis()` 诊断流水线保持不变，FastAPI 适配层位于 `chatkit_app/`，React 前端位于 `chatkit_frontend/`。

先安装后端和前端依赖：

```bash
source .venv/bin/activate
pip install -r requirements.txt
cd chatkit_frontend
npm install
cd ..
```

在第一个终端启动 ChatKit 后端：

```bash
./run_chatkit.sh
```

如果脚本没有执行权限，也可以运行：

```bash
bash run_chatkit.sh
```

在第二个终端启动前端：

```bash
cd chatkit_frontend
npm run dev
```

前端开发服务器固定使用 `43179` 端口，启动后访问 `http://localhost:43179`。
远程使用时，需要同时转发前端 `43179` 端口和后端 `8000` 端口。

前端右上角可选择简体中文或英文作为显示语言。选择结果会同时控制 ChatKit 自带界面、
页面静态文字以及后端消息的展示翻译。前端通过 `X-Display-Language` 请求头传递目标
语言；每个 Agent 完成后，ChatKit 服务端会翻译该阶段的字段标签和字符串内容，再立即
追加到聊天界面。如果切换显示语言，当前线程会按新语言重新加载已有助手消息。

展示翻译不会修改 `make_diagnosis()` 的原始结构化结果。URL、数值、计量单位、医学
编码、枚举值、住院号和 `skill_names` 等机器标识保持不变，其余可见内容按目标语言
翻译。翻译失败时会回退到未翻译内容，不会中断诊断流水线。翻译默认使用
`OPENAI_MODEL`，可在 `.env` 中单独设置：

```dotenv
CHATKIT_TRANSLATION_MODEL=your_translation_model
```

当前实时粒度为阶段级：`main.py` 产生 `stage_completed` 事件后翻译并展示完整阶段结果，
不进行逐 token 翻译。

## PubMed 检索配置

医学知识检索通过 NCBI E-utilities 查询 PubMed。建议在项目根目录的 `.env` 中配置：

```dotenv
NCBI_API_KEY=your_ncbi_api_key
NCBI_EMAIL=your_email@example.com
NCBI_TOOL=medical_skill_hub
```

程序会统一限制 NCBI 请求频率：未配置 API Key 时默认不超过每秒 3 次，配置后默认
不超过每秒 10 次；`ESearch`、批量 `EFetch` 和临时网络错误均使用指数退避重试。
如需调整，可设置 `NCBI_REQUESTS_PER_SECOND`、`NCBI_MAX_RETRIES`、
`NCBI_RETRY_BASE_SECONDS` 和 `NCBI_TIMEOUT_SECONDS`。

## 相似病例检索

检索规划阶段生成 `similar_case_queries` 英文短语列表，其中同时包含病例中明确记录的
阳性临床表现和阳性辅助检查结果。临床表现包括阳性症状、异常生命体征和体格检查阳性
体征；辅助检查结果包括实验室、影像、内镜、病理和微生物检查结果。相似病例库使用
`database/mimic_iv_similar_case.csv`，并以其中的完整 `discharge_text` 作为检索语料。
BM25 仅对英文和数字分词；统一语料分别执行 BM25 和 Dense Retriever 检索，再通过
RRF 融合排名。最终按相关性顺序输出最多 10 条病例对应的 `hadm_id`、
`long_title`（输出字段为 `discharge_disease`）和 `discharge_text`。前端的相似病例
检索结果只展示住院号和出院疾病，完整出院记录仅作为外部参考证据传入最终诊断和诊断
判断阶段，不会被视为当前患者已经存在的临床事实。
启用 `--debug` 时，终端的标准错误流会输出统一检索语料的 BM25 和 Dense Top-K 排名，
包括查询文本、住院号、出院疾病和原始分数；未执行的检索分支会输出跳过原因。
ChatKit 前端也会在每轮相似病例检索完成后展示同一组两路排名及跳过原因，该展示通过
阶段进度事件传递，不要求后端启用 `debug`。

Dense Retriever 默认使用 `BAAI/bge-m3`，首次运行时由 Transformers 加载模型，并将
病例库向量缓存到 `database/mimic_iv_similar_case_embeddings.pt`。模型默认在 CPU 上运行，
可通过 `SIMILAR_CASE_EMBEDDING_DEVICE` 设置为 `cpu`、`cuda` 或 `auto`；其中 `auto`
会在 CUDA 可用时使用 GPU。其他配置可通过
`MIMIC_IV_CASE_PATH`、`SIMILAR_CASE_TOP_K`、`SIMILAR_CASE_EMBEDDING_MODEL`、
`SIMILAR_CASE_EMBEDDING_CACHE_PATH` 和 `SIMILAR_CASE_EMBEDDING_BATCH_SIZE` 调整配置。

## 最终诊断证据引用

最终诊断结果的 `evidence` 先按本地指南检索结果的原始顺序保存完整指南证据，再追加
PubMed 证据，并使用 `[1]`、`[2]` 等序号统一连续编号。指南证据保持
`skill name：guideline evidence` 格式；PubMed 证据保持
`PubMed PMID <PMID>（<论文标题>）：<相关摘要证据>` 格式。

`supporting_evidence` 中的患者事实如果使用编号后的指南或 PubMed 证据解释其诊断意义，
以及 `recommended_next_steps` 使用这些证据提出后续建议时，会在条目末尾添加对应的
`[1]` 或 `[1][2]` 引用。患者事实仍必须来自当前病例，外部证据不能替代或冒充患者
已经存在的临床事实。

## 医疗声明

本 demo 仅用于技术演示和辅助分析，不能替代临床医生诊断、治疗建议或线下医疗评估。
