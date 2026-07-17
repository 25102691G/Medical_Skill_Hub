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

检索规划阶段生成 `similar_case_queries`，其中的 `clinical_manifestations` 和
`examination_results` 使用英文短语，分别与 `database/mimic_iv_case.csv` 中的同名列
进行匹配。`clinical_manifestations` 提取病例中明确记录的阳性临床特征，包括阳性
症状、异常生命体征和体格检查阳性体征；`examination_results` 仅提取明确记录的阳性
辅助检查结果，包括实验室、影像、内镜、病理和微生物检查结果。两个字段互斥，辅助
检查结果只归入 `examination_results`，不能在 `clinical_manifestations` 中重复。
BM25 仅对英文和数字分词；每个字段分别执行 BM25 和 Dense Retriever 检索，
再通过 RRF 融合各路排名。最终按相关性顺序输出最多 10 条病例对应的 `hadm_id`、
`long_title`（输出字段为 `discharge_disease`）和 `discharge_text`。前端的相似病例
检索结果只展示住院号和出院疾病，完整出院记录仅作为外部参考证据传入最终诊断和诊断
判断阶段，不会被视为当前患者已经存在的临床事实。
启用 `--debug` 时，终端的标准错误流会分别输出两个检索字段的 BM25 和 Dense Top-K
排名，包括查询文本、住院号、出院疾病和原始分数；未执行的检索分支会输出跳过原因。
ChatKit 前端也会在每轮相似病例检索完成后展示同一组四路排名及跳过原因，该展示通过
阶段进度事件传递，不要求后端启用 `debug`。

Dense Retriever 默认使用 `BAAI/bge-m3`，首次运行时由 Transformers 加载模型，并将
病例库向量缓存到 `database/mimic_iv_case_embeddings.pt`。模型默认在 CPU 上运行，
可通过 `SIMILAR_CASE_EMBEDDING_DEVICE` 设置为 `cpu`、`cuda` 或 `auto`；其中 `auto`
会在 CUDA 可用时使用 GPU。其他配置可通过
`MIMIC_IV_CASE_PATH`、`SIMILAR_CASE_TOP_K`、`SIMILAR_CASE_EMBEDDING_MODEL`、
`SIMILAR_CASE_EMBEDDING_CACHE_PATH` 和 `SIMILAR_CASE_EMBEDDING_BATCH_SIZE` 调整配置。

## 最终诊断证据引用

最终诊断结果的 `evidence` 按本地指南检索结果的原始顺序保存完整证据，并使用 `[1]`、
`[2]` 等序号标记。`supporting_evidence` 中的患者事实如果使用指南解释其诊断意义，
以及 `recommended_next_steps` 使用指南提出后续建议时，会在条目末尾添加对应的
`[1]` 或 `[1][2]` 引用。患者事实仍必须来自当前病例，指南证据不能替代或冒充患者
已经存在的临床事实。

## 医疗声明

本 demo 仅用于技术演示和辅助分析，不能替代临床医生诊断、治疗建议或线下医疗评估。
