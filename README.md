# 消化内科医疗诊断 Agent Demo

## 环境准备

使用 Python 3.10 创建项目内虚拟环境（scispaCy 暂不支持 Python 3.13）：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

下载模型：
```bash
HF_ENDPOINT=https://hf-mirror.com \
.venv/bin/huggingface-cli download BAAI/bge-m3 \
  --local-dir models/bge-m3 \
  --max-workers 4

HF_ENDPOINT=https://hf-mirror.com \
.venv/bin/huggingface-cli download ncbi/MedCPT-Cross-Encoder \
  --local-dir models/MedCPT-Cross-Encoder \
  --max-workers 4
```

## 运行方式

`run_batch_main.sh` 和 `run_chatkit.sh` 共用项目根目录 `.env` 中的
`DIAGNOSIS_PROVIDER`，可设置为 `openai` 或 `deepseek`。
两个入口也共用对应的 API Key 和模型名称：OpenAI 使用 `OPENAI_API_KEY` 和
`OPENAI_MODEL`，DeepSeek 使用 `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL` 和
`DEEPSEEK_BASE_URL`。例如：

```dotenv
DIAGNOSIS_PROVIDER=deepseek
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_THINKING=true
```

`DEEPSEEK_THINKING` 控制诊断流水线中的 DeepSeek 请求是否启用深度思考，
设置为 `true` 时启用，设置为 `false` 时关闭，默认值为 `true`。

项目中的 DeepSeek Chat Completions 调用统一使用 JSON Output：请求传入
`response_format={"type":"json_object"}` 和与用途相匹配的 `max_tokens`，提示词包含
JSON 输出格式示例，响应按对应结构解析。Markdown、普通文本及二分类等原始业务输出会
先封装在 JSON 字段中，解析后再恢复为原有返回类型。指南检索完成本地工具调用后，在
最终响应中输出符合 `GuidelineSearchResult` Schema 的 JSON 对象。

## 批量运行

`batch_main.py` 读取通过 `--input` 指定的 CSV，使用
`discharge_text_before_disposition` 作为 `case_text` 运行完整诊断流水线。使用
`--limit` 控制本次处理的病例数量，使用 `--workers` 控制同时诊断的病例数：

```bash
.venv/bin/python batch_main.py \
  --input database/mimic_test_case.csv \
  --limit 10 \
  --workers 4
```

`--workers` 默认为 `1`。批处理使用常驻 worker 队列保持指定并发数，成功结果按病例
完成顺序写入 JSONL。

单个病例先由预处理模块并发执行两次相互隔离的 LLM 调用：一次仅根据原始 `case_text`
生成 `llm_hypotheses`，另一次仍仅根据原始 `case_text` 提取 `positive_features`。随后使用
`positive_features` 执行一次相似病例检索。Search Planning 接收原始病例、两项预处理结果
以及相似病例的疾病名称和 ICD code，按 ICD code 将纯 LLM 候选与相似病例候选去重合并，
并生成 5 至 10 条 PubMed 检索词。全部合并候选疾病都必须由至少一条检索词覆盖；当模型
遗漏候选疾病时，程序会补充包含该疾病英文名称的检索词。Search Planning 完成后，PubMed
检索与指南检索并行执行。

`run_batch_main.sh` 会读取 `.env` 中的 `DIAGNOSIS_PROVIDER`，支持 `openai` 和
`deepseek`。对应的 API Key、模型名称和 DeepSeek 地址使用项目根目录 `.env` 中的配置。
例如切换为 DeepSeek：

```dotenv
DIAGNOSIS_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_THINKING=true
```

结果逐条写入
`output/batch/<输入文件名>_<limit>_<时间戳>.jsonl`；未指定 `--limit` 时使用 `all`。
例如输入 `mimic_test_case_hernia.csv` 且 `--limit 5` 时，输出文件名类似
`mimic_test_case_hernia_5_20260729_145433_369954.jsonl`。每行对应一个成功完成的病例，
包含 `subject_id`、`hadm_id`、`icd_code`、`long_title`、`llm_hypotheses_result`、
`positive_features_result` 和 `multi_round_diagnosis`。`multi_round_diagnosis.is_multi_round`
表示是否进入了第二轮，
`multi_round_diagnosis.rounds` 按轮次保存每一轮的 `round`、
`search_planning_result`、`similar_case_retrieval_result`、按 skill 分组的
`guideline_search_result` 和结构化 `diagnosis_result`。未使用 skill 时，
`guideline_search_result.unused_reason` 记录具体原因；该字段仅用于观察 skill 调用过程，
`guideline_search_result.skill_names` 显式列出本轮实际使用的全部 skill 名称，不会传给
最终诊断或多轮诊断判断。如果第二轮触发纠正诊断，该轮保存纠正后的
`diagnosis_result`。预处理结果和相似病例检索在每个病例中只执行一次，第二轮复用这些
结果。`search_planning_result.hypotheses` 先保留最多五个纯 LLM 候选，再按检索排名追加
相似病例候选，并按规范化 ICD code 去重；父级编码与更具体的子级编码同时出现时只保留
子级编码，最多保留十个候选。最终诊断只对该合并候选集
进行重排，指南和 PubMed 证据仅用于解释患者信息和调整排序，不会生成候选集之外的新
ICD。相似病例的疾病名称、ICD code 和匹配文本仍会传入最终诊断。未进入最终前五的合并
候选记录在
`diagnosis_result.excluded_planning_candidates`，并包含支持排除的患者级反证。

每项诊断均使用 `icd_code` 保存移除小数点后的完整三至七字符 ICD-10-CM 编码，并使用
`category_name` 保存候选集中与该编码对应的英文名称。最终结果固定包含五个按 1 至 5 排序
且 ICD 编码互不重复的候选；编码及名称必须原样来自候选并集。单个病例失败时，错误
会输出到终端，脚本继续处理下一条病例。单个 PubMed 查询在完成配置的重试后仍遇到网络错误时，
该查询按空结果处理，不会导致整个病例失败。

## 诊断结果评估

`evaluate.py` 对多轮批量诊断结果中的主诊断 ICD code 进行直接匹配。`batch_main.py` 将输入
CSV 中代表本次住院主诊断的 `icd_code` 写入每条结果，评估脚本以该字段为金标准，并遍历
`multi_round_diagnosis.rounds`。每一轮分别提取以下六组前五项 ICD code：

- `multi_round_diagnosis.rounds[].search_planning_result.hypotheses[].icd_code`
- `multi_round_diagnosis.rounds[].similar_case_retrieval_result.bm25[].icd_code`
- `multi_round_diagnosis.rounds[].similar_case_retrieval_result.embedding[].icd_code`
- `multi_round_diagnosis.rounds[].similar_case_retrieval_result.rrf[].icd_code`
- `multi_round_diagnosis.rounds[].similar_case_retrieval_result.rerank[].icd_code`
- `multi_round_diagnosis.rounds[].diagnosis_result.topk_diagnoses[].icd_code`

金标准和预测编码均先去除首尾空白、转为大写并移除小数点。`disease` 指标取前三个
字符进行匹配，例如 `K50`、`K50.1` 和 `K501` 均按 `K50` 比较；`subcategory`
指标取前四个字符进行匹配，例如 `K50.1` 和 `K501` 均按 `K501` 比较。评估过程不调用
LLM。

可以通过 `run_evaluate.sh` 传入批处理结果：

```bash
bash run_evaluate.sh output/batch/<输入文件名>_<limit>_<时间戳>.jsonl
```

也可以直接运行 Python 并指定输入 JSONL：

```bash
.venv/bin/python evaluate.py \
  --input output/batch/<输入文件名>_<limit>_<时间戳>.jsonl
```

评估结果固定写入
`output/evaluate/<输入文件名>_evaluation.jsonl`。每条评估结果会实时写入输出文件。
每条病例结果中的 `round_evaluations` 保存各轮六组诊断的预测 ICD code，以及
`disease` 和 `subcategory` 排名，分别匹配编码前三位和前四位。程序结束时会在输出文件末尾写入
`total`、`rounds` 和
`final_result`：`rounds` 统计实际进入各轮病例的 Recall@1、Recall@3 和 Recall@5，
`final_result` 使用每个病例最后一轮的评估结果汇总两个指标的相同 Recall。
汇总记录还会写入 `skill_usage`，其使用情况取自每个病例最后一轮的
`diagnosis_result`。没有匹配编码时，该病例在对应诊断组的三个 Recall 中都记为未命中。

## ChatKit 聊天界面

项目提供基于 ChatKit 的自托管聊天界面。FastAPI 适配层位于 `chatkit_app/`，React 前端位于 `chatkit_frontend/`。

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

诊断供应商和模型使用项目根目录的 `.env` 配置。例如：

```dotenv
DIAGNOSIS_PROVIDER=openai
OPENAI_MODEL=gpt-5.5
```

修改 `.env` 后需要重新启动 ChatKit 后端。所选供应商用于搜索规划、知识检索、指南检索、
最终诊断和诊断结果判断等完整诊断流程。
OpenAI 使用 Agents SDK 原生结构化输出；DeepSeek 使用 API JSON Output，并在本地按相同的
Pydantic Schema 解析，因此两种供应商保持相同的阶段输出结构。
指南检索阶段中，OpenAI 使用 Sandbox Skills 读取本地指南，DeepSeek 使用标准 function
tools 搜索和读取同一套 `skills/` 资源；两条路径生成相同的 `GuidelineSearchResult`。

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
远程使用时，需要同时转发前端 `43179` 端口和后端 `8005` 端口。

前端右上角可选择简体中文或英文作为显示语言。选择结果会同时控制 ChatKit 自带界面、
页面静态文字以及后端消息的展示翻译。前端通过 `X-Display-Language` 请求头传递目标
语言；每个 Agent 完成后，ChatKit 服务端会翻译该阶段的字段标签和字符串内容，再立即
追加到聊天界面。如果切换显示语言，当前线程会按新语言重新加载已有助手消息。

展示翻译不会修改诊断流水线的原始结构化结果。URL、数值、计量单位、医学
编码、枚举值、住院号和 `skill_names` 等机器标识保持不变，其余可见内容按目标语言
翻译。长文本会按完整行分段翻译；翻译失败时会显示提示并附上未翻译原文，不会中断
诊断流水线。翻译固定使用 DeepSeek，不随 `DIAGNOSIS_PROVIDER` 切换，并通过 `.env`
单独设置模型：

```dotenv
CHATKIT_TRANSLATION_MODEL=deepseek-v4-pro
```

翻译使用 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_BASE_URL`。因此即使诊断切换为 OpenAI，
ChatKit 展示翻译仍然使用 DeepSeek。

当前实时粒度为阶段级：`main.py` 产生 `stage_completed` 事件后翻译并展示完整阶段结果，
不进行逐 token 翻译。

## PubMed 检索配置

医学知识检索通过 NCBI E-utilities 查询 PubMed。建议在项目根目录的 `.env` 中配置：

每轮知识检索使用全部 5 至 10 条文献查询并发检索，每条最多返回 3 篇文献。检索结果只保留
PMID、标题、独立摘要 section 和 URL；非结构化 `AbstractText` 作为唯一 section 保留，
结构化 `AbstractText` 按原顺序保留各 section 的未经改写正文，不拼接 section，也不保留
Label 或 NlmCategory。Python 会删除 PMID、标题或摘要 section 为空的整条结果，但不删除
重复 PMID。之后由诊断供应商对应的模型分别选择与检索词相关的摘要 section；重复出现的
PMID 会被重点考虑。最终结果由 Python 根据模型选择的 PMID 和 section index 映射回未经
模型改写的原始 PubMed 摘要 section。

```dotenv
NCBI_API_KEY=your_ncbi_api_key
NCBI_EMAIL=your_email@example.com
NCBI_TOOL=medical_skill_hub
```

程序会统一限制 NCBI 请求频率：未配置 API Key 时默认不超过每秒 3 次，配置后默认
不超过每秒 10 次；`ESearch`、批量 `EFetch` 和临时网络错误均使用指数退避重试。
如需调整，可设置 `NCBI_REQUESTS_PER_SECOND`、`NCBI_MAX_RETRIES`、
`NCBI_RETRY_BASE_SECONDS` 和 `NCBI_TIMEOUT_SECONDS`。

## 指南 Skill 编译与检索

批量编译前，需要将每份 PDF 放入人工确认的疾病类别目录：

```text
guidelines/
├── 非感染性小肠炎和结肠炎/
│   ├── 中国克罗恩病诊治指南（2023年·广州）.pdf
│   └── 中国溃疡性结肠炎诊治指南（2023年·西安）.pdf
├── 肠的其他疾病/
│   └── ...
└── 肝疾病/
    └── ...
```

运行：

```bash
./run_compile_skill.sh
```

编译脚本递归读取分类目录。直接放在 `guidelines/` 根目录下的 PDF 不会编译，而是提示
先完成分类。MinerU 输出按相同类别保存：

```text
mineru/<类别>/<PDF 文件名>/auto/...
```

生成的 skill 仍保持 Codex 原生的扁平目录结构：

```text
skills/<PDF 文件名>/
├── SKILL.md
├── agents/openai.yaml
├── references/
└── scripts/
```

`SKILL.md` 的 description 由编译器写入人工确认的类别，再追加指南全文生成的具体疾病
名称、英文名、常用缩写、适用范围和触发边界。使用单个 PDF 或已有 MinerU Markdown
编译时，需要通过 `--category` 明确指定类别。

指南检索使用 `search_planning_result.hypotheses` 选择所有与候选疾病直接对应的 skills，
不设置固定 skill 数量，也不会仅因症状、检查结果或宽泛的消化内科词汇重合而选择其他
疾病类别的 skill。skill 所要求的早期、遗传性、转移部位、并发症、操作或其他限定条件
必须明确出现在候选诊断集合中；只有一般疾病候选时不会触发范围更窄的 skill，也不会从阳性
特征中推断这些限定条件。选择完成后，仅使用
`positive_features_result.positive_features` 在这些 skills 内定位诊断标准、
鉴别诊断、确认或排除检查及下一步建议；推荐索引只用于定位，证据和指南诊断结论均以
`guideline-full-text.md` 核实后的内容为依据。每个 skill 的 `guideline_evidence` 和
`guideline_diagnosis` 保存在同一个 `skill_results` 项目中。`search_queries` 只用于
PubMed 检索。最终诊断使用 Search Planning 的合并候选集，并结合患者信息、PubMed 证据、
相似病例匹配文本和完整指南结果进行排序。

DeepSeek function tools 在每次指南检索中只允许列举一次 skill；每个选中 skill 只允许
读取一次 `SKILL.md`、搜索一次推荐索引、额外搜索一次指南全文，并最多读取两个指南全文
区间。重复或超额调用会被拒绝并要求立即提交已有结果。单次关键词搜索最多返回 120 行，
避免宽泛阳性特征使工具上下文无限增长。达到 agent turn 上限时，该阶段明确记录为检索
未完成，并在失败原因中保留已经访问过的 skill 名称。

DeepSeek 指南结果首次解析失败时固定重试一次。非空但不符合 Schema 的结果只进行一次
无工具 JSON 修复，保留已有 skill、指南证据并补齐缺失字段；空响应使用新的工具调用状态
完整重跑一次指南检索。agent turn 超限不会重试，第二次仍无法解析时记录首次与重试错误。

## 相似病例检索

预处理模块的独立特征提取调用生成 `positive_features` 英文短语列表，其中同时包含病例中明确记录的
阳性临床表现和阳性辅助检查结果。临床表现包括阳性症状、异常生命体征和体格检查阳性
体征；辅助检查结果包括实验室、影像、内镜、病理和微生物检查结果。相似病例库使用
`database/mimic_similar_case.csv`，并使用其中结构化出院记录 section 的非空内容，不再
检索完整 `discharge_text`。每个 section 先按 BGE tokenizer 切分为每块 510 个正文
token，使加入模型特殊 token 后的 Dense 输入长度约为 512 token。BM25 和 Dense
Retriever 都以这些 chunks 为检索语料。BM25 仅对英文和数字分词，并排除原始分数小于
等于 0 的 chunk；两路检索各自先检索 Top-N chunks，再按 `hadm_id` 聚合为病例分数。
病例只有一个命中 chunk 时使用其最高分；至少有两个命中 chunk 时使用
`best_score + 0.2 * second_best_score`，每个病例最多使用两个 chunks。两路病例排名
随后执行病例级 RRF，某病例未进入其中一路的候选排名时，该分支不为其计算 RRF 分数。
RRF 生成 Top-20 候选病例，再使用 `ncbi/MedCPT-Cross-Encoder` 对全部候选病例重新
排序。每个候选病例的 reranker 文档仅由检索命中的 Top-2 chunks 组成：先按 BM25 和
Dense 的 chunk 排名执行 RRF，再去重选出前两个 chunk；reranker 不读取或输入完整
`discharge_text`。reranker 完成后按 `discharge_disease` 聚合，相同诊断标签保留分数
最高的代表病例，最终输出前五个不同诊断标签。批量诊断结果中的
`similar_case_retrieval_result` 按 `bm25`、`embedding`、`rrf` 和 `rerank` 保存四个阶段
各自的 Top 5 疾病及对应 ICD code；前三个阶段不保存 section，只有 `rerank` 中的每个结果
额外保存实际用于重排的 `sections`。主诊断流程只读取 `rerank`，新增的前三组结果仅用于
观察、调试和评估。`sections` 是各标签代表病例实际参与 reranker 的 Top-2 chunks，包含原
section 名称和 chunk 内容，并作为外部参考证据传入最终诊断和诊断判断阶段，不会被视为
当前患者已经存在的临床事实。reranker
模型加载或推理失败时会记录错误日志，回退到 RRF 排名后执行相同的标签级聚合。
批量运行时，相似病例检索在进程内串行使用共享 tokenizer 和模型，其他诊断阶段仍按
`--workers` 设置并发执行。
启用 `--debug` 时，终端的标准错误流会输出 BM25、Dense、RRF 和 Reranker 排名。
BM25/Dense 明细包括查询文本、住院号、出院疾病、病例聚合分数及命中的 Top-2
chunks；RRF 明细额外包括最终 RRF 分数、两路候选排名及两路各自命中的 Top-2
chunks，未进入某路候选排名时该路排名为 `null`；Reranker 明细包括标签级代表病例的
相关性分数和实际输入模型的 chunks。未执行的检索分支会输出跳过原因。
ChatKit 前端也会在相似病例检索完成后展示排名及跳过原因，该展示通过
阶段进度事件传递，不要求后端启用 `debug`。

如需单独测试“阳性特征预处理 → 相似病例检索”模块，可在 `.env` 中通过 `INPUT` 指定输入
CSV，然后运行：

```bash
./run_similar_case_main.sh
```

脚本从 `.env` 读取 `INPUT`，并通过脚本中的 `LIMIT` 和 `WORKERS` 分别控制尝试处理的
CSV 数据条数和并行病例数。直接运行 Python 入口时，`--limit` 和 `--workers` 均接受
大于 0 的整数，其中 `--workers` 默认为 `1`。JSONL 结果仍按输入 CSV 顺序写入；并行
运行时，终端中的 BM25/Dense 排名调试信息可能交错显示。

输入 CSV 需要包含 `subject_id`、`hadm_id`、`long_title` 和
`discharge_text_before_disposition`。程序先调用预处理模块生成
`positive_features`，再执行 BM25、Dense Retriever、RRF 和 Reranker，结果写入
`output/similar_case/similar_case_results_<timestamp>.jsonl`。每条成功记录包含原病例标识、
`positive_features_result.positive_features`、BM25/Dense/RRF/Reranker 排名明细
`similar_case_retrieval_rankings`，以及 Reranker 排序后的 `discharge_disease`、
`icd_code` 和 `Sections`。
独立模块的输出不保存 `llm_hypotheses`、`search_queries`，也不保存相似病例的
完整 `discharge_text`。运行时终端仍会输出四路排名明细及跳过原因。

可运行以下脚本，使用与 `evaluate.py` 相同的模型判断提示词，分别评估 BM25、Dense 和
RRF 融合结果相对于 `long_title` 金标准诊断的 Recall@1、Recall@3 和 Recall@5：

```bash
./run_evaluate_similar_case.sh
```

评估明细和三组汇总指标写入 `output/evaluate/`。输入文件、评估模型和并发数分别由
`run_evaluate_similar_case.sh` 中的 `INPUT`、`MODEL` 和 `WORKERS` 指定。

BM25 首次运行时对所有非空 section chunks 分词并将索引缓存到
`database/mimic_similar_case_bm25.pkl`；病例库和 chunk 缓存 schema 未变化时，后续
运行直接加载该索引。Dense Retriever 默认使用 `BAAI/bge-m3`，首次运行时由
Transformers 加载模型，并将 chunk 向量缓存到
`database/mimic_similar_case_embeddings.pt`。模型和向量相似度计算默认在 GPU 上运行，
可通过 `SIMILAR_CASE_EMBEDDING_DEVICE` 设置为 `cpu`、`cuda` 或 `auto`；其中 `auto`
会在 CUDA 可用时使用 GPU。其他配置可通过
`MIMIC_IV_CASE_PATH`、`SIMILAR_CASE_TOP_K`、`SIMILAR_CASE_BM25_CANDIDATE_K`、
`SIMILAR_CASE_DENSE_CANDIDATE_K`、`SIMILAR_CASE_EMBEDDING_MODEL`、
`SIMILAR_CASE_EMBEDDING_CACHE_PATH`、`SIMILAR_CASE_EMBEDDING_BATCH_SIZE`、
`SIMILAR_CASE_RRF_CANDIDATE_K`、`SIMILAR_CASE_RERANKER_MODEL`、
`SIMILAR_CASE_RERANKER_BATCH_SIZE` 和 `SIMILAR_CASE_RERANKER_DEVICE` 调整配置。
BM25/Dense 候选数默认均为 `50`，RRF 候选病例数默认为 `20`；reranker device
支持 `cpu`、`cuda` 和 `auto`。离线运行时，可将 `SIMILAR_CASE_RERANKER_MODEL`
设置为本地 MedCPT 模型目录。

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
