# 消化内科医疗诊断 Medical Skill Hub

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

## 批量运行

`batch_main.py` 读取通过 `--input` 指定的 CSV，使用
`discharge_text_before_disposition` 作为 `case_text` 运行完整诊断流水线。使用
`--limit` 控制本次处理的病例数量。单个病例的完整诊断流水线失败时，最多执行
3 次（首次执行加 2 次重试），终端会显示每次失败的阶段、错误原因和重试轮次。
阳性特征和最终诊断的结构化输出校验失败时，当前阶段仍会先根据具体错误纠正一次。
成功结果写入 JSONL；最终失败的病例只在终端报告，不另外生成错误文件：

```bash
.venv/bin/python batch_main.py \
  --input database/mimic_test_case.csv \
  --limit 10
```

## 诊断结果评估

`evaluate.py` 对多轮批量诊断结果中的主诊断 ICD code 进行直接匹配。

可以通过 `run_evaluate.sh` 传入批处理结果：

```bash
bash run_evaluate.sh output/batch/<输入文件名>_<limit>_<时间戳>.jsonl
```

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
bash run_chatkit.sh
```

在第二个终端启动前端：

```bash
cd chatkit_frontend
npm run dev
```

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

## 指南 Skill 编译与检索

批量编译前，将所有 PDF 直接放入 `guidelines/`：

```text
guidelines/
├── 中国克罗恩病诊治指南（2023年·广州）.pdf
├── 中国溃疡性结肠炎诊治指南（2023年·西安）.pdf
└── ...
```

运行：

```bash
bash run_compile_skill.sh
```

默认同时编译 2 份指南，但 MinerU 的 CUDA 解析固定为单任务，避免争抢显存；模型请求可与
下一份 PDF 的解析重叠执行。需要调整并发数时显式传入输入目录和 `--workers`：

```bash
bash run_compile_skill.sh --pdfs ./guidelines --workers 3
```

编译器会将 MinerU 生成的 Markdown 直接保存为 `references/guideline-full-text.md`，并为
每个非空原文段落生成稳定的全文行号块。LLM 生成的 `references/recommendations-index.md`
中每个条目都引用对应原文块。每个 Skill 同时包含 `scripts/search_guideline.py`；执行器按照
`SKILL.md` 先用脚本读取完整章节目录，由 LLM 根据病例语义选择相关章节，再用脚本批量
返回这些章节的完整索引条目。LLM 从条目中选择直接支持判断的原文位置，最后用同一脚本
一次批量读取对应全文块。脚本只负责确定性的文件定位与读取，不使用关键词打分或 Top-K
决定医学相关性，同时避免通过多轮 Shell 命令把大型索引分段搬入模型上下文。
编译器还会把指南的主要疾病及适用范围和有实质性诊断内容支持的明确鉴别疾病写入
`SKILL.md` 描述。系统先选择主要疾病与病例候选直接对应的 Skill，再从这些直接命中的
Skill 出发，将其明确鉴别疾病正向匹配到其他 Skill 的主要疾病；该扩展只执行一层，不会
因为病例候选出现在某个 Skill 的鉴别疾病中而反向选择，也不会从扩展得到的 Skill 继续
递归。亚型、分期、遗传性、转移部位、并发症、妊娠等限定条件必须与候选明确匹配。

Skill 匹配完成后，每个选中的 Skill 都由独立的原生 Skill 执行器处理。执行器先调用
`load_skill`，完整读取该 Skill 的 `SKILL.md`，再严格按照其中定义的工作流程和资源说明
检索证据；外层调度仅负责逐个隔离运行和合并结果，不规定 Skill 内部必须使用的索引、
全文或脚本。匹配结果会分别保存直接命中的 Skill、匹配候选，以及一层正向扩展的来源
Skill 和鉴别疾病。匹配阶段固定为一次模型调用，每个 Skill 执行器最多使用 12 次模型调用；
单个 Skill 失败不会丢弃其他 Skill 已成功返回的结果，turn 耗尽原因会保留模型调用数、
工具调用序列、命令摘要及相同参数的精确重复调用次数。

## 医疗声明

本 demo 仅用于技术演示和辅助分析，不能替代临床医生诊断、治疗建议或线下医疗评估。
