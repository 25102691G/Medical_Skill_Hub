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
`--limit` 控制本次处理的病例数量：

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
中每个条目都引用对应原文块；检索时完整读取索引进行语义匹配，再按块ID读取全文核实。
编译器还会把指南的主要疾病及适用范围和有实质性诊断内容支持的明确鉴别疾病写入
`SKILL.md` 描述。病例候选直接对应主要疾病，或出现在明确鉴别疾病中时，都会选择该
Skill。对于主要疾病被病例候选直接命中的 Skill，系统还会选择其明确鉴别疾病各自对应
的主要疾病 Skill；该扩展只执行一层，不从反向匹配或扩展得到的 Skill 继续递归。所有
选中的 Skill 均执行相同的完整检索流程，且不限制匹配数量。

## 医疗声明

本 demo 仅用于技术演示和辅助分析，不能替代临床医生诊断、治疗建议或线下医疗评估。
