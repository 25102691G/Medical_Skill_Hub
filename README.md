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

## 相似病例检索

相似病例库默认使用 `database/mimic_similar.csv`。病例预处理将当前患者信息提取为
`present_illness_history`、`past_medical_history`、`physical_exam`、`family_history` 和
`pertinent_results` 五个字段，每个查询字段只与病例库中的同名字段匹配。

五个字段分别执行 BM25 和 BGE-M3 检索，再通过加权 RRF 合并病例排名。字段权重依次为：

* `pertinent_results`、`present_illness_history`：3；
* `physical_exam`、`past_medical_history`：2；
* `family_history`：1。

混合 RRF 中 Dense 和 BM25 的检索分支权重分别为 0.7 和 0.3，Dense 作为主要语义检索
通道，BM25 用于补充关键词和医学术语召回。

当前患者缺少某字段时跳过该字段，并根据实际非空字段归一化权重。加权 RRF 后按 ICD 编码
保留不同疾病，并直接输出 RRF Top 5。每个RRF候选仅保留实际进入对应字段BM25或Dense
候选集的最佳匹配切片，不使用任意切片补位。
结构化病例信息不包含阴性症状、阴性家族史、正常生命体征以及阴性或正常检查结果。
Dense 编码的最大输入长度为 1024 token，病例库文本仍按 510 个正文 token 切片。
最终结果保留 BM25、Embedding 和 RRF 三个相似病例阶段，供批量评估使用。向量缓存默认写入
`database/mimic_similar_embeddings.pt`，BM25 缓存根据病例库文件名自动生成。

使用已有批量结果中的固定 `positive_features_result` 单独测试相似病例模块，不会调用 LLM：

```bash
bash run_similar_case.sh
```

脚本默认读取 `output/batch/sample5_test_nobhc_75_20260813_111639_669877.jsonl`，结果写入
`output/similar_case/`，并汇总 BM25、Embedding 和 RRF 的 ICD 3 位、4 位及精确编码
Recall@1/3/5。可通过命令行调整融合权重、字段权重、候选数量和 Dense 最大输入长度：

```bash
bash run_similar_case.sh \
  --dense-weight 0.7 \
  --bm25-weight 0.3 \
  --dense-max-length 1024
```

传入其他批量结果时使用：

```bash
bash run_similar_case.sh --input output/batch/<batch-result>.jsonl
```

批量运行多组检索参数：

```bash
bash run_similar_case.sh --experiments similar_case_experiments.json
```

`similar_case_experiments.json` 中每个实验可以覆盖 Dense/BM25 融合权重、五个字段权重、
候选数量和 Dense 最大输入长度；未填写的参数继承命令行默认值。多组结果写入同一个时间戳
目录，每组生成独立 JSONL，同时生成 `summary.json` 和终端 RRF 横向对比表。

字段权重使用 `section_weights` 设置，例如：

```json
{
  "name": "result_focused",
  "section_weights": {
    "present_illness_history": 2,
    "pertinent_results": 4
  },
  "dense_candidate_k": 100,
  "rrf_candidate_k": 30
}
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
全文或脚本。匹配结果会分别保存直接命中的 Skill，以及一层正向扩展的来源 Skill 和
鉴别疾病。直接匹配使用一次模型调用；随后将所有去重后的直接命中 Skill 合并为
一次鉴别疾病映射，目标 catalog 只提供主要疾病及适用范围。代码校验扩展来源与目标
Skill 名并对目标 Skill 全局去重。每个病例最多并发执行 3 个 Skill，每个 Skill 执行器
最多使用 12 次模型调用；单个 Skill 失败不会丢弃其他 Skill 已成功返回的结果，turn 耗尽原因
会保留模型调用数、工具调用序列、命令摘要及相同参数的精确重复调用次数。

## 医疗声明

本 demo 仅用于技术演示和辅助分析，不能替代临床医生诊断、治疗建议或线下医疗评估。
