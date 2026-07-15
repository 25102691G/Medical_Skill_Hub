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

开始诊断后，聊天界面会实时显示当前阶段和诊断轮次。每个 Agent 完成后，会立即
追加带有中文标题和中文字段名称的完整阶段输出；如果进入第二轮，会继续按轮次
依次展示。用于 PubMed 等数据源的检索词、论文标题、URL 和引用原文保留原始语言，
其余解释性内容使用中文。全部阶段结束后，界面会显示格式化的中文最终诊断结果。

## 医疗声明

本 demo 仅用于技术演示和辅助分析，不能替代临床医生诊断、治疗建议或线下医疗评估。
