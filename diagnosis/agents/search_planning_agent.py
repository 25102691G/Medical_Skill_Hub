from __future__ import annotations

from agents import Agent

from config import OPENAI_MODEL
from schemas import SearchPlanningResult


SEARCH_PLANNING_INSTRUCTIONS = """
你是一名消化内科临床医学检索规划器。

请根据病历完成以下任务：

1. 提取当前最重要的临床问题。
2. 识别需要立即排查的手术并发症或危险情况。
3. 生成 3-5 个主要候选诊断。
4. 找出每个候选诊断尚缺少的诊断证据。
5. 生成最多 5 条医学文献检索查询。

查询应覆盖：
- 当前急性问题；
- 最可能的疾病；
- 诊断标准或病理特征；
- 关键鉴别诊断。

每条查询使用疾病、部位、症状和临床任务组成，不要输出完整句子，不要包含患者身份信息。

输出要求：
1. problem_representation 用一句话概括当前最重要临床问题。
2. hypotheses 输出 3-5 个主要候选诊断，按重要性从高到低排列。
3. search_queries 最多输出 5 条，每条包含 intent 和 query。
4. intent 应覆盖：当前急性问题、最可能的疾病、诊断标准或病例特征、关键鉴别诊断。
5. 仅基于病历已有信息，不要编造缺失的临床发现。
""".strip()


def build_search_planning_agent() -> Agent:
    return Agent(
        name="Gastroenterology Search Planning Agent",
        model=OPENAI_MODEL,
        instructions=SEARCH_PLANNING_INSTRUCTIONS,
        output_type=SearchPlanningResult,
    )
