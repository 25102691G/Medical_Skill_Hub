from __future__ import annotations

import json
from collections.abc import Sequence

from agents import Agent

from config import OPENAI_MODEL
from schemas import SearchQueryItem, SimilarCaseRetrievalResult


SIMILAR_CASE_RETRIEVAL_INSTRUCTIONS = """
你是一名消化内科相似病例检索 Agent。

输入是 Gastroenterology Search Planning Agent 生成的 search_queries。

任务：
1. 逐条读取 search_queries 中的 intent 和 query。
2. 判断每条 query 最可能对应的相似病例诊断 diagnosis。
3. 输出每条 query 对应的 diagnosis、匹配理由和置信度。
4. 如果没有真实相似病例库或检索工具结果，不要声称已经检索到真实病例编号、真实患者或真实病例摘要。
5. 在未接入真实病例库时，matched_case_summary 必须说明该结果仅基于 query 语义归纳。

输出要求：
1. diagnosis 使用规范疾病名称，优先使用消化系统疾病名称。
2. supporting_reason 只基于 query 中出现的疾病、部位、症状、诊断标准或鉴别诊断线索。
3. limitations 必须说明当前是否缺少真实病例库证据。
4. 不要输出治疗建议，不要替代临床诊断。
""".strip()


def _search_queries_to_json(search_queries: Sequence[SearchQueryItem]) -> str:
    return json.dumps(
        [query.model_dump() for query in search_queries],
        ensure_ascii=False,
        indent=2,
    )


def build_similar_case_retrieval_prompt(search_queries: Sequence[SearchQueryItem]) -> str:
    return (
        "Search queries from Gastroenterology Search Planning Agent:\n"
        f"{_search_queries_to_json(search_queries)}\n\n"
        "Please infer the corresponding similar-case diagnosis for each query."
    )


def build_similar_case_retrieval_agent() -> Agent:
    return Agent(
        name="Similar Case Retrieval Agent",
        model=OPENAI_MODEL,
        instructions=SIMILAR_CASE_RETRIEVAL_INSTRUCTIONS,
        output_type=SimilarCaseRetrievalResult,
    )
