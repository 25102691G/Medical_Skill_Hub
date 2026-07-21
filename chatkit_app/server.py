from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from agents import Model
from chatkit.server import ChatKitServer
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ErrorEvent,
    ProgressUpdateEvent,
    ThreadItemDoneEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
    UserMessageTextContent,
)
from openai import RateLimitError

from chatkit_app.store import InMemoryChatKitStore
from chatkit_app.translation import (
    DisplayTranslator,
    get_context_display_language,
)
from main import make_diagnosis
from schemas import DiagnosisResult


logger = logging.getLogger(__name__)

DIAGNOSE_COMMANDS = {
    "开始诊断",
    "重新诊断",
    "诊断",
    "start diagnosis",
    "diagnose",
    "/diagnose",
}
CLEAR_COMMANDS = {
    "清空病例",
    "重置病例",
    "clear case",
    "reset case",
    "/clear",
    "/reset",
}
AGENT_DISPLAY_NAMES = {
    "zh-CN": {
        "Search Planning Agent": "检索规划",
        "Knowledge Searcher Agent": "医学知识检索",
        "Similar Case Retrieval Agent": "相似病例检索",
        "Guideline Searcher Agent": "本地指南检索",
        "Digestive Diagnosis Agent": "消化内科诊断分析",
        "Diagnostic Judgement Agent": "诊断结果评估",
    },
    "en": {
        "Search Planning Agent": "search planning",
        "Knowledge Searcher Agent": "medical knowledge retrieval",
        "Similar Case Retrieval Agent": "similar-case retrieval",
        "Guideline Searcher Agent": "local guideline retrieval",
        "Digestive Diagnosis Agent": "gastroenterology diagnosis analysis",
        "Diagnostic Judgement Agent": "diagnostic result assessment",
    },
}
STAGE_DISPLAY_NAMES = {
    "Search Planning Result": "Search Planning Result",
    "Knowledge Search Result": "Medical Knowledge Search Result",
    "Similar Case Retrieval Rankings": "Similar-Case Retrieval Rankings",
    "Similar Case Retrieval Result": "Similar-Case Retrieval Result",
    "Guideline Search Result": "Local Guideline Search Result",
    "Final Diagnosis Result": "Gastroenterology Diagnosis Result",
    "Diagnostic Judgement Result": "Diagnostic Result Assessment",
}
FIELD_DISPLAY_NAMES = {
    "hypotheses": "Candidate Diagnoses",
    "search_queries": "Literature Search Queries",
    "similar_case_queries": "Similar-Case Retrieval Features",
    "used_skill": "Used Local Guideline Material",
    "skill_names": "Guideline Material Identifiers",
    "guideline_evidence": "Guideline Evidence",
    "pubmed_evidence": "PubMed Evidence",
    "pmid": "PubMed PMID",
    "summary": "Summary",
    "evidence": "Evidence",
    "limitations": "Limitations",
    "topk_diagnoses": "Suspected Diagnoses",
    "rank": "Rank",
    "disease": "Disease",
    "confidence": "Support Strength",
    "supporting_evidence": "Supporting Evidence",
    "recommended_next_steps": "Recommended Next Steps",
    "closer_result": "Diagnosis Set Closer to the Case",
    "reason": "Judgement Reason",
    "query": "Search Query",
    "results": "Search Results",
    "title": "Title",
    "source": "Source",
    "published": "Publication Date",
    "content": "Content",
    "metadata": "Metadata",
}
VALUE_DISPLAY_NAMES = {
    "topk_diagnoses": "Gastroenterology Diagnosis Result",
    "hypotheses": "Search-Planning Candidate Diagnoses",
}
STATIC_TEXT = {
    "zh-CN": {
        "case_cleared": "当前线程中的病例信息已清空。请发送新的病例资料。",
        "no_text": "没有读取到文本内容。请发送病例资料。",
        "case_recorded": (
            "已记录这段病例资料，当前累计 {character_count} 个字符。"
            "你可以继续补充检查结果；资料完整后发送“开始诊断”。"
        ),
        "no_case": "当前还没有病例资料。请先发送患者病史、体征和检查结果。",
        "progress": "第 {round_index} 轮：正在进行{agent_name}…",
        "quota": (
            "模型 API 额度不足。请检查 API Key 所属项目的余额、"
            "Billing 和使用预算，更新后重启后端。"
        ),
        "rate_limit": "模型 API 当前达到速率限制，请稍后重试。",
        "pipeline_error": "诊断流水线运行失败，请检查服务端日志后重试。",
    },
    "en": {
        "case_cleared": "The case information in this thread has been cleared. Please send a new case.",
        "no_text": "No text was received. Please send the case information.",
        "case_recorded": (
            "This section has been recorded; the case now contains {character_count} characters. "
            "You may continue adding examination results. Send “start diagnosis” when complete."
        ),
        "no_case": (
            "No case information has been recorded. Please first send the patient history, "
            "physical findings, and examination results."
        ),
        "progress": "Round {round_index}: running {agent_name}…",
        "quota": (
            "The model API quota is insufficient. Check the balance, billing status, and usage "
            "budget for the API key's project, then restart the backend."
        ),
        "rate_limit": "The model API rate limit has been reached. Please try again later.",
        "pipeline_error": "The diagnosis pipeline failed. Check the server logs and try again.",
    },
}


def _extract_user_text(message: UserMessageItem | None) -> str:
    if message is None:
        return ""
    return "".join(
        part.text
        for part in message.content
        if isinstance(part, UserMessageTextContent)
    ).strip()


def _rate_limit_error_code(error: RateLimitError) -> str | None:
    if not isinstance(error.body, dict):
        return None
    details = error.body.get("error", error.body)
    if not isinstance(details, dict):
        return None
    code = details.get("code") or details.get("type")
    return code if isinstance(code, str) else None


def _static_text(language: str, key: str, **values: object) -> str:
    return STATIC_TEXT[language][key].format(**values)


def _format_diagnosis(result: DiagnosisResult) -> str:
    sections = ["## Diagnostic Analysis Result", "", result.summary]

    for item in result.topk_diagnoses:
        sections.extend(
            [
                "",
                f"### {item.rank}. {item.disease} (support strength: {item.confidence}%)",
                "",
                "**Supporting Evidence**",
                *[f"- {evidence}" for evidence in item.supporting_evidence],
                "",
                "**Recommended Next Steps**",
                *[f"- {step}" for step in item.recommended_next_steps],
            ]
        )

    if result.evidence:
        sections.extend(["", "**Evidence**", *[f"- {evidence}" for evidence in result.evidence]])

    if result.used_skill:
        sections.extend(["", "Local guideline material was used to support the diagnosis."])
    return "\n".join(sections)


def _prepare_stage_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            FIELD_DISPLAY_NAMES.get(str(key), str(key)): _prepare_stage_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_prepare_stage_value(item) for item in value]
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return "None"
    if isinstance(value, str):
        return VALUE_DISPLAY_NAMES.get(value, value)
    return value


def _format_stage_result(title: str, content: str) -> str:
    stage_name, separator, round_index = title.partition(" - Round ")
    display_name = STAGE_DISPLAY_NAMES.get(stage_name, "Stage Output Result")
    heading = f"## Round {round_index}: {display_name}" if separator else f"## {display_name}"

    try:
        parsed_content = json.loads(content)
    except json.JSONDecodeError:
        return f"{heading}\n\n{content}"

    if stage_name == "Similar Case Retrieval Rankings":
        if not isinstance(parsed_content, dict):
            return f"{heading}\n\nNo retrieval ranking details are available."
        ranking_groups = parsed_content.get("rankings")
        if not isinstance(ranking_groups, list) or not ranking_groups:
            return f"{heading}\n\nNo retrieval ranking details are available."

        field_names = {
            "similar_case_queries": "Similar-Case Queries",
        }
        sections = [heading]
        for group in ranking_groups:
            if not isinstance(group, dict):
                continue
            query_field = field_names.get(
                str(group.get("query_field", "")),
                str(group.get("query_field", "Unknown Field")),
            )
            method = str(group.get("method", "Unknown Method"))
            query = str(group.get("query", ""))
            sections.extend(
                [
                    "",
                    f"### {query_field} · {method}",
                    "",
                    f"**Query:** {query or 'Empty'}",
                ]
            )
            if group.get("status") == "skipped":
                sections.append(
                    f"**Status:** Skipped — {group.get('skipped_reason') or 'No reason provided.'}"
                )
                continue

            ranking = group.get("ranking")
            if not isinstance(ranking, list) or not ranking:
                sections.append("**Ranking:** No matching cases.")
                continue
            sections.extend(["", "**Ranking:**"])
            for item in ranking:
                if not isinstance(item, dict):
                    continue
                score = item.get("score")
                score_text = f"{score:.6f}" if isinstance(score, (int, float)) else str(score)
                sections.append(
                    (
                        f"{item.get('rank', '-')}. Hospital admission ID: "
                        f"{item.get('hadm_id', '')}; Discharge disease: "
                        f"{item.get('discharge_disease', '')}; Score: {score_text}"
                    )
                )
        return "\n".join(sections)

    if stage_name == "Similar Case Retrieval Result":
        if not isinstance(parsed_content, dict):
            return f"{heading}\n\nNo displayable similar cases were retrieved."
        hadm_ids = parsed_content.get("hadm_id")
        discharge_diseases = parsed_content.get("discharge_disease")
        if not isinstance(hadm_ids, list) or not isinstance(discharge_diseases, list):
            return f"{heading}\n\nNo displayable similar cases were retrieved."
        case_items = [
            (
                str(hadm_id),
                str(discharge_disease),
            )
            for hadm_id, discharge_disease in zip(hadm_ids, discharge_diseases)
        ]
        if not case_items:
            return f"{heading}\n\nNo similar cases were retrieved."
        formatted_cases = "\n".join(
            (
                f"{index}. Hospital admission ID: {hadm_id}\n"
                f"   Discharge disease: {discharge_disease}"
            )
            for index, (hadm_id, discharge_disease) in enumerate(case_items, start=1)
        )
        return f"{heading}\n\n{formatted_cases}"

    prepared_content = _prepare_stage_value(parsed_content)
    if isinstance(prepared_content, str):
        return f"{heading}\n\n{prepared_content}"
    formatted_content = json.dumps(prepared_content, ensure_ascii=False, indent=2)
    return f"{heading}\n\n```json\n{formatted_content}\n```"


class MedicalDiagnosisChatKitServer(ChatKitServer[dict[str, Any]]):
    store: InMemoryChatKitStore

    def __init__(
        self,
        store: InMemoryChatKitStore,
        translator: DisplayTranslator,
        diagnosis_model: str | Model,
    ) -> None:
        super().__init__(store=store)
        self.store = store
        self.translator = translator
        self.diagnosis_model = diagnosis_model

    def _assistant_event(
        self,
        thread: ThreadMetadata,
        text: str,
        context: dict[str, Any],
        *,
        raw_text: str | None = None,
    ) -> ThreadItemDoneEvent:
        item_id = self.store.generate_item_id("message", thread, context)
        self.store.register_raw_assistant_text(item_id, raw_text or text)
        return ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id=item_id,
                thread_id=thread.id,
                created_at=datetime.now(),
                content=[AssistantMessageContent(text=text)],
            )
        )

    async def respond(
        self,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: dict[str, Any],
    ) -> AsyncIterator[ThreadStreamEvent]:
        user_text = _extract_user_text(input_user_message)
        normalized_command = user_text.lower().rstrip("。.!！")
        display_language = get_context_display_language(context)

        if normalized_command in CLEAR_COMMANDS:
            self.store.clear_case_text(thread.id)
            yield self._assistant_event(
                thread,
                _static_text(display_language, "case_cleared"),
                context,
                raw_text=_static_text("en", "case_cleared"),
            )
            return

        should_diagnose = input_user_message is None or normalized_command in DIAGNOSE_COMMANDS
        if not should_diagnose:
            if not user_text:
                yield self._assistant_event(
                    thread,
                    _static_text(display_language, "no_text"),
                    context,
                    raw_text=_static_text("en", "no_text"),
                )
                return

            case_text = self.store.append_case_section(thread.id, user_text)
            yield self._assistant_event(
                thread,
                _static_text(
                    display_language,
                    "case_recorded",
                    character_count=len(case_text),
                ),
                context,
                raw_text=_static_text(
                    "en",
                    "case_recorded",
                    character_count=len(case_text),
                ),
            )
            return

        case_text = self.store.get_case_text(thread.id)
        if not case_text:
            yield self._assistant_event(
                thread,
                _static_text(display_language, "no_case"),
                context,
                raw_text=_static_text("en", "no_case"),
            )
            return

        progress_queue: asyncio.Queue[tuple[str, str, str | None] | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def report_progress(event_type: str, title: str, content: str | None) -> None:
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                (event_type, title, content),
            )

        def run_diagnosis() -> DiagnosisResult:
            try:
                return make_diagnosis(
                    case_text,
                    model=self.diagnosis_model,
                    debug=False,
                    progress_callback=report_progress,
                )
            finally:
                loop.call_soon_threadsafe(progress_queue.put_nowait, None)

        diagnosis_task = asyncio.create_task(asyncio.to_thread(run_diagnosis))
        try:
            while True:
                progress_event = await progress_queue.get()
                if progress_event is None:
                    break

                event_type, title, content = progress_event
                if event_type == "agent_started":
                    agent_name = AGENT_DISPLAY_NAMES[display_language].get(
                        title,
                        "诊断处理" if display_language == "zh-CN" else "diagnostic processing",
                    )
                    yield ProgressUpdateEvent(
                        icon="analytics",
                        text=_static_text(
                            display_language,
                            "progress",
                            round_index=content or "-",
                            agent_name=agent_name,
                        ),
                    )
                elif event_type == "stage_completed" and content is not None:
                    raw_stage_text = _format_stage_result(title, content)
                    translated_stage_text = await self.translator.translate(
                        raw_stage_text,
                        display_language,
                    )
                    yield self._assistant_event(
                        thread,
                        translated_stage_text,
                        context,
                        raw_text=raw_stage_text,
                    )

            result = await diagnosis_task
        except RateLimitError as exc:
            error_code = _rate_limit_error_code(exc)
            logger.warning(
                "Model API request failed for thread %s: code=%s request_id=%s",
                thread.id,
                error_code or "rate_limit_exceeded",
                exc.request_id,
            )
            if error_code == "insufficient_quota":
                yield ErrorEvent(
                    message=_static_text(display_language, "quota"),
                    allow_retry=False,
                )
            else:
                yield ErrorEvent(
                    message=_static_text(display_language, "rate_limit"),
                    allow_retry=True,
                )
            return
        except Exception:
            logger.exception("Diagnosis pipeline failed for thread %s", thread.id)
            yield ErrorEvent(
                message=_static_text(display_language, "pipeline_error"),
                allow_retry=True,
            )
            return

        raw_diagnosis_text = _format_diagnosis(result)
        translated_diagnosis_text = await self.translator.translate(
            raw_diagnosis_text,
            display_language,
        )
        yield self._assistant_event(
            thread,
            translated_diagnosis_text,
            context,
            raw_text=raw_diagnosis_text,
        )
