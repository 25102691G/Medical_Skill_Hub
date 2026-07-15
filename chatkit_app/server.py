from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

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
from main import make_diagnosis
from schemas import DiagnosisResult


logger = logging.getLogger(__name__)

DIAGNOSE_COMMANDS = {"开始诊断", "重新诊断", "诊断", "/diagnose"}
CLEAR_COMMANDS = {"清空病例", "重置病例", "/clear", "/reset"}
AGENT_DISPLAY_NAMES = {
    "Search Planning Agent": "检索规划",
    "Knowledge Searcher Agent": "医学知识检索",
    "Similar Case Retrieval Agent": "相似病例检索",
    "Guideline Searcher Agent": "本地指南检索",
    "Digestive Diagnosis Agent": "消化内科诊断分析",
    "Diagnostic Judgement Agent": "诊断结果评估",
}
STAGE_DISPLAY_NAMES = {
    "Search Planning Result": "检索规划结果",
    "Knowledge Search Result": "医学知识检索结果",
    "Similar Case Retrieval Result": "相似病例检索结果",
    "Guideline Search Result": "本地指南检索结果",
    "Final Diagnosis Result": "消化内科诊断结果",
    "Diagnostic Judgement Result": "诊断结果评估",
}
FIELD_DISPLAY_NAMES = {
    "hypotheses": "候选诊断",
    "search_queries": "文献检索词",
    "used_skill": "是否使用本地指南资料",
    "skill_names": "指南资料标识",
    "guideline_evidence": "指南依据",
    "summary": "总结",
    "limitations": "局限性",
    "topk_diagnoses": "疑似诊断",
    "rank": "排名",
    "disease": "疾病",
    "confidence": "支持强度",
    "supporting_evidence": "支持证据",
    "missing_information": "仍缺少的信息",
    "recommended_next_steps": "建议下一步",
    "safety_note": "安全提示",
    "closer_result": "更接近病例的诊断结果",
    "reason": "判断理由",
    "query": "检索词",
    "results": "检索结果",
    "title": "标题",
    "source": "来源",
    "published": "发表时间",
    "content": "内容",
    "metadata": "元数据",
}
VALUE_DISPLAY_NAMES = {
    "topk_diagnoses": "消化内科诊断结果",
    "hypotheses": "检索规划候选诊断",
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


def _format_diagnosis(result: DiagnosisResult) -> str:
    sections = ["## 诊断分析结果", "", result.summary]

    for item in result.topk_diagnoses:
        sections.extend(
            [
                "",
                f"### {item.rank}. {item.disease}（支持强度 {item.confidence}%）",
                "",
                "**支持证据**",
                *[f"- {evidence}" for evidence in item.supporting_evidence],
                "",
                "**仍缺少的信息**",
                *[f"- {information}" for information in item.missing_information],
                "",
                "**建议下一步**",
                *[f"- {step}" for step in item.recommended_next_steps],
            ]
        )
        if item.guideline_evidence:
            sections.extend(
                [
                    "",
                    "**指南依据**",
                    *[f"- {evidence}" for evidence in item.guideline_evidence],
                ]
            )

    if result.used_skill:
        sections.extend(["", "已使用本地指南资料辅助诊断。"])
    sections.extend(["", f"> {result.safety_note}"])
    return "\n".join(sections)


def _translate_stage_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            FIELD_DISPLAY_NAMES.get(str(key), str(key)): _translate_stage_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_translate_stage_value(item) for item in value]
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return "无"
    if isinstance(value, str):
        return VALUE_DISPLAY_NAMES.get(value, value)
    return value


def _format_stage_result(title: str, content: str) -> str:
    stage_name, separator, round_index = title.partition(" - Round ")
    display_name = STAGE_DISPLAY_NAMES.get(stage_name, "阶段输出结果")
    heading = f"## 第 {round_index} 轮：{display_name}" if separator else f"## {display_name}"

    try:
        parsed_content = json.loads(content)
    except json.JSONDecodeError:
        return f"{heading}\n\n{content}"

    translated_content = _translate_stage_value(parsed_content)
    if isinstance(translated_content, str):
        return f"{heading}\n\n{translated_content}"
    formatted_content = json.dumps(translated_content, ensure_ascii=False, indent=2)
    return f"{heading}\n\n```json\n{formatted_content}\n```"


class MedicalDiagnosisChatKitServer(ChatKitServer[dict[str, Any]]):
    store: InMemoryChatKitStore

    def __init__(self, store: InMemoryChatKitStore) -> None:
        super().__init__(store=store)
        self.store = store

    def _assistant_event(
        self,
        thread: ThreadMetadata,
        text: str,
        context: dict[str, Any],
    ) -> ThreadItemDoneEvent:
        return ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id=self.store.generate_item_id("message", thread, context),
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

        if normalized_command in CLEAR_COMMANDS:
            self.store.clear_case_text(thread.id)
            yield self._assistant_event(
                thread,
                "当前线程中的病例信息已清空。请发送新的病例资料。",
                context,
            )
            return

        should_diagnose = input_user_message is None or normalized_command in DIAGNOSE_COMMANDS
        if not should_diagnose:
            if not user_text:
                yield self._assistant_event(
                    thread,
                    "没有读取到文本内容。请发送病例资料。",
                    context,
                )
                return

            case_text = self.store.append_case_section(thread.id, user_text)
            yield self._assistant_event(
                thread,
                (
                    f"已记录这段病例资料，当前累计 {len(case_text)} 个字符。"
                    "你可以继续补充检查结果；资料完整后发送“开始诊断”。"
                ),
                context,
            )
            return

        case_text = self.store.get_case_text(thread.id)
        if not case_text:
            yield self._assistant_event(
                thread,
                "当前还没有病例资料。请先发送患者病史、体征和检查结果。",
                context,
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
                    round_label = f"第 {content} 轮：" if content is not None else ""
                    display_name = AGENT_DISPLAY_NAMES.get(title, "诊断处理")
                    yield ProgressUpdateEvent(
                        icon="analytics",
                        text=f"{round_label}正在进行{display_name}…",
                    )
                elif event_type == "stage_completed" and content is not None:
                    yield self._assistant_event(
                        thread,
                        _format_stage_result(title, content),
                        context,
                    )

            result = await diagnosis_task
        except RateLimitError as exc:
            error_code = _rate_limit_error_code(exc)
            logger.warning(
                "OpenAI API request failed for thread %s: code=%s request_id=%s",
                thread.id,
                error_code or "rate_limit_exceeded",
                exc.request_id,
            )
            if error_code == "insufficient_quota":
                yield ErrorEvent(
                    message=(
                        "OpenAI API 额度不足。请检查 API Key 所属项目的余额、"
                        "Billing 和使用预算，更新后重启后端。"
                    ),
                    allow_retry=False,
                )
            else:
                yield ErrorEvent(
                    message="OpenAI API 当前达到速率限制，请稍后重试。",
                    allow_retry=True,
                )
            return
        except Exception:
            logger.exception("Diagnosis pipeline failed for thread %s", thread.id)
            yield ErrorEvent(
                message="诊断流水线运行失败，请检查服务端日志后重试。",
                allow_retry=True,
            )
            return

        yield self._assistant_event(thread, _format_diagnosis(result), context)
