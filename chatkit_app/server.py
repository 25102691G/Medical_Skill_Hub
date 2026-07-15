from __future__ import annotations

import asyncio
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
