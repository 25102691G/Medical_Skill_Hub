from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

from agents import Model
from chatkit.server import ChatKitServer
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    CustomTask,
    DurationSummary,
    ErrorEvent,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemRemovedEvent,
    ThreadItemUpdatedEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
    UserMessageTextContent,
    WidgetComponentUpdated,
    WidgetItem,
    Workflow,
    WorkflowItem,
    WorkflowTaskAdded,
    WorkflowTaskUpdated,
)
from chatkit.widgets import BasicRoot, DynamicWidgetComponent
from openai import RateLimitError

from chatkit_app.store import InMemoryChatKitStore
from chatkit_app.translation import (
    DisplayTranslator,
    get_context_display_language,
)
from main import make_diagnosis_pipeline_async
from schemas import DiagnosisPipelineResult, DiagnosisResult


logger = logging.getLogger(__name__)

CHATKIT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "chatkit"

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
        "Preprocessing Agent": "诊断预处理",
        "Search Planning Agent": "检索规划",
        "Knowledge Searcher Agent": "医学知识检索",
        "Similar Case Retrieval Agent": "相似病例检索",
        "Guideline Searcher Agent": "本地指南检索",
        "Digestive Diagnosis Agent": "消化内科诊断分析",
        "Corrective Digestive Diagnosis Agent": "消化内科诊断修正",
        "Diagnostic Judgement Agent": "诊断结果评估",
    },
    "en": {
        "Preprocessing Agent": "case preprocessing",
        "Search Planning Agent": "search planning",
        "Knowledge Searcher Agent": "medical knowledge retrieval",
        "Similar Case Retrieval Agent": "similar-case retrieval",
        "Guideline Searcher Agent": "local guideline retrieval",
        "Digestive Diagnosis Agent": "gastroenterology diagnosis analysis",
        "Corrective Digestive Diagnosis Agent": "gastroenterology diagnosis correction",
        "Diagnostic Judgement Agent": "diagnostic result assessment",
    },
}
AGENT_PHASE_NAMES = {
    "zh-CN": {
        "Preprocessing Agent": "诊断预处理阶段",
        "Search Planning Agent": "检索阶段",
        "Knowledge Searcher Agent": "检索阶段",
        "Similar Case Retrieval Agent": "检索阶段",
        "Guideline Searcher Agent": "检索阶段",
        "Digestive Diagnosis Agent": "诊断阶段",
        "Corrective Digestive Diagnosis Agent": "诊断阶段",
        "Diagnostic Judgement Agent": "诊断结果评估阶段",
        "Report Generation": "诊断结果评估阶段",
    },
    "en": {
        "Preprocessing Agent": "Preprocessing stage",
        "Search Planning Agent": "Retrieval stage",
        "Knowledge Searcher Agent": "Retrieval stage",
        "Similar Case Retrieval Agent": "Retrieval stage",
        "Guideline Searcher Agent": "Retrieval stage",
        "Digestive Diagnosis Agent": "Diagnosis stage",
        "Corrective Digestive Diagnosis Agent": "Diagnosis stage",
        "Diagnostic Judgement Agent": "Diagnostic evaluation stage",
        "Report Generation": "Diagnostic evaluation stage",
    },
}
STAGE_AGENT_NAMES = {
    "Preprocessing Result": "Preprocessing Agent",
    "Similar Case Retrieval Result": "Similar Case Retrieval Agent",
    "Search Planning Result": "Search Planning Agent",
    "Knowledge Search Result": "Knowledge Searcher Agent",
    "Guideline Search Result": "Guideline Searcher Agent",
    "Final Diagnosis Result": "Digestive Diagnosis Agent",
    "Corrective Final Diagnosis Result": "Corrective Digestive Diagnosis Agent",
    "Diagnostic Judgement Result": "Diagnostic Judgement Agent",
}
TRANSLATED_PROGRESS_STAGES = {
    "Preprocessing Result",
    "Similar Case Retrieval Result",
    "Search Planning Result",
    "Final Diagnosis Result",
    "Corrective Final Diagnosis Result",
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
        "progress_without_round": "正在进行{agent_name}…",
        "report_progress": "正在翻译并生成诊断分析结果…",
        "report_title": "生成诊断分析结果",
        "report_completed": "诊断分析结果生成完成",
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
        "progress_without_round": "Running {agent_name}…",
        "report_progress": "Translating and generating the diagnostic analysis result…",
        "report_title": "Generate diagnostic analysis result",
        "report_completed": "Diagnostic analysis result generated",
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


def _format_elapsed_time(seconds: int, language: str) -> str:
    if seconds < 60:
        return f"{seconds}秒" if language == "zh-CN" else f"{seconds}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if language == "zh-CN":
        return f"{minutes}分{remaining_seconds}秒"
    return f"{minutes}m {remaining_seconds}s"


def _format_diagnosis(result: DiagnosisResult) -> str:
    sections = ["## Diagnostic Analysis Result", "", result.summary]

    for item in result.topk_diagnoses:
        sections.extend(
            [
                "",
                (
                    f"### {item.rank}. {item.icd_code} — {item.category_name} "
                    f"(support strength: {item.confidence}%)"
                ),
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


def _format_numbered_details(
    label: str,
    items: list[str],
    overflow_marker: str | None = None,
) -> str:
    if not items:
        return ""

    lines = [f"{index}. {item}" for index, item in enumerate(items, start=1)]
    if overflow_marker:
        lines.append(overflow_marker)
    return f"\n\n{label}\n\n" + "\n".join(lines)


def _format_stage_progress(title: str, content: str, language: str) -> str | None:
    stage_name, separator, round_index = title.partition(" - Round ")
    parsed_content = json.loads(content)
    round_prefix = (
        f"第 {round_index} 轮："
        if language == "zh-CN" and separator
        else f"Round {round_index}: "
        if separator
        else ""
    )

    if stage_name == "Preprocessing Result":
        hypotheses = parsed_content.get("llm_hypotheses", [])
        hypothesis_count = len(hypotheses)
        hypothesis_names = [hypothesis["category_name"] for hypothesis in hypotheses]
        feature_groups = parsed_content.get("positive_features", {})
        feature_count = sum(len(items) for items in feature_groups.values())
        if language == "zh-CN":
            detail = _format_numbered_details("诊断假设：", hypothesis_names)
            return (
                f"诊断预处理完成：生成 {hypothesis_count} 个诊断假设，"
                f"提取 {feature_count} 项结构化病例信息{detail}"
            )
        detail = _format_numbered_details("Diagnostic hypotheses:", hypothesis_names)
        return (
            f"Case preprocessing completed: generated {hypothesis_count} diagnostic hypotheses "
            f"and extracted {feature_count} structured patient findings{detail}"
        )

    if stage_name == "Similar Case Retrieval Result":
        ranking = parsed_content.get("rrf", [])
        diagnosis_names = [case["discharge_disease"] for case in ranking]
        if language == "zh-CN":
            detail = _format_numbered_details("Top5相似病例诊断：", diagnosis_names)
            return f"{round_prefix}相似病例检索完成{detail}"
        detail = _format_numbered_details("Case diagnoses:", diagnosis_names)
        return f"{round_prefix}Similar-case retrieval completed: found {len(ranking)} cases{detail}"

    if stage_name == "Search Planning Result":
        hypothesis_count = len(parsed_content.get("hypotheses", []))
        search_queries = parsed_content.get("search_queries", [])
        query_count = len(search_queries)
        if language == "zh-CN":
            detail = _format_numbered_details("检索式：", search_queries)
            return (
                f"{round_prefix}检索规划完成：合并 {hypothesis_count} 个候选诊断，"
                f"生成 {query_count} 条检索式{detail}"
            )
        detail = _format_numbered_details("Queries:", search_queries)
        return (
            f"{round_prefix}Search planning completed: merged {hypothesis_count} diagnostic "
            f"candidates and generated {query_count} queries{detail}"
        )

    if stage_name == "Knowledge Search Result":
        query_results = parsed_content.get("relevant_pubmed_results", [])
        publications = {}
        for query_result in query_results:
            for result in query_result.get("results", []):
                publications.setdefault(result["pmid"], result["title"])
        section_count = sum(
            len(result.get("abstract_sections", []))
            for query_result in query_results
            for result in query_result.get("results", [])
        )
        if language == "zh-CN":
            if not publications:
                return f"{round_prefix}医学知识检索完成：未检索到相关 PubMed 文献"
            publication_titles = list(publications.values())[:5]
            detail = _format_numbered_details(
                "PubMed 文献：",
                publication_titles,
                "……" if len(publications) > 5 else None,
            )
            return (
                f"{round_prefix}医学知识检索完成：匹配 {len(query_results)} 条检索式，"
                f"保留 {len(publications)} 篇 PubMed 文献、{section_count} 个摘要片段{detail}"
            )
        if not publications:
            return (
                f"{round_prefix}Medical knowledge retrieval completed: "
                "no relevant PubMed articles found"
            )
        publication_titles = list(publications.values())[:5]
        detail = _format_numbered_details(
            "PubMed publications:",
            publication_titles,
            "..." if len(publications) > 5 else None,
        )
        return (
            f"{round_prefix}Medical knowledge retrieval completed: matched {len(query_results)} "
            f"queries and retained {len(publications)} PubMed articles with "
            f"{section_count} abstract sections{detail}"
        )

    if stage_name == "Guideline Search Result":
        skill_results = parsed_content.get("skill_results", [])
        evidence_count = sum(
            len(skill_result.get("guideline_evidence", []))
            for skill_result in skill_results
        )
        if language == "zh-CN":
            if not skill_results:
                return f"{round_prefix}本地指南检索完成：未匹配到可用指南"
            guideline_names = [result["skill_name"] for result in skill_results[:5]]
            detail = _format_numbered_details(
                "指南：",
                guideline_names,
                "……" if len(skill_results) > 5 else None,
            )
            return (
                f"{round_prefix}本地指南检索完成：使用 {len(skill_results)} 份指南，"
                f"提取 {evidence_count} 条相关证据{detail}"
            )
        if not skill_results:
            return (
                f"{round_prefix}Local guideline retrieval completed: "
                "no applicable guideline found"
            )
        guideline_names = [result["skill_name"] for result in skill_results[:5]]
        detail = _format_numbered_details(
            "Guidelines:",
            guideline_names,
            "..." if len(skill_results) > 5 else None,
        )
        return (
            f"{round_prefix}Local guideline retrieval completed: used {len(skill_results)} "
            f"guidelines and extracted {evidence_count} relevant evidence items{detail}"
        )

    if stage_name in {"Final Diagnosis Result", "Corrective Final Diagnosis Result"}:
        diagnoses = parsed_content.get("topk_diagnoses", [])
        diagnosis_names = [diagnosis["category_name"] for diagnosis in diagnoses]
        excluded_candidates = parsed_content.get("excluded_planning_candidates", [])
        stage_label = (
            "消化内科诊断修正"
            if stage_name.startswith("Corrective")
            else "消化内科诊断分析"
        )
        if language == "zh-CN":
            detail = _format_numbered_details("Top5候选诊断：", diagnosis_names)
            excluded_detail = ""
            if excluded_candidates:
                excluded_lines = []
                for index, candidate in enumerate(excluded_candidates, start=1):
                    excluded_lines.extend(
                        [
                            f"{index}. {candidate['category_name']}",
                            "   排除依据：",
                            *[
                                f"   - {evidence}"
                                for evidence in candidate["patient_contrary_evidence"]
                            ],
                        ]
                    )
                excluded_detail = "\n\n排除诊断：\n\n" + "\n".join(excluded_lines)
            return (
                f"{round_prefix}{stage_label}完成{detail}{excluded_detail}"
            )
        stage_label = (
            "Gastroenterology diagnosis correction"
            if stage_name.startswith("Corrective")
            else "Gastroenterology diagnosis analysis"
        )
        detail = _format_numbered_details("Candidate diagnoses:", diagnosis_names)
        excluded_detail = ""
        if excluded_candidates:
            excluded_lines = []
            for index, candidate in enumerate(excluded_candidates, start=1):
                excluded_lines.extend(
                    [
                        f"{index}. {candidate['category_name']}",
                        "   Exclusion evidence:",
                        *[
                            f"   - {evidence}"
                            for evidence in candidate["patient_contrary_evidence"]
                        ],
                    ]
                )
            excluded_detail = "\n\nExcluded diagnoses:\n\n" + "\n".join(excluded_lines)
        return (
            f"{round_prefix}{stage_label} completed: produced {len(diagnoses)} "
            f"candidates{detail}{excluded_detail}"
        )

    if stage_name == "Diagnostic Judgement Result":
        accepted = parsed_content.get("closer_result") == "final_diagnoses"
        if language == "zh-CN":
            outcome = "当前诊断通过" if accepted else "需要补充检索，进入下一轮"
            return f"{round_prefix}诊断结果评估完成：{outcome}"
        outcome = (
            "current diagnosis accepted"
            if accepted
            else "more evidence is needed; proceeding to the next round"
        )
        return f"{round_prefix}Diagnostic result assessment completed: {outcome}"

    return None


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
        preserved_texts: tuple[str, ...] = (),
    ) -> ThreadItemDoneEvent:
        item_id = self.store.generate_item_id("message", thread, context)
        self.store.register_raw_assistant_text(
            item_id,
            raw_text or text,
            preserved_texts,
        )
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
        progress_started_at = loop.time()
        workflow_item = WorkflowItem(
            id=self.store.generate_item_id("workflow", thread, context),
            thread_id=thread.id,
            created_at=datetime.now(),
            workflow=Workflow(
                type="custom",
                tasks=[],
                expanded=True,
            ),
        )
        elapsed_component_id = f"{workflow_item.id}-elapsed"
        elapsed_text_props = {
            "color": "secondary",
            "size": "sm",
        }
        elapsed_item = WidgetItem(
            id=self.store.generate_item_id("message", thread, context),
            thread_id=thread.id,
            created_at=datetime.now(),
            widget=BasicRoot(
                children=[
                    DynamicWidgetComponent(
                        type="Row",
                        gap=1,
                        align="baseline",
                        children=[
                            DynamicWidgetComponent(
                                type="Text",
                                value="已处理" if display_language == "zh-CN" else "Processed",
                                **elapsed_text_props,
                            ),
                            DynamicWidgetComponent(
                                type="Text",
                                id=elapsed_component_id,
                                value=_format_elapsed_time(0, display_language),
                                **elapsed_text_props,
                            ),
                        ],
                    )
                ]
            ),
        )
        active_task_indices: dict[tuple[str, str | None], int] = {}
        workflow_added = False
        workflow_done = False
        elapsed_item_visible = True

        diagnosis_started_at = datetime.now()
        diagnosis_run_id = diagnosis_started_at.strftime("%Y%m%d_%H%M%S_%f")
        output_path = CHATKIT_OUTPUT_DIR / f"diagnosis_{diagnosis_run_id}.json"
        progress_events: list[dict[str, str | None]] = []

        def report_progress(event_type: str, title: str, content: str | None) -> None:
            progress_events.append(
                {
                    "recorded_at": datetime.now().isoformat(),
                    "event_type": event_type,
                    "title": title,
                    "content": content,
                }
            )
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                (event_type, title, content),
            )

        async def run_diagnosis() -> DiagnosisPipelineResult:
            output_record: dict[str, Any] = {
                "diagnosis_run_id": diagnosis_run_id,
                "thread_id": thread.id,
                "started_at": diagnosis_started_at.isoformat(),
                "case_text": case_text,
            }
            try:
                pipeline_result = await make_diagnosis_pipeline_async(
                    case_text,
                    model=self.diagnosis_model,
                    debug=False,
                    progress_callback=report_progress,
                )
                output_record["status"] = "completed"
                output_record["pipeline_result"] = pipeline_result.model_dump(
                    mode="json"
                )
                return pipeline_result
            except Exception as exc:
                output_record["status"] = "failed"
                output_record["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "stage": getattr(exc, "stage", None),
                }
                raise
            finally:
                completed_at = datetime.now()
                output_record["completed_at"] = completed_at.isoformat()
                output_record["duration_seconds"] = (
                    completed_at - diagnosis_started_at
                ).total_seconds()
                output_record["progress_events"] = progress_events
                CHATKIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(output_record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                loop.call_soon(progress_queue.put_nowait, None)

        diagnosis_task = asyncio.create_task(run_diagnosis())
        yield ThreadItemAddedEvent(item=elapsed_item)
        stage_translation_task: asyncio.Task[str] | None = None
        final_translation_task: asyncio.Task[str] | None = None
        try:
            next_elapsed_update = progress_started_at + 1
            while True:
                try:
                    progress_event = await asyncio.wait_for(
                        progress_queue.get(),
                        timeout=max(0, next_elapsed_update - loop.time()),
                    )
                except asyncio.TimeoutError:
                    elapsed_seconds = int(loop.time() - progress_started_at)
                    yield ThreadItemUpdatedEvent(
                        item_id=elapsed_item.id,
                        update=WidgetComponentUpdated(
                            component_id=elapsed_component_id,
                            component=DynamicWidgetComponent(
                                type="Text",
                                id=elapsed_component_id,
                                value=_format_elapsed_time(
                                    elapsed_seconds,
                                    display_language,
                                ),
                                **elapsed_text_props,
                            ),
                        ),
                    )
                    next_elapsed_update = loop.time() + 1
                    continue
                if progress_event is None:
                    break

                event_type, title, content = progress_event
                if event_type == "agent_started":
                    if title in {
                        "Planning Hypotheses Reranker Agent",
                        "Guideline Result Filter Agent",
                    }:
                        continue
                    agent_name = AGENT_DISPLAY_NAMES[display_language].get(
                        title,
                        "诊断处理" if display_language == "zh-CN" else "diagnostic processing",
                    )
                    task = CustomTask(
                        title=AGENT_PHASE_NAMES[display_language].get(
                            title,
                            "诊断阶段" if display_language == "zh-CN" else "Diagnosis stage",
                        ),
                        content=_static_text(
                            display_language,
                            "progress" if content else "progress_without_round",
                            round_index=content or "-",
                            agent_name=agent_name,
                        ),
                        icon="analytics",
                        status_indicator="loading",
                    )
                    task_index = len(workflow_item.workflow.tasks)
                    workflow_item.workflow.tasks.append(task)
                    active_task_indices[(title, content)] = task_index
                    if not workflow_added:
                        workflow_added = True
                        yield ThreadItemAddedEvent(item=workflow_item)
                    else:
                        yield ThreadItemUpdatedEvent(
                            item_id=workflow_item.id,
                            update=WorkflowTaskAdded(
                                task_index=task_index,
                                task=task,
                            ),
                        )
                elif event_type == "stage_completed" and content is not None:
                    stage_name, separator, round_text = title.partition(" - Round ")
                    completed_text = _format_stage_progress(
                        title,
                        content,
                        display_language,
                    )
                    if completed_text is not None:
                        if (
                            display_language == "zh-CN"
                            and stage_name in TRANSLATED_PROGRESS_STAGES
                        ):
                            stage_translation_task = asyncio.create_task(
                                self.translator.translate(
                                    completed_text,
                                    display_language,
                                )
                            )
                            while True:
                                try:
                                    completed_text = await asyncio.wait_for(
                                        asyncio.shield(stage_translation_task),
                                        timeout=max(
                                            0,
                                            next_elapsed_update - loop.time(),
                                        ),
                                    )
                                    stage_translation_task = None
                                    break
                                except asyncio.TimeoutError:
                                    elapsed_seconds = int(
                                        loop.time() - progress_started_at
                                    )
                                    yield ThreadItemUpdatedEvent(
                                        item_id=elapsed_item.id,
                                        update=WidgetComponentUpdated(
                                            component_id=elapsed_component_id,
                                            component=DynamicWidgetComponent(
                                                type="Text",
                                                id=elapsed_component_id,
                                                value=_format_elapsed_time(
                                                    elapsed_seconds,
                                                    display_language,
                                                ),
                                                **elapsed_text_props,
                                            ),
                                        ),
                                    )
                                    next_elapsed_update = loop.time() + 1
                        agent_title = STAGE_AGENT_NAMES[stage_name]
                        task_index = active_task_indices.pop(
                            (agent_title, round_text if separator else None),
                            None,
                        )
                        agent_name = AGENT_DISPLAY_NAMES[display_language][agent_title]
                        completed_task = CustomTask(
                            title=agent_name,
                            content=completed_text,
                            icon="check-circle",
                        )
                        if task_index is None:
                            task_index = len(workflow_item.workflow.tasks)
                            workflow_item.workflow.tasks.append(completed_task)
                            if not workflow_added:
                                workflow_added = True
                                yield ThreadItemAddedEvent(item=workflow_item)
                            else:
                                yield ThreadItemUpdatedEvent(
                                    item_id=workflow_item.id,
                                    update=WorkflowTaskAdded(
                                        task_index=task_index,
                                        task=completed_task,
                                    ),
                                )
                        else:
                            workflow_item.workflow.tasks[task_index] = completed_task
                            yield ThreadItemUpdatedEvent(
                                item_id=workflow_item.id,
                                update=WorkflowTaskUpdated(
                                    task_index=task_index,
                                    task=completed_task,
                                ),
                            )
            pipeline_result = await diagnosis_task
            result = pipeline_result.multi_round_diagnosis.rounds[-1].diagnosis_result
            report_task_index = len(workflow_item.workflow.tasks)
            report_task = CustomTask(
                title=AGENT_PHASE_NAMES[display_language]["Report Generation"],
                content=_static_text(display_language, "report_progress"),
                icon="write",
                status_indicator="loading",
            )
            workflow_item.workflow.tasks.append(report_task)
            if not workflow_added:
                workflow_added = True
                yield ThreadItemAddedEvent(item=workflow_item)
            else:
                yield ThreadItemUpdatedEvent(
                    item_id=workflow_item.id,
                    update=WorkflowTaskAdded(
                        task_index=report_task_index,
                        task=report_task,
                    ),
                )

            raw_diagnosis_text = _format_diagnosis(result)
            preserved_texts = tuple(result.evidence)
            final_translation_task = asyncio.create_task(
                self.translator.translate(
                    raw_diagnosis_text,
                    display_language,
                    preserved_texts,
                )
            )
            while True:
                try:
                    translated_diagnosis_text = await asyncio.wait_for(
                        asyncio.shield(final_translation_task),
                        timeout=1,
                    )
                    break
                except asyncio.TimeoutError:
                    elapsed_seconds = int(loop.time() - progress_started_at)
                    yield ThreadItemUpdatedEvent(
                        item_id=elapsed_item.id,
                        update=WidgetComponentUpdated(
                            component_id=elapsed_component_id,
                            component=DynamicWidgetComponent(
                                type="Text",
                                id=elapsed_component_id,
                                value=_format_elapsed_time(
                                    elapsed_seconds,
                                    display_language,
                                ),
                                **elapsed_text_props,
                            ),
                        ),
                    )
            completed_report_task = CustomTask(
                title=_static_text(display_language, "report_title"),
                content=_static_text(display_language, "report_completed"),
                icon="check-circle",
            )
            workflow_item.workflow.tasks[report_task_index] = completed_report_task
            yield ThreadItemUpdatedEvent(
                item_id=workflow_item.id,
                update=WorkflowTaskUpdated(
                    task_index=report_task_index,
                    task=completed_report_task,
                ),
            )
            elapsed_item_visible = False
            yield ThreadItemRemovedEvent(item_id=elapsed_item.id)
            workflow_item.workflow.summary = DurationSummary(
                duration=int(loop.time() - progress_started_at)
            )
            workflow_item.workflow.expanded = False
            yield ThreadItemDoneEvent(item=workflow_item)
            workflow_done = True
            yield self._assistant_event(
                thread,
                translated_diagnosis_text,
                context,
                raw_text=raw_diagnosis_text,
                preserved_texts=preserved_texts,
            )
        except RateLimitError as exc:
            translation_tasks = [
                task
                for task in (stage_translation_task, final_translation_task)
                if task is not None
            ]
            for task in translation_tasks:
                task.cancel()
            await asyncio.gather(
                *translation_tasks,
                return_exceptions=True,
            )
            error_code = _rate_limit_error_code(exc)
            logger.warning(
                "Model API request failed for thread %s: code=%s request_id=%s",
                thread.id,
                error_code or "rate_limit_exceeded",
                exc.request_id,
            )
            if elapsed_item_visible:
                elapsed_item_visible = False
                yield ThreadItemRemovedEvent(item_id=elapsed_item.id)
            if workflow_item.workflow.tasks and not workflow_done:
                workflow_item.workflow.summary = DurationSummary(
                    duration=int(loop.time() - progress_started_at)
                )
                workflow_item.workflow.expanded = False
                yield ThreadItemDoneEvent(item=workflow_item)
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
            translation_tasks = [
                task
                for task in (stage_translation_task, final_translation_task)
                if task is not None
            ]
            for task in translation_tasks:
                task.cancel()
            await asyncio.gather(
                *translation_tasks,
                return_exceptions=True,
            )
            if elapsed_item_visible:
                elapsed_item_visible = False
                yield ThreadItemRemovedEvent(item_id=elapsed_item.id)
            if workflow_item.workflow.tasks and not workflow_done:
                workflow_item.workflow.summary = DurationSummary(
                    duration=int(loop.time() - progress_started_at)
                )
                workflow_item.workflow.expanded = False
                yield ThreadItemDoneEvent(item=workflow_item)
            logger.exception("Diagnosis pipeline failed for thread %s", thread.id)
            yield ErrorEvent(
                message=_static_text(display_language, "pipeline_error"),
                allow_retry=True,
            )
            return
