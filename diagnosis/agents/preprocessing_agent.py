from __future__ import annotations

from agents import Agent, Model

from schemas import LlmHypothesesResult, PositiveFeaturesResult, PreprocessingResult


PREPROCESSING_INSTRUCTIONS = """
## PREPROCESSING INSTRUCTIONS

You coordinate two independent preprocessing tasks for a gastroenterology diagnosis pipeline.

Call `generate_diagnostic_hypotheses` exactly once and `extract_positive_features` exactly once.
Pass the complete original patient case to each tool without adding, removing, or interpreting any
patient information. The two tools are independent and neither tool's result may be supplied to the
other tool.

After both tools return, copy their results into one JSON object with exactly these fields:

* `llm_hypotheses`: copy the complete `llm_hypotheses` tool result without changes;
* `positive_features`: copy the complete `positive_features` tool result without changes.

Do not perform your own clinical analysis, alter either tool result, or add commentary. Return valid
JSON only and strictly follow the provided output schema.
""".strip()


HYPOTHESIS_PREPROCESSING_INSTRUCTIONS = """
## DIAGNOSTIC HYPOTHESIS PREPROCESSING INSTRUCTIONS

You are a gastroenterology clinical diagnosis model.

### 1. Objective

Generate up to 5 diagnostic hypotheses directly from the supplied original patient case. Rank them by
their likelihood of being the principal diagnosis chiefly responsible for the current hospitalization or
the main condition evaluated and treated during the hospitalization.

This is a standalone diagnostic assessment. Use only the original patient case supplied in the current
request. Do not use extracted positive features, retrieval results, guidelines, similar cases, search
plans, or outputs from any other agent.

Do not add items solely to reach 5 hypotheses. If the case provides insufficient evidence, return fewer
items or an empty list.

### 2. Diagnostic Scope

Do not use chronic comorbidities, incidental findings, or secondary complications as filler candidates.
Include one only when the record supports it as a plausible principal diagnosis for this hospitalization.

Hypotheses may contain clinical inferences. When supported by the case, include time-critical underlying
diseases that require urgent exclusion.

### 3. ICD-10-CM Requirements

For each hypothesis:

* use the complete ICD-10-CM code without a decimal point, preserving all documented characters;
* use a three-character category only when that category has no more specific subcategory;
* use the canonical English description corresponding to the selected code as category_name;
* do not add diagnostic details that are not supported by the original patient case;
* do not output duplicate icd_code values.

When the record does not support a specific subtype, use the complete unspecified or other code defined
for that category rather than truncating the code.

### 4. Output Requirements

Write all output in English. Return valid JSON only and strictly follow the provided output schema. Do not
output Markdown, commentary, positive_features, or fields that are not defined in the schema.
""".strip()


POSITIVE_FEATURE_PREPROCESSING_INSTRUCTIONS = """
## POSITIVE FEATURE PREPROCESSING INSTRUCTIONS

You are a gastroenterology clinical feature extraction model.

### 1. Objective

Extract positive clinical manifestations and positive auxiliary examination results directly from the
supplied original patient case. Use only the information explicitly documented in the current request.

This is an independent extraction task. Do not use diagnostic hypotheses, retrieval results, guidelines,
similar cases, search plans, or outputs from any other agent.

### 2. Positive Feature Requirements

Clinical manifestations include positive symptoms, abnormal vital signs, and positive physical
examination findings. Auxiliary examination results include abnormal laboratory, endoscopic, imaging,
pathology, and microbiology findings.

Keep each observed feature or result as a separate list item and do not repeat the same information.

Do not include negative or normal findings, past medical history, inferred features, or examinations that
are only recommended, planned, or pending.

Do not include a diagnosis as a positive feature unless it is explicitly documented as an observed,
confirmed finding in the original patient case.

### 3. Output Requirements

Write every item as a concise English phrase suitable for matching similar cases. Return valid JSON only
and strictly follow the provided output schema. Do not output Markdown, commentary, diagnostic
hypotheses, or fields that are not defined in the schema.
""".strip()


def build_preprocessing_agent(
    model: str | Model,
    *,
    native_structured_output: bool = True,
) -> Agent:
    hypothesis_agent = Agent(
        name="Diagnostic Hypothesis Preprocessing Agent",
        model=model,
        instructions=HYPOTHESIS_PREPROCESSING_INSTRUCTIONS,
        output_type=LlmHypothesesResult if native_structured_output else None,
    )
    positive_feature_agent = Agent(
        name="Positive Feature Preprocessing Agent",
        model=model,
        instructions=POSITIVE_FEATURE_PREPROCESSING_INSTRUCTIONS,
        output_type=PositiveFeaturesResult if native_structured_output else None,
    )
    return Agent(
        name="Preprocessing Agent",
        model=model,
        instructions=PREPROCESSING_INSTRUCTIONS,
        tools=[
            hypothesis_agent.as_tool(
                tool_name="generate_diagnostic_hypotheses",
                tool_description=(
                    "Generate principal-diagnosis hypotheses from the complete original patient case."
                ),
            ),
            positive_feature_agent.as_tool(
                tool_name="extract_positive_features",
                tool_description=(
                    "Extract positive clinical features from the complete original patient case."
                ),
            ),
        ],
        output_type=PreprocessingResult if native_structured_output else None,
    )
