"""Context-aware chat about uploaded blood test results."""

from __future__ import annotations

import json
from typing import Any

from src.extraction.text_generation import generate_text_chat

CHAT_SYSTEM_PROMPT = """
You are an educational assistant helping a patient understand blood test results.

Rules:
- Use ONLY the patient context, extracted lab values, knowledge-graph enrichment, and grounded
  interpretation notes provided below.
- Do not diagnose, prescribe, or invent medical facts not present in the context.
- Use plain language and say when a clinician should interpret a result in person.
- If the user asks about something not in the context, say you do not have that information.
- Keep answers concise unless the user asks for detail.
""".strip()

_MAX_CONTEXT_CHARS = 8000
_MAX_HISTORY_TURNS = 6


class ResultsChatAssistant:
    def reply(
        self,
        user_message: str,
        chat_history: list[dict[str, str]] | None,
        session: dict[str, Any] | None,
    ) -> str:
        message = (user_message or "").strip()
        if not message:
            return "Please enter a question about your blood test results."

        if not session or not session.get("health_report"):
            return "Upload and analyze a lab report first, then I can answer questions about your results."

        context = build_chat_context(session)
        messages = _build_messages(context, chat_history or [], message)
        try:
            return generate_text_chat(messages)
        except Exception as exc:
            return (
                "I couldn't generate a chat reply with the current model backend. "
                f"Details: {exc}"
            )


def build_chat_context(session: dict[str, Any]) -> str:
    health_report = session.get("health_report") or {}
    extraction = session.get("extraction") or {}
    interpretation = session.get("interpretation") or {}

    patient = health_report.get("patient") or extraction.get("patient") or {}
    markers = health_report.get("markers") or []
    summary = health_report.get("summary") or {}

    lines: list[str] = [
        "=== Patient context ===",
        json.dumps(
            {
                "age": patient.get("age"),
                "age_years": patient.get("age_years"),
                "age_group": patient.get("age_group"),
                "sex": patient.get("sex"),
            },
            indent=2,
        ),
        "",
        "=== Report summary ===",
        json.dumps(summary, indent=2),
        "",
        "=== Extracted markers ===",
    ]

    for marker in markers[:40]:
        lines.append(
            json.dumps(
                {
                    "name": marker.get("display_name") or marker.get("raw_name"),
                    "value": marker.get("value"),
                    "unit": marker.get("unit"),
                    "status": marker.get("status"),
                    "lab_reference_range": marker.get("lab_reference_range"),
                    "comparison_basis": (marker.get("comparison") or {}).get("basis"),
                    "kg_description": ((marker.get("knowledge") or {}).get("description")),
                    "kg_importance": ((marker.get("knowledge") or {}).get("why_important")),
                },
                ensure_ascii=False,
            )
        )

    if interpretation.get("flagged"):
        lines.extend(["", "=== Flagged markers (KB-grounded) ==="])
        for item in interpretation["flagged"]:
            lines.append(json.dumps(item, ensure_ascii=False))

    if interpretation.get("patterns"):
        lines.extend(["", "=== Cross-marker patterns ==="])
        for item in interpretation["patterns"]:
            lines.append(json.dumps(item, ensure_ascii=False))

    if extraction.get("notes"):
        lines.extend(["", "=== Extraction notes ===", json.dumps(extraction["notes"], indent=2)])

    lines.extend(["", "=== Disclaimer ===", interpretation.get("disclaimer", "")])

    context = "\n".join(lines)
    if len(context) <= _MAX_CONTEXT_CHARS:
        return context
    return context[: _MAX_CONTEXT_CHARS - 3].rstrip() + "..."


def _build_messages(
    context: str,
    chat_history: list[dict[str, str]],
    user_message: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Blood test context:\n\n{context}"},
        {
            "role": "assistant",
            "content": "I have your blood test context. Ask me anything about these results.",
        },
    ]

    recent = _recent_chat_turns(chat_history)
    messages.extend(recent)
    messages.append({"role": "user", "content": user_message})
    return messages


def _recent_chat_turns(chat_history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only user follow-up turns, skipping pipeline trace assistant messages."""
    turns: list[dict[str, str]] = []
    for item in chat_history:
        role = item.get("role")
        content = str(item.get("content") or "").strip()
        if not content or role not in {"user", "assistant"}:
            continue
        if role == "assistant" and content.startswith("**Step "):
            continue
        if role == "assistant" and content.startswith("**Analysis pipeline complete."):
            continue
        if role == "assistant" and content.startswith("**Pipeline running"):
            continue
        if role == "assistant" and content.startswith("Upload a lab report"):
            continue
        turns.append({"role": role, "content": content})

    if len(turns) > _MAX_HISTORY_TURNS * 2:
        turns = turns[-(_MAX_HISTORY_TURNS * 2) :]
    return turns
