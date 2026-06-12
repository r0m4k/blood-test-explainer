"""Render the grounded Interpretation into display-ready Markdown / HTML (Phase 3.5).

Pure presentation — every fact comes from src/interpretation.py (which comes from the KB), nothing
is invented here. Framework-agnostic so the app can drop `interpretation_html(tests)` into a
`gr.HTML` (styled cards) or `interpretation_markdown(tests)` into a `gr.Markdown`.
"""

from __future__ import annotations

import html

from src.interpretation import Interpretation, build_interpretation

_STATUS_LABEL = {"low": "Low", "high": "High"}
_STATUS_COLOR = {"low": "#b06b00", "high": "#b22222"}


def interpretation_markdown(tests: list[dict]) -> str:
    """tests (extracted) -> markdown for a gr.Markdown component."""
    return render_markdown(build_interpretation(tests))


def interpretation_html(tests: list[dict]) -> str:
    """tests (extracted) -> styled HTML cards for a gr.HTML component."""
    return render_html(build_interpretation(tests))


def render_markdown(interp: Interpretation) -> str:
    out = ["### What your results may mean", "_Educational information, not a diagnosis._", ""]
    if not interp.has_findings:
        out.append(f"All {interp.normal_count} recognized markers are within their reference ranges.")
        out += ["", f"> {interp.disclaimer}"]
        return "\n".join(out)

    for c in interp.flagged:
        unit = f" {c.unit}" if c.unit else ""
        status = _STATUS_LABEL.get(c.status, c.status)
        out.append(f"**{c.marker} — {c.value}{unit}  ({status}, ref {c.reference_range})**")
        if c.note:
            out.append(c.note)
        if c.questions:
            out.append("Questions for your doctor: " + " ".join(f"_{q}_" for q in c.questions))
        out.append("")

    if interp.patterns:
        out.append("#### Patterns across markers")
        out += [f"- **{p.name}** — {p.note}" for p in interp.patterns]
        out.append("")

    out.append(f"{interp.normal_count} other recognized markers were within range.")
    out += ["", f"> {interp.disclaimer}"]
    return "\n".join(out)


def render_html(interp: Interpretation) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    parts = ['<div style="font-family:system-ui,-apple-system,sans-serif;max-width:760px;">']
    parts.append('<h3 style="margin:0 0 2px;">What your results may mean</h3>')
    parts.append('<div style="color:#6b7280;font-size:13px;margin-bottom:14px;">'
                 'Educational information, not a diagnosis.</div>')

    if not interp.has_findings:
        parts.append(f'<div style="padding:12px;border-radius:10px;background:#f0fdf4;color:#166534;">'
                     f'All {interp.normal_count} recognized markers are within their reference ranges.</div>')
    else:
        for c in interp.flagged:
            color = _STATUS_COLOR.get(c.status, "#374151")
            unit = f" {esc(c.unit)}" if c.unit else ""
            status = _STATUS_LABEL.get(c.status, esc(c.status))
            parts.append(
                f'<div style="border:1px solid #e5e7eb;border-left:4px solid {color};border-radius:10px;'
                f'padding:12px 14px;margin-bottom:10px;">'
                f'<div style="font-weight:600;">{esc(c.marker)} '
                f'<span style="color:{color};">{esc(c.value)}{unit} ({status})</span> '
                f'<span style="color:#9ca3af;font-weight:400;font-size:12px;">ref {esc(c.reference_range)}</span></div>'
            )
            if c.note:
                parts.append(f'<div style="color:#374151;margin-top:4px;font-size:14px;">{esc(c.note)}</div>')
            if c.questions:
                items = "".join(f"<li>{esc(q)}</li>" for q in c.questions)
                parts.append('<div style="margin-top:6px;font-size:13px;color:#6b7280;">'
                             f'Questions for your doctor:<ul style="margin:4px 0 0;padding-left:18px;">{items}</ul></div>')
            parts.append("</div>")

        if interp.patterns:
            parts.append('<h4 style="margin:14px 0 6px;">Patterns across markers</h4>')
            for p in interp.patterns:
                parts.append('<div style="background:#eff6ff;border-radius:8px;padding:10px 12px;margin-bottom:8px;'
                             f'font-size:14px;"><b>{esc(p.name)}</b> — {esc(p.note)}</div>')
        parts.append(f'<div style="color:#6b7280;font-size:13px;margin-top:6px;">'
                     f'{interp.normal_count} other recognized markers were within range.</div>')

    parts.append(f'<div style="margin-top:14px;padding-top:10px;border-top:1px solid #eee;'
                 f'color:#9ca3af;font-size:12px;">{esc(interp.disclaimer)}</div>')
    parts.append("</div>")
    return "".join(parts)
