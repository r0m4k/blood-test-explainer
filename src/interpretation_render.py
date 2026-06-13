"""Render cross-marker interpretation patterns for the health report UI."""

from __future__ import annotations

import html

from src.interpretation import build_interpretation


def patterns_html(tests: list[dict]) -> str:
    """Render ONLY the cross-marker patterns as an HTML block (empty string if none).

    Designed to sit alongside an existing per-marker report (e.g. the knowledge-graph health
    report) so it adds cross-marker reasoning — anemia picture, liver cluster, lipid risk — without
    duplicating per-marker interpretation.
    """
    interp = build_interpretation(tests)
    if not interp.patterns:
        return ""

    def esc(value: object) -> str:
        return html.escape(str(value))

    cards = "".join(
        '<div style="background:#eff6ff;border:1px solid #dbeafe;border-radius:10px;'
        'padding:11px 14px;margin-bottom:8px;font-size:14px;line-height:1.5;">'
        f'<b style="color:#1e3a8a;">{esc(p.name)}</b><br>'
        f'<span style="color:#374151;">{esc(p.note)}</span></div>'
        for p in interp.patterns
    )
    return (
        '<div style="font-family:system-ui,-apple-system,sans-serif;max-width:760px;margin-top:16px;">'
        '<h3 style="margin:0 0 2px;font-size:16px;">Patterns across your markers</h3>'
        '<div style="color:#6b7280;font-size:12px;margin-bottom:10px;">'
        'How several results may relate to one another — educational, not a diagnosis.</div>'
        f"{cards}"
        f'<div style="color:#9ca3af;font-size:11px;margin-top:4px;">{esc(interp.disclaimer)}</div>'
        "</div>"
    )
