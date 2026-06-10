#!/usr/bin/env python3
"""Synthetic lab-report generator → (image, ground-truth JSON) pairs.

This is the training/eval data for fine-tuning MiniCPM-V on extraction. We render realistic,
*format-diverse* lab reports with randomized layouts (column order, fonts, striping, borders,
lab branding) and known values, so every sample has perfect labels and we control diversity.
No real patient data is ever involved.

Each sample → a PNG plus a JSONL row:
    {"image": "images/000123.png",
     "tests": [{"marker","value","unit","reference_range","status","source_text","confidence"}],
     "notes": []}

Usage:
    python train/synth_reports.py --n 2000 --out train/data/synth
    python train/synth_reports.py --n 30 --out eval/data/synth_eval   # smaller held-out set
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.markers import MARKERS, Marker  # noqa: E402

LAB_NAMES = [
    "Northbridge Clinical Labs", "Meridian Diagnostics", "CityHealth Pathology",
    "Aurora Medical Laboratory", "BlueRiver Labs", "Helix Diagnostics", "Summit Pathology",
]
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _round_for(marker: Marker, value: float) -> float:
    hi = marker.ref_high if marker.ref_high is not None else (marker.ref_low or 100)
    if hi >= 100:
        return round(value)
    if hi >= 10:
        return round(value, 1)
    return round(value, 2)


def sample_value(rng: random.Random, marker: Marker, abnormal_p: float = 0.4) -> float:
    """Sample a plausible value; ~abnormal_p of the time push it out of range."""
    lo, hi = marker.ref_low, marker.ref_high
    abnormal = rng.random() < abnormal_p

    if lo is not None and hi is not None:
        if not abnormal:
            v = rng.uniform(lo, hi)
        elif rng.random() < 0.5:
            v = rng.uniform(lo * 0.5, lo * 0.95)
        else:
            v = rng.uniform(hi * 1.05, hi * 1.6)
    elif hi is not None:  # upper-bounded (LDL, Total chol, Triglycerides)
        v = rng.uniform(hi * 0.4, hi * 0.95) if not abnormal else rng.uniform(hi * 1.05, hi * 1.8)
    elif lo is not None:  # lower-bounded (HDL, eGFR)
        v = rng.uniform(lo * 1.05, lo * 1.8) if not abnormal else rng.uniform(lo * 0.4, lo * 0.95)
    else:
        v = rng.uniform(1, 100)
    return _round_for(marker, max(v, 0))


# Layout presets: column order + whether a flag column is shown.
_LAYOUTS = [
    {"cols": ["test", "result", "unit", "ref", "flag"]},
    {"cols": ["test", "result", "unit", "ref"]},
    {"cols": ["test", "result", "ref", "flag"]},       # unit folded into result
    {"cols": ["test", "flag", "result", "unit", "ref"]},
    {"cols": ["test", "result", "ref"]},               # minimal, no flag
    {"cols": ["test", "result", "unit", "flag", "ref"]},
]
_COL_LABEL = {"test": "Test", "result": "Result", "unit": "Units",
              "ref": "Reference Range", "flag": "Flag"}
_FLAG_TEXT = {"low": "L", "high": "H", "normal": ""}

# Palette themes: (header-band fill, accent text color) — diversity across "labs".
_THEMES = [
    ((243, 246, 250), (20, 28, 40)),
    ((237, 244, 238), (24, 54, 36)),
    ((245, 240, 248), (52, 28, 60)),
    ((250, 245, 238), (70, 44, 16)),
    ((255, 255, 255), (15, 15, 15)),   # plain / scanned look
]
_PANEL_TITLES = [
    "Comprehensive Metabolic & Hematology Panel", "Laboratory Report",
    "Blood Test Results", "Clinical Chemistry & CBC", "Pathology Report",
]
# Section headers (by category) are decorations the model must NOT extract as markers — this is
# exactly the real-report failure where "BLOOD INDICES" got read as a marker.
_SECTION_LABEL = {
    "CBC": "Complete Blood Count (CBC)", "Metabolic": "Metabolic Panel",
    "Liver": "Liver Function Tests", "Lipid": "Lipid Profile",
    "Thyroid": "Thyroid Function", "Vitamin": "Vitamins & Iron Studies",
}


def _fmt_ref(marker: Marker, rng: random.Random) -> str:
    """Reference ranges as real labs print them (varied separators / bounds / brackets)."""
    lo, hi = marker.ref_low, marker.ref_high
    if lo is not None and hi is not None:
        sep = rng.choice([" - ", "-", " – ", " to "])
        s = f"{_fmt_num(lo)}{sep}{_fmt_num(hi)}"
        return f"[{s}]" if rng.random() < 0.12 else s
    if hi is not None:
        return rng.choice([f"< {_fmt_num(hi)}", f"<{_fmt_num(hi)}", f"Up to {_fmt_num(hi)}", f"0 - {_fmt_num(hi)}"])
    if lo is not None:
        return rng.choice([f"> {_fmt_num(lo)}", f">{_fmt_num(lo)}", f">= {_fmt_num(lo)}"])
    return ""


def _demo_line(rng: random.Random) -> str:
    age, sex = rng.randint(19, 84), rng.choice(["Male", "Female", "M", "F"])
    return rng.choice([
        f"Patient: [SAMPLE]    Age/Sex: {age}/{sex}    Collected: 2026-03-14",
        f"Name: [SAMPLE]   Age: {age} Years   Sex: {sex}",
        f"[SAMPLE] · {age}{'M' if sex in ('Male', 'M') else 'F'} · Specimen: Serum",
        "Patient: [SAMPLE]    DOB: [SAMPLE]    Collected: 2026-03-14",
    ])


def _make_report(rng: random.Random) -> tuple[list[dict], dict]:
    """Pick markers + values; return (gold tests, layout/style config)."""
    k = rng.randint(6, min(16, len(MARKERS)))
    chosen = rng.sample(MARKERS, k)
    tests = []
    for m in chosen:
        v = sample_value(rng, m)
        status = m.status_for(v)
        # Alias variety: sometimes use an alias as the printed name.
        printed = rng.choice((m.name,) + m.aliases) if (m.aliases and rng.random() < 0.45) else m.name
        tests.append({
            "marker": printed,
            "canonical": m.name,
            "category": m.category,
            "value": _round_for(m, v),
            "unit": m.unit,
            "reference_range": _fmt_ref(m, rng),
            "status": status,
        })
    style = {
        "layout": rng.choice(_LAYOUTS),
        "lab": rng.choice(LAB_NAMES),
        "stripe": rng.random() < 0.55,
        "grid": rng.random() < 0.5,
        "sectioned": rng.random() < 0.55,
        "theme": rng.choice(_THEMES),
        "title": rng.choice(_PANEL_TITLES),
        "demo": _demo_line(rng),
        "base": rng.randint(14, 18),
    }
    return tests, style


def _cell_text(col: str, t: dict, fold_unit: bool) -> str:
    if col == "test":
        return str(t["marker"])
    if col == "result":
        val = _fmt_num(t["value"])
        return f"{val} {t['unit']}" if fold_unit else val
    if col == "unit":
        return str(t["unit"])
    if col == "ref":
        return str(t["reference_range"])
    if col == "flag":
        return _FLAG_TEXT.get(t["status"], "")
    return ""


def _fmt_num(v) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:g}"


def _ordered_rows(tests: list[dict], sectioned: bool) -> list[tuple[str, object]]:
    """Display rows. When sectioned, group markers by category under a section header (a
    decoration that is NOT in the gold), teaching the model to skip such headers."""
    if not sectioned:
        return [("data", t) for t in tests]
    by_cat: dict[str, list[dict]] = {}
    for t in tests:
        by_cat.setdefault(t["category"], []).append(t)
    rows: list[tuple[str, object]] = []
    for cat, items in by_cat.items():
        rows.append(("section", _SECTION_LABEL.get(cat, cat)))
        rows.extend(("data", t) for t in items)
    return rows


def render(tests: list[dict], style: dict) -> tuple[Image.Image, list[dict]]:
    """Render the report to an image; return (image, gold tests with source_text)."""
    cols = list(style["layout"]["cols"])
    fold_unit = "unit" not in cols
    W = 1000
    pad = 48
    base = style["base"]
    band, accent = style["theme"]
    f_h1, f_h2, f_th, f_td = _font(30, True), _font(15), _font(14, True), _font(base)
    f_sec = _font(base + 1, True)

    # Column widths (proportional, tuned per column type).
    weight = {"test": 0.34, "result": 0.18, "unit": 0.14, "ref": 0.26, "flag": 0.08}
    avail = W - 2 * pad
    widths = {c: int(avail * weight[c]) for c in cols}
    scale = avail / sum(widths.values())
    widths = {c: int(w * scale) for c, w in widths.items()}

    display = _ordered_rows(tests, style.get("sectioned", False))
    row_h = base + 16
    header_h = 150
    table_top = header_h + 30
    H = table_top + row_h * (len(display) + 1) + pad

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    # Header band.
    d.rectangle([0, 0, W, header_h], fill=band)
    d.text((pad, 40), style["lab"], font=f_h1, fill=accent)
    d.text((pad, 88), style["demo"], font=f_h2, fill=(90, 100, 115))
    d.text((pad, 112), style["title"], font=f_h2, fill=(90, 100, 115))

    # Column header row.
    x = pad
    y = table_top
    for c in cols:
        d.text((x + 6, y + 2), _COL_LABEL[c], font=f_th, fill=(40, 50, 65))
        x += widths[c]
    d.line([pad, y + row_h - 4, W - pad, y + row_h - 4], fill=(190, 200, 212), width=2)

    # Rows (data + section decorations).
    gold = []
    for di, (kind, payload) in enumerate(display, start=1):
        y = table_top + row_h * di
        if kind == "section":
            d.rectangle([pad, y, W - pad, y + row_h], fill=(232, 237, 243))
            d.text((pad + 6, y + 3), str(payload), font=f_sec, fill=accent)
            continue
        t = payload
        if style["stripe"] and di % 2 == 1:
            d.rectangle([pad, y, W - pad, y + row_h], fill=(248, 250, 252))
        x = pad
        row_pieces = []
        for c in cols:
            text = _cell_text(c, t, fold_unit)
            color = (20, 28, 40)
            if c == "flag" and t["status"] in ("low", "high"):
                color = (170, 50, 50) if t["status"] == "high" else (150, 100, 0)
            d.text((x + 6, y + 3), text, font=f_td, fill=color)
            if text:
                row_pieces.append(text)
            x += widths[c]
        if style["grid"]:
            d.line([pad, y + row_h, W - pad, y + row_h], fill=(228, 233, 240), width=1)
        gold.append({
            "marker": t["canonical"],            # label = canonical name
            "value": _fmt_num(t["value"]),
            "unit": t["unit"],
            "reference_range": t["reference_range"],
            "status": t["status"],
            "source_text": "  ".join(row_pieces),
            "confidence": 1.0,
        })

    return img, gold


def generate(n: int, out_dir: Path, seed: int = 13) -> Path:
    rng = random.Random(seed)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.jsonl"

    with labels_path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            tests, style = _make_report(rng)
            img, gold = render(tests, style)
            rel = f"images/{i:06d}.png"
            img.save(out_dir / rel)
            fh.write(json.dumps({"image": rel, "tests": gold, "notes": []}, ensure_ascii=False) + "\n")
    return labels_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("train/data/synth"))
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    labels = generate(args.n, args.out, args.seed)
    print(f"Wrote {args.n} reports to {args.out} (labels: {labels})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
