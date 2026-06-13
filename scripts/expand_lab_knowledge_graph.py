#!/usr/bin/env python3
"""Expand the lab knowledge graph to cover all canonical markers and attach video URLs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KG_PATH = ROOT / "kb" / "cbc_knowledge_graph.json"
VIDEO_PATH = ROOT / "kb" / "marker_videos.json"

# Canonical marker name -> knowledge-graph test id.
MARKER_IDS: dict[str, str] = {
    "Hemoglobin": "hemoglobin",
    "Hematocrit": "hct",
    "White Blood Cell Count": "wbc",
    "Platelet Count": "plt",
    "Red Blood Cell Count": "rbc",
    "MCV": "mcv",
    "MCH": "mch",
    "MCHC": "mchc",
    "RDW": "rdw_cv",
    "MPV": "mpv",
    "Absolute Neutrophil Count": "neu_absolute",
    "Absolute Lymphocyte Count": "lym_absolute",
    "Absolute Monocyte Count": "mon_absolute",
    "Absolute Eosinophil Count": "eos_absolute",
    "Absolute Basophil Count": "bas_absolute",
    "Band Neutrophils Percent": "band_neutrophils_percent",
    "Reticulocyte Count": "reticulocyte_count",
    "Haptoglobin": "haptoglobin",
    "G6PD": "g6pd",
    "Erythropoietin": "erythropoietin",
    "Glucose": "glucose",
    "Creatinine": "creatinine",
    "eGFR": "egfr",
    "Blood Urea Nitrogen": "bun",
    "Sodium": "sodium",
    "Potassium": "potassium",
    "Chloride": "chloride",
    "Calcium": "calcium",
    "Albumin": "albumin",
    "Total Protein": "total_protein",
    "Globulin": "globulin",
    "Bicarbonate": "bicarbonate",
    "Anion Gap": "anion_gap",
    "Magnesium": "magnesium",
    "Phosphate": "phosphate",
    "Uric Acid": "uric_acid",
    "Serum Iron": "serum_iron",
    "TIBC": "tibc",
    "Transferrin": "transferrin",
    "Transferrin Saturation": "transferrin_saturation",
    "LDH": "ldh",
    "Osmolality": "osmolality",
    "Ammonia": "ammonia",
    "Lactate": "lactate",
    "Homocysteine": "homocysteine",
    "Methylmalonic Acid": "methylmalonic_acid",
    "Cystatin C": "cystatin_c",
    "Prealbumin": "prealbumin",
    "Beta-2 Microglobulin": "beta_2_microglobulin",
    "C-Peptide": "c_peptide",
    "Fructosamine": "fructosamine",
    "Beta-Hydroxybutyrate": "beta_hydroxybutyrate",
    "HbA1c": "hba1c",
    "ALT": "alt",
    "AST": "ast",
    "ALP": "alp",
    "GGT": "ggt",
    "Total Bilirubin": "total_bilirubin",
    "Direct Bilirubin": "direct_bilirubin",
    "Lipase": "lipase",
    "Amylase": "amylase",
    "Total Cholesterol": "total_cholesterol",
    "LDL Cholesterol": "ldl_cholesterol",
    "HDL Cholesterol": "hdl_cholesterol",
    "Triglycerides": "triglycerides",
    "Non-HDL Cholesterol": "non_hdl_cholesterol",
    "Apolipoprotein B": "apolipoprotein_b",
    "Apolipoprotein A-1": "apolipoprotein_a1",
    "Lipoprotein(a)": "lipoprotein_a",
    "TSH": "tsh",
    "Free T4": "free_t4",
    "Free T3": "free_t3",
    "Total T4": "total_t4",
    "Total T3": "total_t3",
    "Anti-TPO Antibodies": "anti_tpo_antibodies",
    "TSH Receptor Antibodies": "tsh_receptor_antibodies",
    "Thyroglobulin Antibodies": "thyroglobulin_antibodies",
    "Vitamin D": "vitamin_d",
    "Vitamin B12": "vitamin_b12",
    "Ferritin": "ferritin",
    "Zinc": "zinc",
    "Copper": "copper",
    "Ceruloplasmin": "ceruloplasmin",
    "Selenium": "selenium",
    "Vitamin E": "vitamin_e",
    "Coenzyme Q10": "coenzyme_q10",
    "Prothrombin Time": "prothrombin_time",
    "INR": "inr",
    "aPTT": "aptt",
    "Fibrinogen": "fibrinogen",
    "D-Dimer": "d_dimer",
    "C-Reactive Protein": "crp",
    "hs-CRP": "hs_crp",
    "ESR": "esr",
    "Procalcitonin": "procalcitonin",
    "Complement C3": "complement_c3",
    "Complement C4": "complement_c4",
    "Rheumatoid Factor": "rheumatoid_factor",
    "Anti-CCP Antibodies": "anti_ccp_antibodies",
    "Immunoglobulin G": "immunoglobulin_g",
    "Immunoglobulin A": "immunoglobulin_a",
    "Immunoglobulin M": "immunoglobulin_m",
    "Immunoglobulin E": "immunoglobulin_e",
    "BNP": "bnp",
    "NT-proBNP": "nt_probnp",
    "Troponin I": "troponin_i",
    "Creatine Kinase": "creatine_kinase",
    "CK-MB": "ck_mb",
    "Myoglobin": "myoglobin",
    "Cortisol": "cortisol",
    "Insulin": "insulin",
    "Testosterone": "testosterone",
    "Free Testosterone": "free_testosterone",
    "Estradiol": "estradiol",
    "Prolactin": "prolactin",
    "FSH": "fsh",
    "LH": "lh",
    "Progesterone": "progesterone",
    "Parathyroid Hormone": "pth",
    "ACTH": "acth",
    "DHEA-S": "dhea_s",
    "Androstenedione": "androstenedione",
    "Anti-Mullerian Hormone": "anti_mullerian_hormone",
    "Beta-hCG": "beta_hcg",
    "SHBG": "shbg",
    "IGF-1": "igf_1",
    "IGF Binding Protein-3": "igfbp_3",
    "PSA": "psa",
    "CEA": "cea",
    "CA-125": "ca_125",
    "CA 19-9": "ca_19_9",
    "Alpha-Fetoprotein": "alpha_fetoprotein",
    "CA 15-3": "ca_15_3",
    "Folate": "folate",
    "Vitamin A": "vitamin_a",
}

EXTRA_ALIAS_UPDATES: dict[str, list[str]] = {
    "gra_absolute": ["Absolute Granulocyte Count", "Granulocytes Absolute", "Abs Granulocytes"],
    "neu_absolute": ["ANC", "Absolute Neutrophil Count", "Abs Neutrophils", "Neutrophils Absolute"],
    "mon_absolute": ["AMC", "Absolute Monocyte Count", "Abs Monocytes", "Monocytes Absolute"],
    "eos_absolute": ["AEC", "Absolute Eosinophil Count", "Abs Eosinophils", "Eosinophils Absolute"],
    "bas_absolute": ["ABC", "Absolute Basophil Count", "Abs Basophils", "Basophils Absolute"],
    "band_neutrophils_percent": ["Band Neutrophil %", "Band Neutrophils", "Band %", "Bands", "Stab Neutrophils"],
    "mpv": ["Mean Platelet Volume"],
    "reticulocyte_count": ["Retic Count", "Retics"],
    "bun": ["BUN", "Urea Nitrogen", "Blood Urea Nitrogen"],
    "egfr": ["GFR", "Estimated GFR"],
    "hba1c": ["A1c", "HgbA1C", "Hemoglobin A1c", "Hemoglobin A1C", "Glycated Hemoglobin"],
    "aptt": ["PTT", "APTT", "Activated Partial Thromboplastin Time"],
    "pth": ["PTH", "Intact PTH", "Parathyroid Hormone"],
    "hs_crp": ["High-Sensitivity CRP", "High Sensitivity C-Reactive Protein"],
    "d_dimer": ["D Dimer"],
}

CATEGORY_GUIDANCE: dict[str, dict[str, list[str]]] = {
    "CBC": {
        "food": [
            "Support healthy blood production with balanced protein, iron, B12, folate, and vitamin C from whole foods.",
            "Stay well hydrated unless a clinician advises fluid restriction.",
        ],
        "exercises": [
            "Use moderate activity as tolerated when blood counts are stable.",
            "Avoid intense training if anemia, infection, or bleeding symptoms are present until evaluated.",
        ],
        "supplements": [
            "Discuss iron, B12, or folate supplementation only when testing supports a deficiency.",
            "Do not start blood-building supplements without clinician guidance.",
        ],
    },
    "Metabolic": {
        "food": [
            "Favor minimally processed meals with vegetables, lean protein, whole grains, and healthy fats.",
            "Limit excess added sugar, alcohol, and high-sodium ultra-processed foods when relevant to the marker.",
        ],
        "exercises": [
            "Aim for regular aerobic activity and resistance training if cleared by a clinician.",
            "Match hydration and recovery to kidney, glucose, or electrolyte concerns.",
        ],
        "supplements": [
            "Use electrolyte, vitamin, or mineral supplements only when labs or diet indicate a need.",
            "Review medications and supplements with a clinician because they can shift metabolic markers.",
        ],
    },
    "Liver": {
        "food": [
            "Limit alcohol and avoid unnecessary hepatotoxic exposures when liver enzymes are abnormal.",
            "Choose balanced meals with vegetables, fiber, and moderate healthy fats.",
        ],
        "exercises": [
            "Stay active within symptom limits; avoid heavy alcohol-related training recovery patterns.",
            "Seek medical review before intense exercise if jaundice or severe abdominal pain is present.",
        ],
        "supplements": [
            "Avoid unverified liver detox products.",
            "Discuss medication, herb, and supplement use because many affect liver tests.",
        ],
    },
    "Lipid": {
        "food": [
            "Emphasize fiber-rich plants, fish, legumes, nuts, and unsaturated fats.",
            "Reduce trans fats, excess saturated fat, and refined carbohydrates when triglycerides or LDL are high.",
        ],
        "exercises": [
            "Use regular aerobic and resistance exercise to support lipid and cardiovascular health.",
            "Maintain consistency rather than extreme short-term training bursts.",
        ],
        "supplements": [
            "Discuss statins, fibrates, omega-3 prescriptions, or other lipid therapies with a clinician.",
            "Do not rely on unproven supplement cocktails for cholesterol management.",
        ],
    },
    "Thyroid": {
        "food": [
            "Ensure adequate iodine and selenium from a balanced diet unless a clinician advises otherwise.",
            "Keep a stable diet around thyroid testing when possible.",
        ],
        "exercises": [
            "Match activity to thyroid symptoms such as fatigue, palpitations, or heat intolerance.",
            "Build up gradually after thyroid treatment changes.",
        ],
        "supplements": [
            "Avoid starting high-dose iodine or thyroid-support supplements without medical supervision.",
            "Take prescribed thyroid medication consistently and separately from interfering foods or supplements.",
        ],
    },
    "Vitamin": {
        "food": [
            "Correct deficiencies first with food sources such as fortified grains, dairy, eggs, fish, legumes, and leafy greens.",
            "Pair nutrient-dense meals with safe sun exposure for vitamin D when appropriate.",
        ],
        "exercises": [
            "Use weight-bearing and muscle-strengthening activity to support bone and metabolic health.",
            "Adjust activity if deficiency symptoms such as fatigue or neuropathy are present.",
        ],
        "supplements": [
            "Supplement only after testing confirms deficiency or insufficiency.",
            "Use clinician-guided dosing, especially for iron, vitamin A, and fat-soluble vitamins.",
        ],
    },
    "Coagulation": {
        "food": [
            "Maintain consistent vitamin K intake if on warfarin, rather than large day-to-day swings.",
            "Use a balanced diet unless anticoagulation counseling specifies otherwise.",
        ],
        "exercises": [
            "Stay active, but use contact-sport caution when bleeding risk is elevated.",
            "Seek urgent care for unexplained bruising, bleeding, or clot symptoms.",
        ],
        "supplements": [
            "Avoid starting aspirin, fish oil, or herbals that affect clotting without clinician review.",
            "Take anticoagulants exactly as prescribed and monitor INR/PT when required.",
        ],
    },
    "Inflammation": {
        "food": [
            "Use an anti-inflammatory dietary pattern rich in vegetables, fruit, legumes, and omega-3 sources.",
            "Limit excess alcohol and ultra-processed foods when inflammation markers are high.",
        ],
        "exercises": [
            "Use regular moderate exercise, which can lower chronic inflammation over time.",
            "Rest during acute infection or inflammatory flares as advised by a clinician.",
        ],
        "supplements": [
            "Treat the underlying cause rather than relying on generic anti-inflammatory supplements.",
            "Discuss persistent abnormal inflammatory markers with a clinician.",
        ],
    },
    "Cardiac": {
        "food": [
            "Follow a heart-healthy diet low in excess sodium and harmful fats when cardiac markers are abnormal.",
            "Limit alcohol and manage blood pressure, glucose, and lipids together.",
        ],
        "exercises": [
            "Use clinician-approved cardiac rehabilitation or gradual aerobic training when safe.",
            "Seek emergency care for chest pain, severe shortness of breath, or syncope.",
        ],
        "supplements": [
            "Do not self-treat suspected heart injury with supplements.",
            "Take prescribed cardiac medications consistently and review interactions.",
        ],
    },
    "Hormone": {
        "food": [
            "Support hormone health with adequate protein, healthy fats, fiber, and micronutrients.",
            "Avoid extreme dieting or rapid weight changes unless medically supervised.",
        ],
        "exercises": [
            "Use resistance training and sleep regularity to support hormonal balance.",
            "Adjust training load during symptomatic hormone disorders.",
        ],
        "supplements": [
            "Avoid unsupervised hormone-boosting products.",
            "Use prescribed hormone therapies only under endocrine or reproductive specialist guidance.",
        ],
    },
    "Oncology": {
        "food": [
            "Follow a balanced, nutrient-dense diet unless oncology care provides specific restrictions.",
            "Limit charred processed meats and excess alcohol when discussing cancer screening markers.",
        ],
        "exercises": [
            "Stay physically active within the limits of prostate or oncology follow-up plans.",
            "Report new urinary, bone, or systemic symptoms promptly.",
        ],
        "supplements": [
            "Do not use high-dose supplements to try to normalize screening markers without specialist input.",
            "Discuss PSA changes with a clinician rather than self-interpreting a single value.",
        ],
    },
}

SEX_SIGNIFICANCE_HIGH = {
    "hemoglobin",
    "rbc",
    "hct",
    "esr",
}

SEX_LOW = {
    "level": "low",
    "summary": "This marker is usually interpreted with age and lab method rather than sex-specific reference intervals.",
    "pipeline_guidance": "Use the age-group interval unless the lab report provides a sex-specific range.",
}

SEX_HIGH_TEMPLATE = {
    "level": "high",
    "summary": "Reference intervals for this marker can differ by sex after puberty.",
    "pipeline_guidance": "Prefer sex-specific lab ranges when available and include clinician context when sex is unknown.",
}


def _round_mid(lo: float, hi: float) -> float:
    return round((lo + hi) / 2, 2)


def _stats_block(lo: float | None, hi: float | None) -> dict[str, dict[str, float]]:
    if lo is None and hi is None:
        lo, hi = 0.0, 1.0
    elif lo is None and hi is not None:
        lo = max(0.0, hi * 0.5)
    elif hi is None and lo is not None:
        hi = lo * 1.5 if lo > 0 else lo + 1.0
    assert lo is not None and hi is not None
    block = {
        "minimal_value": lo,
        "normal_value": _round_mid(lo, hi),
        "maximum_value": hi,
    }
    return {group: dict(block) for group in ("child", "teenager", "adult", "elder")}


def _why_important(name: str, kb_entry: Any) -> str:
    if kb_entry is None:
        return f"Abnormal {name} values can be clinically meaningful and should be interpreted with symptoms, history, and related tests."
    parts = []
    if kb_entry.high:
        parts.append(kb_entry.high)
    if kb_entry.low:
        parts.append(kb_entry.low)
    return " ".join(parts) if parts else f"{name} helps clinicians evaluate related organ systems and disease patterns."


def _build_test(marker: Any, test_id: str, video_url: str, kb_entry: Any) -> dict[str, Any]:
    aliases = list(dict.fromkeys([*marker.aliases]))
    guidance = CATEGORY_GUIDANCE.get(marker.category, CATEGORY_GUIDANCE["Metabolic"])
    sex = SEX_HIGH_TEMPLATE if test_id in SEX_SIGNIFICANCE_HIGH else SEX_LOW
    return {
        "id": test_id,
        "display_name": marker.name,
        "aliases": aliases,
        "category": marker.category,
        "unit": marker.unit,
        "description": f"{marker.name} measures {marker.measures}.",
        "why_important": _why_important(marker.name, kb_entry),
        "sex_significance": dict(sex),
        "instructions_to_improve": {
            "food": list(guidance["food"]),
            "exercises": list(guidance["exercises"]),
            "supplements": list(guidance["supplements"]),
        },
        "statistics_per_group_age": _stats_block(marker.ref_low, marker.ref_high),
        "related_tests": [],
        "source_ids": [],
        "video_url": video_url,
    }


def _merge_aliases(test: dict[str, Any], extra: list[str]) -> None:
    aliases = list(test.get("aliases") or [])
    seen = {a.casefold() for a in aliases}
    for alias in extra:
        if alias.casefold() not in seen:
            aliases.append(alias)
            seen.add(alias.casefold())
    test["aliases"] = aliases


def _remove_aliases(test: dict[str, Any], stale: set[str]) -> None:
    test["aliases"] = [alias for alias in test.get("aliases", []) if alias.casefold() not in stale]


def _refresh_generated_fields(test: dict[str, Any], marker: Any, test_id: str, video_url: str, kb_entry: Any) -> None:
    rebuilt = _build_test(marker, test_id, video_url, kb_entry)
    for key in (
        "display_name",
        "category",
        "unit",
        "description",
        "why_important",
        "sex_significance",
        "instructions_to_improve",
        "statistics_per_group_age",
        "video_url",
    ):
        test[key] = rebuilt[key]
    _merge_aliases(test, rebuilt["aliases"])


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from kb.knowledge_base import KB
    from src.markers import MARKERS

    payload = json.loads(KG_PATH.read_text(encoding="utf-8"))
    video_catalog = json.loads(VIDEO_PATH.read_text(encoding="utf-8"))
    videos: dict[str, str] = video_catalog["videos"]

    existing_by_id = {test["id"]: test for test in payload["tests"]}
    preserved_ids = set(existing_by_id)

    for marker in MARKERS:
        test_id = MARKER_IDS[marker.name]
        video_url = videos.get(test_id, video_catalog.get("default_video_url", ""))
        kb_entry = KB.get(marker.name)
        if test_id in existing_by_id:
            test = existing_by_id[test_id]
            _refresh_generated_fields(test, marker, test_id, video_url, kb_entry)
            _merge_aliases(test, list(marker.aliases))
            continue
        existing_by_id[test_id] = _build_test(marker, test_id, video_url, kb_entry)

    # Keep legacy CBC-only nodes that are not in MARKERS but still useful.
    for legacy_id in ("rdw_sd", "neu_percent", "lym_percent", "mon_percent", "eos_percent", "bas_percent", "gra_absolute"):
        if legacy_id in existing_by_id:
            existing_by_id[legacy_id]["video_url"] = videos.get(legacy_id, existing_by_id[legacy_id].get("video_url", ""))
            _merge_aliases(existing_by_id[legacy_id], EXTRA_ALIAS_UPDATES.get(legacy_id, []))
            if legacy_id == "gra_absolute":
                _remove_aliases(
                    existing_by_id[legacy_id],
                    {
                        "anc",
                        "anc when neutrophil-dominant",
                        "absolute neutrophil count",
                        "abs neutrophils",
                        "neutrophils absolute",
                    },
                )

    for test_id, extra_aliases in EXTRA_ALIAS_UPDATES.items():
        if test_id in existing_by_id:
            _merge_aliases(existing_by_id[test_id], extra_aliases)

    # Add absolute differential markers missing from the legacy graph.
    for marker in MARKERS:
        test_id = MARKER_IDS[marker.name]
        if test_id in preserved_ids or test_id not in {
            "neu_absolute",
            "mon_absolute",
            "eos_absolute",
            "bas_absolute",
            "mpv",
            "reticulocyte_count",
        }:
            continue
        if test_id not in existing_by_id:
            existing_by_id[test_id] = _build_test(
                marker,
                test_id,
                videos.get(test_id, ""),
                KB.get(marker.name),
            )

    ordered_ids = sorted(existing_by_id.keys())
    payload["schema_version"] = "2.0"
    payload["title"] = "Lab Marker Knowledge Graph"
    payload["purpose"] = (
        "Educational knowledge graph for common laboratory markers used by Blood Test Explainer. "
        "It supports explanation agents, not diagnosis or treatment."
    )
    payload["video_url_policy"] = video_catalog.get("description", payload.get("video_url_policy", ""))
    payload["video_catalog_path"] = "kb/marker_videos.json"
    payload["video_notes"] = video_catalog.get("notes", {})
    payload["tests"] = [existing_by_id[test_id] for test_id in ordered_ids]

    KG_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(payload['tests'])} tests to {KG_PATH}")


if __name__ == "__main__":
    main()
