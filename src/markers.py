"""Canonical lab-marker reference.

Single source of truth shared by the synthetic-data generator, the evaluation harness,
and (later) the interpretation knowledge base. Reference ranges are adult, general-population
defaults for synthetic-data generation and flag computation only; the production KB will carry
sex/age-specific ranges with citations. These values are for an educational tool, not diagnosis.

Each marker: canonical name, common aliases (for matching extracted text), unit, an adult
reference interval, a category, and a one-line "what it measures".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Marker:
    name: str
    unit: str
    ref_low: float | None
    ref_high: float | None
    category: str
    measures: str
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def status_for(self, value: float) -> str:
        if self.ref_low is not None and value < self.ref_low:
            return "low"
        if self.ref_high is not None and value > self.ref_high:
            return "high"
        return "normal"

    def ref_range_text(self) -> str:
        if self.ref_low is not None and self.ref_high is not None:
            return f"{_fmt(self.ref_low)} - {_fmt(self.ref_high)}"
        if self.ref_high is not None:
            return f"< {_fmt(self.ref_high)}"
        if self.ref_low is not None:
            return f"> {_fmt(self.ref_low)}"
        return ""


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


# ~30 of the most common markers across CBC, metabolic, lipid, thyroid, and vitamins.
MARKERS: tuple[Marker, ...] = (
    # --- Complete blood count ---
    Marker("Hemoglobin", "g/dL", 13.5, 17.5, "CBC", "oxygen-carrying protein in red blood cells", ("Hgb", "HGB", "Hb")),
    Marker("Hematocrit", "%", 38.8, 50.0, "CBC", "fraction of blood made up of red cells", ("Hct", "HCT", "PCV")),
    Marker("White Blood Cell Count", "10^3/uL", 4.5, 11.0, "CBC", "immune cells that fight infection", ("WBC", "Leukocytes", "WBC Count")),
    Marker("Platelet Count", "10^3/uL", 150, 400, "CBC", "cell fragments that help blood clot", ("Platelets", "PLT")),
    Marker("Red Blood Cell Count", "10^6/uL", 4.5, 5.9, "CBC", "number of oxygen-carrying red cells", ("RBC", "Erythrocytes")),
    Marker("MCV", "fL", 80, 100, "CBC", "average size of red blood cells", ("Mean Corpuscular Volume",)),
    # --- Metabolic panel ---
    Marker("Glucose", "mg/dL", 70, 99, "Metabolic", "blood sugar level", ("Fasting Glucose", "GLU", "Blood Sugar")),
    Marker("Creatinine", "mg/dL", 0.7, 1.3, "Metabolic", "kidney-function waste product", ("Cr", "Serum Creatinine")),
    Marker("eGFR", "mL/min/1.73m2", 90, None, "Metabolic", "estimated kidney filtration rate", ("GFR", "Estimated GFR")),
    Marker("Blood Urea Nitrogen", "mg/dL", 7, 20, "Metabolic", "kidney-function waste product", ("BUN", "Urea Nitrogen")),
    Marker("Sodium", "mmol/L", 135, 145, "Metabolic", "key electrolyte for fluid balance", ("Na", "Na+")),
    Marker("Potassium", "mmol/L", 3.5, 5.1, "Metabolic", "electrolyte vital for heart and muscle", ("K", "K+")),
    Marker("Chloride", "mmol/L", 98, 107, "Metabolic", "electrolyte for fluid and acid balance", ("Cl", "Cl-")),
    Marker("Calcium", "mg/dL", 8.6, 10.3, "Metabolic", "mineral for bones, nerves, and muscle", ("Ca", "Total Calcium")),
    Marker("Albumin", "g/dL", 3.5, 5.0, "Metabolic", "main protein made by the liver", ("ALB",)),
    Marker("Total Protein", "g/dL", 6.0, 8.3, "Metabolic", "total of all blood proteins", ("TP", "Protein, Total")),
    # --- Liver enzymes ---
    Marker("ALT", "U/L", 7, 56, "Liver", "liver enzyme released when liver cells are stressed", ("Alanine Aminotransferase", "SGPT")),
    Marker("AST", "U/L", 10, 40, "Liver", "enzyme from liver and muscle cells", ("Aspartate Aminotransferase", "SGOT")),
    Marker("ALP", "U/L", 44, 147, "Liver", "enzyme from liver and bone", ("Alkaline Phosphatase",)),
    Marker("GGT", "U/L", 9, 48, "Liver", "liver enzyme sensitive to bile and alcohol", ("Gamma-Glutamyl Transferase", "Gamma GT")),
    Marker("Total Bilirubin", "mg/dL", 0.1, 1.2, "Liver", "pigment from red-cell breakdown", ("Bilirubin, Total", "TBIL")),
    # --- Lipid panel ---
    Marker("Total Cholesterol", "mg/dL", None, 200, "Lipid", "total cholesterol in the blood", ("Cholesterol, Total", "TC")),
    Marker("LDL Cholesterol", "mg/dL", None, 100, "Lipid", "'bad' cholesterol that builds in arteries", ("LDL", "LDL-C")),
    Marker("HDL Cholesterol", "mg/dL", 40, None, "Lipid", "'good' cholesterol that clears arteries", ("HDL", "HDL-C")),
    Marker("Triglycerides", "mg/dL", None, 150, "Lipid", "fat circulating in the blood", ("TG", "Trig")),
    # --- Thyroid ---
    Marker("TSH", "mIU/L", 0.4, 4.0, "Thyroid", "pituitary signal that controls the thyroid", ("Thyroid Stimulating Hormone",)),
    Marker("Free T4", "ng/dL", 0.8, 1.8, "Thyroid", "active thyroid hormone, free fraction", ("FT4", "Free Thyroxine")),
    # --- Vitamins / iron ---
    Marker("Vitamin D", "ng/mL", 30, 100, "Vitamin", "vitamin for bone and immune health", ("25-OH Vitamin D", "25-Hydroxyvitamin D", "Vit D")),
    Marker("Vitamin B12", "pg/mL", 200, 900, "Vitamin", "vitamin for nerves and red-cell production", ("B12", "Cobalamin")),
    Marker("Ferritin", "ng/mL", 30, 400, "Vitamin", "stored-iron protein", ("FERR",)),
    Marker("HbA1c", "%", 4.0, 5.6, "Metabolic", "average blood sugar over ~3 months", ("A1c", "Hemoglobin A1c", "Glycated Hemoglobin")),
)


# Fast lookup: any alias or canonical name (casefolded) -> Marker.
_LOOKUP: dict[str, Marker] = {}
for _m in MARKERS:
    _LOOKUP[_m.name.casefold()] = _m
    for _a in _m.aliases:
        _LOOKUP.setdefault(_a.casefold(), _m)


def resolve(name: str) -> Marker | None:
    """Match an extracted marker name (canonical or alias) to a known Marker.

    Real reports print verbose names like "Packed Cell Volume (PCV)" or "Hemoglobin (HB/Hgb)".
    We try the exact name, then the text outside the parentheses, then the abbreviation inside,
    so both the canonical form and the lab's variant resolve to the same marker.
    """
    if not name:
        return None
    key = name.strip().casefold()
    if key in _LOOKUP:
        return _LOOKUP[key]
    m = re.search(r"\(([^)]*)\)", key)
    if m:
        outer = re.sub(r"\([^)]*\)", "", key).strip()
        inner = m.group(1).strip()
        for cand in (outer, inner):
            if cand in _LOOKUP:
                return _LOOKUP[cand]
    return None
