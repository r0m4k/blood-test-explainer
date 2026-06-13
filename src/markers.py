"""Canonical lab-marker reference.

Single source of truth shared by the synthetic-data generator, the evaluation harness,
and the interpretation knowledge base. Reference ranges are adult, general-population defaults
for synthetic-data generation and flag computation only.

CBC marker intervals match `kb/cbc_knowledge_graph.json` → `statistics_per_group_age.adult`
(the age-only fallback used when patient sex is unknown). Sex/age-specific ranges live in the
JSON graph and take precedence in the report pipeline when patient context is available.
These values are for an educational tool, not diagnosis.

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


# ~100 of the most common markers across CBC, metabolic, lipid, thyroid, coagulation, hormones, and vitamins.
MARKERS: tuple[Marker, ...] = (
    # --- Complete blood count ---
    Marker("Hemoglobin", "g/dL", 11.9, 17.7, "CBC", "oxygen-carrying protein in red blood cells", ("Hgb", "HGB", "Hb")),
    Marker("Hematocrit", "%", 35, 52, "CBC", "fraction of blood made up of red cells", ("Hct", "HCT", "PCV")),
    Marker("White Blood Cell Count", "10^3/uL", 3.7, 10.5, "CBC", "immune cells that fight infection", ("WBC", "Leukocytes", "WBC Count", "TLC", "Total Leucocyte Count")),
    Marker("Platelet Count", "10^3/uL", 150, 400, "CBC", "cell fragments that help blood clot", ("Platelets", "PLT")),
    Marker("Red Blood Cell Count", "10^6/uL", 4.0, 6.2, "CBC", "number of oxygen-carrying red cells", ("RBC", "Erythrocytes")),
    Marker("MCV", "fL", 82, 99, "CBC", "average size of red blood cells", ("Mean Corpuscular Volume",)),
    Marker("MCH", "pg", 25, 35, "CBC", "average hemoglobin per red blood cell", ("Mean Corpuscular Hemoglobin",)),
    Marker("MCHC", "g/dL", 32, 36, "CBC", "average hemoglobin concentration in red cells", ("Mean Corpuscular Hemoglobin Concentration",)),
    Marker("RDW", "%", 9.0, 14.5, "CBC", "variation in red blood cell size", ("Red Cell Distribution Width",)),
    Marker("MPV", "fL", 7.5, 11.5, "CBC", "average size of platelets", ("Mean Platelet Volume",)),
    Marker("Absolute Neutrophil Count", "10^3/uL", 1.8, 7.7, "CBC", "count of infection-fighting white cells", ("ANC", "Neutrophils Absolute", "Abs Neutrophils")),
    Marker("Absolute Lymphocyte Count", "10^3/uL", 0.875, 4.8, "CBC", "count of adaptive immune white cells", ("ALC", "Lymphocytes Absolute", "Abs Lymphocytes")),
    Marker("Absolute Monocyte Count", "10^3/uL", 0.2, 0.8, "CBC", "count of cleanup and immune white cells", ("AMC", "Monocytes Absolute", "Abs Monocytes")),
    Marker("Absolute Eosinophil Count", "10^3/uL", 0, 0.5, "CBC", "count of allergy and parasite-related white cells", ("AEC", "Eosinophils Absolute", "Abs Eosinophils")),
    Marker("Absolute Basophil Count", "10^3/uL", 0, 0.2, "CBC", "count of histamine-related white cells", ("ABC", "Basophils Absolute", "Abs Basophils")),
    Marker("Reticulocyte Count", "%", 0.5, 2.5, "CBC", "young red cells recently released from bone marrow", ("Retic Count", "Retics")),
    # --- Metabolic panel ---
    Marker("Glucose", "mg/dL", 70, 99, "Metabolic", "blood sugar level", ("Fasting Glucose", "GLU", "Blood Sugar", "FBS", "RBS", "Fasting Blood Sugar")),
    Marker("Creatinine", "mg/dL", 0.7, 1.3, "Metabolic", "kidney-function waste product", ("Cr", "Serum Creatinine")),
    Marker("eGFR", "mL/min/1.73m2", 90, None, "Metabolic", "estimated kidney filtration rate", ("GFR", "Estimated GFR")),
    Marker("Blood Urea Nitrogen", "mg/dL", 7, 20, "Metabolic", "kidney-function waste product", ("BUN", "Urea Nitrogen")),
    Marker("Sodium", "mmol/L", 135, 145, "Metabolic", "key electrolyte for fluid balance", ("Na", "Na+")),
    Marker("Potassium", "mmol/L", 3.5, 5.1, "Metabolic", "electrolyte vital for heart and muscle", ("K", "K+")),
    Marker("Chloride", "mmol/L", 98, 107, "Metabolic", "electrolyte for fluid and acid balance", ("Cl", "Cl-")),
    Marker("Calcium", "mg/dL", 8.6, 10.3, "Metabolic", "mineral for bones, nerves, and muscle", ("Ca", "Total Calcium")),
    Marker("Albumin", "g/dL", 3.5, 5.0, "Metabolic", "main protein made by the liver", ("ALB",)),
    Marker("Total Protein", "g/dL", 6.0, 8.3, "Metabolic", "total of all blood proteins", ("TP", "Protein, Total")),
    Marker("Globulin", "g/dL", 2.0, 3.5, "Metabolic", "non-albumin blood proteins including antibodies", ("Globulins",)),
    Marker("Bicarbonate", "mmol/L", 22, 28, "Metabolic", "main blood buffer for acid-base balance", ("CO2", "Bicarb", "Total CO2", "Carbon Dioxide")),
    Marker("Anion Gap", "mEq/L", 7, 13, "Metabolic", "calculated gap from electrolytes suggesting acid-base issues", ("AG",)),
    Marker("Magnesium", "mg/dL", 1.7, 2.2, "Metabolic", "electrolyte for nerves, muscle, and heart rhythm", ("Mg", "Mg++")),
    Marker("Phosphate", "mg/dL", 2.5, 4.5, "Metabolic", "mineral for bones, energy, and cell membranes", ("Phosphorus", "PO4", "Inorganic Phosphate")),
    Marker("Uric Acid", "mg/dL", 3.5, 7.2, "Metabolic", "breakdown product of purines; linked to gout", ("UA", "Urate")),
    Marker("Serum Iron", "mcg/dL", 60, 170, "Metabolic", "circulating iron available for red-cell production", ("Iron", "Fe", "Iron, Serum")),
    Marker("TIBC", "mcg/dL", 250, 450, "Metabolic", "blood's capacity to bind and transport iron", ("Total Iron Binding Capacity", "Iron Binding Capacity")),
    Marker("Transferrin Saturation", "%", 20, 50, "Metabolic", "percent of iron-binding sites occupied", ("TSAT", "Iron Saturation")),
    Marker("LDH", "U/L", 140, 280, "Metabolic", "enzyme released when cells are damaged", ("Lactate Dehydrogenase",)),
    Marker("Osmolality", "mOsm/kg", 275, 295, "Metabolic", "concentration of particles in the blood", ("Serum Osmolality",)),
    Marker("Ammonia", "mcg/dL", 15, 45, "Metabolic", "waste product processed by the liver", ("NH3", "Blood Ammonia")),
    Marker("Lactate", "mmol/L", 0.5, 2.0, "Metabolic", "byproduct of anaerobic metabolism", ("Lactic Acid", "Lactate, Blood")),
    Marker("Homocysteine", "umol/L", 5, 15, "Metabolic", "amino acid linked to B-vitamin status and vascular risk", ("Hcy",)),
    Marker("Cystatin C", "mg/L", 0.53, 0.95, "Metabolic", "kidney-function marker less affected by muscle mass", ("CysC",)),
    Marker("Prealbumin", "mg/dL", 20, 40, "Metabolic", "short-lived protein reflecting recent nutrition", ("Transthyretin",)),
    Marker("Beta-2 Microglobulin", "mg/L", 0.7, 1.8, "Metabolic", "small protein from cell turnover; kidney and immune marker", ("B2M", "β2-Microglobulin")),
    # --- Liver enzymes ---
    Marker("ALT", "U/L", 7, 56, "Liver", "liver enzyme released when liver cells are stressed", ("Alanine Aminotransferase", "SGPT")),
    Marker("AST", "U/L", 10, 40, "Liver", "enzyme from liver and muscle cells", ("Aspartate Aminotransferase", "SGOT")),
    Marker("ALP", "U/L", 44, 147, "Liver", "enzyme from liver and bone", ("Alkaline Phosphatase",)),
    Marker("GGT", "U/L", 9, 48, "Liver", "liver enzyme sensitive to bile and alcohol", ("Gamma-Glutamyl Transferase", "Gamma GT")),
    Marker("Total Bilirubin", "mg/dL", 0.1, 1.2, "Liver", "pigment from red-cell breakdown", ("Bilirubin, Total", "TBIL")),
    Marker("Direct Bilirubin", "mg/dL", 0, 0.3, "Liver", "conjugated bilirubin processed by the liver", ("Conjugated Bilirubin", "DBIL")),
    Marker("Lipase", "U/L", 0, 160, "Liver", "pancreatic enzyme for fat digestion", ("LPS",)),
    Marker("Amylase", "U/L", 25, 125, "Liver", "pancreatic and salivary enzyme for starch digestion", ("AMS",)),
    # --- Lipid panel ---
    Marker("Total Cholesterol", "mg/dL", None, 200, "Lipid", "total cholesterol in the blood", ("Cholesterol, Total", "TC")),
    Marker("LDL Cholesterol", "mg/dL", None, 100, "Lipid", "'bad' cholesterol that builds in arteries", ("LDL", "LDL-C")),
    Marker("HDL Cholesterol", "mg/dL", 40, None, "Lipid", "'good' cholesterol that clears arteries", ("HDL", "HDL-C")),
    Marker("Triglycerides", "mg/dL", None, 150, "Lipid", "fat circulating in the blood", ("TG", "Trig")),
    Marker("Non-HDL Cholesterol", "mg/dL", None, 130, "Lipid", "all cholesterol except HDL; atherogenic fraction", ("Non-HDL-C", "Non HDL Cholesterol")),
    Marker("Apolipoprotein B", "mg/dL", None, 90, "Lipid", "protein on LDL and related particles", ("Apo B", "ApoB")),
    Marker("Apolipoprotein A-1", "mg/dL", 120, None, "Lipid", "main protein on HDL particles", ("Apo A-1", "ApoA1")),
    Marker("Lipoprotein(a)", "mg/dL", None, 30, "Lipid", "genetically influenced LDL-like particle", ("Lp(a)", "Lipoprotein a")),
    # --- Thyroid ---
    Marker("TSH", "mIU/L", 0.4, 4.0, "Thyroid", "pituitary signal that controls the thyroid", ("Thyroid Stimulating Hormone",)),
    Marker("Free T4", "ng/dL", 0.8, 1.8, "Thyroid", "active thyroid hormone, free fraction", ("FT4", "Free Thyroxine")),
    Marker("Free T3", "pg/mL", 2.3, 4.2, "Thyroid", "active thyroid hormone, free fraction", ("FT3", "Free Triiodothyronine")),
    Marker("Total T4", "mcg/dL", 4.5, 12.0, "Thyroid", "total thyroxine including bound and free", ("T4", "Thyroxine")),
    Marker("Total T3", "ng/dL", 80, 200, "Thyroid", "total triiodothyronine including bound and free", ("T3", "Triiodothyronine")),
    Marker("Anti-TPO Antibodies", "IU/mL", None, 35, "Thyroid", "antibodies against thyroid peroxidase", ("TPO Antibodies", "Thyroid Peroxidase Antibodies", "Anti-TPO")),
    # --- Vitamins / iron ---
    Marker("Vitamin D", "ng/mL", 30, 100, "Vitamin", "vitamin for bone and immune health", ("25-OH Vitamin D", "25-Hydroxyvitamin D", "Vit D")),
    Marker("Vitamin B12", "pg/mL", 200, 900, "Vitamin", "vitamin for nerves and red-cell production", ("B12", "Cobalamin")),
    Marker("Ferritin", "ng/mL", 30, 400, "Vitamin", "stored-iron protein", ("FERR",)),
    Marker("HbA1c", "%", 4.0, 5.6, "Metabolic", "average blood sugar over ~3 months", ("A1c", "Hemoglobin A1c", "Glycated Hemoglobin")),
    # --- Coagulation ---
    Marker("Prothrombin Time", "seconds", 11, 13.5, "Coagulation", "time for the clotting cascade to form fibrin", ("PT",)),
    Marker("INR", "ratio", 0.9, 1.1, "Coagulation", "standardized prothrombin time for warfarin monitoring", ("International Normalized Ratio",)),
    Marker("aPTT", "seconds", 25, 35, "Coagulation", "time for the intrinsic clotting pathway", ("PTT", "APTT", "Activated Partial Thromboplastin Time")),
    Marker("Fibrinogen", "mg/dL", 200, 400, "Coagulation", "clotting protein and acute-phase reactant", ("Factor I",)),
    Marker("D-Dimer", "ng/mL", None, 500, "Coagulation", "breakdown product of clots; elevated when clotting is active", ("D Dimer",)),
    # --- Inflammation / immune ---
    Marker("C-Reactive Protein", "mg/L", None, 10, "Inflammation", "general marker of inflammation", ("CRP",)),
    Marker("hs-CRP", "mg/L", None, 3.0, "Inflammation", "high-sensitivity CRP for cardiovascular risk", ("High-Sensitivity CRP", "High Sensitivity C-Reactive Protein")),
    Marker("ESR", "mm/hr", 0, 20, "Inflammation", "rate red cells settle; nonspecific inflammation marker", ("Erythrocyte Sedimentation Rate", "Sed Rate")),
    Marker("Procalcitonin", "ng/mL", None, 0.1, "Inflammation", "marker that rises with bacterial infection", ("PCT",)),
    Marker("Complement C3", "mg/dL", 90, 180, "Inflammation", "complement protein in immune activation", ("C3",)),
    Marker("Complement C4", "mg/dL", 10, 40, "Inflammation", "complement protein in immune activation", ("C4",)),
    Marker("Rheumatoid Factor", "IU/mL", None, 14, "Inflammation", "antibody sometimes seen in autoimmune arthritis", ("RF",)),
    # --- Cardiac ---
    Marker("BNP", "pg/mL", None, 100, "Cardiac", "hormone released when the heart is stretched", ("B-Type Natriuretic Peptide", "Brain Natriuretic Peptide")),
    Marker("Troponin I", "ng/mL", None, 0.04, "Cardiac", "heart-muscle protein released with injury", ("TnI", "High-Sensitivity Troponin I")),
    Marker("Creatine Kinase", "U/L", 30, 200, "Cardiac", "enzyme from muscle including heart and skeletal", ("CK", "CPK", "Creatine Phosphokinase")),
    Marker("CK-MB", "ng/mL", None, 5, "Cardiac", "heart-enriched fraction of creatine kinase", ("CKMB", "Creatine Kinase-MB")),
    # --- Hormones ---
    Marker("Cortisol", "mcg/dL", 6, 18, "Hormone", "stress hormone from the adrenal glands", ("AM Cortisol", "Serum Cortisol")),
    Marker("Insulin", "uIU/mL", 2.6, 24.9, "Hormone", "hormone that lowers blood sugar", ("Fasting Insulin",)),
    Marker("Testosterone", "ng/dL", 300, 1000, "Hormone", "androgen sex hormone", ("Total Testosterone",)),
    Marker("Estradiol", "pg/mL", 15, 350, "Hormone", "primary estrogen sex hormone", ("E2", "Estrogen")),
    Marker("Prolactin", "ng/mL", None, 20, "Hormone", "pituitary hormone for lactation and more", ("PRL",)),
    Marker("FSH", "mIU/mL", 1.5, 12.4, "Hormone", "pituitary signal for egg and sperm production", ("Follicle Stimulating Hormone",)),
    Marker("LH", "mIU/mL", 1.5, 9.3, "Hormone", "pituitary signal for ovulation and testosterone", ("Luteinizing Hormone",)),
    Marker("Progesterone", "ng/mL", 0.2, 25, "Hormone", "hormone that supports the uterine lining", ("P4",)),
    Marker("Parathyroid Hormone", "pg/mL", 15, 65, "Hormone", "hormone that regulates blood calcium", ("PTH", "Intact PTH")),
    Marker("ACTH", "pg/mL", 7, 63, "Hormone", "pituitary signal that drives cortisol production", ("Adrenocorticotropic Hormone",)),
    Marker("SHBG", "nmol/L", 10, 80, "Hormone", "protein that binds sex hormones in the blood", ("Sex Hormone Binding Globulin",)),
    Marker("IGF-1", "ng/mL", 115, 355, "Hormone", "growth factor reflecting growth-hormone activity", ("Insulin-Like Growth Factor 1", "Somatomedin C")),
    # --- Oncology / screening ---
    Marker("PSA", "ng/mL", None, 4.0, "Oncology", "prostate-specific protein used in screening", ("Prostate Specific Antigen",)),
    # --- Vitamins / iron (continued) ---
    Marker("Folate", "ng/mL", 3, 20, "Vitamin", "B vitamin needed for DNA and red-cell production", ("Folic Acid", "Serum Folate")),
    Marker("Vitamin A", "mcg/dL", 30, 65, "Vitamin", "fat-soluble vitamin for vision and immunity", ("Retinol",)),
)


# Lab qualifiers we strip when matching ("Serum Sodium" == "Sodium", "Total WBC Count" == "WBC").
_QUALIFIERS = frozenset((
    "serum", "plasma", "blood", "total", "count", "counts", "level", "levels",
    "estimation", "absolute", "fasting", "random", "s", "p", "the",
))


def _normalize(name: str) -> str:
    """Collapse a printed marker name to a comparable core: drop parentheticals + punctuation,
    normalise British spelling, remove lab qualifiers, and sort tokens (word order varies)."""
    s = name.casefold().strip()
    s = re.sub(r"\([^)]*\)", " ", s)                         # drop parentheticals
    s = s.replace("haemo", "hemo").replace("haema", "hema")  # British -> US
    s = s.replace("leuco", "leuko").replace("oe", "e")
    s = re.sub(r"[^a-z0-9 ]", " ", s)                        # punctuation -> space
    tokens = sorted(t for t in s.split() if t and t not in _QUALIFIERS)
    return " ".join(tokens)


# Fast lookups: exact (casefolded) and normalized.
_LOOKUP: dict[str, Marker] = {}
_NORM_LOOKUP: dict[str, Marker] = {}
for _m in MARKERS:
    _LOOKUP[_m.name.casefold()] = _m
    _NORM_LOOKUP.setdefault(_normalize(_m.name), _m)
    for _a in _m.aliases:
        _LOOKUP.setdefault(_a.casefold(), _m)
        _NORM_LOOKUP.setdefault(_normalize(_a), _m)


def _resolve_one(name: str) -> Marker | None:
    key = name.strip().casefold()
    if not key:
        return None
    if key in _LOOKUP:
        return _LOOKUP[key]
    m = re.search(r"\(([^)]*)\)", key)
    if m:
        outer = re.sub(r"\([^)]*\)", "", key).strip()
        inner = m.group(1).strip()
        for cand in (outer, inner):
            if cand in _LOOKUP:
                return _LOOKUP[cand]
    norm = _normalize(name)
    if norm and norm in _NORM_LOOKUP:
        return _NORM_LOOKUP[norm]
    return None


def resolve(name: str) -> Marker | None:
    """Match an extracted marker name (canonical/alias/variant) to a known Marker.

    Handles real-report variety: exact name, the text inside/outside parentheses, a normalized
    form that ignores lab qualifiers (Serum/Total/Count/…), punctuation, word order, and British
    spelling, and slash/comma-joined names like "PCV / Hematocrit" or "Total WBC Count / TLC".
    """
    if not name:
        return None
    for cand in [name, *re.split(r"[/,;|]", name)]:
        marker = _resolve_one(cand)
        if marker is not None:
            return marker
    return None
