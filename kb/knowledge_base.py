"""Grounded interpretation knowledge base (Phase 3).

The model does NOT invent medical facts. It extracts values (vision) and then *phrases* the facts
stored here. Every interpretation the app shows is grounded in this file, which is in turn based on
the reference material under kb/references/ (psap_reference_values.pdf, nbme_reference_values.pdf)
and standard general-population clinical references.

This is an EDUCATIONAL tool, not a diagnosis. Entries describe common, well-established
associations only ("a high value may be associated with ..."), never a diagnosis or treatment.
Canonical marker names + reference ranges live in src/markers.py (single source of truth).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.markers import Marker, resolve

DISCLAIMER = (
    "This is general educational information, not a medical diagnosis. Only a qualified clinician "
    "can interpret your results in the context of your history, symptoms, and other tests."
)


@dataclass(frozen=True)
class MarkerKB:
    """What a high/low value commonly relates to, plus questions to bring to a doctor."""

    high: str
    low: str
    questions: tuple[str, ...] = field(default_factory=tuple)


# Per-marker grounded facts. "" for a direction means it is not typically clinically flagged there
# (e.g. a low LDL is generally favorable). Keep statements associative and educational.
KB: dict[str, MarkerKB] = {
    # --- Complete blood count ---
    "Hemoglobin": MarkerKB(
        high="May be associated with dehydration, smoking, living at high altitude, or less commonly the body making too many red cells.",
        low="A low hemoglobin is the hallmark of anemia, which can stem from iron, B12 or folate deficiency, blood loss, or chronic disease.",
        questions=("Could this explain my tiredness or shortness of breath?", "Should we check iron, B12, or folate?"),
    ),
    "Hematocrit": MarkerKB(
        high="Often tracks with hemoglobin; may reflect dehydration or increased red-cell production.",
        low="Usually moves with hemoglobin and points toward anemia or recent blood loss.",
        questions=("Does this match my hemoglobin result?",),
    ),
    "White Blood Cell Count": MarkerKB(
        high="A high white count commonly accompanies infection or inflammation, physical stress, or certain medications; rarely it reflects a blood disorder.",
        low="A low white count can follow viral infections, some medications, or affect the immune system's ability to fight infection.",
        questions=("Could a recent infection explain this?", "Do we need to recheck it once I'm well?"),
    ),
    "Platelet Count": MarkerKB(
        high="May rise with inflammation, infection, iron deficiency, or after the spleen is removed.",
        low="A low platelet count can increase bruising or bleeding and may relate to infections, medications, or the bone marrow.",
        questions=("Should I avoid anything that thins my blood until this is checked?",),
    ),
    "Red Blood Cell Count": MarkerKB(
        high="May reflect dehydration, smoking, or increased red-cell production.",
        low="Often part of the anemia picture alongside low hemoglobin/hematocrit.",
        questions=("Does this fit with my hemoglobin and MCV?",),
    ),
    "MCV": MarkerKB(
        high="A high MCV (large red cells) is classically associated with B12 or folate deficiency, alcohol, or thyroid issues.",
        low="A low MCV (small red cells) is classically associated with iron deficiency or thalassemia.",
        questions=("Given my MCV, should we look for an iron or a B12 cause?",),
    ),
    # --- Metabolic panel ---
    "Glucose": MarkerKB(
        high="An elevated fasting glucose can indicate prediabetes or diabetes, or simply that the sample was not fasting.",
        low="A low glucose can cause shakiness or lightheadedness and may relate to fasting, medications, or how the sample was handled.",
        questions=("Was this a fasting sample?", "Should we confirm with an HbA1c?"),
    ),
    "Creatinine": MarkerKB(
        high="A high creatinine suggests the kidneys are filtering less efficiently; muscle mass, dehydration, and some medications also raise it.",
        low="A low creatinine is usually not a concern and can reflect lower muscle mass.",
        questions=("What does this mean for my kidney function (eGFR)?",),
    ),
    "eGFR": MarkerKB(
        high="A higher eGFR generally indicates better kidney filtration.",
        low="A low eGFR indicates reduced kidney filtration and is usually interpreted together with creatinine over time.",
        questions=("Is this a one-off or a trend?", "Should any of my medications be dose-adjusted?"),
    ),
    "Blood Urea Nitrogen": MarkerKB(
        high="BUN can rise with dehydration, a high-protein diet, or reduced kidney function.",
        low="A low BUN is rarely significant; it can reflect low protein intake or overhydration.",
        questions=("Does my BUN-to-creatinine ratio suggest dehydration?",),
    ),
    "Sodium": MarkerKB(
        high="High sodium usually reflects dehydration or fluid balance, not dietary salt alone.",
        low="Low sodium is one of the most common electrolyte abnormalities and relates to fluid balance, medications, or hormones.",
        questions=("Could my medications or fluid intake be affecting this?",),
    ),
    "Potassium": MarkerKB(
        high="High potassium can affect heart rhythm and may relate to kidney function or medications; it can also be falsely high from the blood draw.",
        low="Low potassium can cause muscle weakness or cramps and often relates to fluid loss or diuretics.",
        questions=("Should this be rechecked given how it affects the heart?",),
    ),
    "Chloride": MarkerKB(
        high="Usually moves with sodium and acid-base balance.",
        low="Usually moves with sodium and acid-base balance.",
        questions=("Does this fit with my sodium and bicarbonate?",),
    ),
    "Calcium": MarkerKB(
        high="High calcium can relate to the parathyroid glands, vitamin D, or other conditions and is worth following up.",
        low="Low calcium can relate to vitamin D, the parathyroid glands, or low albumin (which carries calcium).",
        questions=("Should we check vitamin D or parathyroid hormone?",),
    ),
    "Albumin": MarkerKB(
        high="A high albumin most often reflects dehydration.",
        low="A low albumin can reflect nutrition, inflammation, or liver/kidney conditions.",
        questions=("Could this be related to my diet or another result?",),
    ),
    "Total Protein": MarkerKB(
        high="May reflect dehydration or increased production of certain proteins.",
        low="May reflect nutrition, liver, or kidney factors.",
        questions=("Does this fit with my albumin level?",),
    ),
    # --- Liver enzymes ---
    "ALT": MarkerKB(
        high="ALT is fairly liver-specific; elevations can follow fatty liver, alcohol, medications, or viral hepatitis.",
        low="A low ALT is not typically a concern.",
        questions=("Could a medication or fatty liver explain this?", "Should we recheck in a few weeks?"),
    ),
    "AST": MarkerKB(
        high="AST rises with liver stress but also comes from muscle; the AST/ALT pattern helps point to a cause.",
        low="A low AST is not typically a concern.",
        questions=("Does the AST/ALT ratio suggest a specific cause?",),
    ),
    "ALP": MarkerKB(
        high="A high alkaline phosphatase can come from the liver/bile ducts or from bone; growth and pregnancy also raise it.",
        low="A low ALP is uncommon and rarely significant.",
        questions=("Is this from my liver or my bones?",),
    ),
    "GGT": MarkerKB(
        high="GGT is sensitive to bile-duct issues and alcohol; it helps clarify whether a high ALP is from the liver.",
        low="A low GGT is not a concern.",
        questions=("Does my GGT help explain my ALP?",),
    ),
    "Total Bilirubin": MarkerKB(
        high="A mildly high bilirubin is often benign (e.g. Gilbert's syndrome) but can relate to the liver or red-cell breakdown.",
        low="A low bilirubin is not a concern.",
        questions=("Is this mild and stable, or does it need follow-up?",),
    ),
    # --- Lipid panel ---
    "Total Cholesterol": MarkerKB(
        high="A high total cholesterol contributes to cardiovascular risk and is best read alongside LDL, HDL, and your overall risk.",
        low="A low total cholesterol is generally not a concern.",
        questions=("What is my overall cardiovascular risk?", "Diet/lifestyle first, or is medication warranted?"),
    ),
    "LDL Cholesterol": MarkerKB(
        high="LDL ('bad' cholesterol) is the main driver of plaque buildup; targets depend on your personal risk.",
        low="A low LDL is generally favorable.",
        questions=("What LDL target is right for my risk level?",),
    ),
    "HDL Cholesterol": MarkerKB(
        high="A higher HDL ('good' cholesterol) is generally protective.",
        low="A low HDL is associated with higher cardiovascular risk; exercise and not smoking help raise it.",
        questions=("Would lifestyle changes help raise my HDL?",),
    ),
    "Triglycerides": MarkerKB(
        high="High triglycerides relate to diet, alcohol, weight, and blood-sugar control, and add to cardiovascular risk.",
        low="A low triglyceride level is generally not a concern.",
        questions=("Was this fasting?", "Would diet changes help?"),
    ),
    # --- Thyroid ---
    "TSH": MarkerKB(
        high="A high TSH usually signals an underactive thyroid (the body asking for more hormone).",
        low="A low TSH usually signals an overactive thyroid.",
        questions=("Should we confirm with a Free T4?", "Could symptoms like fatigue or weight change relate to this?"),
    ),
    "Free T4": MarkerKB(
        high="A high Free T4 supports an overactive thyroid picture.",
        low="A low Free T4 supports an underactive thyroid picture.",
        questions=("How does this fit with my TSH?",),
    ),
    # --- Vitamins / iron ---
    "Vitamin D": MarkerKB(
        high="A very high vitamin D is uncommon and usually from supplements.",
        low="Low vitamin D is common and relates to bone and immune health; sunlight and diet/supplements affect it.",
        questions=("Should I supplement, and at what dose?",),
    ),
    "Vitamin B12": MarkerKB(
        high="A high B12 is usually from supplements and rarely a concern on its own.",
        low="Low B12 can cause fatigue, nerve symptoms, and large red cells (high MCV); diet and absorption matter.",
        questions=("Could this explain my tiredness or tingling?", "Do we need to check absorption?"),
    ),
    "Ferritin": MarkerKB(
        high="Ferritin rises with inflammation as well as iron overload, so a high value is read in context.",
        low="A low ferritin is the most specific sign of low iron stores and a common cause of anemia.",
        questions=("Does my ferritin explain my hemoglobin and MCV?", "Should we look for a source of iron loss?"),
    ),
    "HbA1c": MarkerKB(
        high="A high HbA1c reflects higher average blood sugar over ~3 months and is used to screen for prediabetes and diabetes.",
        low="A low HbA1c is generally not a concern.",
        questions=("Am I in the prediabetes range?", "What changes would lower this?"),
    ),
}


@dataclass(frozen=True)
class Pattern:
    """A cross-marker pattern the interpretation layer can surface (Phase 3.3)."""

    name: str
    when: str  # human-readable trigger description (logic lives in the interpretation module)
    note: str


# Cross-marker patterns: clusters that mean more together than any single value.
PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        "Anemia picture",
        "low Hemoglobin with low Hematocrit (and often low RBC)",
        "These low red-cell values together suggest anemia; MCV and ferritin help point to the cause (iron vs B12/folate).",
    ),
    Pattern(
        "Iron-deficiency pattern",
        "low Ferritin with low MCV (microcytic) and low Hemoglobin",
        "Small red cells plus low iron stores are a classic iron-deficiency pattern worth discussing with a clinician.",
    ),
    Pattern(
        "B12/folate pattern",
        "high MCV (macrocytic) with low Vitamin B12",
        "Large red cells alongside low B12 point toward a B12 or folate cause rather than iron.",
    ),
    Pattern(
        "Liver cluster",
        "elevated ALT and/or AST, sometimes with ALP and GGT",
        "Several liver enzymes elevated together raise the question of liver stress; the AST/ALT and ALP/GGT patterns help localize it.",
    ),
    Pattern(
        "Lipid / cardiovascular risk",
        "high LDL or Triglycerides with low HDL",
        "This lipid combination raises cardiovascular risk and is best interpreted with your overall risk profile.",
    ),
    Pattern(
        "Kidney-function pattern",
        "high Creatinine with low eGFR (and sometimes high BUN)",
        "Together these suggest reduced kidney filtration; trend over time matters more than a single reading.",
    ),
    Pattern(
        "Thyroid pattern",
        "high TSH with low Free T4 (underactive) or low TSH with high Free T4 (overactive)",
        "TSH and Free T4 read together indicate whether the thyroid is under- or over-active.",
    ),
    Pattern(
        "Glycemic pattern",
        "high Glucose with high HbA1c",
        "A high spot glucose backed by a high HbA1c is a stronger signal of impaired blood-sugar control than either alone.",
    ),
)


def interpret(marker_name: str, status: str) -> str | None:
    """Return the grounded educational note for a marker at a given status, or None.

    `status` is one of 'high', 'low', 'normal' (as produced by Marker.status_for / extraction).
    Returns None when there is nothing flagged to say (normal, or no benign-direction note).
    """
    marker = resolve(marker_name)
    if marker is None:
        return None
    entry = KB.get(marker.name)
    if entry is None:
        return None
    if status == "high":
        return entry.high or None
    if status == "low":
        return entry.low or None
    return None


def questions_for(marker_name: str) -> tuple[str, ...]:
    """Doctor-questions to surface for a flagged marker (Phase 3.4)."""
    marker = resolve(marker_name)
    if marker is None:
        return ()
    entry = KB.get(marker.name)
    return entry.questions if entry else ()


def coverage() -> tuple[int, int]:
    """(markers with KB entries, total canonical markers) — for a quick completeness check."""
    from src.markers import MARKERS

    covered = sum(1 for m in MARKERS if m.name in KB)
    return covered, len(MARKERS)
