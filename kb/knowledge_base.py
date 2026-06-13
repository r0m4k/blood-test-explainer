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
    "MCH": MarkerKB(
        high="A high MCH usually tracks with large red cells (high MCV) and similar causes such as B12 or folate deficiency.",
        low="A low MCH usually tracks with small red cells (low MCV) and points toward iron deficiency.",
        questions=("Does my MCH fit with my MCV and hemoglobin?",),
    ),
    "MCHC": MarkerKB(
        high="A high MCHC may reflect spherocytosis or dehydration-related concentration.",
        low="A low MCHC is common in iron-deficiency anemia where cells carry less hemoglobin.",
        questions=("Could iron studies explain my low MCHC?",),
    ),
    "RDW": MarkerKB(
        high="A high RDW means red cells vary more in size; it often appears early in iron, B12, or folate deficiency.",
        low="A low RDW is usually not clinically significant.",
        questions=("Does a high RDW suggest a mixed or early deficiency?",),
    ),
    "MPV": MarkerKB(
        high="A high MPV means larger platelets, which can appear when the marrow is making new platelets quickly.",
        low="A low MPV is usually not a concern on its own.",
        questions=("Does my MPV fit with my platelet count?",),
    ),
    "Absolute Neutrophil Count": MarkerKB(
        high="A high neutrophil count often accompanies bacterial infection, inflammation, or physical stress.",
        low="A low neutrophil count raises infection risk and may follow viruses, medications, or bone-marrow issues.",
        questions=("Could an infection explain my neutrophil count?", "If low, do I need extra precautions?"),
    ),
    "Absolute Lymphocyte Count": MarkerKB(
        high="A high lymphocyte count may follow viral infections or certain immune conditions.",
        low="A low lymphocyte count can follow stress, steroids, or immune conditions.",
        questions=("Was this drawn during or after an illness?",),
    ),
    "Absolute Monocyte Count": MarkerKB(
        high="A high monocyte count may appear during recovery from infection or with chronic inflammation.",
        low="A low monocyte count is rarely significant on its own.",
        questions=("Does this fit with a recent or ongoing infection?",),
    ),
    "Absolute Eosinophil Count": MarkerKB(
        high="A high eosinophil count is classically linked to allergies, asthma, or parasitic infection.",
        low="A low eosinophil count is usually not a concern.",
        questions=("Could allergies or asthma explain this?",),
    ),
    "Absolute Basophil Count": MarkerKB(
        high="A high basophil count is uncommon and may relate to allergy or certain blood disorders.",
        low="A low basophil count is usually not significant.",
        questions=("Is this a persistent finding worth rechecking?",),
    ),
    "Band Neutrophils Percent": MarkerKB(
        high="A high band neutrophil percentage may suggest a left shift, often seen when the marrow is releasing immature neutrophils during infection, inflammation, or physiologic stress.",
        low="A low band neutrophil percentage is usually expected and not clinically significant on its own.",
        questions=("Does this fit with my total WBC and absolute neutrophil count?",),
    ),
    "Reticulocyte Count": MarkerKB(
        high="A high reticulocyte count means the marrow is making extra red cells, often after blood loss or hemolysis.",
        low="A low reticulocyte count in anemia suggests the marrow is not keeping up with red-cell need.",
        questions=("Is my body replacing red cells appropriately?",),
    ),
    "Haptoglobin": MarkerKB(
        high="A high haptoglobin may rise with inflammation because it is an acute-phase protein.",
        low="A low haptoglobin supports red-cell breakdown when interpreted with LDH, bilirubin, and reticulocytes.",
        questions=("Does this fit with LDH, bilirubin, and reticulocyte count?",),
    ),
    "G6PD": MarkerKB(
        high="A high G6PD level is usually not clinically important by itself.",
        low="A low G6PD level can predispose red cells to hemolysis after certain infections, foods, or medications.",
        questions=("Should I avoid any oxidant medications or foods?",),
    ),
    "Erythropoietin": MarkerKB(
        high="A high erythropoietin may reflect the body's response to anemia or low oxygen levels.",
        low="A low or inappropriately normal erythropoietin can contribute to anemia in kidney disease.",
        questions=("Is this appropriate for my hemoglobin level?",),
    ),
    # --- Metabolic panel ---
    "Glucose": MarkerKB(
        high="An elevated fasting or random glucose can indicate prediabetes or diabetes, or simply that the sample timing and recent food intake matter.",
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
    "Globulin": MarkerKB(
        high="A high globulin may reflect increased antibodies from infection, inflammation, or immune conditions.",
        low="A low globulin may reflect reduced antibody production or liver disease.",
        questions=("Does the albumin/globulin ratio need follow-up?",),
    ),
    "Bicarbonate": MarkerKB(
        high="A high bicarbonate or total CO2 may reflect metabolic alkalosis or compensation for lung/respiratory issues.",
        low="A low bicarbonate or total CO2 may reflect metabolic acidosis, severe diarrhea, or other acid-base stress.",
        questions=("Does this fit with my other electrolytes and symptoms?",),
    ),
    "Anion Gap": MarkerKB(
        high="A high anion gap often points to acid buildup from ketones, lactate, toxins, or kidney failure.",
        low="A low anion gap is uncommon and usually less urgent.",
        questions=("Could this relate to dehydration, diabetes, or kidney function?",),
    ),
    "Magnesium": MarkerKB(
        high="A high magnesium is uncommon outside supplements or severe kidney impairment.",
        low="A low magnesium can cause cramps, tremor, or heart rhythm issues and often accompanies low potassium.",
        questions=("Should we recheck magnesium and potassium together?",),
    ),
    "Phosphate": MarkerKB(
        high="A high phosphate may relate to kidney disease, low parathyroid activity, or cell breakdown.",
        low="A low phosphate may relate to malnutrition, alcohol, or overcorrection of vitamin D.",
        questions=("Does this fit with my calcium, PTH, vitamin D, and kidney results?",),
    ),
    "Uric Acid": MarkerKB(
        high="A high uric acid is associated with gout and kidney stones and may rise with diet, alcohol, or kidney disease.",
        low="A low uric acid is usually not a concern.",
        questions=("Could diet or medications be contributing?",),
    ),
    "Serum Iron": MarkerKB(
        high="A high serum iron may reflect supplements, hemochromatosis, or recent infusion.",
        low="A low serum iron often accompanies iron-deficiency anemia but varies with recent meals and inflammation.",
        questions=("Should we interpret this with ferritin and TIBC?",),
    ),
    "TIBC": MarkerKB(
        high="A high TIBC often means the body is trying to bind more iron during iron deficiency.",
        low="A low TIBC may appear with inflammation or iron overload.",
        questions=("Does TIBC fit with my ferritin and transferrin saturation?",),
    ),
    "Transferrin": MarkerKB(
        high="A high transferrin often appears when the body is trying to transport more iron during iron deficiency.",
        low="A low transferrin may appear with inflammation, liver disease, malnutrition, or iron overload.",
        questions=("Does transferrin fit with ferritin, serum iron, and TIBC?",),
    ),
    "Transferrin Saturation": MarkerKB(
        high="A high saturation may suggest iron overload or excess intake.",
        low="A low saturation is common in iron deficiency.",
        questions=("Is this consistent with my ferritin and hemoglobin?",),
    ),
    "LDH": MarkerKB(
        high="LDH is a nonspecific marker of cell damage from hemolysis, liver injury, muscle injury, or malignancy.",
        low="A low LDH is not typically a concern.",
        questions=("What might be causing cell turnover or damage?",),
    ),
    "Osmolality": MarkerKB(
        high="A high osmolality may reflect dehydration, high blood sugar, or excess sodium.",
        low="A low osmolality may reflect overhydration or low sodium.",
        questions=("Does this match my sodium and glucose?",),
    ),
    "Ammonia": MarkerKB(
        high="A high ammonia may relate to liver disease and can affect mental status.",
        low="A low ammonia is not typically significant.",
        questions=("Should liver function be evaluated if ammonia is high?",),
    ),
    "Lactate": MarkerKB(
        high="A high lactate may reflect poor tissue oxygen delivery, sepsis, or strenuous exercise at draw time.",
        low="A low lactate is not a concern.",
        questions=("Was the sample handled promptly?", "Could infection or low blood pressure explain this?"),
    ),
    "Homocysteine": MarkerKB(
        high="A high homocysteine may relate to low B vitamins and is linked to vascular risk in research.",
        low="A low homocysteine is generally favorable.",
        questions=("Should we check B12 and folate?",),
    ),
    "Methylmalonic Acid": MarkerKB(
        high="A high methylmalonic acid supports functional vitamin B12 deficiency, especially when B12 is borderline.",
        low="A low methylmalonic acid is usually not a concern.",
        questions=("Does this clarify whether my B12 level is truly low?",),
    ),
    "Cystatin C": MarkerKB(
        high="A high cystatin C suggests reduced kidney filtration, similar to creatinine but less muscle-dependent.",
        low="A low cystatin C is usually not a concern.",
        questions=("How does this compare with my creatinine and eGFR?",),
    ),
    "Prealbumin": MarkerKB(
        high="A high prealbumin is uncommon and may reflect steroids or kidney loss of protein.",
        low="A low prealbumin may reflect recent poor nutrition or inflammation.",
        questions=("Could nutrition or inflammation be affecting this?",),
    ),
    "Beta-2 Microglobulin": MarkerKB(
        high="A high beta-2 microglobulin may reflect increased cell turnover, kidney impairment, or certain blood conditions.",
        low="A low beta-2 microglobulin is not typically significant.",
        questions=("Should this be interpreted with kidney function?",),
    ),
    "C-Peptide": MarkerKB(
        high="A high C-peptide means the pancreas is making more insulin and may appear with insulin resistance or insulin-producing tumors.",
        low="A low C-peptide suggests reduced natural insulin production, especially when glucose is high.",
        questions=("Was glucose measured at the same time?",),
    ),
    "Fructosamine": MarkerKB(
        high="A high fructosamine suggests higher average blood sugar over the prior two to three weeks.",
        low="A low fructosamine is usually not a concern but can be affected by low blood proteins.",
        questions=("Is this being used because HbA1c may be unreliable for me?",),
    ),
    "Beta-Hydroxybutyrate": MarkerKB(
        high="A high beta-hydroxybutyrate indicates increased ketone production and can be important in diabetes or prolonged fasting.",
        low="A low beta-hydroxybutyrate is expected when ketone production is not increased.",
        questions=("Could this relate to diabetes, fasting, or a low-carbohydrate diet?",),
    ),
    # --- Liver enzymes ---
    "ALT": MarkerKB(
        high="ALT is usually interpreted as a liver enzyme, but it can also rise with skeletal muscle injury; fatty liver, alcohol, medications, viral hepatitis, CK, AST, and GGT help provide context.",
        low="A low ALT is not typically a concern.",
        questions=("Could a medication or fatty liver explain this?", "Should we recheck in a few weeks?"),
    ),
    "AST": MarkerKB(
        high="AST rises with liver stress but also comes from skeletal muscle; CK and GGT can help separate muscle from liver or bile-duct patterns.",
        low="A low AST is not typically a concern.",
        questions=("Does the AST/ALT ratio suggest a specific cause?",),
    ),
    "ALP": MarkerKB(
        high="A high alkaline phosphatase can come from the liver/bile ducts or from bone; growth and pregnancy also raise it.",
        low="A low ALP is uncommon and rarely significant.",
        questions=("Is this from my liver or my bones, and does GGT help clarify it?",),
    ),
    "GGT": MarkerKB(
        high="GGT is sensitive to bile-duct issues and alcohol; it helps clarify whether a high ALP is liver-related, while a normal GGT with high ALP can point more toward bone or skeletal sources.",
        low="A low GGT is not a concern.",
        questions=("Does my GGT help explain my ALP?",),
    ),
    "Total Bilirubin": MarkerKB(
        high="A mildly high bilirubin is often benign (e.g. Gilbert's syndrome) but can relate to the liver or red-cell breakdown.",
        low="A low bilirubin is not a concern.",
        questions=("Is this mild and stable, or does it need follow-up?",),
    ),
    "Direct Bilirubin": MarkerKB(
        high="A high direct bilirubin suggests the liver or bile ducts are not processing bilirubin normally.",
        low="A low direct bilirubin is not a concern.",
        questions=("Does this point to a liver or bile-duct issue?",),
    ),
    "Lipase": MarkerKB(
        high="A high lipase commonly points to pancreatic inflammation but can rise with other abdominal conditions.",
        low="A low lipase is not typically significant.",
        questions=("Could abdominal pain relate to this lipase?",),
    ),
    "Amylase": MarkerKB(
        high="A high amylase may reflect pancreatic or salivary-gland inflammation.",
        low="A low amylase is rarely significant.",
        questions=("Should lipase be checked alongside amylase?",),
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
    "Non-HDL Cholesterol": MarkerKB(
        high="Non-HDL cholesterol captures all atherogenic particles and adds to cardiovascular risk.",
        low="A low non-HDL cholesterol is generally favorable.",
        questions=("What non-HDL target fits my overall risk?",),
    ),
    "Apolipoprotein B": MarkerKB(
        high="Apo B reflects the number of LDL-like particles and is linked to plaque risk.",
        low="A low Apo B is generally favorable.",
        questions=("How does Apo B compare with my LDL?",),
    ),
    "Apolipoprotein A-1": MarkerKB(
        high="A higher Apo A-1 is generally associated with more HDL and lower cardiovascular risk.",
        low="A low Apo A-1 may accompany low HDL and higher risk.",
        questions=("Would exercise or not smoking help raise HDL/Apo A-1?",),
    ),
    "Lipoprotein(a)": MarkerKB(
        high="Lp(a) is largely genetic and adds cardiovascular risk independent of LDL.",
        low="A low Lp(a) is generally favorable.",
        questions=("Does my family history fit with a high Lp(a)?",),
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
    "Free T3": MarkerKB(
        high="A high Free T3 supports an overactive thyroid picture.",
        low="A low Free T3 may appear in underactive thyroid or severe illness.",
        questions=("How does Free T3 fit with my TSH and Free T4?",),
    ),
    "Total T4": MarkerKB(
        high="A high Total T4 may reflect hyperthyroidism or high thyroid-binding proteins.",
        low="A low Total T4 may reflect hypothyroidism or binding-protein changes.",
        questions=("Should Free T4 be used for interpretation?",),
    ),
    "Total T3": MarkerKB(
        high="A high Total T3 may reflect hyperthyroidism.",
        low="A low Total T3 may appear in hypothyroidism or non-thyroidal illness.",
        questions=("Does this match my TSH and Free T3?",),
    ),
    "Anti-TPO Antibodies": MarkerKB(
        high="Anti-TPO antibodies suggest autoimmune thyroid disease such as Hashimoto's.",
        low="A negative or low Anti-TPO is expected in most people without autoimmune thyroid disease.",
        questions=("Could this explain my thyroid symptoms or TSH changes?",),
    ),
    "TSH Receptor Antibodies": MarkerKB(
        high="A high TSH receptor antibody result supports autoimmune Graves-type thyroid stimulation in the right thyroid pattern.",
        low="A low result makes TSH-receptor antibody activity less likely.",
        questions=("Does this fit with my TSH, Free T4, and Free T3?",),
    ),
    "Thyroglobulin Antibodies": MarkerKB(
        high="A high thyroglobulin antibody result may support autoimmune thyroid disease and can interfere with thyroglobulin monitoring.",
        low="A low result is expected when these antibodies are absent.",
        questions=("Could this interfere with thyroglobulin interpretation?",),
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
    "Zinc": MarkerKB(
        high="A high zinc level is uncommon and may reflect excess supplementation.",
        low="A low zinc level may relate to poor intake, malabsorption, inflammation, or increased losses.",
        questions=("Am I taking zinc or other mineral supplements?",),
    ),
    "Copper": MarkerKB(
        high="A high copper level can rise with inflammation, pregnancy, estrogen therapy, or copper disorders.",
        low="A low copper level may cause anemia or nerve problems and can occur with excess zinc intake.",
        questions=("Should copper be interpreted with ceruloplasmin and zinc?",),
    ),
    "Ceruloplasmin": MarkerKB(
        high="A high ceruloplasmin may rise with inflammation, pregnancy, or estrogen therapy.",
        low="A low ceruloplasmin can appear in copper deficiency or Wilson disease workups.",
        questions=("Does this fit with copper and liver tests?",),
    ),
    "Selenium": MarkerKB(
        high="A high selenium level is usually related to excess supplementation and can be toxic.",
        low="A low selenium level may reflect poor intake, malabsorption, or severe illness.",
        questions=("Am I taking selenium-containing supplements?",),
    ),
    "Vitamin E": MarkerKB(
        high="A high vitamin E level is usually supplement-related and may affect bleeding risk at high intakes.",
        low="A low vitamin E level may occur with fat malabsorption and can affect nerves and red cells.",
        questions=("Could fat absorption or supplements explain this?",),
    ),
    "Coenzyme Q10": MarkerKB(
        high="A high CoQ10 level is most often supplement-related and may also track with lipoprotein levels.",
        low="A low CoQ10 level may reflect nutritional status, medication context such as statins, or mitochondrial/neuromuscular evaluation context.",
        questions=("Am I taking CoQ10 or statin therapy?", "Is this being monitored for a mitochondrial or neuromuscular reason?"),
    ),
    "HbA1c": MarkerKB(
        high="A high HbA1c reflects higher average blood sugar over ~3 months and is used to screen for prediabetes and diabetes.",
        low="A low HbA1c is generally not a concern.",
        questions=("Am I in the prediabetes range?", "What changes would lower this?"),
    ),
    # --- Coagulation ---
    "Prothrombin Time": MarkerKB(
        high="A prolonged PT means clotting takes longer and may relate to warfarin, liver disease, or clotting-factor deficiency.",
        low="A shorter PT is usually not clinically flagged.",
        questions=("Am I on blood thinners?", "Should INR be checked instead?"),
    ),
    "INR": MarkerKB(
        high="A high INR means slower clotting; it is expected on warfarin but dangerous if unintentionally high.",
        low="A low INR on warfarin may mean under-anticoagulation.",
        questions=("What INR range am I aiming for?",),
    ),
    "aPTT": MarkerKB(
        high="A prolonged aPTT may relate to heparin, lupus anticoagulant, or clotting-factor deficiency.",
        low="A shorter aPTT is rarely flagged alone.",
        questions=("Am I on heparin or do I bruise easily?",),
    ),
    "Fibrinogen": MarkerKB(
        high="A high fibrinogen may reflect inflammation or increased clotting tendency.",
        low="A low fibrinogen may increase bleeding risk.",
        questions=("Could inflammation explain a high fibrinogen?",),
    ),
    "D-Dimer": MarkerKB(
        high="A high D-dimer suggests active clot breakdown but is nonspecific and rises with infection, surgery, or pregnancy.",
        low="A low D-dimer makes significant clotting less likely in the right clinical context.",
        questions=("Was this ordered because of leg pain or shortness of breath?",),
    ),
    # --- Inflammation / immune ---
    "C-Reactive Protein": MarkerKB(
        high="A high CRP indicates inflammation from infection, autoimmune disease, or tissue injury.",
        low="A low CRP is expected in the absence of significant inflammation.",
        questions=("Could a recent infection explain this?",),
    ),
    "hs-CRP": MarkerKB(
        high="An elevated hs-CRP adds to cardiovascular risk even when general CRP is low-grade.",
        low="A low hs-CRP is generally favorable for heart risk.",
        questions=("What lifestyle changes would lower my cardiovascular risk?",),
    ),
    "ESR": MarkerKB(
        high="A high ESR is a nonspecific sign of inflammation, infection, or autoimmune activity.",
        low="A low ESR is usually not a concern.",
        questions=("Does this fit with my symptoms or other inflammatory markers?",),
    ),
    "Procalcitonin": MarkerKB(
        high="A high procalcitonin more specifically suggests bacterial infection.",
        low="A low procalcitonin makes serious bacterial infection less likely.",
        questions=("Was this drawn during a fever or suspected infection?",),
    ),
    "Complement C3": MarkerKB(
        high="A high C3 may appear during acute inflammation.",
        low="A low C3 may appear in active autoimmune disease or complement consumption.",
        questions=("Should this be read with other immune tests?",),
    ),
    "Complement C4": MarkerKB(
        high="A high C4 is less commonly flagged than low values.",
        low="A low C4 may appear in autoimmune conditions such as lupus.",
        questions=("Do my symptoms fit an autoimmune pattern?",),
    ),
    "Rheumatoid Factor": MarkerKB(
        high="A positive rheumatoid factor may appear in rheumatoid arthritis and other conditions.",
        low="A negative rheumatoid factor does not rule out arthritis.",
        questions=("Do my joints hurt or swell, especially in the morning?",),
    ),
    "Anti-CCP Antibodies": MarkerKB(
        high="A high anti-CCP result is more specific for rheumatoid arthritis than rheumatoid factor in the right symptom pattern.",
        low="A low anti-CCP result makes anti-CCP-positive rheumatoid arthritis less likely.",
        questions=("Does this match joint symptoms or imaging?",),
    ),
    "Immunoglobulin G": MarkerKB(
        high="A high IgG may reflect chronic inflammation, infection, autoimmune disease, liver disease, or monoclonal protein.",
        low="A low IgG may reflect immune deficiency, protein loss, or treatment effects.",
        questions=("Should this be read with serum protein electrophoresis or vaccine responses?",),
    ),
    "Immunoglobulin A": MarkerKB(
        high="A high IgA may reflect inflammation, liver disease, mucosal immune activation, or monoclonal protein.",
        low="A low IgA can be selective and may affect interpretation of some celiac tests.",
        questions=("Does low IgA affect any antibody tests I had?",),
    ),
    "Immunoglobulin M": MarkerKB(
        high="A high IgM may appear with recent immune stimulation or some monoclonal protein disorders.",
        low="A low IgM may reflect immune deficiency or treatment effects.",
        questions=("Is this a persistent pattern across immunoglobulins?",),
    ),
    "Immunoglobulin E": MarkerKB(
        high="A high IgE is commonly associated with allergies, eczema, asthma, or parasite exposure.",
        low="A low IgE is usually not clinically significant.",
        questions=("Could allergy or asthma explain this?",),
    ),
    # --- Cardiac ---
    "BNP": MarkerKB(
        high="A high BNP suggests the heart is under strain, as in heart failure or fluid overload.",
        low="A low BNP makes significant heart failure less likely.",
        questions=("Do I have shortness of breath or leg swelling?",),
    ),
    "NT-proBNP": MarkerKB(
        high="A high NT-proBNP suggests heart strain or heart failure, but age, kidney function, and rhythm affect interpretation.",
        low="A low NT-proBNP makes heart failure less likely in the right clinical context.",
        questions=("How should this be interpreted with my age and kidney function?",),
    ),
    "Troponin I": MarkerKB(
        high="A high troponin indicates heart-muscle injury and needs urgent clinical evaluation.",
        low="A low troponin is expected when there is no heart injury.",
        questions=("Was chest pain or pressure present when this was drawn?",),
    ),
    "Creatine Kinase": MarkerKB(
        high="A high CK may reflect skeletal muscle injury from exercise, trauma, statins, or neuromuscular disease; it can also rise with heart muscle damage in the right context.",
        low="A low CK is not typically significant.",
        questions=("Did I exercise heavily before the blood draw?", "Am I on a statin?", "Is there known muscle disease or weakness?"),
    ),
    "CK-MB": MarkerKB(
        high="A high CK-MB raises concern for heart-muscle injury when troponin is also elevated.",
        low="A low CK-MB is expected without heart injury.",
        questions=("Was this checked because of chest symptoms?",),
    ),
    "Myoglobin": MarkerKB(
        high="A high myoglobin may reflect skeletal or heart muscle injury and is less specific than troponin.",
        low="A low myoglobin is expected without recent muscle injury.",
        questions=("Could exercise, trauma, or muscle symptoms explain this?",),
    ),
    # --- Hormones ---
    "Cortisol": MarkerKB(
        high="A high cortisol may reflect stress, steroids, Cushing syndrome, or lab timing.",
        low="A low cortisol may relate to adrenal insufficiency and can cause fatigue or low blood pressure.",
        questions=("Was this a morning sample?", "Am I on steroid medications?"),
    ),
    "Insulin": MarkerKB(
        high="A high insulin may reflect insulin resistance even when glucose is still normal.",
        low="A low insulin may appear in type 1 diabetes or long-standing type 2 diabetes.",
        questions=("Should we assess insulin resistance or diabetes risk?",),
    ),
    "Testosterone": MarkerKB(
        high="A high testosterone may relate to supplements, tumors, or polycystic ovary syndrome in women.",
        low="A low testosterone may cause fatigue, low libido, or muscle loss in men.",
        questions=("Could symptoms match my testosterone level?",),
    ),
    "Free Testosterone": MarkerKB(
        high="A high free testosterone may reflect androgen exposure or increased androgen production; interpretation depends strongly on sex, age, symptoms, and SHBG.",
        low="A low free testosterone may relate to hypogonadism, pituitary signaling, chronic illness, steroid exposure, or high SHBG.",
        questions=("How does this compare with total testosterone, SHBG, LH, and FSH?",),
    ),
    "Estradiol": MarkerKB(
        high="A high estradiol may relate to ovarian function, obesity, or hormone therapy.",
        low="A low estradiol may relate to menopause, ovarian failure, or low body weight.",
        questions=("Where am I in my cycle or menopause status?",),
    ),
    "Prolactin": MarkerKB(
        high="A high prolactin may cause menstrual changes or milk production and can come from pituitary issues or medications.",
        low="A low prolactin is usually not a concern.",
        questions=("Am I on medications that raise prolactin?",),
    ),
    "FSH": MarkerKB(
        high="A high FSH in women often suggests reduced ovarian reserve or menopause; in men it may suggest testicular failure.",
        low="A low FSH may relate to pituitary or hypothalamic issues.",
        questions=("Are fertility or menopause questions relevant?",),
    ),
    "LH": MarkerKB(
        high="A high LH may appear at menopause or with polycystic ovary syndrome depending on context.",
        low="A low LH may relate to pituitary or hypothalamic causes of low sex hormones.",
        questions=("Should LH be read with FSH and estradiol or testosterone?",),
    ),
    "Progesterone": MarkerKB(
        high="A high progesterone may reflect the luteal phase, pregnancy, or supplementation.",
        low="A low progesterone may relate to anovulation or luteal-phase deficiency.",
        questions=("What day of my cycle was this drawn?",),
    ),
    "Parathyroid Hormone": MarkerKB(
        high="A high PTH may drive calcium up in primary hyperparathyroidism or rise appropriately when calcium is low, phosphate is shifted, or vitamin D is low.",
        low="A low PTH may appear after parathyroid surgery or with high calcium from other causes.",
        questions=("How does PTH fit with my calcium and vitamin D?",),
    ),
    "ACTH": MarkerKB(
        high="A high ACTH may appear when the adrenal glands are underactive or in certain tumors.",
        low="A low ACTH may appear with steroid use or pituitary causes of low cortisol.",
        questions=("Should ACTH be read with cortisol?",),
    ),
    "DHEA-S": MarkerKB(
        high="A high DHEA-S suggests increased adrenal androgen production and may appear in PCOS or adrenal disorders.",
        low="A low DHEA-S can occur with adrenal insufficiency, aging, or steroid exposure.",
        questions=("Does this fit with testosterone and symptoms?",),
    ),
    "Androstenedione": MarkerKB(
        high="A high androstenedione may reflect increased adrenal or ovarian androgen production.",
        low="A low androstenedione is usually interpreted only in endocrine context.",
        questions=("Should this be paired with DHEA-S and testosterone?",),
    ),
    "Anti-Mullerian Hormone": MarkerKB(
        high="A high AMH may appear with a high ovarian follicle count, including some PCOS patterns.",
        low="A low AMH may suggest lower ovarian reserve for age, but it does not predict natural fertility by itself.",
        questions=("How should this be interpreted for my age and fertility goals?",),
    ),
    "Beta-hCG": MarkerKB(
        high="A high beta-hCG most commonly reflects pregnancy, but it is also used in some pregnancy complications and tumor monitoring.",
        low="A low or negative beta-hCG makes pregnancy less likely at the time tested.",
        questions=("Should this be repeated to assess the trend?",),
    ),
    "SHBG": MarkerKB(
        high="A high SHBG binds more testosterone and estrogen, lowering their free fractions.",
        low="A low SHBG is linked to insulin resistance and higher free androgen activity.",
        questions=("Should free testosterone be calculated?",),
    ),
    "IGF-1": MarkerKB(
        high="A high IGF-1 may reflect excess growth hormone.",
        low="A low IGF-1 may reflect growth-hormone deficiency or malnutrition.",
        questions=("Are height, hands, or jaw changes relevant?",),
    ),
    "IGF Binding Protein-3": MarkerKB(
        high="A high IGFBP-3 can accompany increased growth-hormone activity but must be interpreted with age, puberty stage, IGF-1, and lab method.",
        low="A low IGFBP-3 may support reduced growth-hormone activity or malnutrition when it matches IGF-1 and growth history.",
        questions=("Does this match IGF-1 and growth-pattern concerns?",),
    ),
    # --- Oncology / screening ---
    "PSA": MarkerKB(
        high="A high PSA may come from prostate enlargement, infection, recent procedures, or less commonly cancer.",
        low="A low PSA is expected and does not rule out all prostate conditions.",
        questions=("Could recent exercise or infection have raised PSA?", "Is repeat testing planned?"),
    ),
    "CEA": MarkerKB(
        high="A high CEA may be followed in some cancers but can also rise with smoking, inflammation, or liver disease.",
        low="A low CEA is expected and is most useful when compared with prior values during monitoring.",
        questions=("Is this being used for monitoring rather than screening?",),
    ),
    "CA-125": MarkerKB(
        high="A high CA-125 may be followed in ovarian cancer care but can also rise with benign gynecologic or inflammatory conditions.",
        low="A low CA-125 is expected and does not rule out disease by itself.",
        questions=("What trend or imaging is this being compared with?",),
    ),
    "CA 19-9": MarkerKB(
        high="A high CA 19-9 may be followed in pancreaticobiliary cancer care but can also rise with bile duct blockage or inflammation.",
        low="A low CA 19-9 is expected; some people do not make this marker even with disease.",
        questions=("Is bilirubin or bile duct blockage affecting this result?",),
    ),
    "Alpha-Fetoprotein": MarkerKB(
        high="A high AFP may be used in liver or germ-cell tumor evaluation and can also rise in pregnancy or liver disease.",
        low="A low AFP is expected outside pregnancy and monitoring contexts.",
        questions=("Should this be interpreted with liver imaging or pregnancy status?",),
    ),
    "CA 15-3": MarkerKB(
        high="A high CA 15-3 may be used to monitor some breast cancers but is not specific enough to diagnose cancer alone.",
        low="A low CA 15-3 is expected and is mainly useful as a trend during monitoring.",
        questions=("Is this being compared with prior values?",),
    ),
    "Folate": MarkerKB(
        high="A high folate is usually from supplements or fortified foods.",
        low="A low folate can cause anemia similar to B12 deficiency and affects DNA synthesis.",
        questions=("Could low folate explain my MCV or anemia?",),
    ),
    "Vitamin A": MarkerKB(
        high="A very high vitamin A is usually from supplements and can be toxic.",
        low="A low vitamin A may affect vision and immunity and relates to diet or malabsorption.",
        questions=("Am I taking vitamin A supplements?",),
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
    Pattern(
        "Iron studies pattern",
        "low Ferritin with low Serum Iron and low Transferrin Saturation",
        "Low iron stores with low circulating iron and saturation strongly supports iron deficiency as a cause of anemia.",
    ),
    Pattern(
        "Infection / inflammation cluster",
        "high White Blood Cell Count with high C-Reactive Protein or high Procalcitonin",
        "Together these suggest an active inflammatory or infectious process worth clinical correlation.",
    ),
    Pattern(
        "Coagulation concern",
        "prolonged Prothrombin Time or high INR with prolonged aPTT",
        "Multiple clotting tests abnormal together raise bleeding risk and medication or liver causes should be reviewed.",
    ),
    Pattern(
        "Cardiac strain pattern",
        "high BNP with high Troponin I",
        "Elevated heart-strain and injury markers together warrant urgent clinical assessment.",
    ),
    Pattern(
        "Pancreatic enzyme pattern",
        "high Lipase with high Amylase",
        "Both pancreatic enzymes elevated together more strongly suggest pancreatic inflammation than either alone.",
    ),
    Pattern(
        "Autoimmune thyroid pattern",
        "high Anti-TPO Antibodies with abnormal TSH",
        "Thyroid autoantibodies plus abnormal TSH suggest autoimmune thyroid disease rather than a transient lab variation.",
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
