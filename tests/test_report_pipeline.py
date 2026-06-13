import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.knowledge_graph import LabKnowledgeGraph  # noqa: E402
from src.openbmb_client import ExtractionResult  # noqa: E402
from src.report_pipeline import build_health_report, normalize_patient, parse_reference_interval  # noqa: E402
from app import _final_range_position, _final_status_for_marker, _preferred_marker_label  # noqa: E402


def _result(tests, patient=None):
    return ExtractionResult(
        patient=patient or {},
        tests=tests,
        notes=[],
        raw_response="{}",
        request_summary={},
    )


def test_patient_age_and_sex_are_normalized():
    patient = normalize_patient({"age": "25y 10m 26d", "sex": "Female"})
    assert patient["age_group"] == "adult"
    assert patient["sex"] == "female"
    assert patient["age_years"] and patient["age_years"] > 25


def test_reference_interval_parser_handles_common_lab_formats():
    assert parse_reference_interval("3.5-5.50") == {"low": 3.5, "high": 5.5}
    assert parse_reference_interval("Up to 15") == {"low": None, "high": 15.0}
    assert parse_reference_interval("< 20") == {"low": None, "high": 20.0}


def test_report_enriches_extracted_marker_with_knowledge_graph():
    report = build_health_report(
        _result(
            [
                {
                    "marker": "RBC",
                    "value": "3.3",
                    "unit": "10^6/uL",
                    "reference_range": "3.5-5.50",
                    "status": "low",
                    "source_text": "RBC 3.3 3.5-5.50",
                    "confidence": 0.91,
                }
            ],
            patient={"age": "25y 10m 26d", "sex": "Female"},
        )
    )
    marker = report["markers"][0]
    assert marker["canonical_id"] == "rbc"
    assert marker["knowledge"]["description"]
    assert marker["comparison"]["basis"] == "lab_reference_range"
    assert marker["status"] == "low"
    assert marker["reference_selection"]["sex"] == "female"
    assert marker["reference_selection"]["values"]["maximum_value"] == 5.2


def test_report_uses_sex_specific_kg_range_when_lab_range_is_missing():
    report = build_health_report(
        _result(
            [
                {
                    "marker": "Hemoglobin",
                    "value": "12.7",
                    "unit": "g/dL",
                    "reference_range": None,
                    "status": "unknown",
                    "confidence": 1,
                }
            ],
            patient={"age_years": 42, "sex": "male"},
        )
    )
    marker = report["markers"][0]
    assert marker["comparison"]["basis"] == "knowledge_graph"
    assert marker["derived_status"] == "low"
    assert marker["status"] == "low"
    assert marker["reference_selection"]["sex"] == "male"


def test_knowledge_graph_has_sex_guidance_for_every_marker():
    graph = LabKnowledgeGraph.load()
    assert graph.tests
    assert all("sex_significance" in test for test in graph.tests)
    assert all(test.get("video_url") for test in graph.tests)
    high = [test for test in graph.tests if test["sex_significance"]["level"] == "high"]
    assert {test["id"] for test in high} == {"hemoglobin", "rbc", "hct", "esr"}
    assert all("sex_specific_statistics_per_group_age" in test for test in high)


def test_report_knowledge_payload_includes_video_url():
    report = build_health_report(
        _result(
            [
                {
                    "marker": "RBC",
                    "value": "4.2",
                    "unit": "10^6/uL",
                    "status": "normal",
                    "confidence": 1,
                }
            ],
            patient={"age_years": 25, "sex": "female"},
        )
    )
    knowledge = report["markers"][0]["knowledge"]
    assert knowledge["video_url"]
    assert "youtube.com" in knowledge["video_url"]


def test_youtube_embed_helpers():
    from app import _youtube_embed_html, _youtube_video_id

    assert _youtube_video_id("https://www.youtube.com/watch?v=abc123XYZ_-") == "abc123XYZ_-"
    assert _youtube_video_id("https://youtu.be/abc123XYZ_-") == "abc123XYZ_-"
    html = _youtube_embed_html("https://www.youtube.com/watch?v=abc123XYZ_-", title="RBC overview")
    assert "youtube.com/embed/abc123XYZ_-" in html
    assert "RBC overview" in html
    assert "No video is available" in _youtube_embed_html(None)


# CBC markers in src/markers.py must match KG adult fallback intervals (fix #3).
_KG_CBC_MARKER_MAP = {
    "Hemoglobin": "hemoglobin",
    "Hematocrit": "hct",
    "Red Blood Cell Count": "rbc",
    "White Blood Cell Count": "wbc",
    "Platelet Count": "plt",
    "MCV": "mcv",
    "MCH": "mch",
    "MCHC": "mchc",
    "RDW": "rdw_cv",
    "Absolute Lymphocyte Count": "lym_absolute",
    "ESR": "esr",
}


def test_knowledge_graph_normal_values_are_midpoints():
    graph = LabKnowledgeGraph.load()
    for test in graph.tests:
        for stats_key in ("statistics_per_group_age",):
            stats = test.get(stats_key) or {}
            for vals in stats.values():
                lo, hi, mid = vals["minimal_value"], vals["maximum_value"], vals["normal_value"]
                assert mid == round((lo + hi) / 2, 2)
        sex_stats = test.get("sex_specific_statistics_per_group_age") or {}
        for group_stats in sex_stats.values():
            for vals in group_stats.values():
                lo, hi, mid = vals["minimal_value"], vals["maximum_value"], vals["normal_value"]
                assert mid == round((lo + hi) / 2, 2)


def test_markers_py_cbc_ranges_match_knowledge_graph():
    from src.markers import MARKERS

    graph = LabKnowledgeGraph.load()
    by_name = {m.name: m for m in MARKERS}
    for marker_name, node_id in _KG_CBC_MARKER_MAP.items():
        marker = by_name[marker_name]
        node = graph.get(node_id)
        adult = node["statistics_per_group_age"]["adult"]
        assert marker.ref_low == adult["minimal_value"], marker_name
        assert marker.ref_high == adult["maximum_value"], marker_name


def test_final_report_bar_uses_kg_min_normal_and_max_values():
    report = build_health_report(
        _result(
            [
                {"marker": "RBC", "value": "3.3", "unit": "10^6/uL", "status": "normal", "confidence": 1},
                {"marker": "WBC", "value": "6.7", "unit": "10^3/uL", "status": "normal", "confidence": 1},
            ],
            patient={"age_years": 25, "sex": "female"},
        )
    )
    rbc, wbc = report["markers"]
    assert _final_status_for_marker(rbc) == "bad"
    assert 6 <= _final_range_position(rbc) <= 30
    assert _final_status_for_marker(wbc) == "ideal"
    assert 68 <= _final_range_position(wbc) <= 94


def test_final_report_prefers_lab_abbreviations_when_extracted():
    assert _preferred_marker_label(
        {"raw_name": "RBC", "display_name": "Red Blood Cell Count"}
    ) == "RBC"
    assert _preferred_marker_label(
        {"raw_name": "Hct", "display_name": "Hematocrit"}
    ) == "Hct"
    assert _preferred_marker_label(
        {"raw_name": "NEU%", "display_name": "Neutrophils Percent"}
    ) == "NEU%"
    assert _preferred_marker_label(
        {"raw_name": "Hemoglobin", "display_name": "Hemoglobin"}
    ) == "Hemoglobin"
