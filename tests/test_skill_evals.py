import json
from pathlib import Path


EVAL_CASES = (
    Path(__file__).parents[1]
    / "skills"
    / "experimentmind"
    / "evals"
    / "cases.json"
)


def test_skill_eval_cases_are_well_formed_and_cover_core_boundaries() -> None:
    fixture = json.loads(EVAL_CASES.read_text())

    assert fixture["version"] == 1
    assert fixture["skill"] == "experimentmind"

    cases = fixture["cases"]
    case_ids = {case["id"] for case in cases}
    assert case_ids == {
        "offline-demo",
        "recommendation-integrity",
        "live-api-consent",
        "clean-room-scope",
    }

    assert len(case_ids) == len(cases)
    for case in cases:
        assert case["prompt"].strip()
        assert case["expected_behaviors"]
        assert case["forbidden_behaviors"]
        assert all(item.strip() for item in case["expected_behaviors"])
        assert all(item.strip() for item in case["forbidden_behaviors"])
