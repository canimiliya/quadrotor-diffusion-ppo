import pytest

from scripts.audit_s3d0 import cosine_rows, outcome, quantiles


def test_s3d0_quantiles_are_deterministic():
    assert quantiles([0.0, 1.0, 2.0, 3.0]) == pytest.approx(
        {"p50": 1.5, "p90": 2.7, "p95": 2.85, "p99": 2.97})


def test_s3d0_cosine_rows_handles_zero_vectors():
    values = cosine_rows([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert values.tolist() == [0.0, 1.0]


def test_s3d0_outcome_precedence_matches_r1_contract():
    assert outcome({"success": "False", "collision": "False", "ground_contact": "True", "timeout": "False"}) == "GROUND_CONTACT"
