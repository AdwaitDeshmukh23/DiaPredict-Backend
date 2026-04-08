"""
DiaPredict-AI — Unit Tests for Scoring Engine
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PredictionRequest
from scoring import calculate_diabetes_risk, calculate_bmi, RiskCategory


def make_request(**kwargs):
    defaults = dict(
        family_history=False,
        exercise_level="Moderate",
        pregnancies=0, glucose=90, blood_pressure=70,
        skin_thickness=20, insulin=80, height=170, weight=65, age=25
    )
    defaults.update(kwargs)
    return PredictionRequest(**defaults)


# ── BMI Tests ────────────────────────────────────────────────────────────────

def test_bmi_calculation():
    assert calculate_bmi(70, 175) == 22.9
    assert calculate_bmi(90, 170) == 31.1


# ── Risk Category Tests ───────────────────────────────────────────────────────

def test_low_risk_young_healthy():
    req = make_request(family_history=False, exercise_level="Active", pregnancies=0, glucose=85, blood_pressure=70,
                       skin_thickness=15, insulin=80, height=170, weight=63, age=22)
    res = calculate_diabetes_risk(req)
    assert res.risk_category == RiskCategory.LOW
    assert res.risk_score < 30


def test_high_risk_diabetic_profile():
    req = make_request(family_history=True, exercise_level="None", pregnancies=8, glucose=200, blood_pressure=100,
                       skin_thickness=40, insulin=500, height=160, weight=95, age=65)
    res = calculate_diabetes_risk(req)
    assert res.risk_category == RiskCategory.HIGH
    assert res.risk_score >= 60


def test_medium_risk_profile():
    req = make_request(family_history=False, pregnancies=3, glucose=130, blood_pressure=82,
                       skin_thickness=28, insulin=150, height=165, weight=78, age=44)
    res = calculate_diabetes_risk(req)
    assert res.risk_category == RiskCategory.MEDIUM


# ── Output Integrity Tests ────────────────────────────────────────────────────

def test_risk_score_bounds():
    req = make_request(glucose=300, age=80, weight=150, height=155, pregnancies=10)
    res = calculate_diabetes_risk(req)
    assert 0 <= res.risk_score <= 100


def test_bmi_in_response():
    req = make_request(height=170, weight=70)
    res = calculate_diabetes_risk(req)
    assert res.bmi == calculate_bmi(70, 170)


def test_suggestions_not_empty():
    req = make_request()
    res = calculate_diabetes_risk(req)
    assert len(res.suggestions) >= 3


def test_high_risk_has_medical_tests():
    req = make_request(glucose=200, age=65, weight=100, height=160, pregnancies=8)
    res = calculate_diabetes_risk(req)
    if res.risk_category == RiskCategory.HIGH:
        assert res.medical_tests is not None
        assert len(res.medical_tests) > 0


def test_score_breakdown_fields():
    req = make_request()
    res = calculate_diabetes_risk(req)
    bd = res.score_breakdown
    assert bd.glucose_score >= 0
    assert bd.bmi_score >= 0
    assert bd.age_score >= 0
