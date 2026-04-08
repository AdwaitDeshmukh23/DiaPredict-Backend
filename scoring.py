"""
DiaPredict-AI — Heuristic Diabetes Scoring Engine
Weights: Family History 25%, BMI 20%, Age 15%, Pregnancies 10%, Glucose 10%, Insulin 10%, BP 5%, SkinThickness 5%
"""
from models import PredictionRequest, PredictionResponse, RiskCategory, ScoreBreakdown
from typing import List, Tuple


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)


def score_exercise(level: str) -> float:
    """Daily exercise level."""
    if level == "None":      return 0.90
    elif level == "Light":   return 0.60
    elif level == "Moderate":return 0.30
    else:                    return 0.10


def score_family_history(has_history: bool) -> float:
    """Family history of diabetes is a very strong predictor."""
    return 0.85 if has_history else 0.1


def score_glucose(glucose: float) -> float:
    """Normalize glucose 0-300 → 0-1. Clinical thresholds based."""
    if glucose == 0:       return 0.2   # Missing -> Neutral baseline
    elif glucose <= 70:    return 0.05
    elif glucose <= 99:    return 0.1 + (glucose - 70) / 29 * 0.15    # Normal
    elif glucose <= 125:   return 0.25 + (glucose - 99) / 26 * 0.25   # Pre-diabetic
    elif glucose <= 199:   return 0.5 + (glucose - 125) / 74 * 0.35   # Diabetic range
    else:                  return 1.0                                    # Very high


def score_bmi(bmi: float) -> float:
    """WHO BMI thresholds."""
    if bmi < 18.5:         return 0.1   # Underweight (low risk for T2D)
    elif bmi <= 24.9:      return 0.15  # Normal
    elif bmi <= 27.4:      return 0.35  # Overweight mild
    elif bmi <= 29.9:      return 0.55  # Overweight
    elif bmi <= 34.9:      return 0.75  # Obese I
    elif bmi <= 39.9:      return 0.90  # Obese II
    else:                  return 1.0   # Obese III


def score_age(age: int) -> float:
    """Age-based risk escalation inspired by ADA guidelines."""
    if age < 25:           return 0.05
    elif age <= 35:        return 0.15
    elif age <= 45:        return 0.35
    elif age <= 55:        return 0.60
    elif age <= 65:        return 0.80
    else:                  return 1.0


def score_insulin(insulin: float) -> float:
    """Serum insulin scoring (0 = not measured → neutral 0.3)."""
    if insulin == 0:       return 0.3   # Missing data → neutral
    elif insulin <= 16:    return 0.1   # Low (possible type 1 risk)
    elif insulin <= 166:   return 0.2   # Normal range
    elif insulin <= 300:   return 0.6   # High — insulin resistance
    else:                  return 1.0   # Very high


def score_pregnancies(pregnancies: int) -> float:
    """Gestational diabetes correlation."""
    if pregnancies == 0:   return 0.0
    elif pregnancies <= 2: return 0.2
    elif pregnancies <= 4: return 0.45
    elif pregnancies <= 6: return 0.70
    else:                  return 1.0


def score_blood_pressure(bp: float) -> float:
    """Diastolic BP thresholds (JNC 8)."""
    if bp == 0:            return 0.3   # Missing
    elif bp < 60:          return 0.1
    elif bp <= 80:         return 0.15  # Normal
    elif bp <= 89:         return 0.45  # Elevated
    elif bp <= 99:         return 0.70  # Stage 1 HTN
    else:                  return 1.0   # Stage 2 HTN


def score_skin_thickness(thickness: float) -> float:
    """Triceps skinfold (proxy for body fat)."""
    if thickness == 0:     return 0.3   # Missing
    elif thickness <= 15:  return 0.1
    elif thickness <= 25:  return 0.3
    elif thickness <= 35:  return 0.6
    elif thickness <= 50:  return 0.8
    else:                  return 1.0


def get_suggestions(risk_category: RiskCategory, bmi: float, glucose: float, age: int, family_history: bool, exercise_level: str) -> List[str]:
    base = []
    if risk_category == RiskCategory.LOW:
        base = [
            "🥗 Maintain a balanced diet rich in vegetables, whole grains, and lean protein.",
            "🏃 Keep up your current activity levels to maintain a healthy metabolism.",
            "📅 Schedule annual check-ups to stay proactive.",
        ]
    elif risk_category == RiskCategory.MEDIUM:
        base = [
            "🩺 Consult a GP within 30 days for a comprehensive metabolic panel.",
            "🍎 Reduce refined sugar and processed carbohydrate intake immediately.",
            "🏋️ Start a structured workout plan: 30 min cardio + strength training 4×/week.",
        ]
        if glucose > 0:
            base.append("📊 Monitor fasting blood glucose weekly and log results.")
    else:  # HIGH
        base = [
            "🚨 Seek urgent medical consultation within 7 days — do not delay.",
            "🔬 Request HbA1c, Fasting Blood Sugar, and Lipid Profile blood tests.",
            "🚫 Eliminate sugar, white rice, sugary drinks, and processed foods entirely.",
            "💊 Discuss medication or preventive therapy options with your endocrinologist.",
            "🏃 Begin daily 45-minute low-impact exercise (walking, swimming, cycling).",
        ]
    # Contextual additions
    if bmi >= 30:
        base.append("⚖️ Your BMI indicates obesity — a medically supervised weight loss program is recommended.")
    if glucose >= 126:
        base.append("🩸 Your glucose level is in the diabetic range — fasting glucose confirmation is essential.")
    if age >= 55:
        base.append("👨‍⚕️ Age increases your risk — more frequent screening (every 6 months) is advised.")
    if family_history:
        base.append("🧬 Your family history indicates genetic predisposition — preventive lifestyle choices are completely essential.")
    if exercise_level in ["None", "Light"]:
        base.append("⚡ Lack of regular vigorous exercise is a major risk factor. Try adding 30 minutes of brisk walking to your daily routine.")
    return base


def get_medical_tests(risk_category: RiskCategory) -> List[str]:
    if risk_category == RiskCategory.HIGH:
        return [
            "HbA1c (Glycated Haemoglobin)",
            "Fasting Blood Sugar (FBS)",
            "Postprandial Blood Sugar (PPBS)",
            "Lipid Profile",
            "Kidney Function Test (KFT)",
            "Urine Microalbumin",
        ]
    elif risk_category == RiskCategory.MEDIUM:
        return [
            "Fasting Blood Sugar (FBS)",
            "HbA1c",
            "BMI & Waist Circumference",
        ]
    return []


def calculate_diabetes_risk(data: PredictionRequest) -> PredictionResponse:
    bmi = calculate_bmi(data.weight, data.height)

    f_score = score_family_history(data.family_history)
    ex_score = score_exercise(data.exercise_level)
    g_score = score_glucose(data.glucose or 0)
    b_score = score_bmi(bmi)
    a_score = score_age(data.age)
    i_score = score_insulin(data.insulin or 0)
    p_score = score_pregnancies(data.pregnancies)
    bp_score = score_blood_pressure(data.blood_pressure or 0)
    sk_score = score_skin_thickness(data.skin_thickness or 0)

    weighted = (
        f_score  * 0.25 +
        b_score  * 0.15 +
        a_score  * 0.15 +
        ex_score * 0.10 +
        p_score  * 0.10 +
        g_score  * 0.10 +
        i_score  * 0.05 +
        bp_score * 0.05 +
        sk_score * 0.05
    )

    risk_score = round(min(weighted * 100, 100), 1)

    if risk_score < 30:
        category = RiskCategory.LOW
        color = "#22C55E"
    elif risk_score < 60:
        category = RiskCategory.MEDIUM
        color = "#F59E0B"
    else:
        category = RiskCategory.HIGH
        color = "#EF4444"

    breakdown = ScoreBreakdown(
        family_history_score=round(f_score * 100, 1),
        exercise_score=round(ex_score * 100, 1),
        glucose_score=round(g_score * 100, 1),
        bmi_score=round(b_score * 100, 1),
        age_score=round(a_score * 100, 1),
        insulin_score=round(i_score * 100, 1),
        pregnancies_score=round(p_score * 100, 1),
        blood_pressure_score=round(bp_score * 100, 1),
        skin_thickness_score=round(sk_score * 100, 1),
    )

    return PredictionResponse(
        risk_score=risk_score,
        risk_category=category,
        risk_color=color,
        bmi=bmi,
        suggestions=get_suggestions(category, bmi, data.glucose or 0, data.age, data.family_history, data.exercise_level),
        score_breakdown=breakdown,
        medical_tests=get_medical_tests(category),
    )
