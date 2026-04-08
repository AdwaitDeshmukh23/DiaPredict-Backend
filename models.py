"""
Pydantic Models for DiaPredict-AI
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class RiskCategory(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PredictionRequest(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    family_history: bool = Field(..., description="Family history of diabetes")
    exercise_level: str = Field(..., description="Daily exercise level (None, Light, Moderate, Active)")
    glucose: Optional[float] = Field(0, ge=0, le=300, description="Plasma glucose concentration (mg/dL)")
    blood_pressure: Optional[float] = Field(0, ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    skin_thickness: Optional[float] = Field(0, ge=0, le=100, description="Triceps skin fold thickness (mm)")
    insulin: Optional[float] = Field(0, ge=0, le=900, description="2-Hour serum insulin (mu U/ml)")
    height: float = Field(..., ge=100, le=250, description="Height in centimeters")
    weight: float = Field(..., ge=20, le=300, description="Weight in kilograms")
    age: int = Field(..., ge=1, le=120, description="Age in years")

    @validator("height")
    def height_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Height must be greater than 0")
        return v

    @validator("weight")
    def weight_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Weight must be greater than 0")
        return v


class ScoreBreakdown(BaseModel):
    family_history_score: float
    exercise_score: float
    glucose_score: float
    bmi_score: float
    age_score: float
    insulin_score: float
    pregnancies_score: float
    blood_pressure_score: float
    skin_thickness_score: float


class PredictionResponse(BaseModel):
    risk_score: float = Field(..., description="Risk percentage 0-100")
    risk_category: RiskCategory
    risk_color: str
    bmi: float
    suggestions: List[str]
    score_breakdown: ScoreBreakdown
    medical_tests: Optional[List[str]] = None
    disclaimer: str = "This tool is for educational and early risk indication purposes only. It does not replace professional medical advice."


class ChatRequest(BaseModel):
    user_message: str = Field(..., min_length=1, max_length=500)
    risk_score: Optional[float] = None
    risk_category: Optional[str] = None
    bmi: Optional[float] = None
    glucose: Optional[float] = None
    age: Optional[int] = None
    blood_pressure: Optional[float] = None
    history_summary: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    source: str = Field(description="'openai' or 'rule_based'")
    disclaimer: str = "This advice is for educational purposes only. Please consult a healthcare professional."
