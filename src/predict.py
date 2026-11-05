"""Run inference for a sample student using saved models."""
import joblib
import pandas as pd
from src.config import ARTIFACTS

sample = pd.DataFrame([{
    "institution_id": 3, "program": "Computer Science", "semester": 4, "age": 22,
    "study_hours_per_day": 4.5, "attendance_pct": 82, "previous_gpa": 7.2,
    "assignment_completion_pct": 88, "internal_marks": 76, "sleep_hours": 7,
    "commute_minutes": 35, "internet_access": "Reliable",
    "parental_education": "Undergraduate", "extracurricular_level": "Medium",
    "financial_pressure": "Low", "part_time_job": "No", "lms_logins": 20,
    "missed_deadlines": 1
}])
score_model = joblib.load(ARTIFACTS / "best_score_model.joblib")
risk_model = joblib.load(ARTIFACTS / "best_risk_model.joblib")
score = float(score_model.predict(sample)[0])
risk_probability = float(risk_model.predict_proba(sample)[0, 1])
print(f"Predicted final score: {score:.1f}/100")
print(f"At-risk probability (classifier): {risk_probability:.1%}")
# Hybrid score blends the learned risk probability with a score-based risk signal.
hybrid_risk = 0.7 * risk_probability + 0.3 * (1 - score / 100)
print(f"Hybrid early-warning index: {hybrid_risk:.1%}")
print("Risk flag:", "AT RISK" if hybrid_risk >= .5 else "NOT FLAGGED")
