"""Generate a reproducible, fully synthetic student dataset."""
import argparse
import numpy as np
import pandas as pd
from src.config import DATA_RAW, N_STUDENTS, RANDOM_STATE

def generate_students(n=N_STUDENTS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    institution_id = rng.integers(0, 10, n)
    program = rng.choice(["Computer Science", "Commerce", "Arts", "Science", "Management"], n,
                         p=[.28, .18, .16, .22, .16])
    semester = rng.integers(1, 7, n)
    age = np.clip(rng.normal(21.5 + 0.25 * (semester - 1), 1.8, n), 18, 30).round().astype(int)

    # Institution-specific environment creates realistic distribution shifts.
    institution_effect = (institution_id - 4.5) * 0.65
    study_hours = np.clip(rng.gamma(2.4, 2.0, n) + institution_effect * .10, 0, 18)
    attendance = np.clip(rng.beta(8, 2.2, n) * 100 + institution_effect, 25, 100)
    previous_gpa = np.clip(rng.normal(6.4 + institution_effect * .05, 1.35, n), 2, 10)
    assignment_completion = np.clip(rng.beta(7, 2, n) * 100 + institution_effect * .6, 10, 100)
    internal_marks = np.clip(0.55 * previous_gpa * 10 + 0.28 * attendance +
                             0.17 * assignment_completion + rng.normal(0, 8, n), 0, 100)
    sleep_hours = np.clip(rng.normal(6.8, 1.15, n), 3, 10)
    commute_minutes = np.clip(rng.gamma(2.0, 22, n), 0, 180)
    internet_access = rng.choice(["Reliable", "Limited", "No access"], n, p=[.76, .20, .04])
    parental_education = rng.choice(["School", "Undergraduate", "Postgraduate"], n,
                                    p=[.42, .40, .18])
    extracurricular = rng.choice(["Low", "Medium", "High"], n, p=[.35, .45, .20])
    financial_pressure = rng.choice(["Low", "Medium", "High"], n, p=[.48, .36, .16])
    part_time_job = rng.choice(["No", "Yes"], n, p=[.78, .22])
    LMS_logins = np.clip(rng.poisson(13 + study_hours * 1.4, n), 0, 100)
    missed_deadlines = np.clip(rng.poisson(2.8 + (100-attendance)/28, n), 0, 20)

    internet_bonus = np.select([internet_access == "Reliable", internet_access == "Limited"],
                               [2.0, -1.0], default=-4.0)
    parent_bonus = np.select([parental_education == "Postgraduate",
                              parental_education == "Undergraduate"],
                             [1.5, .5], default=0.0)
    activity_effect = np.select([extracurricular == "High", extracurricular == "Medium"],
                                [1.2, .5], default=0.0)
    pressure_penalty = np.select([financial_pressure == "High",
                                  financial_pressure == "Medium"],
                                 [-3.0, -1.0], default=0.0)
    job_penalty = np.where(part_time_job == "Yes", -1.7, 0)
    program_effect = pd.Series(program).map({
        "Computer Science": 1.0, "Science": .5, "Management": .2,
        "Commerce": 0.0, "Arts": -.2
    }).to_numpy()

    # Nonlinear interactions: attendance and internal assessment matter together.
    score = (15 + 0.24 * attendance + 1.35 * study_hours +
             2.15 * previous_gpa + .18 * internal_marks +
             .10 * assignment_completion + .65 * sleep_hours -
             .055 * commute_minutes - 1.25 * missed_deadlines +
             internet_bonus + parent_bonus + activity_effect +
             pressure_penalty + job_penalty + program_effect +
             institution_effect + .018 * (attendance * study_hours / 2) +
             rng.normal(0, 7.5, n))
    final_score = np.clip(score, 0, 100).round(2)
    at_risk = (final_score < 60).astype(int)

    df = pd.DataFrame({
        "student_id": [f"STU{i:06d}" for i in range(n)],
        "institution_id": institution_id,
        "program": program,
        "semester": semester,
        "age": age,
        "study_hours_per_day": study_hours.round(2),
        "attendance_pct": attendance.round(2),
        "previous_gpa": previous_gpa.round(2),
        "assignment_completion_pct": assignment_completion.round(2),
        "internal_marks": internal_marks.round(2),
        "sleep_hours": sleep_hours.round(2),
        "commute_minutes": commute_minutes.round(1),
        "internet_access": internet_access,
        "parental_education": parental_education,
        "extracurricular_level": extracurricular,
        "financial_pressure": financial_pressure,
        "part_time_job": part_time_job,
        "lms_logins": LMS_logins,
        "missed_deadlines": missed_deadlines,
        "final_score": final_score,
        "at_risk": at_risk
    })
    # Introduce a small amount of realistic missingness in input features; targets remain complete.
    for col in ["study_hours_per_day", "attendance_pct", "previous_gpa",
                "internet_access", "sleep_hours", "lms_logins"]:
        missing_mask = rng.random(n) < 0.012
        df.loc[missing_mask, col] = np.nan
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=N_STUDENTS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW / "synthetic_students.csv"
    data = generate_students(args.rows, args.seed)
    data.to_csv(out, index=False)
    print(f"Saved {len(data):,} synthetic records to {out}")
    print(data[["final_score", "at_risk"]].describe())
