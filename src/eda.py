"""Create compact exploratory-data-analysis outputs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.config import DATA_RAW, REPORTS, FIGURES

def main():
    df = pd.read_csv(DATA_RAW / "synthetic_students.csv")
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df.describe(include="all").transpose().to_csv(REPORTS / "data_profile.csv")
    df.isna().mean().sort_values(ascending=False).rename("missing_fraction").to_csv(REPORTS / "missing_values.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    df["final_score"].hist(bins=40, ax=ax)
    ax.set(title="Synthetic final-score distribution", xlabel="Final score", ylabel="Students")
    fig.tight_layout(); fig.savefig(FIGURES / "score_distribution.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    df["at_risk"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set(title="Synthetic at-risk class balance", xlabel="At-risk (0=no, 1=yes)", ylabel="Students")
    fig.tight_layout(); fig.savefig(FIGURES / "risk_class_balance.png", dpi=140); plt.close(fig)
    print("EDA reports and figures saved under reports/")

if __name__ == "__main__":
    main()
