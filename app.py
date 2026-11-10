import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"raw"/"synthetic_students.csv"
ART=ROOT/"artifacts"
REP=ROOT/"reports"

st.set_page_config(page_title="Student Success AI",page_icon="🎓",layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {--ink:#e8eef8;--muted:#9aa9bf;--panel:#111d31;--line:#263750;--accent:#62a7ff;}
.stApp {background:linear-gradient(135deg,#081321 0%,#0b1728 55%,#101d32 100%);color:#e8eef8;}
[data-testid="stSidebar"] {background:#0b1728;border-right:1px solid #263750;}
.block-container {padding-top:1.6rem;padding-bottom:3rem;max-width:1500px;}
h1,h2,h3 {color:#f3f7ff !important;letter-spacing:-.02em;}
p,li,label {color:#c8d4e5;}
div[data-testid="stMetric"] {background:#111f34;border:1px solid #263750;padding:16px 18px;border-radius:14px;}
div[data-testid="stMetricLabel"] {color:#9fb1c9;}
div[data-testid="stMetricValue"] {color:#f3f7ff;}
.stButton>button {border-radius:10px;border:1px solid #3b6fa8;background:#173b63;color:white;}
.stButton>button:hover {border-color:#79b6ff;color:white;}
div[data-testid="stAlert"] {border-radius:12px;}
hr {border-color:#263750;}
.small-muted {color:#9fb1c9;font-size:.88rem;}
.hero {padding:24px 26px;border:1px solid #263750;border-radius:18px;background:linear-gradient(110deg,#132944,#101d32);margin-bottom:18px;}
.hero-kicker {color:#7db9ff;text-transform:uppercase;letter-spacing:.14em;font-size:.75rem;font-weight:700;}
.hero h1 {margin:.35rem 0;font-size:2.1rem;}
.hero p {color:#b8c9df;margin:0;}
</style>
""",unsafe_allow_html=True)

def load_data():
    if DATA.exists():
        return pd.read_csv(DATA)
    return None

def load_csv(name):
    p=REP/name
    return pd.read_csv(p) if p.exists() else None

def load_json(name):
    p=REP/name
    return json.loads(p.read_text()) if p.exists() else None

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def model_status():
    return (ART/"best_score_model.joblib").exists(),(ART/"best_risk_model.joblib").exists()

def chart_layout(fig,height=330):
    fig.update_layout(height=height,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(0,0,0,0)",font_color="#dce7f5",margin=dict(l=10,r=10,t=45,b=10),
      legend_title_text="")
    fig.update_xaxes(gridcolor="#263750",zerolinecolor="#263750")
    fig.update_yaxes(gridcolor="#263750",zerolinecolor="#263750")
    return fig

def section(title,sub=None):
    st.subheader(title)
    if sub: st.caption(sub)

with st.sidebar:
    st.markdown("## 🎓 Student Success AI")
    st.caption("Academic performance & early-warning system")
    page=st.radio("NAVIGATION",["Overview","Student Prediction","Model Comparison",
      "Advanced Research","Dataset Explorer"],label_visibility="visible")
    st.divider()
    st.markdown("**PROJECT STATUS**")
    data=load_data()
    score_ok,risk_ok=model_status()
    st.write("🟢 Dataset available" if data is not None else "🔴 Dataset not generated")
    st.write("🟢 Score model ready" if score_ok else "🟠 Score model not trained")
    st.write("🟢 Risk model ready" if risk_ok else "🟠 Risk model not trained")
    st.divider()
    st.caption("MCA Final-Year Project")
    st.caption("Synthetic data • Research prototype")

st.markdown('<div class="hero"><div class="hero-kicker">MACHINE LEARNING RESEARCH PLATFORM</div><h1>Student Success Prediction</h1><p>Understand academic patterns, estimate performance and explore early-warning signals.</p></div>',unsafe_allow_html=True)

if data is None:
    st.warning("Student dataset is not available yet. Generate it first from the project root.")
    st.code("python -m src.generate_data --rows 200000\npython -m src.train\npython -m src.domain_generalization",language="powershell")
    st.stop()

if page=="Overview":
    section("Institution-wide overview","Descriptive statistics from the synthetic student dataset.")
    total=len(data); avg=float(data.final_score.mean()); risk=int(data.at_risk.sum())
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Student records",f"{total:,}")
    c2.metric("Average final score",f"{avg:.1f}/100")
    c3.metric("At-risk records",f"{risk:,}")
    c4.metric("At-risk share",f"{risk/total:.1%}" if total else "—")
    st.caption("These are descriptive dataset statistics, not predictions for individual real students.")
    left,right=st.columns([1.3,1])
    with left:
        fig=px.histogram(data,x="final_score",nbins=30,title="Final score distribution",
                         color_discrete_sequence=["#62a7ff"])
        st.plotly_chart(chart_layout(fig),use_container_width=True)
    with right:
        risk_df=data["at_risk"].value_counts().rename(index={0:"Not at risk",1:"At risk"}).reset_index()
        risk_df.columns=["Category","Students"]
        fig=px.pie(risk_df,names="Category",values="Students",hole=.62,
                   title="Synthetic risk-label distribution",
                   color_discrete_sequence=["#3dbb9a","#e4a64b"])
        st.plotly_chart(chart_layout(fig),use_container_width=True)
    left,right=st.columns(2)
    with left:
        by_program=data.groupby("program",as_index=False).final_score.mean().sort_values("final_score")
        st.plotly_chart(chart_layout(px.bar(by_program,x="final_score",y="program",orientation="h",
          title="Average score by program",color_discrete_sequence=["#62a7ff"])),use_container_width=True)
    with right:
        by_sem=data.groupby("semester",as_index=False).attendance_pct.mean()
        st.plotly_chart(chart_layout(px.line(by_sem,x="semester",y="attendance_pct",markers=True,
          title="Average attendance by semester",color_discrete_sequence=["#48c6a5"])),use_container_width=True)
    st.info("All records are synthetically generated for academic experimentation. Do not interpret these statistics as evidence about a real college.")

elif page=="Student Prediction":
    section("Individual student prediction","Enter a hypothetical student profile. Predictions require both trained model files.")
    score_ok,risk_ok=model_status()
    if not (score_ok and risk_ok):
        st.warning("Models are not trained yet. Run `python -m src.train` after generating the dataset.")
        st.stop()
    with st.form("student_form"):
        st.markdown("#### Academic information")
        a,b,c=st.columns(3)
        institution=a.selectbox("Institution ID",list(range(10)),index=3)
        program=b.selectbox("Program",["Computer Science","Commerce","Arts","Science","Management"])
        semester=c.selectbox("Semester",list(range(1,7)),index=3)
        a,b,c=st.columns(3)
        age=a.slider("Age",18,30,22)
        study=b.slider("Study hours per day",0.0,18.0,4.5,.5)
        attendance=c.slider("Attendance (%)",25,100,82)
        a,b,c=st.columns(3)
        gpa=a.slider("Previous GPA",2.0,10.0,7.2,.1)
        assignment=b.slider("Assignment completion (%)",10,100,88)
        internal=c.slider("Internal marks",0,100,76)
        st.markdown("#### Lifestyle and engagement")
        a,b,c=st.columns(3)
        sleep=a.slider("Sleep hours",3.0,10.0,7.0,.5)
        commute=b.slider("Commute minutes",0,180,35,5)
        logins=c.slider("LMS logins",0,100,20)
        a,b,c=st.columns(3)
        internet=a.selectbox("Internet access",["Reliable","Limited","No access"])
        parent=b.selectbox("Parental education",["School","Undergraduate","Postgraduate"])
        extra=c.selectbox("Extracurricular level",["Low","Medium","High"])
        a,b,c=st.columns(3)
        pressure=a.selectbox("Financial pressure",["Low","Medium","High"])
        job=b.selectbox("Part-time job",["No","Yes"])
        missed=c.slider("Missed deadlines",0,20,1)
        submitted=st.form_submit_button("✨ Predict student outcome",use_container_width=True)
    if submitted:
        row=pd.DataFrame([{"institution_id":institution,"program":program,"semester":semester,"age":age,
          "study_hours_per_day":study,"attendance_pct":attendance,"previous_gpa":gpa,
          "assignment_completion_pct":assignment,"internal_marks":internal,"sleep_hours":sleep,
          "commute_minutes":commute,"internet_access":internet,"parental_education":parent,
          "extracurricular_level":extra,"financial_pressure":pressure,"part_time_job":job,
          "lms_logins":logins,"missed_deadlines":missed}])
        try:
            score_model=load_model(ART/"best_score_model.joblib")
            risk_model=load_model(ART/"best_risk_model.joblib")
            predicted=float(score_model.predict(row)[0])
            probability=float(risk_model.predict_proba(row)[0,1])
            st.divider()
            x,y=st.columns(2)
            x.metric("Predicted final score",f"{predicted:.1f}/100")
            y.metric("Predicted at-risk probability",f"{probability:.1%}")
            st.progress(min(max(probability,0),1),text="Estimated risk probability")
            if probability>=.5:
                st.warning("The model flags this hypothetical profile for possible academic support.")
            else:
                st.success("The model does not flag this hypothetical profile at the default 0.50 threshold.")
            st.caption("This is an experimental model output, not a diagnosis or decision about a real student. Validate the model and threshold before any practical use.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page=="Model Comparison":
    section("Model benchmark","Metrics are shown only after the training pipeline has completed.")
    reg=load_csv("regression_model_comparison.csv"); clf=load_csv("classification_model_comparison.csv")
    if reg is None or clf is None:
        st.warning("Comparison reports are missing. Run `python -m src.train`.")
        st.stop()
    l,r=st.columns(2)
    with l:
        st.markdown("#### Regression leaderboard")
        st.dataframe(reg,use_container_width=True,hide_index=True)
        fig=px.bar(reg.sort_values("RMSE"),x="RMSE",y="model",orientation="h",
                   title="RMSE (lower is better)",color_discrete_sequence=["#62a7ff"])
        st.plotly_chart(chart_layout(fig,400),use_container_width=True)
    with r:
        st.markdown("#### Classification leaderboard")
        st.dataframe(clf,use_container_width=True,hide_index=True)
        fig=px.bar(clf.sort_values("f1"),x="f1",y="model",orientation="h",
                   title="F1-score (higher is better)",color_discrete_sequence=["#48c6a5"])
        st.plotly_chart(chart_layout(fig,400),use_container_width=True)
    st.download_button("Download regression results",reg.to_csv(index=False),"regression_results.csv","text/csv")
    st.download_button("Download classification results",clf.to_csv(index=False),"classification_results.csv","text/csv")

elif page=="Advanced Research":
    section("Advanced ML experiments","Hyperparameter optimization, model fusion and held-out-institution evaluation.")
    summary=load_json("training_summary.json")
    if summary:
        a,b,c=st.columns(3)
        a.metric("Training records",f'{summary.get("train_records",0):,}')
        b.metric("Test records",f'{summary.get("test_records",0):,}')
        c.metric("Dataset records",f'{summary.get("records",0):,}')
        st.markdown("#### Selected models")
        st.write("Best regression model:",summary.get("best_regression_model","—"))
        st.write("Best classification model:",summary.get("best_classification_model","—"))
        st.markdown("#### Best Random Forest tuning configuration")
        st.json(summary.get("tuning_best_params",{}))
    else:
        st.info("Training summary will appear after `python -m src.train`.")
    st.divider()
    st.markdown("#### Domain generalization")
    dg=load_csv("domain_generalization.csv")
    if dg is not None:
        st.dataframe(dg,use_container_width=True,hide_index=True)
        st.caption("Institutions 0–7 are used as source domains; institutions 8–9 are held out for evaluation. This is a stress test, not proof of universal generalization.")
        fig=px.bar(dg,x="model",y="RMSE",title="Held-out institution RMSE",
                   color_discrete_sequence=["#e4a64b"])
        st.plotly_chart(chart_layout(fig),use_container_width=True)
    else:
        st.warning("Domain report not found. Run `python -m src.domain_generalization`.")
    st.divider()
    st.markdown("#### Feature importance")
    st.caption("Feature importance is not yet calculated in this baseline release. Add permutation importance or SHAP on a held-out sample before presenting feature rankings.")

elif page=="Dataset Explorer":
    section("Dataset explorer","Filter and inspect the generated synthetic records.")
    st.caption(f"{len(data):,} records loaded.")
    c1,c2,c3=st.columns(3)
    programs=sorted(data.program.dropna().unique().tolist())
    selected_program=c1.multiselect("Program",programs,default=programs)
    semesters=c2.multiselect("Semester",sorted(data.semester.unique().tolist()),default=sorted(data.semester.unique().tolist()))
    risk_filter=c3.selectbox("Risk label",["All","At risk","Not at risk"])
    view=data[data.program.isin(selected_program)&data.semester.isin(semesters)]
    if risk_filter=="At risk": view=view[view.at_risk==1]
    elif risk_filter=="Not at risk": view=view[view.at_risk==0]
    st.write(f"Showing {len(view):,} matching records")
    st.dataframe(view.head(1000),use_container_width=True,hide_index=True)
    st.download_button("Download filtered records",view.to_csv(index=False),"student_records_filtered.csv","text/csv")
    st.caption("The table is limited to the first 1,000 matching rows for browser responsiveness; the download includes all filtered rows.")
