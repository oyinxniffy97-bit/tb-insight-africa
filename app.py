import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
from who_burden_tab import render_who_burden_tab
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TB Insight Africa",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #f8fafc; }
    
    .hero {
        background: linear-gradient(135deg, #1A3A5C 0%, #0d2137 50%, #C0392B 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .hero p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .hero .tagline {
        background: rgba(255,255,255,0.15);
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-bottom: 1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .risk-card-high {
        background: linear-gradient(135deg, #C0392B, #e74c3c);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(192,57,43,0.3);
    }
    
    .risk-card-medium {
        background: linear-gradient(135deg, #F39C12, #f1c40f);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(243,156,18,0.3);
    }
    
    .risk-card-low {
        background: linear-gradient(135deg, #27AE60, #2ecc71);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(39,174,96,0.3);
    }
    
    .risk-title {
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.9;
    }
    
    .risk-label {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .risk-score {
        font-size: 1rem;
        opacity: 0.85;
    }

    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #C0392B;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #C0392B;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.3rem;
    }

    .biomarker-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
        border-top: 3px solid #1A3A5C;
    }
    
    .recommendation-box {
        background: #f0f7ff;
        border: 1px solid #bdd7f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .disclaimer {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.8rem;
        color: #795548;
        margin-top: 1rem;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1A3A5C;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    .team-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        text-align: center;
    }

    .footer {
        background: #1A3A5C;
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 3rem;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 8px !important;
        border: 2px solid #e2e8f0 !important;
        font-size: 1.1rem !important;
        padding: 0.5rem !important;
    }
    
    div[data-testid="stNumberInput"] input:focus {
        border-color: #1A3A5C !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1A3A5C, #C0392B) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(26,58,92,0.3) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def classify_risk(esr, crp, uric_acid, sex="M", age=35):
    score = 0
    breakdown = {}

    # ESR scoring — age/sex-specific normal ranges (Sox & Liang, 1986)
    if sex == "M":
        normal_esr = 15 if age <= 50 else 20
    else:
        normal_esr = 20 if age <= 50 else 30

    if esr > 60:
        score += 2
        breakdown['ESR'] = ('High', 2, f'{esr:.1f} mm/hr — markedly elevated (>60 mm/hr)')
    elif esr > normal_esr:
        score += 1
        breakdown['ESR'] = ('Moderate', 1, f'{esr} mm/hr — above normal for age/sex ({normal_esr} mm/hr)')
    else:
        breakdown['ESR'] = ('Normal', 0, f'{esr} mm/hr — within normal range (<{normal_esr} mm/hr)')

    # CRP scoring (Sproston & Ashworth, 2018)
    if crp > 50:
        score += 2
        breakdown['CRP'] = ('High', 2, f'{crp:.1f} mg/L — severe inflammation (>50 mg/L)')
    elif crp > 6:
        score += 1
        breakdown['CRP'] = ('Moderate', 1, f'{crp:.1f} mg/L — mild-to-moderate inflammation')
    else:
        breakdown['CRP'] = ('Normal', 0, f'{crp:.1f} mg/L — within normal range (<6 mg/L)')

    # Uric Acid scoring — ELEVATED, not low, per Shin et al. 2023 / Sautin & Johnson 2008
    ua_threshold = 7.0 if sex == "M" else 6.0
    ua_borderline = ua_threshold * 0.85

    if uric_acid > ua_threshold:
        score += 2
        breakdown['Uric Acid'] = ('High', 2, f'{uric_acid:.1f} mg/dL — hyperuricemia (>{ua_threshold} mg/dL)')
    elif uric_acid > ua_borderline:
        score += 1
        breakdown['Uric Acid'] = ('Borderline', 1, f'{uric_acid} mg/dL — borderline elevated')
    else:
        breakdown['Uric Acid'] = ('Normal', 0, f'{uric_acid} mg/dL — within normal range')

    # Risk category
    if score >= 4:
        category, color = "HIGH", "high"
        action = "⚠️ Urgent clinical evaluation required. Refer for sputum smear microscopy and GeneXpert testing immediately."
        next_steps = [
            "Refer immediately for sputum smear microscopy",
            "Request GeneXpert MTB/RIF test",
            "Chest X-ray recommended",
            "Isolate patient pending confirmation",
            "Notify clinician urgently"
        ]
    elif score >= 2:
        category, color = "MEDIUM", "medium"
        action = "🔍 Clinical follow-up required. Repeat biomarkers in 2 weeks and monitor closely."
        next_steps = [
            "Schedule follow-up appointment within 2 weeks",
            "Repeat ESR and CRP after 14 days",
            "Review patient's symptom history",
            "Consider chest X-ray if symptoms persist",
            "Monitor weight and temperature"
        ]
    else:
        category, color = "LOW", "low"
        action = "✅ Low TB risk based on current biomarkers. Routine monitoring advised."
        next_steps = [
            "Routine monitoring — no urgent action needed",
            "Repeat tests if new symptoms develop",
            "Maintain general health screening schedule",
            "Patient education on TB warning signs"
        ]

    return score, category, color, action, next_steps, breakdown


# ─── SEX/AGE-AWARE SYNTHETIC BIOMARKER SAMPLING ──────────────────────────────
# Older version generated biomarkers from fixed ranges per risk group, but ESR and
# Uric Acid thresholds are sex-specific in classify_risk() — a fixed range could
# straddle the female threshold while sitting safely under the male one (or vice versa),
# silently pushing patients into the wrong risk category and weakening MEDIUM-tier
# ML performance. This version samples each biomarker to hit an exact scoring tier
# (0/1/2 points) computed from the SAME sex/age-specific thresholds classify_risk() uses,
# so the intended risk group always matches the actual computed category.

# Tier combinations (ESR tier, CRP tier, UA tier) that sum to each target risk band
LOW_COMBOS    = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]                                   # score 0-1
MEDIUM_COMBOS = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2)]    # score 2-3
HIGH_COMBOS   = [(2, 2, 0), (2, 0, 2), (0, 2, 2), (2, 1, 1), (1, 2, 1),
                 (1, 1, 2), (2, 2, 1), (2, 1, 2), (1, 2, 2), (2, 2, 2)]                          # score 4-6

def _sample_marker(tier, kind, normal_esr=None, ua_threshold=None, ua_borderline=None):
    """Sample a biomarker value guaranteed to land in the given scoring tier (0/1/2 points)."""
    if kind == "esr":
        if tier == 0:
            return round(np.random.uniform(5, max(6, normal_esr - 1)), 1)
        elif tier == 1:
            return round(np.random.uniform(normal_esr + 1, 60), 1)
        else:
            return round(np.random.uniform(61, 135), 1)
    elif kind == "crp":
        if tier == 0:
            return round(np.random.uniform(1, 6), 1)
        elif tier == 1:
            return round(np.random.uniform(6.5, 50), 1)
        else:
            return round(np.random.uniform(51, 95), 1)
    else:  # uric acid
        lo_bound = 2.0
        if tier == 0:
            return round(np.random.uniform(lo_bound, max(lo_bound + 0.5, ua_borderline - 0.1)), 1)
        elif tier == 1:
            return round(np.random.uniform(ua_borderline + 0.1, ua_threshold - 0.1), 1)
        else:
            return round(np.random.uniform(ua_threshold + 0.1, ua_threshold + 3.5), 1)

def sample_patient_biomarkers(group, sex, age):
    """Generate one synthetic patient's ESR/CRP/Uric Acid that reliably lands in `group`
    (low/medium/high) once run through classify_risk(), for the given sex and age."""
    normal_esr = (15 if age <= 50 else 20) if sex == "M" else (20 if age <= 50 else 30)
    ua_threshold = 7.0 if sex == "M" else 6.0
    ua_borderline = ua_threshold * 0.85

    combo_pool = {"low": LOW_COMBOS, "medium": MEDIUM_COMBOS, "high": HIGH_COMBOS}[group]
    esr_tier, crp_tier, ua_tier = combo_pool[np.random.randint(len(combo_pool))]

    esr = _sample_marker(esr_tier, "esr", normal_esr=normal_esr)
    crp = _sample_marker(crp_tier, "crp")
    ua = _sample_marker(ua_tier, "ua", ua_threshold=ua_threshold, ua_borderline=ua_borderline)
    return esr, crp, ua

# ─── GENERATE SYNTHETIC COHORT ────────────────────────────────────────────────
@st.cache_data
# ─── GENERATE SYNTHETIC COHORT (CORRECTED UA DIRECTION) ──────────────────────
@st.cache_data
def generate_cohort(n=500):
    np.random.seed(42)
    records = []

    states = ["Lagos","Kano","Abuja","Rivers","Oyo","Enugu",
              "Kaduna","Delta","Borno","Anambra","Ekiti","Osun",
              "Sokoto","Bauchi","Plateau","Imo","Abia","Cross River"]

    groups = (
        [("low", 0.4)] * int(n * 0.4) +
        [("medium", 0.35)] * int(n * 0.35) +
        [("high", 0.25)] * int(n * 0.25)
    )
    np.random.shuffle(groups)

    for i, (group, _) in enumerate(groups[:n]):
        sex = "M" if i % 2 == 0 else "F"
        age = np.random.randint(18, 66)
        state = np.random.choice(states)

        esr, crp, ua = sample_patient_biomarkers(group, sex, age)

        score, category, _, _, _, _ = classify_risk(esr, crp, ua, sex, age)
        records.append({
            "Patient ID": f"P{i+1:04d}",
            "Age": age,
            "Sex": sex,
            "State": state,
            "ESR (mm/hr)": esr,
            "CRP (mg/L)": crp,
            "Uric Acid (mg/dL)": ua,
            "Risk Score": score,
            "Risk Category": category,
            "True Group": group.capitalize()
        })

    return pd.DataFrame(records)


@st.cache_data
def train_ml_models(n=1000):
    np.random.seed(99)
    records = []
    groups = (["low"] * int(n*0.40) + ["medium"] * int(n*0.35) + ["high"] * int(n*0.25))
    np.random.shuffle(groups)
    for i, group in enumerate(groups[:n]):
        sex = "M" if i % 2 == 0 else "F"
        age = np.random.randint(18, 66)
        esr, crp, ua = sample_patient_biomarkers(group, sex, age)
        score, category, _, _, _, _ = classify_risk(esr, crp, ua, sex, age)
        records.append({"ESR": esr, "CRP": crp, "Uric_Acid": ua, "Risk_Category": category})
    df_ml = pd.DataFrame(records)
    le = LabelEncoder()
    df_ml["Risk_Encoded"] = le.fit_transform(df_ml["Risk_Category"])
    X = df_ml[["ESR", "CRP", "Uric_Acid"]]
    y = df_ml["Risk_Encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    cm_lr = confusion_matrix(y_test, lr_pred)
    cm_dt = confusion_matrix(y_test, dt_pred)
    return lr, dt, le, X_test, y_test, lr_pred, dt_pred, lr_acc, dt_acc, df_ml, cm_lr, cm_dt

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

# HERO
st.markdown("""
<div class="hero">
    <div class="tagline">🌍 Voice for Health Africa Initiative</div>
    <h1>🫁 TB Insight Africa</h1>
    <p>A proof-of-concept health intelligence tool using routine laboratory biomarkers<br>
    to support early tuberculosis detection across Africa</p>
</div>
""", unsafe_allow_html=True)

# STATS ROW
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<div class="stat-card">
        <div class="stat-number">10.6M</div>
        <div class="stat-label">Global TB Cases (2022)</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="stat-card">
        <div class="stat-number">23.4%</div>
        <div class="stat-label">Africa's Share of Global Burden</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="stat-card">
        <div class="stat-number">479K</div>
        <div class="stat-label">Nigeria TB Cases (2022)</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""<div class="stat-card">
        <div class="stat-number">#6</div>
        <div class="stat-label">Nigeria's Global TB Rank</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔬 Risk Classifier",
    "🤖 ML Model",
    "📊 Data Dashboard",
    "🧬 Biomarker Guide",
    "👥 Our Team",
    "📖 About",
    "🌍 Regional Context"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RISK CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">TB Risk Classifier</div>', unsafe_allow_html=True)
    st.markdown("Enter a patient's routine laboratory values below to generate an instant TB risk assessment.")

    with st.expander("ℹ️ What is this, in plain terms?"):
        st.markdown("""
        This is a **scorecard**, not artificial intelligence. It works the same way a clinician
        already scores risk manually — each biomarker result (ESR, CRP, uric acid, etc.) is checked
        against known WHO/clinical reference ranges, and points are added up to give a **Risk Score out of 6**.

        - No "learning" happens here — the rules are fixed and based on published thresholds.
        - You can trace exactly *why* a patient got a given score.
        - Think of it like a checklist a doctor could do on paper — this just does it instantly.

        👉 For the version that uses **Machine Learning** (a model trained on many past patient examples),
        see the **🤖 ML Model** tab.
        """)
    
    col_form, col_result = st.columns([1, 1.2], gap="large")
    
    with col_form:
        st.markdown("#### Patient Information")
        
        p_sex = st.selectbox("Patient Sex", ["Male", "Female"])
        p_age = st.number_input("Age (years)", min_value=1, max_value=100, value=35)
        
        st.markdown("#### Biomarker Values")
        
        p_esr = st.number_input(
            "ESR — Erythrocyte Sedimentation Rate (mm/hr)",
            min_value=0.0, max_value=200.0, value=45.0, step=0.5,
            help="Normal: Male <15, Female <20 mm/hr"
        )
        
        p_crp = st.number_input(
            "CRP — C-Reactive Protein (mg/L)",
            min_value=0.0, max_value=300.0, value=12.0, step=0.5,
            help="Normal: <6mg/L"
        )
        
        p_ua = st.number_input(
    "Uric Acid (mg/dL)",
    min_value=0.0, max_value=15.0, value=3.8, step=0.1,
    help="Normal: Male 3.5-7.0, Female 2.5-6.0 mg/dL"
)
        
        classify_btn = st.button("🔍 Classify TB Risk", use_container_width=True)
    
    with col_result:
        if classify_btn:
            sex_code = "M" if p_sex == "Male" else "F"
            score, category, color, action, next_steps, breakdown = classify_risk(
                p_esr, p_crp, p_ua, sex_code, p_age
            )
            
            # Risk card
            st.markdown(f"""
            <div class="risk-card-{color}">
                <div class="risk-title">TB Risk Assessment</div>
                <div class="risk-label">{category} RISK</div>
                <div class="risk-score">Risk Score: {score} / 6</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 6], 'tickwidth': 1},
                    'bar': {'color': "#C0392B" if color=="high" else ("#F39C12" if color=="medium" else "#27AE60")},
                    'steps': [
                        {'range': [0, 2], 'color': '#d5f5e3'},
                        {'range': [2, 4], 'color': '#fdebd0'},
                        {'range': [4, 6], 'color': '#fadbd8'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(t=30, b=0, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Biomarker breakdown
            st.markdown("**Biomarker Breakdown:**")
            for marker, (status, pts, detail) in breakdown.items():
                icon = "🔴" if pts == 2 else ("🟡" if pts == 1 else "🟢")
                st.markdown(f"{icon} **{marker}:** {detail} → **{pts} pts**")
            
            # Recommendation
            st.markdown(f"""
            <div class="recommendation-box">
                <strong>Clinical Recommendation:</strong><br>{action}
                <br><br><strong>Next Steps:</strong>
                <ul>{''.join(f'<li>{s}</li>' for s in next_steps)}</ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="disclaimer">
                ⚠️ <strong>Disclaimer:</strong> This tool is a proof-of-concept only. 
                Results are based on a rule-based scoring model trained on published literature 
                and synthetic data. It is not a diagnostic tool and does not replace clinical 
                judgment. Always consult a qualified healthcare professional.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👈 Enter patient biomarker values and click **Classify TB Risk** to generate an assessment.")
            
            # Reference ranges
            st.markdown("#### Quick Reference — Normal Ranges")
            ref_data = {
    "Biomarker": ["ESR (Male)", "ESR (Female)", "CRP", "Uric Acid (Male)", "Uric Acid (Female)"],
    "Normal Range": ["< 15 mm/hr", "< 20 mm/hr", "< 6 mg/L", "3.5–7.0 mg/dL", "2.5–6.0 mg/dL"],
    "TB-Suggestive": ["> 60 mm/hr", "> 60 mm/hr", "> 50 mg/L", "> 7.0 mg/dL", "> 6.0 mg/dL"]
}
            st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🤖 Machine Learning Model</div>', unsafe_allow_html=True)
    st.markdown("Two supervised ML classifiers trained on **1,000 synthetic patients** generated from published biomarker literature.")

    with st.expander("ℹ️ How is this different from the Risk Classifier tab?"):
        st.markdown("""
        The **🔬 Risk Classifier** tab uses fixed, human-written rules (like a scoring checklist).
        This tab is different: it's a model that **learned patterns on its own** from 1,000 example
        patient cases, instead of following rules a person wrote down.

        - It was shown many example patients (with known outcomes) and learned which
          *combinations* of ESR, CRP, and uric acid tend to go with higher TB risk —
          including subtle patterns a fixed checklist might miss.
        - It's less "explainable" step-by-step than the scorecard, but often more accurate
          because it can weigh multiple biomarkers together, not just one at a time.
        - The data used here is **synthetic** (computer-generated from published literature),
          not real patient records — so treat this tab as a research/demo preview of what
          ML could offer once trained on real clinical data.

        In short: **Risk Classifier = a checklist. ML Model = a pattern-detector.**
        Both are shown side-by-side so you can compare how they score the same patient.
        """)

    lr, dt, le, X_test, y_test, lr_pred, dt_pred, lr_acc, dt_acc, df_ml, cm_lr, cm_dt = train_ml_models(1000)

    # Model accuracy cards
    st.markdown("### Model Performance")
    ma1, ma2, ma3 = st.columns(3)
    with ma1:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number" style="color:#27AE60">{lr_acc*100:.1f}%</div>
            <div class="stat-label">Logistic Regression Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with ma2:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number" style="color:#1A3A5C">{dt_acc*100:.1f}%</div>
            <div class="stat-label">Decision Tree Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with ma3:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number" style="color:#C0392B">1,000</div>
            <div class="stat-label">Synthetic Patients Trained On</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Precision / Recall / F1 per class — accuracy alone can hide weak spots in a health-screening
    # tool, especially for an imbalanced middle class, so this is shown explicitly rather than
    # only reporting a single headline accuracy number.
    st.markdown("### Precision & Recall by Risk Class")
    st.caption(
        "Accuracy alone can look good while quietly failing one risk tier. Recall (sensitivity) "
        "matters most for HIGH risk — it tells us how many true high-risk patients the model actually catches."
    )

    from sklearn.metrics import precision_recall_fscore_support
    prec_lr, rec_lr, f1_lr, support_lr = precision_recall_fscore_support(y_test, lr_pred, labels=range(len(le.classes_)))
    prec_dt, rec_dt, f1_dt, support_dt = precision_recall_fscore_support(y_test, dt_pred, labels=range(len(le.classes_)))

    metrics_df = pd.DataFrame({
        "Risk Class": le.classes_,
        "LR Precision": [f"{p:.1%}" for p in prec_lr],
        "LR Recall (Sensitivity)": [f"{r:.1%}" for r in rec_lr],
        "DT Precision": [f"{p:.1%}" for p in prec_dt],
        "DT Recall (Sensitivity)": [f"{r:.1%}" for r in rec_dt],
        "Test Set Size": support_lr,
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Note:</strong> Synthetic patients are generated with sex/age-specific biomarker
        thresholds matching the Risk Classifier's own scoring logic, so all three risk tiers
        (LOW/MEDIUM/HIGH) are reliably represented and learnable. Even so, this is
        <strong>synthetic data</strong>, not real patient records — genuine clinical validation
        requires ethically approved data such as the ISTH cohort currently in progress.
    </div>
    """, unsafe_allow_html=True)

    # Confusion matrices
    st.markdown("### Confusion Matrices")
    cm_col1, cm_col2 = st.columns(2)
    labels = le.classes_

    with cm_col1:
        fig_cm_lr = px.imshow(
            cm_lr,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels, y=labels,
            title="Logistic Regression — Confusion Matrix",
            color_continuous_scale=["#d5f5e3", "#1A3A5C"],
            text_auto=True
        )
        fig_cm_lr.update_layout(height=350)
        st.plotly_chart(fig_cm_lr, use_container_width=True)

    with cm_col2:
        fig_cm_dt = px.imshow(
            cm_dt,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels, y=labels,
            title="Decision Tree — Confusion Matrix",
            color_continuous_scale=["#fadbd8", "#C0392B"],
            text_auto=True
        )
        fig_cm_dt.update_layout(height=350)
        st.plotly_chart(fig_cm_dt, use_container_width=True)

    # Feature importance
    st.markdown("### Feature Importance — Decision Tree")
    features = ["ESR (mm/hr)", "CRP (mg/L)", "Uric Acid (mg/dL)"]
    importances = dt.feature_importances_
    fig_fi = px.bar(
        x=importances,
        y=features,
        orientation="h",
        title="Which Biomarker Drives TB Risk Prediction Most?",
        labels={"x": "Importance Score", "y": "Biomarker"},
        color=importances,
        color_continuous_scale=["#d6e4f0", "#C0392B"]
    )
    fig_fi.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Live ML prediction
    st.markdown("### Live ML Prediction")
    st.markdown("Compare rule-based scoring vs ML model prediction:")

    ml_col1, ml_col2 = st.columns(2)
    with ml_col1:
        ml_esr = st.number_input("ESR (mm/hr)", min_value=0.0, max_value=200.0, value=85.0, step=0.5, key="ml_esr")
        ml_crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=300.0, value=55.0, step=0.5, key="ml_crp")
        ml_ua  = st.number_input("Uric Acid (mg/dL)", min_value=0.0, max_value=15.0, value=1.9, step=0.1, key="ml_ua")
        ml_btn = st.button("🤖 Run ML Prediction", use_container_width=True)

    with ml_col2:
        if ml_btn:
            input_data = pd.DataFrame([[ml_esr, ml_crp, ml_ua]], columns=["ESR", "CRP", "Uric_Acid"])
            lr_result = le.inverse_transform(lr.predict(input_data))[0]
            dt_result = le.inverse_transform(dt.predict(input_data))[0]
            rule_score, rule_cat, rule_color, _, _, _ = classify_risk(ml_esr, ml_crp, ml_ua)

            color_map = {"HIGH": "#C0392B", "MEDIUM": "#F39C12", "LOW": "#27AE60"}

            st.markdown(f"""
            <div style="background:white;padding:1.5rem;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08)">
                <p style="font-weight:700;color:#1A3A5C;margin:0 0 1rem">Prediction Results:</p>
                <p>🔬 <strong>Rule-Based Score:</strong> 
                    <span style="color:{color_map.get(rule_cat,'#000')};font-weight:700">{rule_cat}</span> 
                    ({rule_score}/6 pts)</p>
                <p>📈 <strong>Logistic Regression:</strong> 
                    <span style="color:{color_map.get(lr_result,'#000')};font-weight:700">{lr_result}</span></p>
                <p>🌳 <strong>Decision Tree:</strong> 
                    <span style="color:{color_map.get(dt_result,'#000')};font-weight:700">{dt_result}</span></p>
                <hr>
                <p style="font-size:0.8rem;color:#666;margin:0">
                When all three methods agree, confidence is highest.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👈 Enter biomarker values and click **Run ML Prediction** to compare models.")

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Note:</strong> Models are trained on synthetic data generated from published 
        clinical literature. Real-world validation requires ethically approved patient data. 
        Algorithms used: Logistic Regression and Decision Tree (scikit-learn).
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Synthetic Patient Cohort Dashboard</div>', unsafe_allow_html=True)
    st.caption("Visualising 500 synthetic patients generated from published biomarker literature. For proof-of-concept demonstration only.")
    
    df = generate_cohort(500)
    
    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    high_count = len(df[df['Risk Category'] == 'HIGH'])
    med_count = len(df[df['Risk Category'] == 'MEDIUM'])
    low_count = len(df[df['Risk Category'] == 'LOW'])
    
    m1.metric("Total Patients", "500")
    m2.metric("🔴 High Risk", high_count, f"{high_count/5:.0f}%")
    m3.metric("🟡 Medium Risk", med_count, f"{med_count/5:.0f}%")
    m4.metric("🟢 Low Risk", low_count, f"{low_count/5:.0f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        # Risk distribution pie
        risk_counts = df['Risk Category'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Patient Risk Distribution",
            color=risk_counts.index,
            color_discrete_map={"HIGH": "#C0392B", "MEDIUM": "#F39C12", "LOW": "#27AE60"}
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_c2:
        # Biomarker means by group
        means = df.groupby('Risk Category')[['ESR (mm/hr)', 'CRP (mg/L)', 'Uric Acid (mg/dL)']].mean().round(1)
        fig_bar = px.bar(
            means.reset_index(),
            x='Risk Category',
            y=['ESR (mm/hr)', 'CRP (mg/L)'],
            title="Mean ESR & CRP by Risk Group",
            barmode='group',
            color_discrete_map={'ESR (mm/hr)': '#1A3A5C', 'CRP (mg/L)': '#C0392B'},
            category_orders={"Risk Category": ["LOW", "MEDIUM", "HIGH"]}
        )
        fig_bar.update_layout(height=350)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    col_c3, col_c4 = st.columns(2)
    
    with col_c3:
        # ESR vs CRP scatter
        fig_scatter = px.scatter(
            df, x='ESR (mm/hr)', y='CRP (mg/L)',
            color='Risk Category',
            title="ESR vs CRP — Risk Stratification",
            color_discrete_map={"HIGH": "#C0392B", "MEDIUM": "#F39C12", "LOW": "#27AE60"},
            opacity=0.6,
            category_orders={"Risk Category": ["LOW", "MEDIUM", "HIGH"]}
        )
        fig_scatter.update_layout(height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col_c4:
        # Uric acid distribution
        fig_ua = px.box(
            df, x='Risk Category', y='Uric Acid (mg/dL)',
            title="Uric Acid Distribution by Risk Group",
            color='Risk Category',
            color_discrete_map={"HIGH": "#C0392B", "MEDIUM": "#F39C12", "LOW": "#27AE60"},
            category_orders={"Risk Category": ["LOW", "MEDIUM", "HIGH"]}
        )
        fig_ua.update_layout(height=350)
        st.plotly_chart(fig_ua, use_container_width=True)
    
    # State distribution
    state_risk = df[df['Risk Category'] == 'HIGH']['State'].value_counts().head(10)
    fig_state = px.bar(
        x=state_risk.index, y=state_risk.values,
        title="High-Risk Patients by Nigerian State (Top 10)",
        labels={'x': 'State', 'y': 'High Risk Count'},
        color=state_risk.values,
        color_continuous_scale=['#fadbd8', '#C0392B']
    )
    fig_state.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_state, use_container_width=True)
    
    # Data table
    with st.expander("📋 View Raw Patient Data"):
        st.dataframe(
            df.drop('True Group', axis=1),
            use_container_width=True,
            hide_index=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BIOMARKER GUIDE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Biomarker Science Guide</div>', unsafe_allow_html=True)
    
    col_b1, col_b2, col_b3 = st.columns(3)
    
    with col_b1:
        st.markdown("""
        <div class="biomarker-card">
            <h3>🔬 ESR</h3>
            <p><strong>Erythrocyte Sedimentation Rate</strong></p>
            <p>ESR measures how quickly red blood cells settle. In TB, chronic inflammation causes proteins to coat red cells, making them settle faster.</p>
            <hr>
            <p>🟢 Normal Male: &lt;15 mm/hr</p>
            <p>🟢 Normal Female: &lt;20 mm/hr</p>
            <p>🟡 Elevated: 20–50 mm/hr</p>
            <p>🔴 TB-suggestive: &gt;60 mm/hr</p>
            <p>🔴 Active TB: &gt;100 mm/hr</p>
            <hr>
            <small><em>Source: University of Maiduguri Teaching Hospital Study (PMID 20415078)</em></small>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b2:
        st.markdown("""
        <div class="biomarker-card">
            <h3>🧪 CRP</h3>
            <p><strong>C-Reactive Protein</strong></p>
            <p>CRP is produced by the liver in response to inflammation. It rises sharply in active TB and drops with effective treatment — making it a monitoring marker too.</p>
            <hr>
            <p>🟢 Normal: &lt;5 mg/L</p>
            <p>🟡 Mild-to-Moderate: 6–50 mg/L</p>
            <p>🔴 Severe: >50 mg/L</p>
            <hr>
            <small><em>Source:Sproston & Ashworth, 2018; Uganda Diagnostic Study — 78% sensitivity (PMID 33087439)</em></small>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b3:
        st.markdown("""
        <div class="biomarker-card">
            <h3>⚗️ Uric Acid</h3>
            <p><strong>The Nuanced Biomarker</strong></p>
            <p>Active TB causes tissue damage that releases uric acid as a byproduct, acting as a damage signal (DAMP) that activates inflammatory pathways. Elevated uric acid, not low, is the expected pattern in untreated active TB.</p>
            <hr>
            <p>🟢 Normal Male: 3.5–7.0 mg/dL</p>
            <p>🟢 Normal Female: 2.5–6.0 mg/dL</p>
            <p>🟡 Borderline Elevated: 6.0–7.0 mg/dL (M) / 5.1–6.0 mg/dL (F)</p>
            <p>🔴 Hyperuricemia: &gt;7.0 mg/dL (M) / &gt;6.0 mg/dL (F)</p>
            <hr>
            <small><em>Source: Cameroon Study — 58.3% hyperuricemia in TB patients (IJMY 2018); Sautin &amp; Johnson, 2008</em></small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<p>🟡 Borderline Elevate
    <div class="recommendation-box">
        <strong>🧠 Why Combine All Three?</strong><br><br>
        ESR alone is sensitive but not TB-specific — it rises in many conditions. 
        CRP adds specificity to the inflammatory signal. 
        Uric Acid provides a third dimension, reflecting metabolic disturbance unique to active TB. 
        Combined, the three biomarkers create a multi-signal risk score that is significantly more 
        informative than any single marker alone. This mirrors established clinical reasoning in 
        Chemical Pathology: <strong>no single marker is enough — the pattern matters.</strong>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TEAM
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Our Team</div>', unsafe_allow_html=True)
    
    t1, t2, t3, t4 = st.columns(4)
    
    team = [
        ("👨‍🏫", "Dr. Dic-Ijiewere Ebenezer", "Academic Authority",
         "Head of Department, Chemical Pathology, Ambrose Alli University — published researcher in biomarker and oxidative stress science, providing academic credibility and methodology oversight"),
        ("👩‍🔬", "Adese Abosede Oluwanifemi", "Founder",
         "Medical Laboratory Science Student | Data Scientist | Machine Learning"),
        ("👨‍💻", "Ibhate Destiny", "Co-Founder",
         "Medical Laboratory Science Student | Machine Learning | AI Engineer"),
        ("👩‍⚕️", "Aishat Afunlehin", "Microbiologist & Researcher",
         "Microbiologist with research expertise supporting biomarker validation and clinical interpretation")
    ]
    
    for col, (icon, name, role, desc) in zip([t1, t2, t3, t4], team):
        with col:
            st.markdown(f"""
            <div class="team-card">
                <div style="font-size:3rem">{icon}</div>
                <h4 style="color:#1A3A5C;margin:0.5rem 0 0.2rem">{name}</h4>
                <p style="color:#C0392B;font-size:0.8rem;font-weight:600;margin:0">{role}</p>
                <p style="font-size:0.78rem;color:#666;margin-top:0.5rem">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">About TB Insight Africa</div>', unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("""
        ### The Problem
        Tuberculosis kills over **1.25 million people annually**. Africa carries 25% of the global 
        burden, with Nigeria ranking 6th in the world with an estimated 510,000 new cases each year.
        
        The most critical gap is not treatment — it is **detection**. Thousands of patients visit 
        laboratories daily across Nigeria, generating routine biomarker data that is collected, 
        recorded, and then ignored.
        
        ### Our Solution
        TB Insight Africa uses three routine laboratory biomarkers — **ESR, CRP, and Uric Acid** — 
        already collected in most Nigerian laboratories, to automatically classify patients as 
        Low, Medium, or High TB risk.
        
        No new equipment. No additional cost. Just smarter use of existing data.
        """)
    
    with col_a2:
        st.markdown("""
        ### What Makes Us Different
        
        | Existing Solutions | TB Insight Africa |
        |---|---|
        | Require expensive GeneXpert | Uses existing lab data |
        | Specialist infrastructure | Works in any lab |
        | Single biomarker focus | Multi-signal approach |
        | Urban-centric | Scalable to rural settings |
        
        ### Current Stage
        This is a **proof-of-concept** built on:
        - Published clinical literature (PubMed)
        - WHO Global TB Report 2025 data
        - 500 synthetic patient records
        - Expert input from PhD-level advisor
        
        Real-world clinical validation will follow ethical approval submission to a Nigerian teaching hospital.
        
        ### Vision
        TB Insight Africa is the first product of **Voice for Health Africa** — 
        a Pan-African health intelligence initiative transforming routine laboratory 
        data into actionable insights that improve healthcare outcomes across the continent.
        """)


# FOOTER
st.markdown("""
<div class="footer">
    <h3>🫁 TB Insight Africa</h3>
    <p>A Voice for Health Africa Initiative</p>
    <p style="opacity:0.7;font-size:0.85rem">
        Proof-of-concept tool | Built on WHO 2025 data & published clinical literature<br>
        Not a diagnostic tool. For demonstration and research purposes only.
    </p>
    <p style="opacity:0.5;font-size:0.75rem;margin-top:1rem">
        © 2026 TB Insight Africa | Contact: abosedethelabgirl@gmail.com
    </p>
</div>
""", unsafe_allow_html=True)
with tab7:
    render_who_burden_tab()
