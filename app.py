import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap

# ============ BASIC SETUP ============
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")
st.title("🏥 Heart Disease ML Dashboard")
st.caption("Hager • Mai • Menna • Shahd")

@st.cache_resource
def load_artifacts():
    xgb = joblib.load("models/xgb_model.pkl")
    rf = joblib.load("models/rf_model.pkl")
    lg = joblib.load("models/lg_model.pkl")
    dbscan = joblib.load("models/dbscan_model.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")          # سكالر واحد فقط
    return xgb, rf, lg, dbscan, kmeans, scaler

xgb_model, rf_model, lg_model, dbscan_model, kmeans_model, scaler = load_artifacts()

# ============ SIDEBAR: GLOBAL CONTROLS ============
st.sidebar.header("⚙ Settings")

mode = st.sidebar.radio(
    "Choose action",
    ["Single patient prediction", "Cluster exploration", "Batch upload"],
)

model_choice = st.sidebar.selectbox(
    "Choose model",
    ["XGBoost", "Random Forest", "Logistic Regression"],
)

model_map = {
    "XGBoost": xgb_model,
    "Random Forest": rf_model,
    "Logistic Regression": lg_model,
}
current_model = model_map[model_choice]

# ============ FEATURES ============
# الأعمدة المستخدمة في الكلاسترينج (لازم تطابق تدريب KMeans)
cluster_features = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
]

# الأعمدة اللي اتعمل لها scaling في تدريب نماذج التصنيف
num_cols = ["age", "height", "weight"]

# ============ HAGER: PATIENT FORM (11 FEATURES) ============
def patient_form():
    st.subheader("👤 Patient clinical data")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", 20, 90, 50)
        height = st.number_input("Height (cm)", 140, 210, 170)
        weight = st.number_input("Weight (kg)", 40, 160, 75)
    with c2:
        ap_hi = st.number_input("Systolic BP (ap_hi)", 80, 240, 120)
        ap_lo = st.number_input("Diastolic BP (ap_lo)", 40, 160, 80)
        gender = st.selectbox(
            "Gender", [1, 2],
            format_func=lambda x: "Male" if x == 1 else "Female",
        )
    with c3:
        cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
        gluc = st.selectbox("Glucose", [1, 2, 3])
        smoke = st.checkbox("Smoker")
        alco = st.checkbox("Alcohol")
        active = st.checkbox("Physically active", True)

    if ap_hi <= ap_lo:
        st.error("Systolic pressure (ap_hi) must be greater than diastolic (ap_lo).")
        return None

    df = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "height": [height],
        "weight": [weight],
        "ap_hi": [ap_hi],
        "ap_lo": [ap_lo],
        "cholesterol": [cholesterol],
        "gluc": [gluc],
        "smoke": [int(smoke)],
        "alco": [int(alco)],
        "active": [int(active)],
    })
    return df

# ============ COMMON PREPROCESSING ============
def add_default_id_year(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["id"] = 0
    df["year"] = 2020
    cols_order = [
        "id", "age", "gender", "height", "weight",
        "ap_hi", "ap_lo", "cholesterol", "gluc",
        "smoke", "alco", "active", "year",
    ]
    df = df[cols_order]
    return df

def preprocess(df):
    df_proc = df.copy()
    df_proc[num_cols] = scaler.transform(df_proc[num_cols])
    return df_proc

# ============ MAI: REAL-TIME PREDICTION + CI ============
def predict_single(df_raw):
    df_raw = add_default_id_year(df_raw)
    df = preprocess(df_raw)

    proba = current_model.predict_proba(df)[0, 1]

    probs_models = [
        xgb_model.predict_proba(df)[0, 1],
        rf_model.predict_proba(df)[0, 1],
        lg_model.predict_proba(df)[0, 1],
    ]
    mean_p = np.mean(probs_models)
    low = np.percentile(probs_models, 10)
    high = np.percentile(probs_models, 90)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Selected model risk", f"{proba:.1%}")
    with c2:
        st.metric("Ensemble mean risk", f"{mean_p:.1%}")
    with c3:
        st.metric("Confidence interval", f"{low:.0%} – {high:.0%}")

    st.progress(float(proba))
    st.write("⬆ High risk" if proba > 0.5 else "⬇ Low risk")

# ============ MENNA: CLUSTER VISUALIZATION ============
def cluster_view(df_raw):
    st.subheader("🎯 Cluster assignment")

    # نسخة من بيانات المريض
    df_cluster = df_raw.copy()

    # ========== 1) Apply scaler ONLY on numerical columns ==========
    df_scaled = df_cluster.copy()
    df_scaled[num_cols] = scaler.transform(df_scaled[num_cols])

    # ========== 2) Select the full 11 KMeans features ==========
    X_k = df_scaled[cluster_features]   # هنا كل الأعمدة اللي اتدرب عليها KMeans

    # ========== 3) Predict clusters ==========
    k_cluster = int(kmeans_model.predict(X_k)[0])
    d_label = int(dbscan_model.fit_predict(X_k)[0])

    # ========== 4) Display results ==========
    c1, c2 = st.columns(2)
    c1.metric("KMeans cluster", k_cluster)
    c2.metric("DBSCAN label", d_label if d_label != -1 else "Noise")

    # ========== 5) Visualization ==========
    feat_for_plot = df_raw[["age", "ap_hi"]].copy()
    feat_for_plot["cluster"] = k_cluster

    fig = px.scatter(
        feat_for_plot,
        x="age",
        y="ap_hi",
        color="cluster",
        title="Patient position by age & systolic BP",
    )

    st.plotly_chart(fig, use_container_width=True)


# ============ SHAHD: SHAP + BATCH ============
def shap_explain(df_raw):
    st.subheader("🔍 SHAP explanation (XGBoost)")

    df_raw = add_default_id_year(df_raw)
    df = preprocess(df_raw)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(df)

    # Create a stable Matplotlib figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))

    shap.plots.waterfall(shap_values[0], show=False)
    
    st.pyplot(fig)


def batch_mode():
    st.subheader("📂 Batch prediction for multiple patients")
    file = st.file_uploader("Upload CSV with 11 clinical columns", type=["csv"])
    if file is None:
        st.info("Template columns: age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active")
        return
    df_raw = pd.read_csv(file)
    df_raw = add_default_id_year(df_raw)
    df = preprocess(df_raw)
    probs = current_model.predict_proba(df)[:, 1]
    df_raw["risk_proba"] = probs
    st.write("Results:", df_raw.head())
    st.download_button("Download predictions CSV", df_raw.to_csv(index=False), "predictions.csv")

# ============ MAIN LOGIC ============
if mode == "Single patient prediction":
    patient_df = patient_form()
    if patient_df is not None and st.button("🔮 Predict", use_container_width=True):
        predict_single(patient_df)
        cluster_view(patient_df)
        with st.expander("Explain prediction (SHAP)"):
            shap_explain(patient_df)

elif mode == "Cluster exploration":
    st.info("Use the patient form on the left, then click below.")
    patient_df = patient_form()
    if patient_df is not None and st.button("🎯 Show clusters", use_container_width=True):
        cluster_view(patient_df)

else:  # Batch upload
    batch_mode()