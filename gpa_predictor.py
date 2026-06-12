"""
Aplikasi Prediksi Nilai Akhir Semester (post_semester_gpa)

Model ML yang digunakan:
- Linear Regression
- XGBoost

Semua model menggunakan pipeline: SimpleImputer + StandardScaler + Regressor
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Prediksi Nilai Akhir Semester",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_models():
    """Load two regression models (cached)"""
    models = {}
    try:
        models['Linear Regression'] = joblib.load("linier_regression_with_simple_and_standard.pkl")
        models['XGBoost'] = joblib.load("xgboost_with_simple_and_standard.pkl")
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found! Please run ai_train_regression.ipynb first to train and save the models.\n\nError: {e}")
        st.stop()
    return models

models = load_models()

# ============================================================================
# LOAD SAMPLE DATA
# ============================================================================
@st.cache_data
def load_sample_data():
    """Load dataset for preview (cached)"""
    try:
        df = pd.read_csv('ai_student_impact_clean.csv')
        columns_to_show = [
            'major_category', 'year_of_study', 'pre_semester_gpa',
            'weekly_genai_hours', 'primary_use_case', 'prompt_engineering_skill',
            'tool_diversity', 'paid_subscription', 'traditional_study_hours',
            'perceived_ai_dependency', 'institutional_policy',
            'anxiety_level_during_exams', 'post_semester_gpa',
            'skill_retention_score', 'burnout_risk_level'
        ]
        # Filter hanya kolom yang tersedia
        available_cols = [col for col in columns_to_show if col in df.columns]
        return df[available_cols].head(5)  # tampilkan 5 baris pertama
    except FileNotFoundError:
        st.warning("⚠️ File 'ai_student_impact_clean.csv' tidak ditemukan. Sample data tidak dapat ditampilkan.")
        return pd.DataFrame()

sample_df = load_sample_data()

# ============================================================================
# FEATURE NAMES (urutan sesuai training)
# ============================================================================
feature_names = [
    'pre_semester_gpa', 'weekly_genai_hours', 'tool_diversity',
    'traditional_study_hours', 'perceived_ai_dependency', 'anxiety_level_during_exams',
    'skill_retention_score', 'major_category_num', 'primary_use_case_num',
    'paid_subscription_num', 'institutional_policy_num', 'year_of_study_num',
    'prompt_engineering_skill_num', 'burnout_risk_level_num'
]

# ============================================================================
# MAPPING UNTUK INPUT (agar user friendly)
# ============================================================================
major_cat_mapping = {
    "Business (1)": 1,
    "Humanities (2)": 2,
    "Medical (3)": 3,
    "STEM (4)": 4
}
primary_use_mapping = {
    "Copywriting/Drafting (0)": 0,
    "Debugging/Troubleshooting (1)": 1,
    "Other (2)": 2,
    "Ideation (3)": 3,
    "Summarizing/Reading (4)": 4
}
institutional_policy_mapping = {
    "Allowed With Citation (1)": 1,
    "Strict Ban (2)": 2
}
year_of_study_mapping = {
    "Freshman (0)": 0,
    "Sophomore (1)": 1,
    "Junior (2)": 2,
    "Senior (3)": 3
}
prompt_skill_mapping = {
    "Beginner (0)": 0,
    "Intermediate (1)": 1,
    "Advanced (2)": 2
}
burnout_risk_mapping = {
    "Low (0)": 0,
    "Medium (1)": 1,
    "High (2)": 2
}

# ============================================================================
# SIDEBAR: INFO
# ============================================================================
st.sidebar.header("📖 Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini memprediksi **Nilai Akhir Semester (GPA)** mahasiswa "
    "berdasarkan berbagai faktor seperti kebiasaan belajar, penggunaan AI, "
    "tingkat kecemasan, retensi skill, dll.\n\n"
    "**Model yang tersedia:**\n"
    "- Linear Regression (R² ≈ 89.7%)\n"
    "- XGBoost (R² ≈ 90.6%)\n\n"
    "**Rentang GPA:** 0.0 – 4.0"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dibuat dengan Streamlit** | Data Science Project")

# ============================================================================
# HEADER
# ============================================================================
st.title("📊 Prediksi Nilai Akhir Semester")
st.markdown("Masukkan data mahasiswa untuk memprediksi **post_semester_gpa** (skala 0-4)")

# Model selection
model_choice = st.selectbox(
    "Pilih Model Machine Learning:",
    list(models.keys()),
    help="Pilih algoritma yang akan digunakan untuk prediksi"
)

st.markdown("---")

# ============================================================================
# INPUT FORM - Dua kolom
# ============================================================================
st.subheader("🔧 Input Data Mahasiswa")

col1, col2 = st.columns(2)

with col1:
    pre_gpa = st.number_input("Nilai awal semester (pre_semester_gpa)", min_value=0.0, max_value=4.0, value=3.0, step=0.1, format="%.2f")
    weekly_genai = st.number_input("Jam penggunaan GenAI per minggu", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
    tool_diversity = st.number_input("Diversitas tools (jumlah tools AI yang digunakan)", min_value=0, max_value=10, value=2, step=1)
    traditional_study = st.number_input("Jam belajar tradisional per minggu", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
    perceived_ai_dep = st.slider("Persepsi ketergantungan pada AI (1-10)", 1, 10, 5)
    anxiety_level = st.slider("Tingkat kecemasan saat ujian (1-10)", 1, 10, 5)
    skill_retention = st.number_input("Skor retensi skill (0-100)", min_value=0, max_value=100, value=75, step=1)

with col2:
    major_cat_label = st.selectbox("Kategori jurusan", list(major_cat_mapping.keys()))
    major_category_num = major_cat_mapping[major_cat_label]

    primary_use_label = st.selectbox("Penggunaan utama AI", list(primary_use_mapping.keys()))
    primary_use_case_num = primary_use_mapping[primary_use_label]

    paid_sub = st.selectbox("Langganan berbayar?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
    inst_policy_label = st.selectbox("Kebijakan institusi", list(institutional_policy_mapping.keys()))
    institutional_policy_num = institutional_policy_mapping[inst_policy_label]

    year_label = st.selectbox("Tahun studi", list(year_of_study_mapping.keys()))
    year_of_study_num = year_of_study_mapping[year_label]

    prompt_skill_label = st.selectbox("Kemampuan prompt engineering", list(prompt_skill_mapping.keys()))
    prompt_engineering_skill_num = prompt_skill_mapping[prompt_skill_label]

    burnout_label = st.selectbox("Tingkat burnout", list(burnout_risk_mapping.keys()))
    burnout_risk_level_num = burnout_risk_mapping[burnout_label]

# ============================================================================
# BUAT DATAFRAME INPUT
# ============================================================================
input_data = {
    'pre_semester_gpa': pre_gpa,
    'weekly_genai_hours': weekly_genai,
    'tool_diversity': tool_diversity,
    'traditional_study_hours': traditional_study,
    'perceived_ai_dependency': perceived_ai_dep,
    'anxiety_level_during_exams': anxiety_level,
    'skill_retention_score': skill_retention,
    'major_category_num': major_category_num,
    'primary_use_case_num': primary_use_case_num,
    'paid_subscription_num': paid_sub,
    'institutional_policy_num': institutional_policy_num,
    'year_of_study_num': year_of_study_num,
    'prompt_engineering_skill_num': prompt_engineering_skill_num,
    'burnout_risk_level_num': burnout_risk_level_num
}

input_df = pd.DataFrame([input_data])[feature_names]  # urutkan sesuai feature_names

st.markdown("---")
st.subheader("📊 Ringkasan Input")
st.dataframe(input_df, use_container_width=True)

# ============================================================================
# SAMPLE DATA (ditempatkan di sini)
# ============================================================================
if not sample_df.empty:
    st.subheader("📋 Contoh Data (5 baris pertama)")
    st.dataframe(sample_df, use_container_width=True)

# ============================================================================
# PREDIKSI
# ============================================================================
st.markdown("---")
st.subheader("🎯 Hasil Prediksi")

if st.button("🚀 Prediksi Nilai Akhir Semester", type="primary"):
    model = models[model_choice]
    with st.spinner("Model sedang memproses..."):
        prediction = model.predict(input_df)[0]
        # Batasi prediksi antara 0 dan 4 (GPA range)
        prediction = max(0.0, min(4.0, prediction))
        
        st.success(f"### 📈 Prediksi `post_semester_gpa`: **{prediction:.3f}**")
        
        mae_approx = 0.12  # dari hasil evaluasi
        st.metric("Perkiraan Rentang (MAE)", f"± {mae_approx:.2f}")
        
        # Bar chart sederhana untuk visualisasi nilai
        st.write("**Visualisasi Nilai**")
        st.bar_chart(pd.DataFrame({"Prediksi GPA": [prediction]}, index=["post_semester_gpa"]))
        
        st.balloons()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("✅ Model regresi siap digunakan. Pastikan file .pkl ada di direktori yang sama.")