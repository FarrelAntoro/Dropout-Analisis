import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load model
model = joblib.load('Model_Random_Forest_Classifier.joblib')

joblib.dump(model_info, "Model_Random_Forest_Classifier.joblib")

# Judul app
st.title("Prediksi Dropout Mahasiswa - Jaya Jaya Institut")

st.write("Masukkan data mahasiswa untuk memprediksi kemungkinan dropout:")

# Form input data mahasiswa
with st.form("form_prediksi"):
    curriculum_diff = st.number_input("Curriculum Difference", min_value=0, max_value=50, value=0)
    scholarship_holder = st.selectbox("Scholarship Holder (0: Tidak, 1: Ya)", [0, 1])
    curricular_units_1st_sem_approved = st.number_input("Mata Kuliah Lulus Semester 1", min_value=0, max_value=20, value=0)
    curricular_units_1st_sem_enrolled = st.number_input("Mata Kuliah Ambil Semester 1", min_value=0, max_value=20, value=0)
    age_at_enrollment = st.number_input("Usia Saat Daftar", min_value=15, max_value=60, value=18)
    admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, value=100.0)

    submitted = st.form_submit_button("Prediksi Dropout")

# Proses prediksi saat tombol diklik
if submitted:
    # Buat dataframe dari input
    data_input = pd.DataFrame({
        'Curriculum_diff': [curriculum_diff],
        'Scholarship_holder': [scholarship_holder],
        'Curricular_units_1st_sem_approved': [curricular_units_1st_sem_approved],
        'Curricular_units_1st_sem_enrolled': [curricular_units_1st_sem_enrolled],
        'Age_at_enrollment': [age_at_enrollment],
        'Admission_grade': [admission_grade]
    })

    # Prediksi
    prediksi = model.predict(data_input)

    # Tampilkan hasil
    if prediksi[0] == 1:
        st.error("⚠️ Mahasiswa diprediksi Berpotensi Dropout.")
    else:
        st.success("✅ Mahasiswa diprediksi Akan Bertahan.")

# Footer
st.markdown("---")
st.caption("Prototype ini dibuat untuk submission proyek Data Science Dicoding.")
