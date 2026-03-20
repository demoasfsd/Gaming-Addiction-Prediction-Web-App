import streamlit as st
import pandas as pd
import joblib
import json

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Gaming Addiction Predictor", layout="wide")

@st.cache_resource
def load_model_artifacts():
    # โหลด Pipeline และ Metadata
    pipeline = joblib.load("model_artifacts/gaming_model_pipeline.pkl")
    with open("model_artifacts/feature_names.json", "r") as f:
        features = json.load(f)
    with open("model_artifacts/model_metadata.json", "r") as f:
        metadata = json.load(f) # แก้ไขจุด json.json.load เดิม
    return pipeline, features, metadata

try:
    model, feature_names, metadata = load_model_artifacts()
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    st.stop()

# --- 2. ส่วน UI ---
st.title("🎮 Gaming Addiction Prediction Web App")
st.markdown(f"ทำนายระดับการติดเกมโดยใช้โมเดล: **{metadata['model_type']}**")
st.divider()

with st.form("prediction_form"):
    st.subheader("กรุณากรอกข้อมูลพฤติกรรม")
    
    col1, col2 = st.columns(2)
    
    with col1:
        daily_hours = st.number_input("ชั่วโมงการเล่นเกมต่อวัน", min_value=0.0, max_value=24.0, value=2.0)
        sleep_hours = st.number_input("ชั่วโมงการนอน", min_value=0.0, max_value=24.0, value=7.0)
        screen_time = st.number_input("เวลาหน้าจอรวมทั้งหมด (ชั่วโมง)", min_value=0.0, max_value=24.0, value=5.0)
        # แก้ไข Stress Level เป็น Slider 1-10 ตามข้อมูลจริงใน Dataset
        stress_level = st.slider("ระดับความเครียด (Stress Level 1-10)", 1, 10, 5)

    with col2:
        anxiety = st.slider("ระดับความกังวล (Anxiety Score)", 0, 20, 5)
        depression = st.slider("ระดับความซึมเศร้า (Depression Score)", 0, 20, 5)
        loneliness = st.slider("ระดับความเหงา (Loneliness Score)", 0, 20, 5)
        gender = st.selectbox("เพศ (Gender)", ["Male", "Female", "Other"])

    submit = st.form_submit_button("ทำนายผล (Predict)")

# --- 3. ส่วนการทำนายผล ---
if submit:
    # สร้าง Dictionary ให้ชื่อ Key ตรงกับชื่อคอลัมน์ใน CSV ต้นฉบับ
    data_dict = {
        "daily_gaming_hours": daily_hours,
        "sleep_hours": sleep_hours,
        "anxiety_score": anxiety,
        "depression_score": depression,
        "screen_time_total": screen_time,
        "loneliness_score": loneliness,
        "stress_level": stress_level, # ส่งค่าเป็นตัวเลขแล้ว
        "gender": gender
    }
    
    input_df = pd.DataFrame([data_dict])

    # ตรวจสอบว่าคอลัมน์ครบถ้วนตามที่โมเดลต้องการหรือไม่
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0 

    # จัดลำดับคอลัมน์ให้ตรงกับตอน Train
    input_df = input_df[feature_names]

    # ทำนายผล
    try:
        prediction = model.predict(input_df)[0]
        
        st.subheader("ผลการวิเคราะห์:")
        # แสดงเกจ์หรือสีตามระดับ (สมมติระดับ 0-100)
        if prediction > 7:
            st.error(f"ระดับการติดเกมที่ทำนายได้: {prediction:.2f} (ความเสี่ยงสูงมาก)")
        elif prediction > 4:
            st.warning(f"ระดับการติดเกมที่ทำนายได้: {prediction:.2f} (ความเสี่ยงปานกลาง)")
        else:
            st.success(f"ระดับการติดเกมที่ทำนายได้: {prediction:.2f} (ความเสี่ยงต่ำ)")
            
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดขณะทำนาย: {e}")
        st.write("คอลัมน์ที่ส่งไป:", input_df.columns.tolist())