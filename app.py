import streamlit as st
import requests
import pandas as pd
import time
import joblib

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Hệ thống Giám sát & Dự báo Chất lượng Nước",
    page_icon="🌊",
    layout="wide"
)

# Địa chỉ API FastAPI Backend
API_URL = "http://127.0.0.1:8000/predict"

st.title("🌊 Hệ thống Giám sát, Bù dữ liệu khuyết & Dự báo Chất lượng Nước")
st.caption("Đồ án Mini-Project môn Điện toán đám mây | Cloud & AI Pipeline")

# Tạo 2 Tab chính
tab1, tab2 = st.tabs(["🎯 Dự báo theo Thông số", "📈 Giả lập Luồng dữ liệu IoT (Data Stream)"])

# -------------------------------------------------------------
# TAB 1: DỰ BÁO THEO THÔNG SỐ (Dành cho kiểm thử Imputation & Model)
# -------------------------------------------------------------
with tab1:
    st.subheader("Nhập thông số cảm biến đo đạc")
    st.info("💡 Mẹo: Nếu bỏ chọn một thông số, hệ thống sẽ tự động gọi mô hình KNNImputer để bù dữ liệu khuyết!")

    col1, col2, col3 = st.columns(3)

    with col1:
        use_ph = st.checkbox("Sử dụng chỉ số pH", value=True)
        ph_val = st.slider("Chỉ số pH", 0.0, 14.0, 7.5, step=0.1) if use_ph else None

    with col2:
        use_temp = st.checkbox("Sử dụng Nhiệt độ (TEMP)", value=True)
        temp_val = st.slider("Nhiệt độ (°C)", 0.0, 40.0, 25.0, step=0.5) if use_temp else None

    with col3:
        use_bod = st.checkbox("Sử dụng chỉ số BOD", value=True)
        bod_val = st.slider("Nhu cầu Oxy sinh hóa (BOD - mg/L)", 0.0, 20.0, 2.5, step=0.1) if use_bod else None

    if st.button("🚀 Thực hiện Dự báo", type="primary"):
        payload = {
            "pH": ph_val,
            "TEMP": temp_val,
            "BOD": bod_val
        }

        with st.spinner("Đang gửi yêu cầu tới FastAPI Backend..."):
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    res_data = response.json()

                    st.success("Dự báo thành công!")

                    # Hiển thị kết quả bằng Metric Cards
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Oxy hòa tan dự báo (DO)", f"{res_data['predicted_DO_mg_L']} mg/L")
                    m2.metric("Trạng thái chất lượng nước", res_data['water_quality_status'])
                    m3.metric("Giá trị TEMP sau bù khuyết", f"{res_data['imputed_values']['TEMP']} °C")

                    # Chi tiết JSON trả về
                    with st.expander("🔍 Chi tiết dữ liệu JSON trả về từ API"):
                        st.json(res_data)
                else:
                    st.error(f"Lỗi API: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Không thể kết nối tới FastAPI Backend tại {API_URL}. Vui lòng kiểm tra lại lệnh uvicorn!")

# -------------------------------------------------------------
# TAB 2: GIẢ LẬP LUỒNG DỮ LIỆU IOT (DATA STREAMING)
# -------------------------------------------------------------
with tab2:
    st.subheader("Mô phỏng Dữ liệu Cảm biến thời gian thực")
    st.write("Đọc dữ liệu từ `daily_land.csv`, tự động bù khuyết bằng KNNImputer và đẩy luồng IoT về giao diện.")

    if st.button("▶️ Bắt đầu Stream dữ liệu"):
        try:
            # 1. Đọc dữ liệu và xoay thành dạng bảng (Wide format)
            df_stream = pd.read_csv("data/daily_land.csv")
            df_pivoted = df_stream.pivot_table(
                index=['MonitoringLocationIdentifier', 'MonitoringDate'], 
                columns='IndicatorsName', 
                values='Value'
            ).reset_index()

            # 2. Lấy bộ chỉ số và bù dữ liệu khuyết bằng KNNImputer (tránh bị mất dòng như dropna)
            selected_cols = ['pH', 'TEMP', 'BOD', 'DO']
            df_sub = df_pivoted[selected_cols].copy()

            imputer = joblib.load("models/imputer.pkl")
            df_imputed = pd.DataFrame(imputer.transform(df_sub), columns=selected_cols)

            # Lấy 15 dòng dữ liệu đã được làm sạch để chạy mô phỏng
            sample_data = df_imputed.head(15)

            chart_placeholder = st.empty()
            table_placeholder = st.empty()

            records = []

            for idx, row in sample_data.iterrows():
                records.append({
                    "Lần đo": len(records) + 1,
                    "pH": round(row['pH'], 2),
                    "Nhiệt độ": round(row['TEMP'], 2),
                    "BOD": round(row['BOD'], 2),
                    "DO (Oxy hòa tan)": round(row['DO'], 2)
                })
                df_curr = pd.DataFrame(records)

                # Vẽ biểu đồ đường
                with chart_placeholder.container():
                    st.line_chart(df_curr.set_index("Lần đo")[["DO (Oxy hòa tan)", "Nhiệt độ"]])

                # Bảng hiển thị 5 dòng dữ liệu mới nhất
                with table_placeholder.container():
                    st.dataframe(df_curr.tail(5), use_container_width=True)

                time.sleep(1) # Delay 1 giây giữa mỗi lần phát dữ liệu

            st.success("✅ Hoàn thành mô phỏng luồng dữ liệu!")
        except Exception as e:
            st.error(f"Lỗi mô phỏng Stream dữ liệu: {str(e)}")