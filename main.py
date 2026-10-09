from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import os

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Water Quality Monitoring API",
    description="API bù dữ liệu khuyết (Imputation) và dự báo chất lượng nước (DO Prediction)",
    version="1.0"
)

# Đường dẫn tới các file mô hình
MODEL_PATH = "models/model.pkl"
IMPUTER_PATH = "models/imputer.pkl"

# Kiểm tra và load mô hình
if not os.path.exists(MODEL_PATH) or not os.path.exists(IMPUTER_PATH):
    raise FileNotFoundError("Không tìm thấy file model.pkl hoặc imputer.pkl trong thư mục models/")

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

# Định nghĩa cấu trúc dữ liệu đầu vào (Cung cấp pH, TEMP, BOD; các trường có thể bị khuyết/None)
class WaterQualityInput(BaseModel):
    pH: Optional[float] = None
    TEMP: Optional[float] = None
    BOD: Optional[float] = None

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Water Quality API đang hoạt động bình thường!",
        "docs_url": "/docs"
    }

@app.post("/predict")
def predict_water_quality(data: WaterQualityInput):
    try:
        # Chuyển đổi dữ liệu từ Request thành dạng mảng
        input_data = [data.pH, data.TEMP, data.BOD, np.nan] # Thêm NaN giả định cho vị trí DO
        
        # Đưa vào DataFrame
        cols = ['pH', 'TEMP', 'BOD', 'DO']
        df_input = pd.DataFrame([input_data], columns=cols)
        
        # 1. Bù dữ liệu khuyết (Imputation) nếu người dùng truyền thiếu chỉ số
        df_imputed = pd.DataFrame(imputer.transform(df_input), columns=cols)
        
        # Lấy ra các chỉ số sau khi đã xử lý bù khuyết
        pH_clean = float(df_imputed['pH'].iloc[0])
        TEMP_clean = float(df_imputed['TEMP'].iloc[0])
        BOD_clean = float(df_imputed['BOD'].iloc[0])
        
        # 2. Dự báo chỉ số DO
        X_pred = np.array([[pH_clean, TEMP_clean, BOD_clean]])
        predicted_DO = float(model.predict(X_pred)[0])
        
        # 3. Phân loại chất lượng nước sơ bộ dựa trên DO (mg/L)
        if predicted_DO >= 6.5:
            quality_status = "Tốt (Nước sạch, sinh vật phát triển tốt)"
        elif predicted_DO >= 4.0:
            quality_status = "Trung bình (Có dấu hiệu ô nhiễm nhẹ)"
        else:
            quality_status = "Kém (Báo động ô nhiễm nặng)"
            
        return {
            "input_received": {
                "pH": data.pH,
                "TEMP": data.TEMP,
                "BOD": data.BOD
            },
            "imputed_values": {
                "pH": round(pH_clean, 2),
                "TEMP": round(TEMP_clean, 2),
                "BOD": round(BOD_clean, 2)
            },
            "predicted_DO_mg_L": round(predicted_DO, 2),
            "water_quality_status": quality_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý dữ liệu: {str(e)}")