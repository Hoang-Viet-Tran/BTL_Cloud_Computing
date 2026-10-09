import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# 1. Đọc và chuyển đổi dữ liệu từ dạng dọc (Long) sang dạng bảng (Wide)
file_path = "data/daily_land.csv"   # Hoặc "../data/daily_land.csv" nếu chạy từ trong folder notebooks
df_raw = pd.read_csv(file_path)

df_pivoted = df_raw.pivot_table(
    index=['MonitoringLocationIdentifier', 'MonitoringDate'], 
    columns='IndicatorsName', 
    values='Value'
).reset_index()

# Lọc 4 chỉ số quan trọng
selected_cols = ['pH', 'TEMP', 'BOD', 'DO']
df = df_pivoted[selected_cols].copy()

print("=== Số ô trống (NaN) ban đầu ===")
print(df.isnull().sum())

# 2. Xử lý dữ liệu khuyết bằng KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed_array = imputer.fit_transform(df)
df_clean = pd.DataFrame(df_imputed_array, columns=selected_cols)

print("\n=== Số ô trống sau khi xử lý (Imputation) ===")
print(df_clean.isnull().sum())

# 3. Train mô hình RandomForestRegressor dự báo DO
X = df_clean[['pH', 'TEMP', 'BOD']]
y = df_clean['DO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\n=== Kết quả đánh giá mô hình ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# 4. Xuất file mô hình
os.makedirs("models", exist_ok=True)
joblib.dump(imputer, "models/imputer.pkl")
joblib.dump(model, "models/model.pkl")

print("\n✅ THÀNH CÔNG! Đã lưu 2 file: models/imputer.pkl và models/model.pkl")