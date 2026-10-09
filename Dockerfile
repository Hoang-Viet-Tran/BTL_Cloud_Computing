FROM python:3.10-slim

WORKDIR /app

# Copy file requirements và cài đặt môi trường
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code và dữ liệu dự án vào container
COPY . .

# Mở cổng kết nối 8000 (FastAPI) và 8501 (Streamlit)
EXPOSE 8000 8501

# Chạy kịch bản khởi động
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]