#!/bin/bash
# Khởi chạy FastAPI Backend ngầm ở Port 8000
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Khởi chạy Streamlit Frontend ở Port 8501
streamlit run app.py --server.port 8501 --server.address 0.0.0.0