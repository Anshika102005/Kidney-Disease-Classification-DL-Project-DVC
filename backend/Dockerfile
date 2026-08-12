FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

EXPOSE 5000

CMD ["python", "backend/app.py"]




# FROM python:3.8-slim
        # ↓
# Python environment
# 
# COPY requirements.txt
        # ↓
# Dependencies ki list copy
# 
# pip install
        # ↓
# Flask + TensorFlow + required libraries install
# 
# COPY . .
        # ↓
# backend ka code container mein copy
# 
# EXPOSE 5000
        # ↓
# ML API ka port
# 
# CMD ["python", "app.py"]
        # ↓
# Flask ML backend start