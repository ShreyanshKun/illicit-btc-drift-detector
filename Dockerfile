FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The app only needs code, saved models, result tables and the small demo sample.
COPY src ./src
COPY app ./app
COPY models ./models
COPY reports/results ./reports/results
COPY data/processed/app_sample.parquet data/processed/demo_upload.csv ./data/processed/
COPY .streamlit ./.streamlit

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
