# Run the app anywhere with:
#   docker build -t rag-app .
#   docker run --env-file .env -p 8501:8501 rag-app
# Then open http://localhost:8501
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first: Docker caches this layer, so rebuilds are fast when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
