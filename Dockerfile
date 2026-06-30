# Container da demo interativa (Streamlit) do motor de pricing.
# Build:  docker build -t pricing-engine .
# Run:    docker run -p 8501:8501 pricing-engine   ->  http://localhost:8501
FROM python:3.12-slim

WORKDIR /app

# Dependências primeiro (camada cacheável).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código.
COPY . .

EXPOSE 8501

# Na subida: baixa as bases, gera os snapshots processados e sobe o app.
# (download_datasets usa kagglehub/OpenML; precisa de rede no primeiro start.)
CMD ["sh", "-c", "python download_datasets.py && python run_etl.py && python -m streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]
