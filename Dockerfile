FROM python:3.8-slim
WORKDIR /app

# Dependencies
RUN apt-get update && apt-get install -y procps
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install --no-deps RMextract

# Copy files
COPY FRion /app/FRion
COPY setup.py .

# FRion
RUN pip install -e .