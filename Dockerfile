FROM python:3.9-slim

# Only install libgomp1 which is needed for OpenMP support in numpy/opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9473

CMD ["python", "main.py"]
