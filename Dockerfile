FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-semantic.txt ./
RUN pip install --no-cache-dir -r requirements-semantic.txt
COPY . .

