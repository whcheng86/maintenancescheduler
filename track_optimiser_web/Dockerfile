FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 --timeout 0 main:app
