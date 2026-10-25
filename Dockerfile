ARG BASE_IMAGE=public.ecr.aws/docker/library/python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt .
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
    --index-url "$PIP_INDEX_URL" -r requirements.txt

COPY app ./app
COPY data ./data
COPY chroma_db ./chroma_db
COPY .env.example ./.env.example

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
