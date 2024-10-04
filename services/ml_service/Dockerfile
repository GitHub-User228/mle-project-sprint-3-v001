FROM python:3.10.14-slim

LABEL author=${LABEL}

WORKDIR /fastapi_app/ml_service

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/bash", "entrypoints/entrypoint_compose.sh"]