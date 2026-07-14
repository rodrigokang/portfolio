FROM mcr.microsoft.com/azure-functions/python:4-python3.11

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY azure/requirements.txt /tmp/requirements.txt

RUN pip install \
    --no-cache-dir \
    --upgrade pip \
    && pip install \
    --no-cache-dir \
    -r /tmp/requirements.txt

COPY azure/function_app.py /home/site/wwwroot/function_app.py
COPY azure/host.json /home/site/wwwroot/host.json
COPY azure/src /home/site/wwwroot/src

COPY artifacts/sentiment_pipeline.joblib \
    /home/site/artifacts/sentiment_pipeline.joblib
COPY artifacts/sentiment_pipeline_metadata.json \
    /home/site/artifacts/sentiment_pipeline_metadata.json
