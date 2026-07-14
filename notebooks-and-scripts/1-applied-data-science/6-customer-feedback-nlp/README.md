# Understanding Customer Feedback through NLP

A reproducible end-to-end machine learning project for sentiment analysis of customer reviews using classical Natural Language Processing techniques, Azure Functions, Docker, and local Azure service emulation.

The project demonstrates how a trained machine learning model can be transformed into a production-style inference service while remaining fully reproducible on a local development environment.

Unlike many portfolio projects that stop after model evaluation, this repository continues through the deployment stage by packaging the trained model, exposing it through a REST API, containerising the application with Docker, and emulating Azure Storage locally using Azurite.

## Documentation

- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Dataset:** [DATASET.md](DATASET.md)

# Project Overview

Customer reviews contain valuable information about product quality, customer satisfaction, and purchasing behaviour. Extracting this information automatically allows organisations to monitor customer experience at scale and integrate feedback into business decision-making.

This project develops a complete sentiment analysis pipeline using the Amazon Reviews 2023 dataset. The trained model is exported as a Scikit-Learn pipeline and served through an Azure Function that exposes a REST API for inference.

The complete deployment can be executed locally without requiring an Azure subscription.

# Objectives

The project demonstrates the complete lifecycle of a machine learning model:

- Data preparation
- Text preprocessing
- Feature engineering with TF-IDF
- Logistic Regression training
- Model evaluation
- Model serialization
- Metadata versioning
- REST API deployment
- Azure Functions
- Docker containerisation
- Local Azure Storage emulation
- Automated testing

# Dataset

**Dataset**

Amazon Reviews 2023

Category used:

```
All_Beauty
```

Target variable:

| Rating | Sentiment |
|---------|-----------|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

The project uses product-grouped train/test splitting to reduce information leakage between products.

# Repository Structure

```
3-customer-feedback-nlp/

│
├── data/
│
├── artifacts/
│   ├── sentiment_pipeline.joblib
│   └── sentiment_pipeline_metadata.json
│
├── python/
│   └── training notebook
│
├── azure/
│   ├── function_app.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── host.json
│   ├── local.settings.json.example
│   │
│   ├── src/
│   ├── scripts/
│   ├── tests/
│   │
│   ├── requirements.txt
│   └── requirements-dev.txt
│
└── README.md
```

# Machine Learning Pipeline

The exported Scikit-Learn pipeline contains two stages:

```
Raw Review
      │
      ▼
TF-IDF Vectorizer
      │
      ▼
Logistic Regression
      │
      ▼
Sentiment Prediction
```

The entire pipeline is serialized into a single `joblib` artifact to guarantee identical preprocessing during inference.

# Model Metadata

A companion JSON file stores metadata describing the exported model.

The metadata includes:

- model version
- training date
- dataset
- preprocessing configuration
- target definition
- random seed
- evaluation metric
- training sample size

This information is loaded automatically by the Azure Function and exposed through the API.

# REST API

The project exposes two endpoints.

## Health Check

```
GET /api/health
```

Example response

```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_version": "1.0.0"
}
```

## Prediction

```
POST /api/predict
```

Example request

```json
{
    "text": "This product exceeded my expectations."
}
```

Example response

```json
{
    "sentiment": "positive",
    "confidence": 0.938,
    "probabilities": {
        "negative": 0.017,
        "neutral": 0.045,
        "positive": 0.938
    },
    "model_version": "1.0.0"
}
```

# Local Development

Create a virtual environment

```powershell
python3.11 -m venv .venv
```

Activate

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies

```powershell
pip install -r requirements-dev.txt
```

Run tests

```powershell
pytest
```

# Running the Azure Function

Start the Function

```powershell
.\scripts\start_function.ps1
```

Verify the service

```powershell
.\scripts\health_check.ps1
```

Run a prediction

```powershell
.\scripts\invoke_local.ps1
```

# Docker Deployment

Start the complete environment

```powershell
.\scripts\docker_up.ps1
```

This launches:

- Azure Function
- Azurite
- Docker network

Stop

```powershell
.\scripts\docker_down.ps1
```

# Testing

The repository contains automated tests covering:

- predictor behaviour
- metadata loading
- model loading
- Azure Function endpoints
- request validation
- inference output

Execute

```powershell
pytest
```

# Technologies

- Python 3.11
- Scikit-Learn
- Azure Functions
- Azure Functions Core Tools
- Docker
- Docker Compose
- Azurite
- Pytest
- Joblib
- PowerShell

# Reproducibility

The project has been designed to run entirely on a local machine.

No Azure subscription is required.

All infrastructure components are executed locally using Docker and Azurite.

The trained model is included as a serialized artifact, allowing inference without retraining.

# Future Improvements

Potential extensions include:

- Transformer-based sentiment models
- Batch inference endpoints
- Model monitoring
- Continuous deployment to Azure
- OpenAPI documentation
- Authentication
- Request logging
- Automated model retraining