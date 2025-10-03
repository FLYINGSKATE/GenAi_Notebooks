# LLM Model Deployment

This directory contains scripts and configurations for deploying fine-tuned LLM models to various cloud platforms.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployment](#cloud-deployment)
  - [AWS SageMaker](#aws-sagemaker)
  - [Google Cloud AI Platform](#google-cloud-ai-platform)
  - [Azure ML](#azure-ml)
- [Monitoring and Scaling](#monitoring-and-scaling)
- [Security](#security)

## Prerequisites

- Python 3.8+
- Docker
- Kubernetes (for K8s deployment)
- Cloud provider account (AWS, GCP, or Azure)
- Fine-tuned model files

## Local Development

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the example environment file and update with your settings:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Run the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

## Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t llm-api .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file .env llm-api
   ```

## Kubernetes Deployment

1. Create Kubernetes secrets and config maps:
   ```bash
   kubectl create secret generic llm-secrets --from-env-file=.env
   kubectl create configmap llm-config --from-env-file=.env
   ```

2. Apply the Kubernetes manifests:
   ```bash
   kubectl apply -f kubernetes/
   ```

## Cloud Deployment

### AWS SageMaker

1. Install AWS CLI and configure credentials:
   ```bash
   aws configure
   ```

2. Run the SageMaker deployment script:
   ```bash
   python cloud/aws_sagemaker.py
   ```

### Google Cloud AI Platform

1. Install Google Cloud SDK and authenticate:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. Deploy the model:
   ```bash
   gcloud ai-platform models create MODEL_NAME \
     --region=us-central1 \
     --enable-logging \
     --enable-console-logging
   ```

### Azure ML

1. Install Azure CLI and login:
   ```bash
   az login
   az account set --subscription YOUR_SUBSCRIPTION_ID
   ```

2. Deploy the model using Azure ML SDK (see `cloud/azure_ml.py` for details).

## Monitoring and Scaling

- **Metrics Collection**: Prometheus and Grafana for monitoring
- **Logging**: ELK Stack or CloudWatch/Stackdriver
- **Scaling**:
  - Horizontal Pod Autoscaler for Kubernetes
  - Auto Scaling Groups for EC2
  - Managed instance groups for GCP

## Security

- Use HTTPS with valid certificates (Let's Encrypt)
- Implement rate limiting
- Use API keys or OAuth2 for authentication
- Encrypt sensitive data at rest and in transit
- Regularly update dependencies

## Troubleshooting

Check logs:
```bash
# Docker
kubectl logs -f <pod-name>

# Kubernetes
docker logs <container-id>
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
