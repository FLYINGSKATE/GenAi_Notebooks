# LLM Integration Hub

This directory contains various integration examples and deployment scripts for fine-tuned LLM models.

## Table of Contents

- [API Integration](#api-integration)
- [Web Applications](#web-applications)
- [Cloud Deployment](#cloud-deployment)
- [Enterprise Integrations](#enterprise-integrations)
- [Example Use Cases](#example-use-cases)
- [Development](#development)

## API Integration

### FastAPI Server
A production-ready FastAPI server for serving LLM models.

**Features:**
- JWT Authentication
- Model management
- Health checks
- Rate limiting
- Request validation

**Quick Start:**
```bash
cd api_integration/fastapi_app
pip install -r requirements.txt
uvicorn main:app --reload
```

## Web Applications

### React Frontend
Modern, responsive web interface for interacting with the API.

**Features:**
- Real-time chat
- Dark/light mode
- Markdown support
- Adjustable parameters

**Quick Start:**
```bash
cd web_app/react-frontend
npm install
npm start
```

### Streamlit App
Quick prototyping and demonstration interface.

**Quick Start:**
```bash
cd web_app/streamlit
pip install -r requirements.txt
streamlit run app.py
```

## Cloud Deployment

### AWS Deployment
Deploy to AWS ECS/EKS with Terraform.

**Prerequisites:**
- AWS CLI configured
- Terraform installed

**Deploy:**
```bash
cd cloud/aws/terraform
terraform init
terraform apply
```

### Google Cloud Run
Deploy as a serverless container.

**Deploy:**
```bash
cd cloud/gcp
./deploy.sh
```

### Azure Container Apps
Deploy to Azure's managed container service.

**Deploy:**
```bash
cd cloud/azure
az login
./deploy.ps1
```

## Enterprise Integrations

### Snowflake Integration
Example of integrating with Snowflake for data processing.

### Databricks Notebook
Jupyter notebook for Databricks integration.

### Airflow DAG
Example DAG for scheduling and monitoring.

## Example Use Cases

### Chatbot Implementation
End-to-end chatbot with persistent storage.

### RAG System
Retrieval-Augmented Generation implementation.

### Document Processing
Automated document analysis pipeline.

## Development

### Prerequisites
- Python 3.8+
- Node.js 16+
- Docker
- Cloud provider CLI tools

### Local Development
1. Set up Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Start development servers:
   ```bash
   # API Server
   cd api_integration/fastapi_app
   uvicorn main:app --reload

   # React Frontend
   cd web_app/react-frontend
   npm start
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
