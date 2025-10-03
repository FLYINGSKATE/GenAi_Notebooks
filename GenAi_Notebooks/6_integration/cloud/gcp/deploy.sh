#!/bin/bash

# Exit on error
set -e

# Configuration
PROJECT_ID="your-project-id"
SERVICE_NAME="llm-api"
REGION="us-central1"
MODEL_PATH="gs://your-bucket/models/llm"
SECRET_KEY="your-secret-key-here"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com

# Create Artifact Registry repository (if it doesn't exist)
echo "Setting up Artifact Registry..."
gcloud artifacts repositories create docker-repo \
  --repository-format=docker \
  --location=$REGION \
  --description="Docker repository for LLM API" || echo "Repository already exists"

# Build and push Docker image
echo "Building and pushing Docker image..."
gcloud builds submit \
  --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/docker-repo/${SERVICE_NAME}:latest \
  --timeout=3600s

# Create secret for environment variables (if it doesn't exist)
echo "Setting up secrets..."
echo -n "$SECRET_KEY" | gcloud secrets create ${SERVICE_NAME}-secret \
  --data-file=- \
  --replication-policy=automatic \
  --labels=environment=production || echo "Secret already exists"

# Grant Cloud Run service account access to the secret
SERVICE_ACCOUNT="${SERVICE_NUMBER}-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding ${SERVICE_NAME}-secret \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/docker-repo/${SERVICE_NAME}:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars="MODEL_PATH=${MODEL_PATH}" \
  --set-secrets="SECRET_KEY=${SERVICE_NAME}-secret:latest" \
  --cpu=4 \
  --memory=8Gi \
  --min-instances=0 \
  --max-instances=5 \
  --concurrency=10 \
  --timeout=300s \
  --port 8000

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')
echo "Service deployed successfully!"
echo "Service URL: ${SERVICE_URL}"

# Enable auto-scaling
echo "Configuring auto-scaling..."
gcloud alpha run services update ${SERVICE_NAME} \
  --region ${REGION} \
  --cpu-throttling \
  --min-instances=0 \
  --max-instances=10 \
  --concurrency=10

# Set up Cloud SQL if needed (uncomment and configure as needed)
# echo "Setting up Cloud SQL..."
# gcloud sql instances create llm-db \
#   --database-version=POSTGRES_13 \
#   --tier=db-f1-micro \
#   --region=${REGION}

# Set up Cloud Storage bucket for model files (if not already set up)
echo "Setting up Cloud Storage..."
gsutil mb -p ${PROJECT_ID} -l ${REGION} gs://${PROJECT_ID}-models || echo "Bucket already exists"

# Set up monitoring and logging
echo "Setting up monitoring and logging..."
gcloud services enable monitoring.googleapis.com logging.googleapis.com

# Create a service account for the application
echo "Creating service account..."
gcloud iam service-accounts create llm-service-account \
  --display-name="LLM Service Account"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:llm-service-account@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

echo "Deployment completed successfully!"
echo "Service URL: ${SERVICE_URL}"
