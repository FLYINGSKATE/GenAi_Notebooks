"""
AWS SageMaker Deployment Script

This script demonstrates how to deploy a fine-tuned model to AWS SageMaker.
"""
import os
import boto3
import tarfile
from pathlib import Path
import shutil
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model_tarball(model_path: str, output_path: str = "model.tar.gz") -> str:
    """
    Create a tarball of the model for SageMaker.
    
    Args:
        model_path: Path to the fine-tuned model
        output_path: Path to save the tarball
        
    Returns:
        Path to the created tarball
    """
    logger.info(f"Creating tarball for model at {model_path}")
    
    # Create a temporary directory
    temp_dir = Path("temp_model")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Copy model files to temp directory
        model_files = list(Path(model_path).glob("*"))
        for file_path in model_files:
            dest_path = temp_dir / file_path.name
            if file_path.is_file():
                shutil.copy(file_path, dest_path)
            else:
                shutil.copytree(file_path, dest_path)
        
        # Create tarball
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(temp_dir, arcname=".")
            
        logger.info(f"Model tarball created at {output_path}")
        return output_path
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def deploy_to_sagemaker(
    model_tarball: str,
    role_arn: str,
    instance_type: str = "ml.g4dn.xlarge",
    region: str = "us-west-2",
    endpoint_name: str = "llm-endpoint"
):
    """
    Deploy the model to SageMaker.
    
    Args:
        model_tarball: Path to the model tarball
        role_arn: IAM role ARN for SageMaker
        instance_type: SageMaker instance type
        region: AWS region
        endpoint_name: Name for the SageMaker endpoint
    """
    logger.info("Starting SageMaker deployment...")
    
    # Initialize SageMaker session
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # Upload model to S3
    model_s3_uri = sagemaker_session.upload_data(
        path=model_tarball,
        bucket=sagemaker_session.default_bucket(),
        key_prefix="llm-models"
    )
    logger.info(f"Model uploaded to {model_s3_uri}")
    
    # Create HuggingFace model
    huggingface_model = HuggingFaceModel(
        model_data=model_s3_uri,
        role=role_arn,
        transformers_version="4.26.0",
        pytorch_version="1.13.1",
        py_version="py39",
        entry_point="inference.py",  # You'll need to create this
        source_dir="code",  # Directory containing inference.py
    )
    
    # Deploy the model
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    
    logger.info(f"Model deployed to endpoint: {endpoint_name}")
    return predictor

if __name__ == "__main__":
    # Example usage
    model_path = "/path/to/your/fine-tuned/model"
    role_arn = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"
    
    # Create model tarball
    tarball_path = create_model_tarball(model_path)
    
    # Deploy to SageMaker
    deploy_to_sagemaker(
        model_tarball=tarball_path,
        role_arn=role_arn,
        instance_type="ml.g4dn.xlarge",
        region="us-west-2",
        endpoint_name="my-llm-endpoint"
    )
