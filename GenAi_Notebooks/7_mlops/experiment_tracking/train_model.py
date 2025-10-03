"""
Experiment Tracking Example with MLflow

This script demonstrates how to track experiments, log parameters,
metrics, and artifacts using MLflow.
"""
import os
import mlflow
import numpy as np
from datetime import datetime
from pathlib import Path

# Set up MLflow
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

def train_model(params):
    """Mock training function that logs metrics and artifacts"""
    # Log parameters
    mlflow.log_params(params)
    
    # Simulate training
    epochs = params["epochs"]
    learning_rate = params["learning_rate"]
    
    for epoch in range(epochs):
        # Simulate metrics
        train_loss = np.exp(-epoch / epochs) * (1 + np.random.normal(0, 0.1))
        val_loss = train_loss * (1 + np.random.normal(0, 0.05))
        accuracy = 1 - val_loss
        
        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy
        }, step=epoch)
        
        # Log a model checkpoint (mock)
        if epoch % 5 == 0:
            checkpoint_path = f"checkpoints/epoch_{epoch}.pth"
            Path("checkpoints").mkdir(exist_ok=True)
            with open(checkpoint_path, "w") as f:
                f.write(f"Checkpoint at epoch {epoch}")
            mlflow.log_artifact(checkpoint_path)
    
    return {"train_loss": train_loss, "val_loss": val_loss, "accuracy": accuracy}

def main():
    # Experiment setup
    experiment_name = "llm-finetuning"
    mlflow.set_experiment(experiment_name)
    
    # Example hyperparameters
    params = {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "learning_rate": 2e-5,
        "batch_size": 4,
        "epochs": 10,
        "technique": "qlora",
        "dataset": "imdb"
    }
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        print(f"Starting experiment: {experiment_name}")
        print(f"Parameters: {params}")
        
        # Log code state
        mlflow.log_artifact(__file__)
        
        # Train model and log metrics
        metrics = train_model(params)
        
        # Log final metrics
        mlflow.log_metrics({"final_accuracy": metrics["accuracy"]})
        
        # Log model (mock)
        mlflow.pytorch.log_model(
            None,  # Replace with actual model
            "model",
            registered_model_name="llm-classifier"
        )
        
        print(f"Training complete. Final accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
