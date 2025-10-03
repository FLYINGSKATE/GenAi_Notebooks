"""
Comprehensive LLM Fine-Tuning Script

This script provides a modular approach to fine-tuning LLMs with different techniques.
Supports full fine-tuning, LoRA, QLoRA, and adapters.

Usage:
    # Train a model
    python comprehensive_finetuning_enhanced.py --mode train --config config.json
    
    # Evaluate a model
    python comprehensive_finetuning_enhanced.py --mode evaluate --config config.json
    
    # Generate text with a trained model
    python comprehensive_finetuning_enhanced.py --mode inference --config config.json --prompt "Your prompt here"
"""

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, BitsAndBytesConfig,
    pipeline, TextGenerationPipeline
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import evaluate
import numpy as np

# ===== CONFIGURATION =====
class FineTuningConfig:
    """Configuration class for fine-tuning parameters."""
    
    def __init__(self):
        # Model selection
        self.model_name = "meta-llama/Llama-2-7b-hf"  # Base model to fine-tune
        
        # Dataset configuration
        self.dataset_name = "imdb"  # Can be local path or HF dataset
        self.dataset_config = None  # For datasets with configs
        self.text_column = "text"   # Column containing text
        self.label_column = "label" # Column containing labels (if applicable)
        self.max_train_samples = 1000  # Limit training samples for quick testing
        
        # Training parameters
        self.technique = "qlora"  # Options: "full", "lora", "qlora", "adapter"
        self.output_dir = "./results"
        self.num_train_epochs = 3
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 4
        self.learning_rate = 2e-4
        self.max_seq_length = 512
        self.logging_steps = 10
        self.save_steps = 100
        self.warmup_steps = 100
        self.weight_decay = 0.01
        
        # LoRA specific
        self.lora_rank = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.lora_target_modules = ["q_proj", "v_proj"]
        
        # Generation parameters for inference
        self.max_new_tokens = 100
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        self.repetition_penalty = 1.1
        
        # Evaluation
        self.eval_metrics = ["accuracy", "perplexity"]
        self.eval_steps = 100

    def to_dict(self) -> Dict:
        """Convert config to dictionary for saving."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'FineTuningConfig':
        """Create config from dictionary."""
        config = cls()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config

# ===== DATASET HANDLING =====
class DatasetProcessor:
    """Handles dataset loading and preprocessing."""
    
    @staticmethod
    def load_dataset(config: FineTuningConfig) -> Dataset:
        """Load dataset from HuggingFace or local path."""
        print(f"Loading dataset: {config.dataset_name}")
        
        # Handle custom dataset formats
        if Path(config.dataset_name).is_dir():
            # Load from local directory
            data_files = {
                'train': str(Path(config.dataset_name) / 'train.jsonl'),
                'validation': str(Path(config.dataset_name) / 'validation.jsonl'),
                'test': str(Path(config.dataset_name) / 'test.jsonl')
            }
            dataset = load_dataset('json', data_files={k: v for k, v in data_files.items() 
                                                     if Path(v).exists()})
        else:
            # Load from HuggingFace Hub
            dataset = load_dataset(
                config.dataset_name, 
                config.dataset_config,
                split=None  # Returns all splits
            )
        
        # Limit dataset size for quick testing
        if config.max_train_samples:
            for split in list(dataset.keys()):
                if split != 'test':  # Don't limit test set
                    dataset[split] = dataset[split].select(range(min(config.max_train_samples, len(dataset[split]))))
        
        return dataset
    
    @staticmethod
    def preprocess_dataset(dataset: Dataset, tokenizer, config: FineTuningConfig) -> Dataset:
        """Tokenize and prepare dataset for training."""
        print("Preprocessing dataset...")
        
        # Define tokenization function
        def tokenize_function(examples):
            # For text generation tasks
            if config.text_column in examples:
                text = examples[config.text_column]
                if not isinstance(text, str):
                    text = [str(t) for t in text]
                return tokenizer(
                    text,
                    truncation=True,
                    max_length=config.max_seq_length,
                    padding="max_length"
                )
            return examples
        
        # Apply tokenization
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized_datasets

# ===== MODEL SETUP =====
class ModelManager:
    """Handles model loading, setup, and saving."""
    
    @staticmethod
    def load_model(config: FineTuningConfig):
        """Load model with appropriate configuration."""
        print(f"Loading model: {config.model_name}")
        
        # Set up quantization config for QLoRA
        bnb_config = None
        if config.technique == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if bnb_config is None else None,
            trust_remote_code=True
        )
        
        # Apply PEFT techniques
        model = ModelManager._apply_peft(model, config)
        return model
    
    @staticmethod
    def _apply_peft(model, config: FineTuningConfig):
        """Apply Parameter-Efficient Fine-Tuning techniques."""
        if config.technique == "full":
            # Full fine-tuning - no PEFT
            return model
        
        # Prepare model for k-bit training if using QLoRA
        if config.technique == "qlora":
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA or Adapter
        if config.technique in ["lora", "qlora"]:
            peft_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)
        
        elif config.technique == "adapter":
            # Adapter implementation would go here
            raise NotImplementedError("Adapter implementation not shown for brevity")
        
        # Print trainable parameters
        model.print_trainable_parameters()
        return model
    
    @staticmethod
    def save_model(model, tokenizer, output_dir: str):
        """Save model and tokenizer."""
        print(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

# ===== TRAINING =====
class FineTuner:
    """Handles the fine-tuning process."""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
    
    def setup_training(self, train_dataset, eval_dataset=None):
        """Set up trainer with training arguments."""
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = ModelManager.load_model(self.config)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=self.config.eval_steps if eval_dataset is not None else None,
            load_best_model_at_end=eval_dataset is not None,
            save_total_limit=2,
            fp16=torch.cuda.is_available() and not self.config.technique == "qlora",
            bf16=torch.cuda.is_bf16_supported() and self.config.technique == "qlora",
            remove_unused_columns=False,
            report_to=["tensorboard"],
            optim="paged_adamw_32bit" if self.config.technique == "qlora" else "adamw_torch",
            push_to_hub=False
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self._get_collator(),
            compute_metrics=self._get_compute_metrics() if eval_dataset is not None else None
        )
    
    def _get_collator(self):
        """Get data collator for training."""
        def collate_fn(batch):
            return {
                'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in batch]),
                'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in batch]),
                'labels': torch.stack([torch.tensor(f['input_ids']) for f in batch])
            }
        return collate_fn
    
    def _get_compute_metrics(self):
        """Get metrics computation function."""
        metric = evaluate.load("accuracy")
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
        
        return compute_metrics
    
    def train(self):
        """Run training."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_training first.")
        
        print("Starting training...")
        train_result = self.trainer.train()
        
        # Save final model
        if self.trainer.is_world_process_zero():
            ModelManager.save_model(
                self.model,
                self.tokenizer,
                f"{self.config.output_dir}/final_model"
            )
        
        return train_result
    
    def evaluate(self):
        """Run evaluation."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_training first.")
        
        if self.trainer.eval_dataset is None:
            print("No evaluation dataset provided.")
            return None
        
        print("Running evaluation...")
        eval_results = self.trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        return eval_results

# ===== INFERENCE =====
class TextGenerator:
    """Handles text generation with the fine-tuned model."""
    
    def __init__(self, model_path: str, config: Optional[FineTuningConfig] = None):
        """Initialize generator with model and tokenizer."""
        self.config = config or FineTuningConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, prompt: str, **generation_kwargs) -> str:
        """Generate text from prompt."""
        # Update generation parameters with any overrides
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            **generation_kwargs
        }
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode and return generated text
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===== MAIN SCRIPT =====
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "evaluate", "inference"],
                       help="Mode to run: train, evaluate, or inference")
    parser.add_argument("--prompt", type=str, help="Prompt for text generation")
    return parser.parse_args()

def main():
    """Main function for training and evaluation."""
    args = parse_args()
    
    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = FineTuningConfig.from_dict(config_dict)
    else:
        config = FineTuningConfig()
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(f"{config.output_dir}/config.json")
    
    if args.mode in ["train", "evaluate"]:
        # Load and preprocess dataset
        dataset = DatasetProcessor.load_dataset(config)
        tokenized_datasets = DatasetProcessor.preprocess_dataset(
            dataset, 
            AutoTokenizer.from_pretrained(config.model_name), 
            config
        )
        
        # Initialize and run training
        trainer = FineTuner(config)
        trainer.setup_training(
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets.get("validation", None)
        )
        
        if args.mode == "train":
            trainer.train()
        
        if args.mode == "evaluate" or args.mode == "train":
            trainer.evaluate()
    
    elif args.mode == "inference":
        if not args.prompt:
            print("Please provide a prompt with --prompt for inference")
            return
        
        # Initialize generator with the trained model
        model_path = f"{config.output_dir}/final_model"
        if not Path(model_path).exists():
            print(f"Model not found at {model_path}. Please train a model first.")
            return
        
        generator = TextGenerator(model_path, config)
        
        # Generate text
        print("\n=== Generated Text ===")
        output = generator.generate(args.prompt)
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {output}")

if __name__ == "__main__":
    main()
