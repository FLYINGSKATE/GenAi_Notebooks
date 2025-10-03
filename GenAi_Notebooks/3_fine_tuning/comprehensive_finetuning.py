"""
Comprehensive LLM Fine-Tuning Script

This script provides a modular approach to fine-tuning LLMs with different techniques.
Toggle between methods by uncommenting the relevant sections.
"""

# ===== IMPORTS =====
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===== CONFIGURATION =====
# Model selection
model_name = "meta-llama/Llama-2-7b-hf"  # Base model to fine-tune

# Fine-tuning technique (uncomment ONE)
TECHNIQUE = "qlora"  # Options: "full", "lora", "qlora", "adapter"

# Dataset configuration
dataset_name = "imdb"  # Example dataset
text_column = "text"   # Column containing text

# Training parameters
output_dir = "./results"
num_train_epochs = 3
per_device_train_batch_size = 4
learning_rate = 2e-4
max_seq_length = 512

# ===== LOAD AND PREPARE DATASET =====
print("Loading dataset...")
dataset = load_dataset(dataset_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples[text_column],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length"
    )

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ===== MODEL SETUP =====
print("Loading model...")

# Quantization config for QLoRA
if TECHNIQUE == "qlora":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
else:
    bnb_config = None

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply fine-tuning technique
if TECHNIQUE in ["lora", "qlora"]:
    print(f"Applying {TECHNIQUE.upper()} configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    if TECHNIQUE == "qlora":
        model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# ===== TRAINING SETUP =====
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    learning_rate=learning_rate,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    optim="paged_adamw_32bit" if TECHNIQUE == "qlora" else "adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=lambda data: {
        'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
        'labels': torch.stack([torch.tensor(f['input_ids']) for f in data])
    }
)

# ===== TRAINING =====
print("Starting training...")
trainer.train()

# ===== SAVING =====
print("Saving model...")
model.save_pretrained(f"{output_dir}/final_model")
tokenizer.save_pretrained(f"{output_dir}/final_model")

print("Training complete!")
