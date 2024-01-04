from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import pandas as pd 
from datasets import load_dataset
import os

# Check for other heavy processes and terminate if any
os.system("pkill -f heavy_process_name")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-1b1')
model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-1b1')

# load the dataset
dataset = load_dataset('csv', data_files='FINETUNE  - model-prompt.csv')

# Split the dataset
dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['PROMPT'], examples['RESPONSE'], examples['CONTEXT'], examples['INTENT'], padding='max_length', truncation=True, max_length=512)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the tokenized dataset
tokenized_dataset = tokenized_dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = tokenized_dataset['train']
eval_dataset = tokenized_dataset['test']

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=4,  # Added gradient accumulation
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
