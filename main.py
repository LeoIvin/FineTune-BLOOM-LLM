from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import pandas as pd 
from datasets import load_dataset

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-1b1')
model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-1b1')

# load the dataset
dataset = load_dataset('csv', data_files='FINETUNE  - model-prompt.csv')

# Split the dataset
dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# tokenize prompts
def tokenize_function(examples):
    return tokenizer(examples['PROMPT'], 
    truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# responses
def tokenize_responses(examples):
    return tokenizer(examples['RESPONSE'], 
    truncation=True, max_length=512)

tokenized_resp = dataset.map(tokenize_responses, batched=True)

# context
def tokenize_context(examples):
    return tokenizer(examples['CONTEXT'], 
    truncation=True, max_length=512)

tokenized_contx = dataset.map(tokenize_context, batched=True)

# intent
def tokenize_intent(examples):
    return tokenizer(examples['INTENT'], 
    truncation=True, max_length=512)

tokenized_int = dataset.map(tokenize_intent, batched=True)



training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
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