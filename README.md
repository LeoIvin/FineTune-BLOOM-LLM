# Fine-tuning BLOOM Model for Text Generation

This project involves fine-tuning the BLOOM model on a custom dataset for text generation tasks.

## Project Overview

The goal of this project is to fine-tune the BLOOM model on a dataset containing prompts, responses, contexts, and intents to improve its performance in generating contextually accurate and relevant text.

## Key Components

1. **Loading the Model and Tokenizer**
    - Use `AutoTokenizer` and `AutoModelForCausalLM` from the `transformers` library to load the pre-trained BLOOM model.

    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-1b1')
    model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-1b1')
    ```

2. **Loading and Splitting the Dataset**
    - Load the dataset using `load_dataset` from the `datasets` library.
    - Split the dataset into training and evaluation sets.

    ```python
    from datasets import load_dataset

    dataset = load_dataset('csv', data_files='FINETUNE  - model-prompt.csv')
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    ```

3. **Tokenizing the Dataset**
    - Define a tokenization function to process the text data.
    - Apply the tokenization function to the dataset.

    ```python
    def tokenize_function(examples):
        return tokenizer(examples['PROMPT'], examples['RESPONSE'], examples['CONTEXT'], examples['INTENT'], padding='max_length', truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['test']
    ```

4. **Training the Model**
    - Define training arguments using `TrainingArguments`.
    - Use the `Trainer` class to fine-tune the model.

    ```python
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4, 
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
    ```

## Getting Started

### Prerequisites

- Python 3.x
- Transformers library
- Datasets library

### Installing

1. **Clone the repository:**

    ```bash
    git clone https://github.com/LeoIvin/FineTune-BLOOM-LLM.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd fine-tune-bloom
    ```

3. **Install the required libraries:**

    ```bash
    pip install transformers datasets
    ```

4. **Run the script:**

    ```bash
    python fine_tune_bloom.py
    ```

## Usage

The script `fine_tune_bloom.py` loads the model and tokenizer, processes the dataset, and fine-tunes the model. Ensure you have the dataset file `FINETUNE  - model-prompt.csv` in the project directory.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the [BLOOM team](https://huggingface.co/bigscience/bloom) for providing the pre-trained model.

