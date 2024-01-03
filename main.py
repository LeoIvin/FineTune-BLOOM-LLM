# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="bigscience/bloom-7b1")

# Define the prompt
prompt = "What do you know about Nigeria"

# Generate text
output = generator(prompt, max_length=100)


# Print the generated text
print(output[0]['generated_text'])


""" tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m") """
