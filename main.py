# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

# Initialize the text generation pipeline
generator = pipeline('text-generation', model='bigscience/Bloom7b1')

# Define the prompt
prompt = "What do you know about Nigeria"

# Generate text
output = generator(prompt, max_length=100)


# Print the generated text
print(output[0]['generated_text'])


""" tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m") """
