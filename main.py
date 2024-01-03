from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

# Define the prompt
prompt = "What do you know about Nigeria"

# Encode the prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate text with a maximum length of 100
output = model.generate(input_ids, max_length=100)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)
