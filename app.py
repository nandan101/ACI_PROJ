import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr
import re

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"  # Use a larger model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Function to generate text
def generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Function to clean generated text
def clean_generated_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'(\b\w+\b)( \1)+', r'\1', text)  # Remove repetitive phrases
    return text.strip()

# Gradio Interface
def generate_text_with_gradio(prompt):
    generated_text = generate_text(prompt)
    cleaned_text = clean_generated_text(generated_text)
    return cleaned_text

# Create a Gradio interface
interface = gr.Interface(
    fn=generate_text_with_gradio,
    inputs="text",
    outputs="text",
    title="Text Generation",
    description="Enter a prompt to generate text."
)

# Launch the interface
interface.launch()