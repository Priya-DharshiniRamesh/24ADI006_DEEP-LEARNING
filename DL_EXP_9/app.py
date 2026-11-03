
# IMPORTANT: Import spaces before torch
import spaces

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Student Details
student_name = "Priya Dharshini R"
roll_number = "24BAD092"

# Load GPT-2 model
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Generate chatbot response
@spaces.GPU(duration=60)
def generate_response(message, history):

    prompt = ""

    for item in history[-6:]:
        if isinstance(item, dict):
            prompt += f"{item['role']}: {item['content']}\n"
        elif isinstance(item, (list, tuple)):
            prompt += f"User: {item[0]}\n"
            if len(item) > 1 and item[1]:
                prompt += f"Assistant: {item[1]}\n"

    prompt += f"User: {message}\nAssistant:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip() or "Please try another question."


# Create Gradio interface
demo = gr.ChatInterface(
    fn=generate_response,
    title="AI Chatbot - Priya Dharshini R",
    description="Roll Number: 24BAD092 | Transformer-Based LLM Chatbot"
)

demo.launch()
