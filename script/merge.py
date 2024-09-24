# Script to merge LoRA adapters into the base model

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
)

# Load the LoRA model
print("Loading LoRA model...")
lora_model = PeftModel.from_pretrained(
    base_model, "../LLaMA-Factory/eddie", is_trainable=False)

# Merge weights
print("Merging weights...")
merged_model = lora_model.merge_and_unload()

# Save the merged model
print("Saving merged model...")
merged_model.save_pretrained("./eddie", safe_serialization=True)

# Save the tokenizer
print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
tokenizer.save_pretrained("./eddie")

print("Model and tokenizer saved to ./merged_model")
