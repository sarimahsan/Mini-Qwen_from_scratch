import os
import shutil
import json
import argparse
import torch
try:
    from safetensors.torch import save_file
except ImportError:
    print("Please install safetensors using: pip install safetensors")
    raise

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Please install huggingface_hub using: pip install huggingface_hub")
    raise

from transformers import AutoTokenizer

def create_gitattributes(export_dir):
    gitattributes_content = """*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
"""
    with open(os.path.join(export_dir, ".gitattributes"), "w", encoding="utf-8") as f:
        f.write(gitattributes_content)
    print("Created .gitattributes file.")

def create_chat_template(export_dir):
    # Professional ChatML template
    chat_template = """{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}
{% elif message['role'] == 'system' %}
{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}
{% elif message['role'] == 'assistant' %}
{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|im_start|>assistant\\n' }}
{% endif %}"""
    with open(os.path.join(export_dir, "chat_template.jinja"), "w", encoding="utf-8") as f:
        f.write(chat_template)
    print("Created chat_template.jinja file.")

def prepare_and_push():
    parser = argparse.ArgumentParser(description="Convert and push MiniQwen model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="HF Repo ID (e.g. username/mini-qwen)")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Path to local PyTorch checkpoint")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token")
    args = parser.parse_args()

    export_dir = "./hf_export"
    os.makedirs(export_dir, exist_ok=True)
    print(f"Export directory prepared at: {export_dir}")

    # 1. Convert checkpoint to safetensors
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        print("Please train the model first or specify the correct path using --checkpoint.")
        return

    print(f"Loading weights from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    
    safetensors_path = os.path.join(export_dir, "model.safetensors")
    print(f"Saving weights in safetensors format to {safetensors_path}...")
    save_file(state_dict, safetensors_path)

    # 2. Copy the self-contained HF model architecture file
    shutil.copy("modeling_mini_qwen.py", os.path.join(export_dir, "modeling_mini_qwen.py"))
    print("Copied modeling_mini_qwen.py to export directory.")

    # 3. Create config.json
    config_data = {
        "architectures": ["MiniQwenForCausalLM"],
        "model_type": "mini_qwen",
        "auto_map": {
            "AutoConfig": "modeling_mini_qwen.MiniQwenConfig",
            "AutoModelForCausalLM": "modeling_mini_qwen.MiniQwenForCausalLM"
        },
        "vocab_size": 50257,
        "hidden_dim": 256,
        "num_layers": 4,
        "num_heads": 8,
        "num_kv_heads": 2,
        "max_seq_len": 256,
        "tie_word_embeddings": True
    }
    with open(os.path.join(export_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print("Created config.json file.")

    # 4. Create generation_config.json
    generation_config = {
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        "max_length": 256,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "do_sample": True
    }
    with open(os.path.join(export_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(generation_config, f, indent=2)
    print("Created generation_config.json file.")

    # 5. Save tokenizer files
    print("Loading and exporting tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(export_dir)

    # Add custom chat template mapping to the tokenizer config
    tok_config_path = os.path.join(export_dir, "tokenizer_config.json")
    if os.path.exists(tok_config_path):
        with open(tok_config_path, "r", encoding="utf-8") as f:
            tok_config = json.load(f)
        
        # Set chat_template in config using the template string
        tok_config["chat_template"] = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )
        
        with open(tok_config_path, "w", encoding="utf-8") as f:
            json.dump(tok_config, f, indent=2)
            
    # 6. Create extra files
    create_gitattributes(export_dir)
    create_chat_template(export_dir)

    # 7. Create professional README.md for the Model Hub page
    model_card = f"""---
language: en
license: mit
tags:
- mini-qwen
- causal-lm
- text-generation
- custom-architecture
---

# Mini-Qwen Custom Model

This is a compact, custom Qwen-inspired autoregressive language model trained from scratch.

## Model Highlights
- Grouped-Query Attention (GQA)
- Learnable QK RMSNorm for training stability
- Rotary Position Embeddings (RoPE)
- SwiGLU Feed-Forward layers
- Tied Input/Output word embeddings

## Usage
You can load this model directly using standard Hugging Face `transformers` APIs with `trust_remote_code=True`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "{args.repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)

prompt = "Once upon a time, there was a little boy named"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card)
    print("Created model card README.md.")

    # 8. Upload to Hugging Face
    print(f"Initiating upload to Hugging Face Hub repo '{args.repo_id}'...")
    api = HfApi()
    
    # Create the repository if it doesn't exist
    api.create_repo(repo_id=args.repo_id, exist_ok=True, token=args.token)
    
    # Upload folder contents
    api.upload_folder(
        folder_path=export_dir,
        repo_id=args.repo_id,
        repo_type="model",
        token=args.token
    )
    print("Upload completed successfully! Your model is now live on Hugging Face Hub.")

if __name__ == "__main__":
    prepare_and_push()
