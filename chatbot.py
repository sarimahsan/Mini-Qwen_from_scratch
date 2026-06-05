import sys
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Interactive Chatbot for MiniQwen")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Path to local trained PyTorch checkpoint")
    parser.add_argument("--repo_id", type=str, default=None, help="HF Repo ID to download and run (e.g. username/mini-qwen)")
    parser.add_argument("--temp", type=type(0.7), default=0.7, help="Generation temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k tokens filtering")
    parser.add_argument("--top_p", type=type(0.9), default=0.9, help="Top-p/Nucleus filtering")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max new tokens to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model and Tokenizer
    if args.repo_id:
        print(f"Loading model from Hugging Face Hub: '{args.repo_id}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.repo_id)
            model = AutoModelForCausalLM.from_pretrained(args.repo_id, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading model from Hugging Face: {e}")
            sys.exit(1)
    else:
        print(f"Loading local checkpoint from '{args.checkpoint}'...")
        if not os.path.exists(args.checkpoint):
            print(f"Error: Local checkpoint '{args.checkpoint}' not found.")
            print("To run locally, please train a model first or specify --repo_id to load from HF.")
            sys.exit(1)
        
        # Load locally via self-contained modeling file
        try:
            # Add current directory to path to locate modeling_mini_qwen
            sys.path.append(os.getcwd())
            from modeling_mini_qwen import MiniQwenConfig, MiniQwenForCausalLM
            
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            config = MiniQwenConfig(
                vocab_size=len(tokenizer),
                hidden_dim=256,
                num_layers=4,
                num_heads=8,
                num_kv_heads=2,
                max_seq_len=256,
                tie_word_embeddings=True
            )
            model = MiniQwenForCausalLM(config)
            
            state_dict = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state_dict)
            print("Local checkpoint weights loaded successfully.")
        except Exception as e:
            print(f"Error preparing local model: {e}")
            sys.exit(1)

    model = model.to(device)
    model.eval()

    # 2. Start Chat Loop
    print("\n" + "="*50)
    print("      Welcome to the Mini-Qwen Interactive Chat!     ")
    print("   Type 'exit' or 'quit' to end the conversation.    ")
    print("="*50 + "\n")

    history = []
    system_prompt = "You are a helpful, creative assistant. Speak in a friendly tone and tell short stories if asked."

    while True:
        try:
            user_input = input("User: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        history.append(("user", user_input))

        # Format input using ChatML-style template
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        for role, content in history:
            formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        formatted_prompt += "<|im_start|>assistant\n"

        # Encode prompt
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temp,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode output and extract assistant response
        full_tokens = output_ids[0].tolist()
        new_tokens = full_tokens[input_ids.shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Clean up tags if model outputs them explicitly
        response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

        print(f"Assistant: {response}\n")
        history.append(("assistant", response))

        # Keep history reasonable to prevent sequence length overflow
        if len(history) > 10:
            history = history[-10:]

if __name__ == "__main__":
    main()
