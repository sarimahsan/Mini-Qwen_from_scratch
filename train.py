import torch
import torch.nn as nn
from model import MiniQwen
from optimizer import AdamW
from data import get_dataloader
from generate import generate

# 1. Hyperparameters
dataset_name = "wikitext"
dataset_config = "wikitext-2-raw-v1"
tokenizer_name = "gpt2"
seq_len = 128
batch_size = 8
epochs = 3
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# 2. Load dataset and tokenizer
loader, tokenizer = get_dataloader(
    dataset_name=dataset_name,
    dataset_config=dataset_config,
    split="train",
    tokenizer_name=tokenizer_name,
    seq_len=seq_len,
    batch_size=batch_size,
    shuffle=True
)

vocab_size = len(tokenizer)
print(f"Tokenizer vocab size: {vocab_size}")

# 3. Initialize Model
model = MiniQwen(
    vocab_size=vocab_size,
    hidden_dim=256,      # Slightly wider hidden_dim for real text vocabulary
    num_layers=4,        # 4 layers
    num_heads=8,
    num_kv_heads=2,
    max_seq_len=seq_len
)
model = model.to(device)

# 4. Optimizer and Loss Function
optimizer = AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 5. Training Loop
print("Starting training pipeline...")
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)  # (B, T, V)

        # Reshape for loss calculation
        B, T, V = logits.shape
        loss = criterion(
            logits.view(B * T, V),
            y.view(B * T)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0:
            print(f"Epoch {epoch} | Step {step}/{len(loader)} | Loss {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"\n--- Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} ---\n")

# 6. Test Text Generation
print("\nTesting text generation post-training:")
test_prompt = "The future of artificial intelligence is"
input_ids = tokenizer.encode(test_prompt)
prompt_tensor = torch.tensor([input_ids], device=device)

# Generate new tokens
generated_ids = generate(model, prompt_tensor, max_new_tokens=30, device=device)
generated_text = tokenizer.decode(generated_ids[0].tolist())

print(f"Prompt: {test_prompt}")
print(f"Generated text:\n{generated_text}\n")
print("OK")