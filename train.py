import torch
import torch.nn as nn
import math
from model import MiniQwen
from optimizer import AdamW
from data import get_dataloader
from generate import generate

# 1. Hyperparameters & Configuration
dataset_name = "roneneldan/TinyStories"
dataset_config = None  # TinyStories does not use a config name
tokenizer_name = "gpt2"
seq_len = 256          # Increased sequence length for richer contexts
batch_size = 64        # Larger batch size to fully utilize GPU memory
epochs = 1
max_lr = 6e-4          # Peak learning rate
min_lr = 6e-5          # Minimum decayed learning rate
warmup_steps = 200     # Linear warmup steps
weight_decay = 0.01
clip_grad = 1.0        # Gradient clipping limit
max_samples = 40000    # Train subset size (~10M tokens), fast to tokenize and load
val_max_samples = 1000 # Validation subset size
val_interval = 250     # Steps between validation evaluations
save_path = "best_model.pt"

# Performance toggles
use_custom_optimizer = False  # Set to True to use custom scratch AdamW, False for native torch.optim.AdamW (fused)

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"
print(f"Using device: {device} (type: {device_type})")

# 2. Load dataset and tokenizer
print("Loading train dataset...")
train_loader, tokenizer = get_dataloader(
    dataset_name=dataset_name,
    dataset_config=dataset_config,
    split="train",
    tokenizer_name=tokenizer_name,
    seq_len=seq_len,
    batch_size=batch_size,
    shuffle=True,
    max_samples=max_samples
)

print("Loading validation dataset...")
val_loader, _ = get_dataloader(
    dataset_name=dataset_name,
    dataset_config=dataset_config,
    split="validation",
    tokenizer_name=tokenizer_name,
    seq_len=seq_len,
    batch_size=batch_size,
    shuffle=False,
    max_samples=val_max_samples
)

vocab_size = len(tokenizer)
print(f"Tokenizer vocab size: {vocab_size}")

# 3. Initialize Model
model = MiniQwen(
    vocab_size=vocab_size,
    hidden_dim=256,
    num_layers=4,
    num_heads=8,
    num_kv_heads=2,
    max_seq_len=seq_len,
    tie_word_embeddings=True # Tie weights for better generalization and smaller footprint
)
model = model.to(device)

# 4. Optimizer, Scheduler, and Loss Function
if use_custom_optimizer:
    print("Using custom AdamW optimizer implemented from scratch...")
    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
else:
    print("Using PyTorch native AdamW optimizer with fused kernel...")
    use_fused = (device_type == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=weight_decay,
        fused=use_fused
    )

criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda"))

def get_lr(step, total_steps, max_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    if step > total_steps:
        return min_lr
    # Cosine decay
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

total_steps = len(train_loader) * epochs
print(f"Total steps: {total_steps}")

# 5. Training Loop
print("Starting training pipeline...")
best_val_loss = float("inf")
global_step = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for step, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        # Dynamic learning rate update
        current_lr = get_lr(global_step, total_steps, max_lr, min_lr, warmup_steps)
        if use_custom_optimizer:
            optimizer.lr = current_lr
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            logits = model(x)  # (B, T, V)
            B, T, V = logits.shape
            loss = criterion(
                logits.view(B * T, V),
                y.view(B * T)
            )

        # Backward pass with gradient scaling
        if device_type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

        total_loss += loss.item()
        global_step += 1

        if global_step % 20 == 0 or step == 0:
            print(f"Epoch {epoch} | Step {step}/{len(train_loader)} | Loss {loss.item():.4f} | LR {current_lr:.2e}")

        # Periodic Validation
        if global_step % val_interval == 0 or global_step == total_steps:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)
                    with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                        val_logits = model(val_x)
                        B_v, T_v, V_v = val_logits.shape
                        v_loss = criterion(
                            val_logits.view(B_v * T_v, V_v),
                            val_y.view(B_v * T_v)
                        )
                    val_loss += v_loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"\n[Validation] Step {global_step} | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f}")
            
            # Save checkpoint if validation improves
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                print(f"--> Saved new best checkpoint to '{save_path}'\n")
            else:
                print("--> Validation did not improve.\n")
                
            model.train()

    avg_train_loss = total_loss / len(train_loader)
    print(f"\n--- Epoch {epoch} Complete | Avg Train Loss: {avg_train_loss:.4f} ---\n")

# 6. Test Text Generation
print("\nTesting text generation post-training:")
# Load best model checkpoint for generation
if torch.os.path.exists(save_path):
    print(f"Loading best model checkpoint from '{save_path}' for evaluation...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    
test_prompt = "Once upon a time, there was a little puppy named"
input_ids = tokenizer.encode(test_prompt)
prompt_tensor = torch.tensor([input_ids], device=device)

# Generate new tokens
generated_ids = generate(model, prompt_tensor, max_new_tokens=40, device=device)
generated_text = tokenizer.decode(generated_ids[0].tolist())

print(f"Prompt: {test_prompt}")
print(f"Generated text:\n{generated_text}\n")
print("OK")