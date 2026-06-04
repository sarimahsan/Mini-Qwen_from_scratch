import torch
import torch.nn.functional as F

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = torch.full_like(logits, -float("inf"))
    out.scatter_(dim=-1, index=ix, src=v)
    return out

def top_p_logits(logits, p=0.9):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)

    cum_probs = torch.cumsum(probs, dim=-1)

    mask = cum_probs > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = 0

    sorted_logits[mask] = -float("inf")

    return sorted_logits.scatter(-1, sorted_idx, sorted_logits)

@torch.no_grad()
def generate(
    model,
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    top_k=50,
    device="cpu"
):
    model.eval()

    x = prompt.to(device)

    for _ in range(max_new_tokens):

        logits = model(x)          # (B, T, V)
        logits = logits[:, -1, :]  # last token only

        # temperature scaling
        logits = logits / temperature

        # top-k filtering
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_token], dim=1)

    return x

if __name__ == "__main__":

    from model import MiniQwen

    vocab_size = 1000

    model = MiniQwen(
        vocab_size=vocab_size,
        hidden_dim=128,
        num_layers=2,
        num_heads=8,
        num_kv_heads=2
    )

    prompt = torch.tensor([[1, 5, 10, 20]])

    out = generate(model, prompt, max_new_tokens=20)

    print("Generated token IDs:")
    print(out)