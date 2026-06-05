import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class MiniQwenConfig(PretrainedConfig):
    model_type = "mini_qwen"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50257,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        max_seq_len=256,
        tie_word_embeddings=True,
        **kwargs
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len

# ---------------------------
# RMSNorm Implementation
# ---------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

# ---------------------------
# RoPE Position Cache & Apply
# ---------------------------
def build_rope_cache(seq_len, head_dim, base=10000, device="cpu"):
    assert head_dim % 2 == 0, "head_dim must be even"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin

def apply_rope(x, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = x1 * cos[..., ::2] - x2 * sin[..., ::2]
    out[..., 1::2] = x1 * sin[..., ::2] + x2 * cos[..., ::2]
    return out

# ---------------------------
# GQA repeat
# ---------------------------
def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    x = x[:, :, None, :, :]            # (b, h, 1, s, d)
    x = x.expand(b, h, n_rep, s, d)    # repeat
    x = x.reshape(b, h * n_rep, s, d)
    return x

# ---------------------------
# GQA Attention
# ---------------------------
class Qwen3Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.n_rep = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(b, s, self.hidden_dim)
        return self.out_proj(out)

# ---------------------------
# SwiGLU FFN
# ---------------------------
class SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_dim=None):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = int(hidden_dim * 4)
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        gate = self.w_gate(x)
        up = self.w_up(x)
        return self.w_down(F.silu(gate) * up)

# ---------------------------
# TransformerBlock
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = Qwen3Attention(hidden_dim, num_heads, num_kv_heads)
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = SwiGLUFeedForward(hidden_dim)

    def forward(self, x, cos, sin):
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, cos, sin)
        x = x + residual

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x

# ---------------------------
# MiniQwenForCausalLM (HF wrapper)
# ---------------------------
class MiniQwenForCausalLM(PreTrainedModel):
    config_class = MiniQwenConfig
    base_model_prefix = "model"
    _supports_flash_attn_2 = True

    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config.hidden_dim
        self.max_seq_len = config.max_seq_len

        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.hidden_dim, config.num_heads, config.num_kv_heads)
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.embed.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            b, s = input_ids.shape
            x = self.embed(input_ids)
        elif inputs_embeds is not None:
            b, s, _ = inputs_embeds.shape
            x = inputs_embeds
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        # Recompute RoPE cache dynamically
        cos, sin = build_rope_cache(s, self.blocks[0].attn.head_dim, device=x.device)

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
