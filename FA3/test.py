# demo.py
import torch
import flash_attn_ext as F

device = "cuda"
B, T, NH, D = 8, 2048, 8, 128
model_dim = NH * D

# Create inputs in the layout your kernel expects: [B, T, NH*D]
Q = torch.empty(B, T, model_dim, device=device, dtype=torch.bfloat16).random_(-1, 2)
K = torch.empty(B, T, model_dim, device=device, dtype=torch.bfloat16).random_(-1, 2)
V = torch.empty(B, T, model_dim, device=device, dtype=torch.bfloat16).random_(-1, 2)
O = torch.zeros(B, T, model_dim, device=device, dtype=torch.bfloat16)

# Call the extension (runs on current CUDA stream)
F.flash_attention_bf16(Q, K, V, O, B, T, NH, D)

print("O:", O.shape, O.dtype, O.device)
