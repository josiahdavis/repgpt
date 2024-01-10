"""
References: 
1. The original source code from openAI: https://github.com/openai/gpt-2/blob/master/src/model.py
2. Andrej Karpathy's implementation: https://github.com/karpathy/nanoGPT/blob/master/model.py
3. HuggingFace's implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from loguru import logger

class LRScheduler:
    def __init__(self, max_iters, warmup_iters, decay_iters, max_lr, min_lr):
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
        self.max_lr = max_lr
        self.min_lr = min_lr

    def step(self, step):
        if step < self.warmup_iters:
            return self.max_lr * step / self.warmup_iters
        if step > self.decay_iters:
            return self.min_lr

        decay_ratio = (step - self.warmup_iters) / (self.decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

class MultiheadedAttention(nn.Module):
    # Information is gathered from other tokens in the context sequence.
    # The mechanism is the humble pairwise dot product between all tokens combination.
    # A single sequence provides multiple training examples through triangular masking.
    # Input and output dimension is the same, batch_size, context_size, n_embd.
    def __init__(self, head_size, n_head, n_embd, context_size, dropout, bias):
        super().__init__()
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=bias)
        self.query = nn.Linear(n_embd, n_embd, bias=bias)
        self.value = nn.Linear(n_embd, n_embd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, context_size, n_embd = x.shape

        # Compute the {key,query,value} projections for all heads at the same time
        k = self.key(x)     # (batch_size, context_size, n_embd)
        q = self.query(x)   # (batch_size, context_size, n_embd)
        v = self.value(x)   # (batch_size, context_size, n_embd)

        # Compute self-attention for each head independently.
        # Convert the head to a batch dimension by:
        # (1) breaking up the embedding dimension into their individual heads, then
        # (2) swapping the context_size and n_head
        k = k.view(batch_size, context_size, self.n_head, n_embd // self.n_head).transpose(1, 2) # (batch_size, n_head, context_size, n_embd // self.n_head)
        q = q.view(batch_size, context_size, self.n_head, n_embd // self.n_head).transpose(1, 2) # (batch_size, n_head, context_size, n_embd // self.n_head)
        v = v.view(batch_size, context_size, self.n_head, n_embd // self.n_head).transpose(1, 2) # (batch_size, n_head, context_size,  n_embd // self.n_head)

        # Compute the pairwise similarity of all of the tokens in the sequence.
        # Batch dimensions are the training example and the attention head.
        # The mask is applied within the function due to is_causal=True argument.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True) # (batch_size, n_head, context_size,  n_embd // self.n_head)

        # Convert back to 3D shape
        # Contiguous is required here, which refers to creating a new tensor
        out = out.transpose(1, 2).contiguous().view(batch_size, context_size, n_embd)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    # This operates on a per-token level, across the entire embedding space.
    # Information from other tokens is gathered by the dot-product.
    # Then the model needs to "think" on that information it has gathered.
    def __init__(self, n_embd, dropout, bias):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.layer_norm1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.multi_headed_attention = MultiheadedAttention(head_size, config.n_head, config.n_embd, config.context_size, config.dropout, config.bias)
        self.layer_norm2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.feed_forward = FeedForward(config.n_embd, config.dropout, config.bias)

    def forward(self, x):
        x = x + self.multi_headed_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_size = config.context_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.context_size, config.n_embd)
        self.blocks = nn.Sequential(*[Transformer(config) for _ in range(config.n_layer)])
        self.layer_norm_final = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Convert the output back to the vocabulary space and use weight tying.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)
        # > "We scale the weights of residual layers at initialization by a factor of 1/ N where N is the number of residual layers." - GPT2 paper
        # I take 'Residual layer' to mean the final linear layer before the residual skip connection.
        # There are 2 skip connections in each transformer block.
        # One around the self-attention block, the other around the feed-forward block.
        n_residual_layers = 2 * config.n_layer
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("linear2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(n_residual_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # Implement forward pass for training (Loss calculation is performed)
        batch_size, context_size = idx.shape
        device = idx.device
        tok_emb = self.token_embedding_table(idx)  # (batch_size, context_size, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(context_size, dtype=torch.long, device=device))  # (context_size, n_embd)
        x = tok_emb + pos_emb  # (batch_size, context_size, n_embd)
        x = self.blocks(x)  # (batch_size, context_size, n_embd)
        x = self.layer_norm_final(x)  # (batch_size, context_size, n_embd)
        logits = self.lm_head(x)  # (batch_size, context_size, vocab_size)
        batch_size, context_size, vocab_size = logits.shape
        logits = logits.view(batch_size * context_size, vocab_size)
        targets = targets.view(batch_size * context_size)
        loss = F.cross_entropy(logits, targets)
        return logits, loss