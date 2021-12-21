import torch
from torch import nn
import numpy as np


class GenerativeAttention(nn.Module):
    
    def __init__(self, hidden_size: int = 128, n_heads: int = 4, attn_pd: float = 0.1, resid_pd: float = 0.1):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key   = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.proj  = nn.Linear(hidden_size, hidden_size)
        self.n_heads = n_heads
        if (hidden_size % n_heads) != 0:
            raise ValueError(
                f"hidden_size mut be a integer multiple of n_heads, got {hidden_size} / {n_heads} = {hidden_size/n_heads}"
            )
        self.head_dim = hidden_size // n_heads
        
        self.attn_dropout = nn.Dropout(attn_pd)
        self.resid_dropout = nn.Dropout(resid_pd)
        
        self.init_weights()
        
    def init_weights(self):
        for module in [self.query, self.key, self.value, self.proj]:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def split_for_heads(self, inputs, num_heads, head_dim):
        new_shape = inputs.size()[:-1] + (num_heads, head_dim)
        inputs = inputs.reshape(*new_shape)
        inputs = inputs.permute(0, 2, 1, 3)
        return inputs
    
    def merge_from_heads(self, inputs, num_heads, head_dim):
        inputs = inputs.permute(0, 2, 1, 3)
        new_shape = inputs.size()[:-2] + (num_heads * head_dim,)
        inputs = inputs.reshape(*new_shape)
        return inputs
    
    def attn(self, query, key, value, mask=None):
        bs = query.size(0)
        seq_len = query.size(-2)
        dim_k = key.size(-1)
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(dim_k)
        
        lm_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.uint8, device=scores.device)
        ).view(
            1, 1, seq_len, seq_len
        ).bool()
        
        scores = torch.where(lm_mask, scores, torch.tensor(-1e4, dtype=scores.dtype, device=scores.device))
        
        if mask is not None:
            scores = torch.where(mask, scores, torch.tensor(-1e4, dtype=scores.dtype, device=scores.device))
            
        scores = torch.softmax(scores, dim=-1)
        scores = scores.type(value.dtype)
        scores = self.attn_dropout(scores)
        
        return torch.matmul(scores, value)
    
    def forward(self, inputs, mask=None):
        q = self.split_for_heads(self.query(inputs), self.n_heads, self.head_dim)
        k = self.split_for_heads(self.key(inputs), self.n_heads, self.head_dim)
        v = self.split_for_heads(self.value(inputs), self.n_heads, self.head_dim)
        
        scores = self.attn(q, k, v, mask)
        scores = self.merge_from_heads(scores, self.n_heads, self.head_dim)
        scores = self.proj(scores)
        scores = self.resid_dropout(scores)
        
        return scores
    
class FeedForward(nn.Module):
    
    def __init__(self, hidden_size: int = 128, resid_pd: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(resid_pd)
        
    
    def init_weights(self):
        for module in [self.input_proj, self.out_proj]:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, inputs):
        x = self.input_proj(inputs)
        x = nn.functional.mish(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x
    
class GenerativeLayer(nn.Module):
    
    def __init__(self, hidden_size: int = 128, n_heads: int = 128, attn_pd: float = 0.1, resid_pd: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = GenerativeAttention(hidden_size=hidden_size, n_heads=n_heads, attn_pd=attn_pd, resid_pd=resid_pd)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ff = FeedForward(hidden_size=hidden_size, resid_pd=resid_pd)
        
        self.init_weights()
        
    def init_weights(self):
        for module in [self.attn, self.ff]:
            module.init_weights()
            
        for ln in [self.ln1, self.ln2]:
            ln.weight.data.fill_(1.0)
            ln.bias.data.zero_()
            
    def forward(self, inputs, mask=None):
        residual = inputs
        x = self.ln1(inputs)
        attn = self.attn(x, mask)
        x = residual + attn
        
        residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = residual + x
        return x
    

class GenerativeTransformer(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        hidden_size: int = 128,
        n_layers: int = 6,
        n_heads: int = 4,
        embd_pd: float = 0.1,
        attn_pd: float = 0.1,
        resid_pd: float = 0.1,
        padding_idx: int = 0
    ):
    
        super().__init__()
        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx
        )

        self.position_embeddings = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=hidden_size,
            padding_idx=padding_idx
        )

        self.emb_dropout = nn.Dropout(embd_pd)
        self.layers = nn.ModuleList([
            GenerativeLayer(
                hidden_size=hidden_size,
                n_heads=n_heads,
                attn_pd=attn_pd,
                resid_pd=resid_pd
            ) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(hidden_size)
        self.output_embeddings = nn.Linear(hidden_size, vocab_size, bias=False)
        self.output_embeddings.weight = self.word_embeddings.weight

        self.init_weights()
    
    def init_weights(self):
        for module in [self.word_embeddings, self.position_embeddings]:
            module.weight.data.normal_(mean=0.0, std=0.02)
            module.weight.data[module.padding_idx].zero_()
            
        for module in self.layers:
            module.init_weights()
            
        self.ln_final.weight.data.fill_(1.0)
        self.ln_final.bias.data.zero_()
        
    def forward(self, input_ids, mask=None, position_ids=None):
        
        bs = input_ids.size(0)
        seq_len = input_ids.size(-1)
        
        if mask is not None:
            mask = mask.view(bs, -1)
            mask = mask[:, None, None, :]
            
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
            position_ids = position_ids.repeat(bs, 1)
            
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        hidden_state = word_embeddings + position_embeddings
        hidden_state = self.emb_dropout(hidden_state)
        
        for layer in self.layers:
            hidden_state = layer(hidden_state, mask)
            
        hidden_state = self.ln_final(hidden_state)
        
        output = self.output_embeddings(hidden_state)
        return output

class CrossEntropyForLM(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.base_loss = nn.CrossEntropyLoss()
        
    def forward(self, prediction, target):
        shift_logits = prediction[..., :-1, :].contiguous()
        shift_labels = target[..., 1:].contiguous()
        # Flatten the tokens
        loss = self.base_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss
