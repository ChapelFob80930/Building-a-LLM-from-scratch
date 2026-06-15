import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out%num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length), diagonal=1)
        )
        
# x: (B, T, D_in)

# Q,K,V projection
# → (B, T, D_out)

# Split heads
# → (B, T, H, head_dim)

# Transpose
# → (B, H, T, head_dim)

# Attention scores
# → (B, H, T, T)

# Weighted values
# → (B, H, T, head_dim)

# Merge heads
# → (B, T, D_out)
        
    def forward(self, x):
        
        # B = batch size
        # T = sequence length (number of tokens)
        # H = number of attention heads
        # head_dim = head dimension
        # D_in = input embedding dimension
        # D_out = model dimension
        
        
        # Input tensor: (B, T, D_in)
        # B = batch size, T = number of tokens, D_in = input embedding dimension
        b, num_tokens, d_in = x.shape
        
        # Linear projections to Query, Key, Value
        # (B, T, D_in) @ (D_in, D_out) -> (B, T, D_out)
        queries = self.W_query(x)   # (B, T, D_out) -> each token's query vector
        keys = self.W_key(x)        # (B, T, D_out) -> each token's key vector
        values = self.W_value(x)    # (B, T, D_out) -> each token's value vector
        
        # Split the model dimension across multiple attention heads
        # (B, T, D_out) -> (B, T, H, head_dim)
        # H = number of heads
        # head_dim = head dimension (D_out / H)
        # Each token now has a separate representation for each head
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)   # (B, T, H, head_dim) -> each token's representation in each head
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)         # (B, T, H, head_dim) -> each token's key representation in each head
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)     # (B, T, H, head_dim) -> each token's value representation in each head
        
        
        # Move head dimension before tokens to compute attention independently per head
        # (B, T, H, head_dim) -> (B, H, T, head_dim)
        # Now each head processes the full sequence separately
        queries = queries.transpose(1,2)  # (B, H, T, head_dim) -> each head's queries for all tokens
        keys = keys.transpose(1,2)        # (B, H, T, head_dim) -> each head's keys for all tokens 
        values = values.transpose(1,2)    # (B, H, T, head_dim) -> each head's values for all tokens
        
        
        # Compute attention scores
        # (B, H, T, head_dim) @ (B, H, head_dim, T) -> (B, H, T, T)
        # For each head, each token computes similarity with every other token
        attn_scores = queries @ keys.transpose(2,3)    
        # (B, H, T, T) -> attention score of each token attending to every token in each head
        
        # Apply causal mask to prevent tokens from attending to future tokens
        # mask shape: (T, T) -> broadcasted to (B, H, T, T)
        ## mask_bool = self.mask.bool()[:num_tokens, :num_tokens] 
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        
        # Normalize attention scores into probabilities
        # softmax over the tokens being attended to
        # (B, H, T, T) -> attention weights for each token attending to every token in each head
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) 
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum of value vectors
        # (B, H, T, T) @ (B, H, T, head_dim) -> (B, H, T, head_dim)
        # Each token aggregates information from all tokens in the sequence
        context_vec = (attn_weights @ values).transpose(1,2)
        # Move heads back after token dimension
        # (B, H, T, head_dim) -> (B, T, H, head_dim) -> # each token's output representation from each head
        
        # Concatenate all heads
        # (B, T, H, head_dim) -> (B, T, D_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # Each token now has a combined representation from all heads
        
        # Final linear projection to mix information from different heads
        # (B, T, D_out) @ (D_out, D_out) -> (B, T, D_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec