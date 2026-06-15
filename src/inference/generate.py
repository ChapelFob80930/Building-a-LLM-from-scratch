import torch

def generate(model, idx, max_new_tokens, context_size, temperature = 1.0, top_k = None, eos_id = None):

    for _ in range(max_new_tokens):
        idx_trimmed = idx[:, -context_size:] # Trim context to fit within model's context size
        
        with torch.no_grad():
            logits = model(idx_trimmed)
        
        logits = logits[:,-1,:] # Get logits for the last token in the sequence
    
        if top_k is not None:
            top_k_logits, _ = torch.topk(logits, top_k)
            min_top_k_logit = top_k_logits[:,-1] # Get the lowest logit in the top-k for each batch item
            logits = torch.where(
                condition = logits < min_top_k_logit, # Compare with the lowest logit in the top-k
                input = torch.tensor(float("-inf")).to(logits.device), # Set to -inf if condition is true
                other=logits # Keep original logit if condition is false
            )
        
        if temperature > 0.0:
            logits = logits/temperature
            probas = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probas, num_samples=1)
        
        else:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        
        # print(f"Next token ID: {next_token_id.item()}") # Debugging line to print the next token ID
        
        if next_token_id == eos_id:
            break
        
        idx = torch.cat([idx, next_token_id], dim=-1)
        
        # idx = torch.cat((idx, next_token_id), dim=1)

        # if eos_id is not None and (next_token_id == eos_id).any():
        #     break
        
    return idx
    