import torch

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    token_list = token_ids.squeeze(0).tolist() #as we will get a stacked tensor of shape (1, seq_len) we need to remove the batch dimension 
    return tokenizer.decode(token_list)