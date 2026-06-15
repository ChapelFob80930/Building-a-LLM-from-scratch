import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Autoregressive text generation using greedy decoding.

    idx shape: (batch, seq_len)
    returns: (batch, seq_len + max_new_tokens)
    """

    for _ in range(max_new_tokens):

        # crop to model context window
        idx_cond = idx[:, -context_size:]

        # forward pass
        with torch.no_grad():
            logits = model(idx_cond)

        # take logits for last token
        logits = logits[:, -1, :]  # (batch, vocab_size)

        # convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # greedy token selection
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        # append token
        idx = torch.cat((idx, idx_next), dim=1)

    return idx