# 🧠 Building a Large Language Model From Scratch

> A hands-on implementation of LLM internals — from raw text to token embeddings — following Sebastian Raschka's *Build a Large Language Model From Scratch*. Every step is coded from first principles to build genuine intuition, not just API fluency.

---

## 📁 Repository Structure

```
├── working_with_text_data.ipynb                  # Ch. 2 — Tokenization, vocabulary, embeddings
├── self_attention.ipynb                          # Ch. 3 — Self-attention & multi-head attention
├── GPT_implmentation.ipynb                       # Ch. 4 — Full GPT model architecture
├── GPT_Model_Training_and_Evaluation.ipynb       # Ch. 5 — Pretraining, loss, decoding strategies
├── finetuning_for_classification.ipynb           # Ch. 6 — Spam classifier fine-tuning
├── finetuing_for_instructions.ipynb              # Ch. 7 — Instruction fine-tuning (Alpaca format)
├── Adding_Bells_and_Whistles_to_the_Training_Loop.ipynb  # Appendix A — LR warmup, cosine decay, gradient clipping
├── building_a_tokenizer_from_scratch.ipynb       # Bonus — BPE tokenizer from scratch (Karpathy-style)
├── Parameter-efficient-Finetuning-with-LoRA.ipynb        # Appendix E — LoRA parameter-efficient fine-tuning
├── the-verdict.txt                               # Sample corpus (short story by Edith Wharton)
└── README.md
```

> **Note:** More chapters will be added progressively. This README grows with the repo.

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Chapter 2 — Working with Text Data](#-chapter-2--working-with-text-data)
   - [What is Tokenization?](#what-is-tokenization)
   - [Step 1 — Simple Regex Tokenization](#step-1--simple-regex-tokenization)
   - [Step 2 — Building a Vocabulary](#step-2--building-a-vocabulary)
   - [Step 3 — SimpleTokenizerV1 (Encode & Decode)](#step-3--simpletokenizerv1-encode--decode)
   - [Step 4 — Handling Unknown Words (SimpleTokenizerV2)](#step-4--handling-unknown-words-simpletokenizerv2)
   - [Step 5 — BPE Tokenization with tiktoken](#step-5--bpe-tokenization-with-tiktoken)
   - [Step 6 — Input–Target Pairs for Next-Token Prediction](#step-6--inputtarget-pairs-for-next-token-prediction)
   - [Step 7 — GPTDatasetV1 and the DataLoader](#step-7--gptdatasetv1-and-the-dataloader)
   - [Step 8 — Token Embeddings](#step-8--token-embeddings)
   - [Step 9 — Positional Embeddings](#step-9--positional-embeddings)
   - [Step 10 — Final Input Embeddings](#step-10--final-input-embeddings)
3. [Chapter 3 — Self-Attention & Multi-Head Attention](#-chapter-3--self-attention--multi-head-attention)
   - [What is Self-Attention?](#what-is-self-attention)
   - [Step 1 — Attention Scores via Dot Product](#step-1--attention-scores-via-dot-product)
   - [Step 2 — Softmax Normalisation → Attention Weights](#step-2--softmax-normalisation--attention-weights)
   - [Step 3 — Context Vectors](#step-3--context-vectors)
   - [Step 4 — Full Attention Matrix (All Tokens at Once)](#step-4--full-attention-matrix-all-tokens-at-once)
   - [Step 5 — Learnable Q, K, V Projections](#step-5--learnable-q-k-v-projections)
   - [Step 6 — Scaled Dot-Product Attention](#step-6--scaled-dot-product-attention)
   - [Step 7 — SelfAttention Classes (v1 & v2)](#step-7--selfattention-classes-v1--v2)
   - [Step 8 — Causal (Masked) Attention](#step-8--causal-masked-attention)
   - [Step 9 — Dropout on Attention Weights](#step-9--dropout-on-attention-weights)
   - [Step 10 — CausalAttention Module](#step-10--causalattention-module)
   - [Step 11 — Multi-Head Attention Wrapper (Naive)](#step-11--multi-head-attention-wrapper-naive)
   - [Step 12 — Efficient MultiHeadAttention (Single Projection)](#step-12--efficient-multiheadattention-single-projection)
4. [Chapter 4 — GPT Model Architecture](#-chapter-4--gpt-model-architecture)
   - [GPT-2 Configuration](#gpt-2-configuration)
   - [Step 1 — DummyGPTModel (Architecture Skeleton)](#step-1--dummygptmodel-architecture-skeleton)
   - [Step 2 — Layer Normalization](#step-2--layer-normalization)
   - [Step 3 — GELU Activation](#step-3--gelu-activation)
   - [Step 4 — Feed-Forward Network](#step-4--feed-forward-network)
   - [Step 5 — Residual (Skip) Connections](#step-5--residual-skip-connections)
   - [Step 6 — Transformer Block](#step-6--transformer-block)
   - [Step 7 — Full GPTModel](#step-7--full-gptmodel)
   - [Step 8 — Parameter Count & Weight Tying](#step-8--parameter-count--weight-tying)
   - [Step 9 — GPT-2 Model Variants](#step-9--gpt-2-model-variants)
   - [Step 10 — Autoregressive Text Generation](#step-10--autoregressive-text-generation)
5. [Chapter 5 — Pretraining & Evaluation](#-chapter-5--pretraining--evaluation)
   - [Step 1 — Cross-Entropy Loss from First Principles](#step-1--cross-entropy-loss-from-first-principles)
   - [Step 2 — Perplexity](#step-2--perplexity)
   - [Step 3 — Data Split and DataLoaders](#step-3--data-split-and-dataloaders)
   - [Step 4 — Loss Calculation Functions](#step-4--loss-calculation-functions)
   - [Step 5 — The Training Loop](#step-5--the-training-loop)
   - [Step 6 — Saving and Loading Models](#step-6--saving-and-loading-models)
   - [Step 7 — Loading Pretrained GPT-2 Weights](#step-7--loading-pretrained-gpt-2-weights)
   - [Step 8 — Decoding Strategies](#step-8--decoding-strategies)
6. [Chapter 6 — Fine-Tuning for Classification](#-chapter-6--fine-tuning-for-classification)
   - [Step 1 — Dataset Preparation](#step-1--dataset-preparation)
   - [Step 2 — SpamDataset and DataLoaders](#step-2--spamdataset-and-dataloaders)
   - [Step 3 — Modifying the GPT Head for Classification](#step-3--modifying-the-gpt-head-for-classification)
   - [Step 4 — Training and Evaluation](#step-4--training-and-evaluation)
   - [Step 5 — Inference](#step-5--inference)
7. [Chapter 7 — Instruction Fine-Tuning](#-chapter-7--instruction-fine-tuning)
   - [Step 1 — The Alpaca Format](#step-1--the-alpaca-format)
   - [Step 2 — InstructionDataset and Custom Collate Function](#step-2--instructiondataset-and-custom-collate-function)
   - [Step 3 — The ignore_index Trick](#step-3--the-ignore_index-trick)
   - [Step 4 — Fine-Tuning GPT-2 Medium](#step-4--fine-tuning-gpt-2-medium)
   - [Step 5 — LLM-as-Judge Evaluation](#step-5--llm-as-judge-evaluation)
8. [Appendix A — Advanced Training Loop Techniques](#-appendix-a--advanced-training-loop-techniques)
   - [Step 1 — Linear LR Warmup](#step-1--linear-lr-warmup)
   - [Step 2 — Cosine Annealing Decay](#step-2--cosine-annealing-decay)
   - [Step 3 — Gradient Clipping](#step-3--gradient-clipping)
   - [Step 4 — Full train_model Function](#step-4--full-train_model-function)
9. [Bonus — Building a BPE Tokenizer from Scratch](#-bonus--building-a-bpe-tokenizer-from-scratch)
   - [Why Tokenization Matters](#why-tokenization-matters)
   - [Step 1 — Unicode and Bytes Fundamentals](#step-1--unicode-and-bytes-fundamentals)
   - [Step 2 — BPE Core Algorithm (Functional)](#step-2--bpe-core-algorithm-functional)
   - [Step 3 — Training Loop and Compression Ratio](#step-3--training-loop-and-compression-ratio)
   - [Step 4 — Decode and Encode](#step-4--decode-and-encode)
   - [Step 5 — GPT-2 Regex Pre-tokenization](#step-5--gpt-2-regex-pre-tokenization)
   - [Step 6 — Inspecting Real Tokenizers](#step-6--inspecting-real-tokenizers)
   - [Step 7 — BasicTokenizer Class](#step-7--basictokenizer-class)
   - [Step 8 — RegexTokenizer Class (GPT-4 Style)](#step-8--regextokenizer-class-gpt-4-style)
   - [Step 9 — SentencePiece Comparison](#step-9--sentencepiece-comparison)
   - [Step 10 — Vocab Size Tradeoffs](#step-10--vocab-size-tradeoffs)
10. [Appendix E — Parameter-Efficient Fine-Tuning with LoRA](#-appendix-e--parameter-efficient-fine-tuning-with-lora)
    - [The Problem: Full Fine-Tuning is Expensive](#the-problem-full-fine-tuning-is-expensive)
    - [Step 1 — LoRALayer](#step-1--loralayer)
    - [Step 2 — LinearWithLoRA](#step-2--linearwithlora)
    - [Step 3 — Replacing All Linear Layers](#step-3--replacing-all-linear-layers)
    - [Step 4 — Freeze Base, Train LoRA](#step-4--freeze-base-train-lora)
    - [Step 5 — Results](#step-5--results)
11. [Key Concepts Glossary](#-key-concepts-glossary)
12. [Skills Demonstrated](#-skills-demonstrated)
13. [Environment](#-environment)

---

## 🚀 Project Overview

Large Language Models (LLMs) like GPT are often treated as black boxes. This project tears the box open.

Rather than calling `openai.ChatCompletion.create(...)` and moving on, we build every component by hand:

- A **tokenizer** that converts raw text into integer token IDs
- A **vocabulary** that maps every unique token to a number
- A **sliding-window dataset** that creates (input, target) pairs for next-token prediction
- **Token embeddings** that convert IDs into dense vectors a neural network can process
- **Positional embeddings** that inject word-order information into those vectors
- **Self-attention** from dot-product scores through to multi-head causal attention
- A complete **GPT-2-class model** with layer norm, GELU, feed-forward blocks, residual connections, and autoregressive text generation

The end goal is a complete, trained GPT-style model — built neuron by neuron.

---

## 📒 Chapter 2 — Working with Text Data

**Notebook:** `working_with_text_data.ipynb`  
**Corpus:** `the-verdict.txt` — a short story by Edith Wharton (~20,479 characters)

---

### What is Tokenization?

A neural network cannot process raw text strings. It can only process numbers. **Tokenization** is the process of converting text into a sequence of integers.

The full pipeline looks like this:

```
Raw Text  →  Tokens (strings)  →  Token IDs (integers)  →  Embeddings (vectors)
```

There are different granularities at which you can tokenize:

| Level        | Example input: `"don't"` | Tokens          |
|--------------|--------------------------|-----------------|
| Character    | —                        | `d, o, n, ', t` |
| Word         | —                        | `don't`         |
| Subword (BPE)| —                        | `don`, `'t`     |

Modern LLMs (GPT-2, GPT-3, LLaMA) use **Byte-Pair Encoding (BPE)**, a subword method. We start with a simple word-level approach and build up to BPE.

---

### Step 1 — Simple Regex Tokenization

**File section:** Cells 2–7

We use Python's `re` module to split text on whitespace and punctuation:

```python
import re

# Split on whitespace only → punctuation stays glued to words
result = re.split(r'(\s)', text)
# ['Hello,', ' ', 'world.', ' ', ...]

# Split on punctuation AND whitespace → punctuation becomes its own token
result = re.split(r'([,.]|\s)', text)
# ['Hello', ',', '', ' ', 'world', '.', ...]

# Strip empty strings
result = [item for item in result if item.strip()]
# ['Hello', ',', 'world', '.', ...]
```

The final regex used on the full corpus handles a wider set of punctuation:

```python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
```

**Result:** The 20,479-character story becomes **4,690 tokens**.

**Why keep punctuation as tokens?**  
Punctuation changes meaning. A model needs to learn that `"word."` signals end-of-sentence differently from `"word,"`. Treating `,`, `.`, `--`, `?`, etc. as first-class tokens lets the model attend to them.

---

### Step 2 — Building a Vocabulary

**File section:** Cells 8–9

A **vocabulary** is a bijective mapping between every unique token string and a unique integer.

```python
all_words = sorted(set(preprocessed))  # deduplicate and sort alphabetically
vocab = {token: integer for integer, token in enumerate(all_words)}
```

**Result:** 1,130 unique tokens in the vocabulary.

Sorting alphabetically ensures **deterministic, reproducible** token IDs — the same word always gets the same ID. A few entries:

```
('!', 0)
(',', 5)
('--', 6)
('.', 7)
('Claude', 26)       ← appears in the story
('Gisburn', 38)
```

We also create the **reverse mapping** (int → string) for decoding:

```python
int_to_str = {i: s for s, i in vocab.items()}
```

---

### Step 3 — SimpleTokenizerV1 (Encode & Decode)

**File section:** Cell 10–12

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [token.strip() for token in preprocessed if token.strip()]
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove space before punctuation: "hello ," → "hello,"
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```

**Encode** converts a string to a list of integers.  
**Decode** converts a list of integers back to a string, cleaning up spacing around punctuation with a regex substitution.

**Limitation of V1:** Any word not seen in the training corpus causes a `KeyError` crash. Real-world text has infinite vocabulary coverage; our 1,130-word vocab doesn't.

---

### Step 4 — Handling Unknown Words (SimpleTokenizerV2)

**File section:** Cells 14–19

We add two **special tokens** to the vocabulary:

| Token            | Purpose                                                              |
|-----------------|----------------------------------------------------------------------|
| `<\|endoftext\|>` | Signals the boundary between two unrelated documents in a batch     |
| `<\|unk\|>`       | Replaces any word not found in the vocabulary (out-of-vocabulary token) |

```python
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
# Vocab size grows: 1130 → 1132
```

V2 adds OOV handling:

```python
preprocessed = [
    token if token in self.str_to_int else "<|unk|>"
    for token in preprocessed
]
```

**Joining two documents:**

```python
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
# → "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
```

**Decode output:** `<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.`  
("Hello" and "palace" weren't in The Verdict's vocabulary.)

**Why `<|endoftext|>` matters:** LLMs are trained on concatenated documents. Without a separator, the model might hallucinate connections between a news article and a recipe that happen to be adjacent in the training batch.

---

### Step 5 — BPE Tokenization with tiktoken

**File section:** Cells 20–24

Our handcrafted tokenizer has a 1,130-word vocabulary and can't handle unknown words gracefully. Real LLMs use **Byte-Pair Encoding (BPE)**, which:

1. Starts with a vocabulary of individual bytes (256 tokens)
2. Iteratively merges the most frequent adjacent byte-pair into a new token
3. Repeats until the target vocabulary size is reached (GPT-2 uses **50,257 tokens**)

We use OpenAI's `tiktoken` library, which implements GPT-2's BPE tokenizer:

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, ...]

strings = tokenizer.decode(integers)
# "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
```

**Key advantage — no unknown tokens:**

```python
test_text = "Akwirw ier"   # completely made-up string
test_ids = tokenizer.encode(test_text)
# [33901, 86, 343, 86, 220, 959]

for i in test_ids:
    print(tokenizer.decode([i]))
# Ak / w / ir / w / (space) / ier
```

BPE decomposes any string into subword units. `someunknownPlace` is tokenized into familiar subpieces rather than being replaced with `<|unk|>`. This is critical for handling names, code, scientific terms, and any domain-specific text.

**The Verdict tokenized:** 20,479 characters → **5,145 BPE tokens** (vs. 4,690 with our regex tokenizer — BPE is more granular on some subwords).

---

### Step 6 — Input–Target Pairs for Next-Token Prediction

**File section:** Cells 26–32

LLMs are trained via **next-token prediction** (also called causal language modeling). Given a sequence of tokens, predict the next one.

For each position `i` in a sequence, the input is everything up to position `i`, and the target is token at position `i+1`:

```python
enc_sample = enc_text[50:]   # skip first 50 tokens for illustration
context_size = 4

x = enc_sample[:context_size]   # [290, 4920, 2241, 287]
y = enc_sample[1:context_size+1] # [4920, 2241, 287, 257]
```

Unpacked into individual training examples:

```
[290]              →  4920       (predict next token given 1 token of context)
[290, 4920]        →  2241       (predict given 2 tokens)
[290, 4920, 2241]  →  287        (predict given 3 tokens)
[290, 4920, 2241, 287] → 257     (predict given 4 tokens = full context)
```

In decoded text:

```
" and"                         →  " established"
" and established"             →  " himself"
" and established himself"     →  " in"
" and established himself in"  →  " a"
```

**One sequence of length `N` produces `N` training examples** — this is what makes language modeling so data-efficient. You get massive amounts of supervision signal from a single document.

---

### Step 7 — GPTDatasetV1 and the DataLoader

**File section:** Cells 47–56

We formalize the above with a PyTorch `Dataset` and `DataLoader`.

**The sliding window approach:**

```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
```

**Parameters:**

| Parameter    | Effect                                                                 |
|-------------|------------------------------------------------------------------------|
| `max_length` | Context window size — how many tokens the model sees at once           |
| `stride`     | How many tokens to shift the window by for the next sample             |

**stride = 1** → maximum overlap, one new token per sample. More training examples, but highly redundant.  
**stride = max_length** → no overlap, each token appears in exactly one sample. Efficient, less redundant.

```
Tokens: [A, B, C, D, E, F, G, H]   max_length=4

stride=1:  [A,B,C,D], [B,C,D,E], [C,D,E,F], ...   ← lots of overlap
stride=4:  [A,B,C,D], [E,F,G,H]                    ← no overlap
stride=2:  [A,B,C,D], [C,D,E,F], [E,F,G,H]         ← partial overlap
```

**The DataLoader factory:**

```python
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                          shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader
```

`drop_last=True` discards the final batch if it's smaller than `batch_size`. This prevents shape mismatches during training, where all batches must be uniform.

**Example output (batch_size=8, max_length=4, stride=4):**

```
Inputs:
tensor([[  40,  367, 2885, 1464],   ← 8 sequences
        [1807, 3619,  402,  271],
        ...])

Targets:
tensor([[  367, 2885, 1464, 1807],  ← same 8 sequences, shifted by 1
        [3619,  402,  271, 10899],
        ...])
```

---

### Step 8 — Token Embeddings

**File section:** Cells 57–66

Token IDs are integers. Neural networks need **floating-point vectors** — `torch.nn.Embedding` provides the lookup table:

```python
vocab_size = 6
output_size = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_size)

print(embedding_layer.weight)
# tensor([[ 0.3374, -0.1778, -0.1690],  ← row 0 (token 0's embedding)
#         [ 0.9178,  1.5810,  1.3010],  ← row 1
#         [ 1.2753, -0.2010, -0.1606],  ← row 2
#         [-0.4015,  0.9666, -1.1481],  ← row 3
#         [-1.1589,  0.3255, -0.6315],  ← row 4
#         [-2.8400, -0.7849, -1.4096]], ← row 5
```

Looking up token ID 3:

```python
embedding_layer(torch.tensor([3]))
# tensor([[-0.4015,  0.9666, -1.1481]])  ← row 3 of the weight matrix
```

`nn.Embedding` is **literally a matrix lookup** — calling it with index `[3]` returns the 3rd row. The weights are learned during training so that semantically similar words end up with similar embedding vectors.

**For the real GPT-2 scale:**

```python
vocab_size = 50257    # GPT-2's full BPE vocabulary
output_dim = 256      # embedding dimension (GPT-2 uses 768)

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
```

With a batch of 8 sequences of length 4:

```python
token_embeddings = token_embedding_layer(inputs)
# Shape: [8, 4, 256]  → (batch_size, seq_len, embedding_dim)
```

Each token ID has been replaced with a 256-dimensional learned vector.

---

### Step 9 — Positional Embeddings

**File section:** Cells 68

**The problem:** `nn.Embedding` is a pure lookup table. Token ID 42 always maps to the same vector, whether it's the 1st or 100th word in the sentence. The model has **no sense of word order**.

Consider: *"The dog bit the man"* vs *"The man bit the dog"* — same tokens, completely different meaning. Without position information, a transformer cannot distinguish these.

**The solution:** Add a second learnable embedding that encodes *position*:

```python
context_length = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# torch.arange(4) = [0, 1, 2, 3]  ← position indices
# Shape: [4, 256]  → one 256-dim vector per position
```

Position 0 always gets the same vector, position 1 always gets the same vector, etc. These are **learned**, meaning the model figures out during training how to best encode positional information.

> **Alternative:** Original "Attention is All You Need" paper uses fixed sinusoidal positional encodings. GPT-2 and most modern LLMs use learned positional embeddings, as we do here.

---

### Step 10 — Final Input Embeddings

**File section:** Cell 69

Token embeddings and positional embeddings are **added element-wise** to produce the final input representation:

```python
input_embeddings = token_embeddings + pos_embeddings
# token_embeddings shape: [8, 4, 256]
# pos_embeddings shape:   [4, 256]    ← broadcast across batch dimension
# result shape:           [8, 4, 256]
```

PyTorch's broadcasting automatically expands `pos_embeddings` from `[4, 256]` to `[8, 4, 256]` so the same positional vector is added to every sequence in the batch.

**Final shape:** `[batch_size, seq_len, embedding_dim]` = `[8, 4, 256]`

This tensor is the **input to the transformer**. Each of the 8×4 = 32 tokens is now represented as a 256-dimensional vector that encodes both *what* the token is (token embedding) and *where* it appears (positional embedding).

**The full pipeline, assembled:**

```
Raw Text
   ↓
tiktoken BPE Tokenizer
   ↓
Token IDs  [batch=8, seq=4]  e.g. [[40, 367, 2885, 1464], ...]
   ↓
nn.Embedding(50257, 256)  ← Token Embedding Layer
   ↓
Token Embeddings  [8, 4, 256]
   +
nn.Embedding(4, 256)  ← Positional Embedding Layer
   ↓
Final Input Embeddings  [8, 4, 256]  ← Ready for Transformer
```

---

## 📒 Chapter 3 — Self-Attention & Multi-Head Attention

**Notebook:** `self_attention.ipynb`

---

### What is Self-Attention?

Self-attention is the core mechanism that makes transformers powerful. It answers the question: **for each token in a sequence, how much should it "attend to" every other token?**

Unlike an RNN that reads tokens one at a time left-to-right, self-attention lets every token look at every other token *simultaneously*, in parallel. This is what enables transformers to capture long-range dependencies efficiently.

**Intuition:** When reading *"The animal didn't cross the street because it was too tired"*, to resolve what *"it"* refers to, the model must look back at *"animal"*. Self-attention gives the model a mechanism to learn this connection.

---

### Step 1 — Attention Scores via Dot Product

**Input:** 6 tokens, each represented as a 3-dimensional embedding vector.

```python
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # "Your"    (x¹)
    [0.55, 0.87, 0.66],  # "journey" (x²)
    [0.57, 0.85, 0.64],  # "starts"  (x³)
    [0.22, 0.58, 0.33],  # "with"    (x⁴)
    [0.77, 0.25, 0.10],  # "one"     (x⁵)
    [0.05, 0.80, 0.55]   # "step"    (x⁶)
])  # shape: [6, 3]
```

To compute how much token 2 ("journey") attends to all other tokens, we take the **dot product** of its embedding (the *query*) with every other token's embedding (the *keys*):

```python
query = inputs[1]   # "journey"
attention_scores_2 = torch.tensor([torch.dot(x_i, query) for x_i in inputs])
# → [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]
```

**Why dot product?** If two vectors point in similar directions (similar meaning/context), their dot product is large. If they're orthogonal (unrelated), it's near zero. High dot product = high relevance.

---

### Step 2 — Softmax Normalisation → Attention Weights

Raw dot-product scores can be any real number. We need them to be a **probability distribution** (non-negative, summing to 1) so we can use them as mixing weights:

```python
# Naive normalisation (dividing by sum)
attn_weights = attention_scores_2 / attention_scores_2.sum()
# → [0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656]  (sums to 1.0)

# Proper softmax (exponential normalisation — handles negative scores too)
attn_weights = torch.softmax(attention_scores_2, dim=0)
# → [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]
```

**Why softmax over simple division?** Softmax handles negative scores gracefully (e`^x` is always positive) and amplifies the differences between large and small scores, making the distribution more "peaky" — the most relevant tokens get disproportionately more weight.

---

### Step 3 — Context Vectors

The **context vector** for token 2 is a weighted sum of *all* token embeddings, using the attention weights:

```python
context_vector = sum(attn_weights[i] * inputs[i] for i in range(6))
# → tensor([0.4419, 0.6515, 0.5683])
```

This vector is a blend of information from every token in the sequence, weighted by relevance to "journey". It's richer than the original embedding because it incorporates context from surrounding words.

---

### Step 4 — Full Attention Matrix (All Tokens at Once)

Instead of computing one token's attention at a time, we compute the full attention matrix with a single matrix multiplication:

```python
# All pairwise dot products in one operation
attn_scores = inputs @ inputs.T   # shape: [6, 6]

# Softmax over each row (each token's attention distribution over all tokens)
attn_weights = torch.softmax(attn_scores, dim=-1)  # shape: [6, 6]

# All context vectors in one operation
all_context_vecs = attn_weights @ inputs  # shape: [6, 3]
```

`dim=-1` means softmax is applied along the last dimension (columns), so each **row** sums to 1.0. Row `i` is token `i`'s attention distribution over all tokens.

---

### Step 5 — Learnable Q, K, V Projections

The attention we've built so far is **symmetric** — the score of token A attending to token B equals B attending to A (because we're just doing `inputs @ inputs.T`). Real transformers break this symmetry with three separate learned weight matrices:

| Matrix | Role | Intuition |
|--------|------|-----------|
| **W_query (Wq)** | Projects input into "what am I looking for?" space | The question |
| **W_key (Wk)** | Projects input into "what do I contain?" space | The label on the filing cabinet |
| **W_value (Wv)** | Projects input into "what do I return if selected?" space | The contents of the filing cabinet |

```python
d_in, d_out = 3, 2   # project from 3-dim to 2-dim

W_query = nn.Parameter(torch.rand(d_in, d_out))
W_key   = nn.Parameter(torch.rand(d_in, d_out))
W_value = nn.Parameter(torch.rand(d_in, d_out))

query_2 = x_2 @ W_query   # token 2's query vector
keys    = inputs @ W_key   # all tokens' key vectors
values  = inputs @ W_value # all tokens' value vectors
```

Now attention scores are `query @ keys.T` — asymmetric, because Q and K come from different learned projections. Token A's relevance to token B is no longer the same as B's relevance to A.

---

### Step 6 — Scaled Dot-Product Attention

Raw dot products grow in magnitude as embedding dimension increases (more dimensions = larger sums). Large scores push softmax into saturation, where gradients vanish. The fix: **scale by the square root of the key dimension**:

```python
d_k = keys.shape[-1]   # dimension of the key vectors
attn_scores = query_2 @ keys.T
attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=0)
```

This is the formula from the landmark *"Attention is All You Need"* paper. Dividing by `√d_k` keeps the dot products in a range where softmax produces meaningful gradients regardless of model size.

---

### Step 7 — SelfAttention Classes (v1 & v2)

**SelfAttention_v1** — using raw `nn.Parameter`:

```python
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys    = x @ self.W_key
        values  = x @ self.W_value
        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values
```

**SelfAttention_v2** — replacing `nn.Parameter` with `nn.Linear(bias=False)`:

```python
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key   = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)
        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values
```

Both are mathematically equivalent. `nn.Linear` is preferred in practice: it integrates with PyTorch optimisers, weight-initialisation schemes, and `model.parameters()` cleanly. The key difference in weight layout: `nn.Parameter` stores `(d_in, d_out)`, while `nn.Linear.weight` stores `(d_out, d_in)` (transposed), which is why we assign `sa_v1.W_key.data = sa_v2.W_key.weight.data.T` to prove equivalence.

---

### Step 8 — Causal (Masked) Attention

Standard self-attention lets every token look at every other token — including *future* tokens. For language modelling (predicting the next word), a model must **never see tokens that come after the current position**. This would be data leakage.

**The fix:** apply a **causal mask** (lower-triangular matrix) that blocks future positions:

```python
context_length = 6
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
# tensor([[0., 1., 1., 1., 1., 1.],
#         [0., 0., 1., 1., 1., 1.],
#         ...])   ← 1s mark positions to block

# Set masked positions to -infinity BEFORE softmax
attn_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
```

**Why `-inf`?** Because `softmax(−∞) = 0`. Future positions receive exactly zero attention weight. After masking, each token can only attend to itself and previous tokens.

```
Position 0: attends to [token 0]
Position 1: attends to [token 0, token 1]
Position 3: attends to [token 0, token 1, token 2, token 3]
...
```

This is what makes GPT a **causal** (autoregressive) language model, as opposed to bidirectional models like BERT.

---

### Step 9 — Dropout on Attention Weights

**Dropout** randomly zeroes a fraction of attention weights during training and rescales the survivors by `1/(1 - p)` to preserve expected values:

```python
dropout = nn.Dropout(0.5)
attn_weights = dropout(attn_weights)
# Example: half the weights become 0, survivors are doubled
```

**Why apply dropout to attention weights?** It prevents the model from over-relying on any single attention pattern, forcing it to learn more distributed representations. It's disabled at inference (`model.eval()`).

---

### Step 10 — CausalAttention Module

All the pieces — Q/K/V projections, scaled attention, causal masking, dropout — assembled into one clean PyTorch module:

```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)         # (B, T, T)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values                         # (B, T, d_out)
```

**`register_buffer`** stores the mask as a non-trainable tensor that automatically moves to the correct device (CPU/GPU) with the model. Using `.transpose(1, 2)` instead of `.T` is necessary for batched inputs with shape `(B, T, d)`.

---

### Step 11 — Multi-Head Attention Wrapper (Naive)

Running multiple independent attention heads in parallel and concatenating their outputs:

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
        # Output shape: (B, T, num_heads * d_out)
```

**Why multiple heads?** Each head can specialise in a different type of relationship: one head might learn syntax (subject–verb agreement), another might learn coreference (pronoun → noun). Concatenating gives the model access to all these relationship types simultaneously.

**Problem with this naive approach:** each head runs its own separate Q/K/V projection — `num_heads` separate forward passes. This is computationally inefficient.

---

### Step 12 — Efficient MultiHeadAttention (Single Projection)

The efficient implementation uses a **single large projection** and then *splits* it across heads — all heads compute attention in parallel via tensor reshaping:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out    = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads   # dimension per head

        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)   # mix heads after concat
        self.dropout  = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Single large projection → (B, T, d_out)
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        # Split d_out across heads → (B, T, H, head_dim) → (B, H, T, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores per head → (B, H, T, T)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted values → (B, H, T, head_dim) → (B, T, H, head_dim) → (B, T, d_out)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)

        # Output projection to mix cross-head information
        return self.out_proj(context_vec)
```

**Tensor shape walkthrough for GPT-2 Small (B=2, T=6, d_out=768, H=12):**

```
Input:          (2, 6, 3)
After W_query:  (2, 6, 768)
After .view():  (2, 6, 12, 64)    ← split 768 into 12 heads × 64 dims
After .T(1,2):  (2, 12, 6, 64)   ← heads move to dim 1
Attn scores:    (2, 12, 6, 6)    ← each head: every token vs every token
After values:   (2, 12, 6, 64)
After .T(1,2):  (2, 6, 12, 64)   ← heads back to dim 2
After .view():  (2, 6, 768)       ← concatenate heads
After out_proj: (2, 6, 768)       ← final output
```

The `out_proj` (output projection) is a final linear layer that mixes information across the concatenated head outputs — allowing the model to learn which combinations of head outputs matter.

---

## 📒 Chapter 4 — GPT Model Architecture

**Notebook:** `GPT_implmentation.ipynb`

---

### GPT-2 Configuration

All hyperparameters are stored in a single config dictionary, mirroring how real research code is structured:

```python
GPT_CONFIG_124M = {
    "vocab_size":       50257,   # BPE vocabulary (tiktoken gpt2)
    "context_length":   1024,    # max tokens the model can attend to at once
    "emb_dim":          768,     # embedding dimension (token + positional)
    "n_heads":          12,      # number of attention heads
    "n_layers":         12,      # number of stacked transformer blocks
    "drop_rate":        0.1,     # dropout probability
    "qkv_bias":         False    # no bias in Q/K/V projections (GPT-2 default)
}
```

**Critical relationship:** `head_dim = emb_dim / n_heads = 768 / 12 = 64`. This must divide evenly. The value 64 is common across many transformer architectures because it provides good numerical stability for scaled dot-product attention.

**Attention complexity** grows as O(n²) with context length — doubling the context quadruples the attention computation. This is why long-context models (GPT-4, Claude) require significant architectural innovations.

---

### Step 1 — DummyGPTModel (Architecture Skeleton)

Before building the real components, we verify the **tensor shapes** flow correctly through a skeleton model with placeholder blocks:

```python
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb    = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb    = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb   = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm_layer = DummyLayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_emb + pos_emb)
        x = self.trf_blocks(x)
        x = self.norm_layer(x)
        return self.out_head(x)   # → (batch, seq_len, vocab_size)
```

**Output shape:** `[2, 4, 50257]` — for each of the 2 sequences, at each of the 4 positions, a score for every token in the vocabulary. These are called **logits** (raw unnormalised scores before softmax).

---

### Step 2 — Layer Normalization

**The problem:** as data flows through many layers, the distribution of activations can shift wildly. Large values cause exploding gradients; tiny values cause vanishing gradients. Training becomes unstable.

**Layer Norm** fixes this by normalising each token's embedding vector to have mean ≈ 0 and variance ≈ 1, independently for each token:

```
LayerNorm(x) = γ · (x − μ) / √(σ² + ε) + β
```

| Symbol | Meaning |
|--------|---------|
| `μ` | Mean across the embedding dimension for this token |
| `σ²` | Variance across the embedding dimension for this token |
| `ε` | Small constant (1e-5) for numerical stability — prevents division by zero |
| `γ` (scale) | Learnable parameter, initialised to 1 — lets model amplify/suppress dims |
| `β` (shift) | Learnable parameter, initialised to 0 — lets model offset normalised values |

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps   = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))   # γ
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # β

    def forward(self, x):
        mean   = x.mean(dim=-1, keepdim=True)
        var    = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

**Key difference from Batch Norm:** Layer Norm normalises across the *feature/embedding dimension* for each individual token independently. Batch Norm normalises across the *batch dimension* for each feature. Layer Norm is preferred in NLP because sequence lengths vary across examples in a batch.

**Why learnable γ and β?** Pure normalisation forces every distribution to mean=0, var=1, which is unnecessarily restrictive. The learnable scale and shift restore the model's freedom to represent any distribution while still benefiting from the stability of normalisation.

---

### Step 3 — GELU Activation

Transformers replaced ReLU with **GELU** (Gaussian Error Linear Unit):

```
GELU(x) = x · Φ(x)
```

where `Φ(x)` is the CDF of the standard normal distribution. In practice, the fast approximation is used:

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)
        ))
```

**ReLU vs GELU:**

| Property | ReLU | GELU |
|----------|------|------|
| Negative inputs | Hard zero (dead neurons) | Smooth suppression |
| Gradient at x < 0 | Exactly 0 (gradient vanishes) | Small but non-zero |
| Shape | Sharp corner at 0 | Smooth, differentiable everywhere |
| Training stability | Can cause dead neurons | Better gradient flow in deep nets |

GELU is used in GPT-2, BERT, PaLM, and LLaMA. The smooth gradient for small negative values helps information flow during backpropagation in very deep networks.

---

### Step 4 — Feed-Forward Network

Every transformer block contains a 2-layer feed-forward network (FFN) applied **independently to each token position**:

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # expand:  768 → 3072
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])   # project: 3072 → 768
        )

    def forward(self, x):
        return self.layers(x)   # shape preserved: (B, T, emb_dim)
```

**Why expand 4×?** The intermediate dimension `4 × emb_dim` (3072 for GPT-2 Small) gives the network room to learn richer nonlinear feature transformations. Research has found this 4× expansion factor to be a sweet spot — larger doesn't reliably help, smaller hurts quality.

**Role of FFN vs attention:** Self-attention *mixes information across tokens* (which tokens are relevant to each other). The FFN *transforms the representation of each token independently* (what to do with the mixed information). Together they form the full token-processing pipeline.

---

### Step 5 — Residual (Skip) Connections

**The vanishing gradient problem:** in very deep networks, gradients become exponentially smaller as they backpropagate through many layers. By layer 12 or 24, the gradient reaching early layers is essentially zero — those layers don't learn.

**Residual connections** (skip connections) solve this by providing a gradient highway:

```python
# Without residual — gradient must pass through every layer
x = layer(x)

# With residual — gradient can bypass the layer entirely
shortcut = x
x = layer(x)
x = x + shortcut   # ← residual connection
```

**Gradient flow comparison (5 layers, measured by mean absolute gradient):**

```
Without skip connections:
  Layer 0: 0.000688   ← gradient almost zero — not learning
  Layer 4: 0.024815

With skip connections:
  Layer 0: 0.555774   ← strong gradient — learning effectively
  Layer 4: 3.140750
```

Without residuals, the earliest layers get ~800× smaller gradients than the final layer. With residuals, gradients remain large throughout the network. This is what makes training 12–96 layer GPT models practical.

---

### Step 6 — Transformer Block

The transformer block combines all components: Layer Norm, Multi-Head Attention, Feed-Forward, Residuals, and Dropout:

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn    = MultiHeadAttention(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
                                          context_length=cfg["context_length"],
                                          dropout=cfg["drop_rate"], num_heads=cfg["n_heads"],
                                          qkv_bias=cfg["qkv_bias"])
        self.ffn     = FeedForward(cfg)
        self.norm1   = LayerNorm(cfg["emb_dim"])
        self.norm2   = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention sub-layer with Pre-LayerNorm and residual
        shortcut = x
        x = self.norm1(x)      # normalise BEFORE attention (Pre-LN)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + shortcut       # residual connection

        # FFN sub-layer with Pre-LayerNorm and residual
        shortcut = x
        x = self.norm2(x)      # normalise BEFORE FFN (Pre-LN)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut       # residual connection

        return x   # shape unchanged: (B, T, emb_dim)
```

**Pre-LayerNorm vs Post-LayerNorm:** The original "Attention is All You Need" paper applied LayerNorm *after* the attention/FFN (Post-LN). GPT-2 switched to applying it *before* (Pre-LN). Pre-LN trains more stably in very deep networks because the residual path is always "clean" — gradients flow through the addition without passing through a normalisation layer.

```
Input
  │
  ├──→ LayerNorm → Attention → Dropout ──→ Add ← ────────────── shortcut
  │                                          │
  ├──→ LayerNorm → FFN → Dropout ────────→ Add ← ────────────── shortcut
  │
Output (same shape as Input)
```

---

### Step 7 — Full GPTModel

All components assembled into the complete model:

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb          = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb          = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb         = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm       = LayerNorm(cfg["emb_dim"])
        self.out_head         = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        x = self.drop_emb(
            self.tok_emb(in_idx) + self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        )
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)   # (B, T, vocab_size)
```

**Complete forward pass for batch of 2 sequences of length 4:**

```
in_idx              (2, 4)            ← integer token IDs
tok_emb             (2, 4, 768)       ← token embedding lookup
pos_emb             (4, 768)          ← positional embedding (broadcast over batch)
tok_emb + pos_emb   (2, 4, 768)       ← combined input representation
× 12 TransformerBlocks               ← each preserves (2, 4, 768)
final_norm          (2, 4, 768)
out_head            (2, 4, 50257)     ← logit for every vocab token at every position
```

---

### Step 8 — Parameter Count & Weight Tying

```python
total_params = sum(p.numel() for p in model.parameters())
# → 163,009,536  (163M parameters)
```

**Why more than 124M?** The 124M figure uses **weight tying** — the token embedding matrix `(50257 × 768)` and the output projection `(50257 × 768)` share the same weights. This is a standard GPT-2 optimisation that saves ~38M parameters:

```python
# Without weight tying: 163,009,536 params
# Minus output head:
params_without_output_head = total_params - sum(p.numel() for p in model.out_head.parameters())
# → 124,412,160  ≈ 124M  ← the advertised number
```

**Why share weights?** The token embedding maps token IDs to semantic vectors; the output head maps semantic vectors back to token scores. They're doing the inverse operation on the same semantic space, so sharing makes sense — and it acts as a regulariser.

**Parameter breakdown for GPT-2 Small:**

| Component | Parameters |
|-----------|-----------|
| Token embedding | 38,597,376 |
| Positional embedding | 786,432 |
| Feed-forward networks (×12) | 56,669,184 |
| Attention layers (×12) | 28,320,768 |
| Other (norms, biases) | ~1M |

---

### Step 9 — GPT-2 Model Variants

The same `GPTModel` class handles all GPT-2 sizes by changing the config:

| Model | `emb_dim` | `n_heads` | `n_layers` | Parameters |
|-------|-----------|-----------|------------|------------|
| Small | 768 | 12 | 12 | ~124M |
| Medium | 1024 | 16 | 24 | ~355M |
| Large | 1280 | 20 | 36 | ~774M |
| XL | 1600 | 25 | 48 | ~1.5B |

All produce output shape `(batch, seq_len, 50257)` regardless of size — they differ only in internal capacity.

---

### Step 10 — Autoregressive Text Generation

**Greedy decoding** — generating one token at a time:

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # Crop input to context window (model can't attend beyond context_size)
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)   # (B, T, vocab_size)

        # Take the logits at the LAST position only (that's the next token prediction)
        logits = logits[:, -1, :]   # (B, vocab_size)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Greedy: pick the highest-probability token
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)   # (B, 1)

        # Append to growing sequence
        idx = torch.cat((idx, idx_next), dim=1)   # (B, T+1)

    return idx
```

**Running inference:**

```python
model.eval()   # disables dropout for deterministic output

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)            # [15496, 11, 314, 716]
encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # (1, 4) — add batch dim

out = generate_text_simple(model, encoded_tensor, max_new_tokens=6,
                            context_size=GPT_CONFIG_124M["context_length"])

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# → "Hello, I am Featureiman Byeswick Exit In"
```

The output is gibberish because the model is **randomly initialised** — it hasn't been trained yet. The architecture is correct; it just hasn't learned any language yet. Pre-training on billions of tokens is what gives the model coherent outputs.

**Why greedy decoding?** It's the simplest strategy — always pick the most probable next token. Real inference uses more sophisticated strategies like **temperature scaling**, **top-k sampling**, and **nucleus (top-p) sampling** to produce more diverse, natural-sounding text. These will be covered in later chapters.

---

## 📒 Chapter 5 — Pretraining & Evaluation

**Notebook:** `GPT_Model_Training_and_Evaluation.ipynb`

---

### Step 1 — Cross-Entropy Loss from First Principles

The training objective for language models is **cross-entropy loss** — a measure of how surprised the model is by the true next tokens. We derive it step by step before using PyTorch's built-in version:

```
1. Forward pass → logits   shape: (batch, seq_len, vocab_size)
2. softmax(logits) → probabilities
3. Index into probabilities at the target token positions → target probs
4. torch.log(target_probs) → log probabilities (negative values)
5. Average → avg_log_prob (e.g. -10.57)
6. Multiply by -1 → loss = 10.57
```

**Why cross-entropy?** A perfectly confident correct prediction gives log(1.0) = 0 loss. Random guessing over 50,257 tokens gives log(1/50257) ≈ 10.8 — exactly what we observe at initialisation.

**PyTorch's shortcut** does all of this in one call:

```python
loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1),  # (batch*seq_len, vocab_size)
    targets.flatten()      # (batch*seq_len,)
)
```

`flatten(0, 1)` merges the batch and sequence dimensions so the function sees a flat list of independent token predictions.

---

### Step 2 — Perplexity

**Perplexity** is cross-entropy loss exponentiated back to probability scale:

```python
perplexity = torch.exp(loss)
# At init: exp(10.57) ≈ 38,783
```

Perplexity is interpretable: it represents the **effective vocabulary size the model is confused by**. A perplexity of 38,783 means the model is as uncertain as if it were choosing uniformly over ~38k tokens. A well-trained model on English text might reach perplexity ≈ 20–50.

---

### Step 3 — Data Split and DataLoaders

The corpus is split into 90% training / 10% validation **at the character level** (before tokenisation), so there is no information leakage from validation tokens appearing in training context windows:

```python
train_ratio = 0.90
split_idx   = int(train_ratio * len(text_data))
train_data  = text_data[:split_idx]
val_data    = text_data[split_idx:]
```

DataLoaders use `stride = context_length` (non-overlapping windows) and `shuffle=True` for training, `shuffle=False` for validation to ensure reproducible validation metrics.

---

### Step 4 — Loss Calculation Functions

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss   = torch.nn.functional.cross_entropy(
                 logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Average loss over num_batches batches (all if None)."""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    num_batches = num_batches or len(data_loader)
    num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
        else:
            break
    return total_loss / num_batches
```

`num_batches` is a **speed knob**: during training we only evaluate on a few batches (`eval_iter=1`) to stay fast, then do a full evaluation at the end.

---

### Step 5 — The Training Loop

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                        num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()  # batch_size × seq_len
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(...)
                # Log losses

        generate_and_print_sample(...)   # qualitative check after each epoch
```

**Observed training behaviour over 10 epochs on The Verdict:**

| Epoch | Train Loss | Val Loss | Sample output |
|-------|-----------|---------|---------------|
| 1 | 8.16 | 8.34 | `,,,,,,,.` |
| 3 | 5.37 | 6.38 | `and to the to the of the picture.` |
| 7 | 2.03 | 6.15 | `"Yes--quite insensible to the irony..."` |
| 10 | 0.56 | 6.37 | Exact passage from training text |

The widening gap between train and val loss is the **overfitting signature** — the model is memorising the tiny training corpus (~5k tokens) rather than generalising. Real pretraining uses billions of tokens where overfitting is not a concern.

**Optimizer:** `AdamW` with `lr=0.0004` and `weight_decay=0.1`. AdamW adds L2 regularisation directly to the weight update rule (rather than the gradient), which is better for transformer training.

---

### Step 6 — Saving and Loading Models

**Model weights only** (for inference):

```python
torch.save(model.state_dict(), "model.pth")

model_loaded = GPTModel(GPT_CONFIG_124M)
model_loaded.load_state_dict(torch.load("model.pth"))
```

**Model + optimizer state** (to resume training):

```python
torch.save({
    "model_state_dict":     model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "model_and_optimizer.pth")

checkpoint = torch.load("model_and_optimizer.pth")
model_loaded.load_state_dict(checkpoint["model_state_dict"])
optimizer_loaded.load_state_dict(checkpoint["optimizer_state_dict"])
```

Saving the **optimizer state** is critical for resuming training: optimisers like AdamW maintain per-parameter momentum estimates (`m`) and velocity estimates (`v`) that take thousands of steps to warm up. Starting from a fresh optimiser state would cause a large loss spike at resume.

---

### Step 7 — Loading Pretrained GPT-2 Weights

Training a GPT-2 class model from scratch on The Verdict is a toy exercise. The real power comes from loading **OpenAI's pretrained GPT-2 weights**:

```python
# download_and_load_gpt2() fetches official TF checkpoint and converts to numpy
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# params keys: wte (token emb), wpe (pos emb), blocks (transformer blocks),
#              g (final norm scale γ), b (final norm shift β)
```

The `assign()` helper verifies shapes before writing:

```python
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.from_numpy(right).to(device))
```

Weight loading requires careful attention to **transpositions**: OpenAI's TF checkpoint stores projection weights as `(d_in, d_out)` but PyTorch's `nn.Linear` stores them as `(d_out, d_in)`, so all weight matrices are transposed during loading.

**Impact:** The pretrained model achieves train loss ≈ 3.75 and val loss ≈ 3.56 on The Verdict **without any fine-tuning** — far better than our from-scratch model after 10 epochs.

Two config differences from our default must be set to match OpenAI's pretrained checkpoint:
- `context_length: 1024` (GPT-2's actual context window, not our 256 shortcut)
- `qkv_bias: True` (OpenAI used bias vectors in Q/K/V projections)

---

### Step 8 — Decoding Strategies

The `generate()` function is upgraded from simple greedy decoding to support **temperature scaling** and **top-k filtering**:

```python
def generate(model, idx, max_new_tokens, context_size,
             temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Top-k: set all logits below the k-th largest to -inf
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_top = top_logits[:, -1]
            logits = torch.where(logits < min_top, torch.tensor(-inf), logits)

        # Temperature: divide before softmax to control peakiness
        if temperature > 0.0:
            logits  = logits / temperature
            probs   = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)   # stochastic
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # greedy

        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

**Temperature effects (illustrated):**

| Temperature | Effect | When to use |
|------------|--------|-------------|
| `0.0` | Greedy — always picks highest-probability token | Deterministic, factual tasks |
| `0.1` | Very confident — distribution sharply peaked | When accuracy matters most |
| `1.0` | Unchanged softmax distribution | Balanced creativity/coherence |
| `1.4` | Flat — low-probability tokens more likely | Creative/diverse generation |
| `5.0` | Near-uniform — almost random | Rarely useful |

**Top-k filtering** acts as a safety net: even with high temperature, it prevents the model from picking tokens with near-zero probability (typos, gibberish). Setting `top_k=3` restricts sampling to only the 3 most probable tokens.

**Combining them:** `temperature=1.4, top_k=3` — high temperature for diversity but top-k guardrail for coherence — often produces the best qualitative outputs.

---

## 📒 Chapter 6 — Fine-Tuning for Classification

**Notebook:** `finetuning_for_classification.ipynb`

---

### Step 1 — Dataset Preparation

The **UCI SMS Spam Collection** dataset contains 5,572 SMS messages labelled `ham` (legitimate) or `spam`. It is heavily imbalanced: 4,825 ham vs 747 spam.

**Class balancing:** We randomly downsample the majority class so both classes have exactly 747 examples. Training on imbalanced data causes models to default to predicting the majority class, achieving high accuracy while being useless.

```python
num_spam = df[df["Label"] == "spam"].shape[0]          # 747
ham_sampled = df[df["Label"] == "ham"].sample(num_spam) # 747
balanced_df = pd.concat([ham_sampled, df[df["Label"] == "spam"]])
```

Labels are encoded as integers: `ham → 0`, `spam → 1`.

**Split:** 70% train / 10% validation / 20% test.

---

### Step 2 — SpamDataset and DataLoaders

```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()  # from training set
        else:
            self.max_length = max_length
            self.encoded_texts = [e[:self.max_length] for e in self.encoded_texts]  # truncate

        # Pad all sequences to max_length
        self.encoded_texts = [
            e + [pad_token_id] * (self.max_length - len(e))
            for e in self.encoded_texts
        ]
```

**Key design decisions:**
- `max_length` is determined from the **training set** and applied to val/test sets. This ensures test data can't "leak" its length distribution.
- Padding uses `<|endoftext|>` (token ID 50256), which the model already knows — this is fine since padded positions won't determine the output anyway.
- Training set max token length: **120** — well within GPT-2's 1024 context limit.

---

### Step 3 — Modifying the GPT Head for Classification

The pretrained GPT-2 has an output head `(768 → 50,257)` mapping to vocabulary logits. For binary classification we swap it for a `(768 → 2)` head:

```python
# Step 1: Freeze ALL pretrained parameters
for param in model.parameters():
    param.requires_grad = False

# Step 2: Replace output head (automatically gets requires_grad=True)
model.out_head = torch.nn.Linear(in_features=768, out_features=2)

# Step 3: Unfreeze the last transformer block + final norm
for param in model.transformer_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
```

**Why unfreeze the last block?** Research shows that fine-tuning only the new head gives worse results than also allowing the last transformer block to adapt. The early layers capture universal linguistic features; the last block learns task-specific representations. Unfreezing more layers risks catastrophic forgetting of pretrained knowledge.

**Which token's output to use?** We use the **last token's** output vector `logits[:, -1, :]`:

```
Causal attention → each token only sees itself + preceding tokens
Last token → has attended to the entire sequence
∴ Last token's representation aggregates the most information about the full input
```

This is the standard approach for decoder-only transformers used for classification.

---

### Step 4 — Training and Evaluation

```python
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct, total = 0, 0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if num_batches and i >= num_batches: break
        with torch.no_grad():
            logits = model(input_batch.to(device))[:, -1, :]  # last token
        predicted = torch.argmax(logits, dim=-1)
        correct += (predicted == target_batch.to(device)).sum().item()
        total   += len(target_batch)
    return correct / total
```

**Training results (5 epochs, AdamW lr=5e-5):**

| Metric | Score |
|--------|-------|
| Training accuracy | 97.21% |
| Validation accuracy | 97.99% |
| **Test accuracy** | **96.00%** |

The model achieves strong generalisation with minimal fine-tuning — testament to the quality of pretrained representations.

---

### Step 5 — Inference

```python
def classify_review(text, model, tokenizer, device, max_length, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)[:max_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    tensor_in = torch.tensor(input_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_in)[:, -1, :]
    label = torch.argmax(logits, dim=-1).item()
    return "spam" if label == 1 else "not spam"
```

```python
classify_review("You are a winner you have been specially selected...", ...)
# → "spam"

classify_review("Hey, just wanted to check if we're still on for dinner...", ...)
# → "not spam"
```

---

## 📒 Chapter 7 — Instruction Fine-Tuning

**Notebook:** `finetuing_for_instructions.ipynb`

---

### Step 1 — The Alpaca Format

**Instruction fine-tuning** teaches a model to follow human instructions (e.g. "Translate this sentence", "Summarise this document") rather than just continue text. The **Alpaca format** is the most widely used template:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Identify the correct spelling of the following word.

### Input:
Ocassion

### Response:
The correct spelling is 'Occasion.'
```

The `### Input:` section is **optional** — for instructions with no additional context (e.g. "What is an antonym of 'complicated'?"), it is omitted.

```python
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text
```

**Dataset:** 1,100 instruction-response pairs. Split: 85% train (935), 5% val (55), 10% test (110).

---

### Step 2 — InstructionDataset and Custom Collate Function

Each example is the **full formatted text** (instruction + response) encoded as token IDs. Because examples vary in length, a custom collate function is needed to batch them together.

**The challenge:** Standard batching requires all sequences to be the same length. For instruction tuning, sequences range from ~50 to 200+ tokens.

**Three-stage collate function evolution:**

**Draft 1** — Pad inputs to batch max length:

```python
# Pads all sequences to the longest in the batch
padded = item + [pad_token_id] * (batch_max_length - len(item) - 1)
inputs = torch.tensor(padded[:-1])
```

**Draft 2** — Create input/target pairs (shifted by 1):

```python
inputs  = padded[:-1]   # token 0 to N-1
targets = padded[1:]    # token 1 to N  (next-token prediction)
```

**Final version** — Mask padding in targets with `-100`:

```python
# -100 is the ignore_index for cross_entropy; padding positions don't contribute to loss
mask = targets == pad_token_id
indices = torch.nonzero(mask).squeeze()
if indices.numel() > 1:
    targets[indices[1:]] = ignore_index  # keep first pad, mask the rest
```

---

### Step 3 — The ignore_index Trick

**Why mask padding?** If the model is trained to predict `<|endoftext|>` at every padded position, it learns to generate end-of-sequence tokens at arbitrary points. Masking them with `-100` tells `cross_entropy` to ignore those positions entirely.

**Proof of concept:**

```python
# Logits for 3 tokens, 2 classes
logits_1  = torch.tensor([[-1.0, 1.0], [-0.5, 1.5]])
targets_1 = torch.tensor([0, 1])
loss_1 = cross_entropy(logits_1, targets_1)  # 1.127

# Adding a third token that should be ignored
logits_2  = torch.tensor([[-1.0, 1.0], [-0.5, 1.5], [-0.5, 1.5]])
targets_3 = torch.tensor([0, 1, -100])  # -100 = ignored
loss_3 = cross_entropy(logits_2, targets_3)

assert loss_1 == loss_3  # True ✓ — the third position didn't affect loss
```

This also explains why we only mask from the **second** padding token onward: the first pad position still carries signal (it's the model learning when to end generation).

---

### Step 4 — Fine-Tuning GPT-2 Medium

We use **GPT-2 Medium (355M parameters)** — larger capacity is better for following diverse instructions:

```python
BASE_CONFIG = {
    "vocab_size":    50257,
    "context_length": 1024,
    "drop_rate":      0.0,   # no dropout for fine-tuning
    "qkv_bias":       True,
    "emb_dim":        1024,
    "n_layers":       24,
    "n_heads":        16,
}
```

**Before fine-tuning (pretrained GPT-2 Medium on instruction input):**

```
Input: "Convert the active sentence to passive: 'The chef cooks the meal every day.'"
Output: "A utensil is useful when defining each reply — but written data is simply useless..."
```

The model produces fluent text but completely ignores the instruction — it's a completion model, not an instruction follower.

**After 2 epochs of instruction fine-tuning (~12 minutes on GPU):**

```
Input: "Convert the active sentence to passive: 'The chef cooks the meal every day.'"
Output: "The meal is cooked every day by the chef."
```

**Training configuration:**
- Optimizer: AdamW, `lr=5e-5`, `weight_decay=0.1`
- Batch size: 8
- Eval freq: every 5 steps
- Final train loss: ~0.29, val loss: ~0.66

The gap between train and val loss reflects the small dataset (935 training examples); real instruction fine-tuning uses 50k–500k examples.

---

### Step 5 — LLM-as-Judge Evaluation

Automated metrics (BLEU, ROUGE) struggle with open-ended instruction responses. A better evaluation strategy is **LLM-as-Judge**: use a powerful model (here, Llama 3 via Ollama) to score the fine-tuned model's responses:

```python
def generate_model_scores(json_data, json_key, model="llama3"):
    for entry in tqdm(json_data):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}` "
            f"on a scale from 0 to 100. Respond with the integer number only."
        )
        score = query_model(prompt, model)
        scores.append(int(score.strip()))
    return scores
```

**Results on 110 test examples:**
- Average LLM-judge score: **46.60 / 100**
- This baseline reflects the small training set; larger datasets consistently push scores above 80.

**Why use an LLM as judge?** Human evaluation is expensive and slow. LLM judges correlate well with human judgements for factual and instruction-following tasks, and provide a scalable automated signal for iterating on fine-tuning data and training recipes.

---

## 📒 Appendix A — Advanced Training Loop Techniques

**Notebook:** `Adding_Bells_and_Whistles_to_the_Training_Loop.ipynb`

The basic training loop from Ch. 5 (constant LR, no gradient stabilisation) is fine for toy experiments but diverges or converges poorly when scaled up. This appendix adds three industrial-strength techniques that are standard in modern LLM training runs.

---

### Step 1 — Linear LR Warmup

**Why warm up?** At the start of training, the model is randomly initialised — weight gradients are large and chaotic. Jumping straight to a high learning rate causes early instability or divergence. Warming up gradually grows the LR from a small initial value to its peak over a fixed number of steps, giving the optimiser time to "orient" itself before taking large steps.

```python
n_epochs    = 15
initial_lr  = 0.0001   # start very small
peak_lr     = 0.01     # ramp up to this
warmup_steps = 20      # over this many steps

lr_increment = (peak_lr - initial_lr) / warmup_steps  # per-step increment

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
global_step = -1

for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        global_step += 1

        if global_step < warmup_steps:
            # linearly interpolate from initial_lr to peak_lr
            lr = initial_lr + global_step * lr_increment
        else:
            lr = peak_lr          # flat thereafter (replaced by cosine in Step 2)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
```

**LR schedule (warmup only):**
```
Step 0    →  lr = 0.0001   (initial)
Step 10   →  lr = 0.0055   (midpoint)
Step 19   →  lr = 0.01     (peak)
Step 20+  →  lr = 0.01     (flat)
```

The warmup curve is **linear** — each step adds a fixed `lr_increment`. The resulting plot shows a clean ramp followed by a plateau.

---

### Step 2 — Cosine Annealing Decay

**Why decay?** Keeping the LR at its peak after warmup tends to overshoot local minima in later epochs. A **cosine decay** smoothly reduces the LR from `peak_lr` to `min_lr` following a half-cosine curve — fast decay at first, then slowing down as it approaches the minimum. This is empirically superior to step decay or linear decay for language models.

**Formula:**

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

Where:
- `t` = steps completed since end of warmup
- `T` = total training steps minus warmup steps
- `η_max` = `peak_lr`
- `η_min` = `min_lr`

```python
import math

min_lr = 0.1 * initial_lr   # floor: don't decay below 10% of initial

for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        global_step += 1

        if global_step < warmup_steps:
            lr = initial_lr + global_step * lr_increment
        else:
            # progress ∈ [0, 1] over post-warmup training
            progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)
            # cosine decay
            lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
```

**LR schedule (warmup + cosine decay):**
```
Step 0   → 0.0001  (warmup start)
Step 20  → 0.01    (peak, warmup complete)
Step 110 → ~0.005  (cosine mid-point)
Step 200 → 0.00001 (min_lr floor)
```

The resulting schedule has three phases: **linear ramp → cosine decay → floor**. This is precisely the schedule used in the Chinchilla and LLaMA training papers.

---

### Step 3 — Gradient Clipping

**Why clip?** Deep networks occasionally produce very large gradients ("gradient spikes") due to specific unlucky batches. If these spikes update the weights without restraint, they can shatter the model's learned representations — a phenomenon called **gradient explosion**. Gradient clipping caps the global gradient norm to a maximum value (`max_norm=1.0`) before the optimiser step, harmlessly scaling down any spike.

```python
# After loss.backward(), before optimizer.step():
loss = calc_loss_batch(input_batch, target_batch, model, device)
loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Computes global norm of all gradients; if > 1.0, scales every gradient
# by 1.0 / global_norm so the norm exactly equals 1.0.

optimizer.step()
```

**Effect on this run:**
```python
# Before clipping:
find_highest_gradient(model)  # → tensor(0.0344)

# After clipping:
find_highest_gradient(model)  # → tensor(0.0155)
```

In practice, clipping is typically applied **only after warmup** (gradients are expected to be large at step 0). During warmup the LR is tiny so even large gradients result in small updates.

| Technique | When to apply | Effect |
|-----------|--------------|--------|
| LR warmup | First N steps | Prevents early divergence |
| Cosine decay | After warmup | Smooth convergence to minimum |
| Gradient clipping | After warmup (or always) | Prevents gradient explosion spikes |

---

### Step 4 — Full `train_model` Function

All three techniques are unified in a single `train_model` function that replaces the simple loop from Ch. 5:

```python
def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context,
                warmup_steps=10, initial_lr=3e-5, min_lr=1e-6, peak_lr=5e-4):

    tokenizer = tiktoken.get_encoding("gpt2")
    train_losses, val_losses, num_tokens_seen, track_lrs = [], [], [], []

    total_training_steps = len(train_loader) * n_epochs
    lr_increment = (peak_lr - initial_lr) / warmup_steps
    global_step, tokens_seen = -1, 0

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # 1. Learning rate schedule
            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_increment
            else:
                progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)
                lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(optimizer.param_groups[0]["lr"])

            # 2. Forward + backward
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # 3. Gradient clipping (only after warmup)
            if global_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            tokens_seen += input_batch.numel()

            # 4. Periodic evaluation + sample generation
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                num_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, num_tokens_seen, track_lrs
```

**Results (15 epochs, `the-verdict.txt`):**

```
Ep 1 (Iter 000000): Train loss 10.971, Val loss 10.938
Ep 5 (Iter 000040): Train loss 4.590, Val loss 6.434
Ep 10 (Iter 000085): Train loss 1.146, Val loss 6.180
Ep 15 (Iter 000130): Train loss 0.467, Val loss 6.327
```

Sample generation after epoch 10:
```
Every effort moves you?  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"
He laughed again, and threw back his head to look up at the sketch of the donkey.
```

The model achieves near-perfect training fit (train loss ~0.47) while the val loss stabilises around 6.3, showing the small corpus is memorised rather than generalised — expected at this scale. The cosine LR curve is clearly visible in the tracked `track_lrs` values across all training steps.

---

---

## 📒 Bonus — Building a BPE Tokenizer from Scratch

**Notebook:** `building_a_tokenizer_from_scratch.ipynb`

This standalone notebook builds a fully working Byte-Pair Encoding tokenizer from first principles — starting from raw Unicode, climbing through the BPE algorithm step by step, then packaging the logic into two progressively more capable classes: `BasicTokenizer` (clean BPE) and `RegexTokenizer` (GPT-4 style with regex pre-tokenisation). A SentencePiece comparison and a vocab-size tradeoff analysis round off the notebook.

> Inspired by Andrej Karpathy's "Let's build the GPT Tokenizer" lecture and the [minBPE](https://github.com/karpathy/minbpe) repository.

---

### Why Tokenization Matters

Before writing a single line of code the notebook catalogues every major class of LLM failure that traces back to tokenization:

| Failure Mode | Root Cause |
|---|---|
| Can't spell words | Tokens rarely align with individual letters |
| Can't reverse a string | Tokens don't have a predictable character structure |
| Worse at non-English | Non-ASCII scripts get fragmented into many more bytes |
| Bad at arithmetic | Digits are merged unpredictably across token boundaries |
| GPT-2 struggled with Python | Indentation whitespace consumed separate tokens |
| Abrupt halt on `<\|endoftext\|>` | Special token triggers end-of-sequence logic |
| "SolidGoldMagikarp" bug | Token existed in vocab but never appeared in training data → undefined embedding |
| YAML cheaper than JSON | YAML formatting uses fewer tokens per unit of information |

This list reframes tokenization from a boring preprocessing step into a critical design decision with deep downstream consequences.

---

### Step 1 — Unicode and Bytes Fundamentals

Every string in Python is a sequence of Unicode code points. The first move is to understand how those code points map to bytes:

```python
ord('h')   # → 104 (Unicode code point for 'h')

[ord(x) for x in "您好 🙏"]
# → [24744, 22909, 32, 128591]  (code points, not bytes)

list("您好 🙏".encode("utf-8"))
# → [230, 130, 168, 229, 165, 189, 32, 240, 159, 153, 143]
# each non-ASCII code point expands to 2–4 bytes
```

**Why UTF-8?** UTF-8 encodes every Unicode code point as 1–4 bytes. ASCII characters (0–127) encode to a single byte — identical to their ASCII values. Non-ASCII characters expand to multi-byte sequences based on their code point range:

| Byte pattern | Code point range | Bytes used |
|---|---|---|
| `0xxxxxxx` | U+0000–U+007F (ASCII) | 1 |
| `110xxxxx 10yyyyyy` | U+0080–U+07FF | 2 |
| `1110xxxx 10yyyyyy 10zzzzzz` | U+0800–U+FFFF | 3 |
| `11110xxx 10yyyyyy …` | U+10000–U+10FFFF | 4 |

**Why this matters for BPE:** BPE operates on bytes (integers 0–255), not on code points or characters. This means the initial vocabulary is exactly 256 tokens — one per possible byte value. Every text, in every language, can be represented with these 256 base tokens. The BPE merges then learn efficient multi-byte abbreviations on top.

---

### Step 2 — BPE Core Algorithm (Functional)

Two helper functions form the algorithmic core. Everything else in the notebook builds on them.

**`get_stats(ids)` — count consecutive pair frequencies:**

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):          # slide a window of size 2
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# Example:
get_stats([104, 101, 108, 108, 111])
# {(104,101): 1, (101,108): 1, (108,108): 1, (108,111): 1}
# The pair (108, 108) = "ll" has frequency 1 here; on a large corpus it would be much higher
```

**`merge_pair(ids, pair, new_idx)` — replace every occurrence of a pair with a new token index:**

```python
def merge_pair(ids, pair, new_idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(new_idx)   # replace the pair
            i += 2                    # skip both elements
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# Example:
merge_pair([5, 6, 6, 7, 9, 1], (6, 6), 99)
# → [5, 99, 7, 9, 1]   (the "66" pair replaced by 99)
```

**Why these two functions are enough:** BPE training is just repeated application of `get_stats` → pick best pair → `merge_pair`. No neural network, no gradient — pure frequency statistics.

---

### Step 3 — Training Loop and Compression Ratio

The training loop applies the two functions iteratively to learn a fixed number of merges:

```python
vocab_size       = 276    # target vocabulary size
number_of_merges = vocab_size - 256   # = 20 (starting from 256 base bytes)

tokens = text.encode("utf-8")
ids    = list(map(int, tokens))       # working copy as list of ints

merges = {}   # dict: (int, int) → int  (pair → new token id)

for i in range(number_of_merges):
    stats    = get_stats(ids)
    top_pair = max(stats, key=stats.get)     # most frequent pair
    new_idx  = 256 + i                       # assign next available id
    ids      = merge_pair(ids, top_pair, new_idx)
    merges[top_pair] = new_idx
    print(f"Merging {top_pair} → {new_idx}, count: {stats[top_pair]}")
```

**Token compression:**
```
tokens length before merges: 24,597
length after merges:          22,162
compression ratio:            1.11X
```

Each merge reduces total token count. With more merges (larger `vocab_size`), compression grows substantially — GPT-2's 50,000-token vocabulary achieves roughly 3–4× compression on English text.

**Why start from 256?** All 256 single-byte values are pre-loaded as base tokens. Merges only ever *add* tokens (indices 256, 257, 258, …) — they never remove the originals. This guarantees the tokenizer can always represent any byte sequence, even unseen text.

---

### Step 4 — Decode and Encode

**Building the vocabulary from merges:**

```python
# Map every token id → its byte sequence
vocab = {idx: bytes([idx]) for idx in range(256)}   # base: each byte maps to itself

for (idx1, idx2), new_idx in merges.items():
    vocab[new_idx] = vocab[idx1] + vocab[idx2]       # new token = concatenation of its two parents
```

**`decode(ids)` — tokens → text:**

```python
def decode(ids):
    text_bytes = bytes().join(vocab[idx] for idx in ids)
    return text_bytes.decode("utf-8", errors="replace")
```

Lookup each token id in `vocab`, concatenate the bytes, then UTF-8 decode. The `errors="replace"` guard handles the rare case where a sequence of tokens doesn't form valid UTF-8 at a boundary.

**`encode(text)` — text → tokens:**

```python
def encode(text):
    tokens = list(map(int, text.encode("utf-8")))   # start as raw bytes

    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # find the pair with the lowest merge index (earliest merge wins)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))

        if pair not in merges:
            break                                   # nothing left to merge
        tokens = merge_pair(tokens, pair, merges[pair])

    return tokens
```

**Why `min` instead of `max`?** During encoding we must apply merges in the same order they were *created* during training. Picking the pair with the smallest merge index (the earliest, most frequent pair) replicates the greedy training order. Using `float('inf')` as a default ensures pairs not in `merges` are never chosen.

**Round-trip verification:**
```python
decode(encode("hello world!")) == "hello world!"   # True
decode(encode(text)) == text                        # True (full corpus)
decode(encode("您好 🙏 (formal hello in Mandarin Chinese!)")) == "您好 🙏..."  # True
```

---

### Step 5 — GPT-2 Regex Pre-tokenization

The BPE algorithm above has a critical flaw: it will happily merge bytes *across* word boundaries. For example, a trailing space becomes part of the next word's token (e.g. `" the"` becomes one merged unit), and contractions like `"don't"` get split unpredictably. GPT-2 solves this with a regex pre-split step.

**GPT-2 regex pattern:**

```python
import regex as re

gpt2pat = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
```

This pattern splits text into **chunks** before BPE ever sees it. Each chunk is encoded independently — merges can never span a chunk boundary.

**What each alternative matches:**

| Pattern | Matches | Example |
|---|---|---|
| `'s\|'t\|'re\|'ve\|'m\|'ll\|'d` | English contractions | `'s`, `'t`, `'ll` |
| `?\p{L}+` | Optional leading space + letters | `Hello`, ` world` |
| `?\p{N}+` | Optional leading space + digits | `123`, ` 42` |
| `?[^\s\p{L}\p{N}]+` | Optional leading space + punctuation | `!!!`, `???` |
| `\s+(?!\S)` | Whitespace not followed by non-whitespace | Trailing spaces |
| `\s+` | Any other whitespace | Newlines, tabs |

**Effect on Python code:** Indentation spaces each form their own chunk rather than merging with the identifier that follows — fixing GPT-2's notorious Python indentation problem.

```python
print(re.findall(gpt2pat, "Hello've world123 how's are you!!!?   "))
# ["Hello", "'ve", " world", "123", " how", "'s", " are", " you", "!!!?", "   "]

# Python code example:
print(re.findall(gpt2pat, "    if i % 3 == 0:"))
# ["    ", "if", " i", " %", " 3", " ==", " 0", ":"]
```

---

### Step 6 — Inspecting Real Tokenizers

**GPT-2 vocabulary structure:**
```python
import json
with open("encoder.json", 'r') as f:
    encoder = json.load(f)

len(encoder)   # 50,257
# = 256 raw byte tokens
#   + 50,000 BPE merge tokens
#   + 1 special token (<|endoftext|>)

encoder['<|endoftext|>']  # → 50256  (the last index)
```

**tiktoken comparison (GPT-2 vs GPT-4):**

```python
import tiktoken

enc_gpt2 = tiktoken.get_encoding("gpt2")
enc_gpt4 = tiktoken.get_encoding("cl100k_base")

enc_gpt2.encode("    hello world!!!")
# → [220, 220, 220, 220, 31373, 995, 10185]
# (4 separate space tokens, then hello/world/!!!)

enc_gpt4.encode("    hello world!!!")
# → [262, 24748, 1917, 12340]
# (4 spaces merged into one token, different byte-pair merges)
```

**GPT-4 tokenizer on multilingual text:**
```python
enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode("안녕하세요 👋 (hello in Korean!)")
# Korean characters each tokenise to multiple tokens; emoji to 2 tokens

enc.decode(ids) == "안녕하세요 👋 (hello in Korean!)"  # True
enc.n_vocab   # 100,277
```

**Special token handling:**
```python
enc.encode("<|endoftext|>hello world", allowed_special="all")
# The special token must be explicitly allowed, otherwise it is tokenised as regular text
```

---

### Step 7 — BasicTokenizer Class

The functional BPE code is refactored into a clean class with the standard `train / encode / decode` interface, trained on `taylorswift.txt`:

```python
class BasicTokenizer:
    def __init__(self):
        self.vocab = {}           # bytes → int (for debugging)
        self.merges = {}          # (int, int) → int
        self.reverse_vocab = {}   # int → bytes (for decode)

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_pair(self, ids, pair, new_idx):
        # same as the functional version above
        ...

    def train(self, text, vocab_size, verbose=False):
        tokens = text.encode("utf-8")
        ids    = list(map(int, tokens))

        for i in range(vocab_size - 256):
            stats    = self.get_stats(ids)
            if not stats: break
            top_pair = max(stats, key=stats.get)
            ids      = self.merge_pair(ids, top_pair, 256 + i)
            self.merges[top_pair] = 256 + i

        # Build reverse_vocab from the learned merges
        self.reverse_vocab = {idx: bytes([idx]) for idx in range(256)}
        for (idx1, idx2), new_idx in self.merges.items():
            self.reverse_vocab[new_idx] = self.reverse_vocab[idx1] + self.reverse_vocab[idx2]

        self.vocab = {v: k for k, v in self.reverse_vocab.items()}

    def encode(self, text):
        ids = list(map(int, text.encode("utf-8")))
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair  = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges: break
            ids   = self.merge_pair(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        text_bytes = bytes().join(self.reverse_vocab.get(idx, b'') for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")
```

**Training and verification:**
```python
tokenizer = BasicTokenizer()
tokenizer.train(text, vocab_size=1000, verbose=True)

decoded = tokenizer.decode(tokenizer.encode("hello hello hello"))
# → "hello hello hello"   ✓

decoded = tokenizer.decode(tokenizer.encode("hello world!!!? (안녕하세요!) lol123 😉"))
# → "hello world!!!? (안녕하세요!) lol123 😉"   ✓
```

---

### Step 8 — RegexTokenizer Class (GPT-4 Style)

`RegexTokenizer` extends `BasicTokenizer` with the critical addition of regex pre-tokenization — matching GPT-4's `cl100k_base` design:

```python
class RegexTokenizer:
    def __init__(self):
        self.merges          = {}
        self.reverse_vocab   = {}
        self.vocab           = {}
        # GPT-4's cl100k_base regex pattern (Unicode-aware)
        self.pattern         = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern)
```

**Key difference from `BasicTokenizer` — `train()` splits first:**

```python
def train(self, text, vocab_size, verbose=False):
    token_chunks = self.compiled_pattern.findall(text)   # split into pre-token chunks
    ids = [list(chunk.encode("utf-8")) for chunk in token_chunks]  # encode each chunk separately

    for i in range(vocab_size - 256):
        stats = {}
        for chunk_ids in ids:
            self.get_stats(chunk_ids, counts=stats)   # aggregate stats ACROSS chunks
        if not stats: break

        pair    = max(stats, key=stats.get)
        new_idx = 256 + i
        ids     = [self.merge_pair(chunk_ids, pair, new_idx) for chunk_ids in ids]  # merge WITHIN each chunk only
        self.merges[pair]      = new_idx
        self.reverse_vocab[new_idx] = self.reverse_vocab[pair[0]] + self.reverse_vocab[pair[1]]
```

**Why `get_stats` takes an optional `counts` dict:** To accumulate pair statistics across all chunks in a single pass without re-allocating a new dictionary each time — a significant performance optimisation when the corpus has thousands of chunks.

**Encoding a chunk respects the training order:**

```python
def encode_chunks(self, chunk):
    chunk_ids = list(map(int, chunk))
    while len(chunk_ids) >= 2:
        stats = self.get_stats(chunk_ids)
        pair  = min(stats, key=lambda p: self.merges.get(p, float('inf')))
        if pair not in self.merges: break
        chunk_ids = self.merge_pair(chunk_ids, pair, self.merges[pair])
    return chunk_ids

def encode(self, text):
    token_chunks = self.compiled_pattern.findall(text)
    ids = []
    for chunk in token_chunks:
        chunk_bytes = chunk.encode("utf-8")
        ids.extend(self.encode_chunks(chunk_bytes))
    return ids
```

**Results — trained on `taylorswift.txt` with `vocab_size=1000`:**
```python
text    = "hello world!!!? (안녕하세요!) lol123 😉"
encoded = regexTokenizer.encode(text)
decoded = regexTokenizer.decode(encoded)
print(decoded == text)  # True

# Compare against GPT-4:
enc = tiktoken.get_encoding("cl100k_base")
gpt4_ids = enc.encode(text)
# Same round-trip guarantee; different token ids due to different training corpus
```

**RegexTokenizer vs BasicTokenizer comparison:**

| Feature | BasicTokenizer | RegexTokenizer |
|---|---|---|
| Pre-tokenization | None — BPE runs on full text | Regex splits text into chunks first |
| Cross-boundary merges | Yes (spaces merge into words) | No (merges constrained within chunks) |
| Contraction handling | Unpredictable | `'s`, `'t`, `'ll` etc. are separate chunks |
| Matches | — | GPT-4 `cl100k_base` design |

---

### Step 9 — SentencePiece Comparison

SentencePiece is the other dominant tokenizer library — used in Llama, Mistral, and T5. Its key architectural difference:

| | tiktoken / this notebook | SentencePiece |
|---|---|---|
| BPE operates on | **UTF-8 bytes** (0–255) | **Unicode code points** directly |
| Rare character handling | Any byte is valid → no unknowns | `character_coverage` hyperparameter; rare code points → UNK or byte fallback |
| Byte fallback | Implicit (bytes are native) | Explicit (`byte_fallback=True`) |

**Llama 2 SentencePiece training settings:**
```python
options = dict(
    model_type       = "bpe",
    vocab_size       = 400,           # Llama 2 uses 32,000
    normalization_rule_name = "identity",  # no normalisation — preserves raw text
    remove_extra_whitespaces = False,
    character_coverage       = 0.99995,    # keep 99.995% of characters by frequency
    byte_fallback            = True,       # encode rare chars as raw bytes
    split_digits             = True,       # prevent digit merging
    split_by_unicode_script  = True,       # don't merge across scripts (Latin/CJK/etc.)
    split_by_whitespace      = True,
    add_dummy_prefix         = True,       # prepend space to first word (Llama quirk)
    unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
)
```

**`add_dummy_prefix=True`** is a Llama-specific quirk: a leading space is prepended so that the first word and any mid-sentence word tokenise identically. This means encoding `"hello"` and `" hello"` produce the same token.

---

### Step 10 — Vocab Size Tradeoffs

Every design choice in tokenizer training feeds into a fundamental tension:

**Why large vocabulary is costly:**
- Input embedding matrix grows: `V × d_model` parameters
- Output projection (softmax) grows: same `V × d_model`
- Rare tokens appear infrequently → embeddings undertrained (sparse gradient signal)
- Overly specific merges lose compositionality

**Why small vocabulary is costly:**
- Sequences get longer → quadratic attention cost
- More tokens needed to represent the same information

**BPE merge granularity example:**

| Vocab size | "machine learning" tokenisation |
|---|---|
| Tiny (~300) | `"m" "a" "c" "h" "i" "n" "e" " " "l" "e" "a" "r" "n" "i" "n" "g"` |
| Medium (~1000) | `"machine" " learning"` |
| Large (~50k) | `"machine"` + `" learning"` (or even `" machine learning"` as one token) |
| Huge | `" machinelearningmodel"` — too specific, undertrained |

**The compositionality argument:** A model that sees `"machine"` and `"learning"` as separate tokens can transfer knowledge across compounds (`"machine translation"`, `"learning rate"`, etc.). A model that sees `"machinelearning"` as one opaque token cannot.

**Modern LLM vocab size choices:**

| Model | Vocab size | Tokenizer |
|---|---|---|
| GPT-2 | 50,257 | tiktoken BPE |
| GPT-4 | 100,277 | tiktoken BPE (cl100k_base) |
| Llama 2 | 32,000 | SentencePiece BPE |
| Mistral 7B | 32,000 | SentencePiece BPE |

The sweet spot empirically lands between 32k–100k — large enough for efficient sequence lengths, small enough that every token receives meaningful training signal.

---

## 📒 Appendix E — Parameter-Efficient Fine-Tuning with LoRA

**Notebook:** `Parameter-efficient-Finetuning-with-LoRA.ipynb`

### The Problem: Full Fine-Tuning is Expensive

Full fine-tuning updates every weight in a 124M-parameter model. For larger models (7B, 70B), this requires enormous GPU memory and compute. **LoRA (Low-Rank Adaptation)** offers a principled way to fine-tune with a tiny fraction of the parameters — by injecting small trainable low-rank matrices alongside frozen pretrained weights.

**Parameter comparison for GPT-2 Small:**

| Method | Trainable Parameters | % of Total |
|--------|---------------------|------------|
| Full fine-tuning | 124,441,346 | 100% |
| LoRA (rank=16) | 2,666,528 | **2.1%** |

This notebook reuses the spam classification task from Ch. 6, applying LoRA on top of the frozen pretrained GPT-2 weights.

---

### Step 1 — LoRALayer

The mathematical insight behind LoRA: instead of updating a weight matrix `W` (shape `d_in × d_out`) directly, we add a **low-rank perturbation** `ΔW = A × B` where:
- `A` has shape `(d_in, rank)` — initialised with Kaiming uniform
- `B` has shape `(rank, d_out)` — initialised to **zeros** (so LoRA starts as an identity transform)
- `rank << d_in, d_out` — the bottleneck that enforces low-rank structure

The full adapted weight is: `W_adapted = W_original + alpha * (A @ B)`

```python
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))  # zero init!
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)
        # Shape: (B, T, in_dim) @ (in_dim, rank) @ (rank, out_dim) = (B, T, out_dim)
```

**Why zero-initialise B?** At the start of training, `A @ B = 0`, so the LoRA path contributes nothing. The model begins exactly where the pretrained checkpoint left off and gradually learns the adaptation. If A were initialised to zero instead, gradients would be identically zero — no learning possible.

**Alpha scaling:** `alpha` controls how aggressively LoRA adapts relative to the base model. Setting `alpha = rank` (common default) means the effective learning rate of the LoRA path scales consistently regardless of rank choice.

---

### Step 2 — LinearWithLoRA

`LinearWithLoRA` is a drop-in wrapper that keeps the original `nn.Linear` frozen and adds a parallel `LoRALayer`:

```python
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear         # original frozen Linear layer
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
        # base output  +  low-rank adaptation
```

**Forward pass breakdown:**
```
x: (B, T, 768)
self.linear(x):   (B, T, 768) — frozen base computation
self.lora(x):     (B, T, 16) @ (16, 768) = (B, T, 768) — trainable delta
output:           (B, T, 768) — sum
```

---

### Step 3 — Replacing All Linear Layers

A single recursive function walks every `nn.Linear` in the model tree and replaces it with `LinearWithLoRA`:

```python
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)  # recurse into sub-modules
```

After calling `replace_linear_with_lora(model, rank=16, alpha=16)`, every `W_key`, `W_query`, `W_value`, `out_proj`, and FFN linear layer becomes a `LinearWithLoRA`. The model structure becomes:

```
MultiHeadAttention(
  (W_key): LinearWithLoRA(
    (linear): Linear(768 → 768, frozen)
    (lora):   LoRALayer()   ← trainable
  )
  ...
)
```

**Where do the 2.67M parameters come from?**
- 12 transformer blocks × 6 linear layers per block = 72 LinearWithLoRA wrappers
- Each adds 2 × (in_dim × rank) parameters (A and B matrices)
- Plus the classification head: 1 LinearWithLoRA
- Total ≈ 2,666,528 trainable LoRA parameters

---

### Step 4 — Freeze Base, Train LoRA

The protocol:

```python
# 1. First, freeze ALL parameters
for param in model.parameters():
    param.requires_grad = False

print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# → 0 (fully frozen)

# 2. Then inject LoRA (creates NEW trainable A and B Parameters)
replace_linear_with_lora(model, rank=16, alpha=16)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# → 2,666,528 (only LoRA parameters train)
```

The base weights remain completely frozen — their gradients are never computed, saving memory and compute. Only the tiny A and B matrices are updated during fine-tuning.

**Training setup (same as Ch. 6):**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
# num_epochs = 5, batch_size = 8, spam dataset (balanced, 70/10/20 split)
```

---

### Step 5 — Results

**Training progress (5 epochs):**
```
Ep 1 (Step 000000): Train loss 3.867, Val loss 3.505
Ep 1 (Step 000050): Train loss 0.394, Val loss 0.371
Ep 1 (Step 000100): Train loss 0.131, Val loss 0.269
Ep 2 (Step 000150): Train loss 0.120, Val loss 0.087
...
Ep 5 (Step 000600): Train loss 0.000, Val loss 0.112
Training completed in 1.68 minutes.
```

**Final accuracy:**

| Split | Accuracy |
|-------|---------|
| Train | 99.42% |
| Validation | 100.00% |
| **Test** | **96.33%** |

LoRA achieves **96.33% test accuracy** — comparable to full fine-tuning from Ch. 6 (97%) — using only **2.1% of the total parameters** and training in **under 2 minutes**.

**Why LoRA works:** The pretrained GPT-2 weights already encode rich representations of language. Fine-tuning for spam classification only requires a small *correction* to those representations — exactly what a low-rank delta can capture. The frozen base handles syntax, semantics, and world knowledge; the LoRA matrices learn the task-specific "direction" in weight space.

**Comparison to full fine-tuning (Ch. 6):**

| Metric | Full Fine-Tuning (Ch. 6) | LoRA (Appendix E) |
|--------|--------------------------|-------------------|
| Trainable params | 124M | **2.67M** |
| Test accuracy | 97% | **96.33%** |
| Memory saved | — | ~98% fewer param updates |
| Use case | Small models / ample GPU | Large models / limited GPU |

---

## 📚 Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **Token** | The atomic unit of text a model processes. Can be a character, word, or subword. |
| **Tokenization** | The process of splitting raw text into tokens and converting them to integer IDs. |
| **Vocabulary** | A fixed mapping from token strings to unique integers. |
| **BPE (Byte-Pair Encoding)** | A subword tokenization algorithm that merges frequent byte pairs iteratively to build a vocabulary. Used by GPT-2, GPT-3, LLaMA. |
| **`<\|endoftext\|>`** | A special delimiter token inserted between documents in a training corpus to prevent cross-document context leakage. |
| **`<\|unk\|>`** | A special token used as a placeholder for out-of-vocabulary words. BPE doesn't need this; word-level tokenizers do. |
| **Embedding** | A dense, learned vector representation of a discrete symbol (token or position). |
| **`nn.Embedding`** | A PyTorch module that acts as a learnable lookup table: maps integer indices to float vectors. |
| **Token Embedding** | A learned vector representing *what* a token is (its semantic identity). |
| **Positional Embedding** | A learned vector representing *where* a token appears in the sequence (its position). |
| **Context Length / Window** | The maximum number of tokens a model can attend to at once. GPT-2 uses 1024. |
| **Stride** | The step size for the sliding window when creating training samples from a corpus. |
| **Next-Token Prediction** | The core self-supervised training objective: predict the next token given the preceding tokens. Also called causal language modeling. |
| **Sliding Window** | A technique to create overlapping (input, target) pairs by moving a fixed-size window across a token sequence. |
| **`DataLoader`** | PyTorch utility that batches, shuffles, and streams samples from a `Dataset` to the training loop. |
| **Batch Size** | Number of independent sequences processed in parallel in one forward pass. |
| **Broadcasting** | PyTorch's automatic expansion of tensors with compatible shapes for arithmetic operations (e.g., adding `[4, 256]` to `[8, 4, 256]`). |
| **Self-Attention** | A mechanism where each token computes how much it should attend to every other token in the sequence using learned Query, Key, Value projections. |
| **Query (Q)** | A learned projection of a token representing "what am I looking for?" |
| **Key (K)** | A learned projection of a token representing "what do I contain?" |
| **Value (V)** | A learned projection of a token representing "what do I return if attended to?" |
| **Attention Score** | The dot product of a Query with a Key — measures relevance between two tokens. |
| **Attention Weight** | An attention score after softmax normalisation — a probability in [0, 1] summing to 1 over all positions. |
| **Context Vector** | A weighted sum of Value vectors using attention weights — a token's representation enriched with information from all relevant tokens. |
| **Scaled Dot-Product Attention** | Attention scores divided by `√d_k` before softmax to prevent gradient saturation in large dimensions. |
| **Causal Mask** | An upper-triangular mask of `-inf` values that prevents tokens from attending to future positions. Essential for autoregressive language modelling. |
| **Multi-Head Attention** | Running several attention mechanisms in parallel, each with different Q/K/V projections, then concatenating their outputs. |
| **Head Dimension** | `emb_dim / num_heads` — the size of each attention head's Q/K/V vectors. |
| **Layer Normalization** | Normalises each token's embedding to mean=0, var=1 across the embedding dimension, with learnable scale (γ) and shift (β). |
| **GELU** | Gaussian Error Linear Unit — a smooth activation function used in transformers. Outperforms ReLU by allowing small gradients for negative inputs. |
| **Feed-Forward Network (FFN)** | A 2-layer MLP inside each transformer block: expands `emb_dim → 4×emb_dim` via GELU, then projects back. Applied independently to each token. |
| **Residual / Skip Connection** | Adding a layer's input directly to its output (`x = layer(x) + x`). Provides gradient highways through deep networks, enabling stable training. |
| **Pre-LayerNorm** | Applying LayerNorm *before* attention/FFN (GPT-2 style). More stable than Post-LayerNorm in deep networks. |
| **Transformer Block** | The core repeating unit of a GPT model: Pre-LN → Multi-Head Attention → Residual → Pre-LN → FFN → Residual. |
| **Logits** | Raw, unnormalised output scores from the final linear layer. Shape `(B, T, vocab_size)`. Apply softmax for probabilities. |
| **Weight Tying** | Sharing the token embedding matrix and the output projection matrix. Reduces parameter count by ~38M in GPT-2 Small. |
| **Autoregressive Generation** | Generating text one token at a time, appending each predicted token to the input before predicting the next. |
| **Greedy Decoding** | Always selecting the highest-probability next token. Deterministic but can produce repetitive text. |
| **`model.eval()`** | Switches PyTorch model to inference mode — disables dropout and batch normalisation updates. |
| **Cross-Entropy Loss** | The primary training loss for language models: measures the model's surprise at the true next token. Computed as negative average log probability of correct tokens. |
| **Perplexity** | `exp(cross_entropy_loss)` — interpretable metric representing effective vocabulary size the model is uncertain about. Lower is better. |
| **AdamW** | Adam optimiser with decoupled L2 weight decay. Standard for transformer training. |
| **Weight Decay** | Regularisation technique that penalises large weights, discouraging overfitting. |
| **Overfitting** | Model memorises training data instead of learning generalisable patterns. Evidenced by widening train/val loss gap. |
| **Temperature Scaling** | Dividing logits by a temperature parameter before softmax to control randomness of sampling. T < 1 = more confident; T > 1 = more random. |
| **Top-k Sampling** | Restricting token sampling to the k highest-probability tokens, preventing selection of very unlikely tokens. |
| **Instruction Fine-Tuning** | Training a pretrained language model to follow human instructions using labelled instruction-response pairs. |
| **Alpaca Format** | A widely used instruction template with `### Instruction:`, optional `### Input:`, and `### Response:` sections. |
| **ignore_index** | Value (-100 by default) passed to `cross_entropy` to mask positions that should not contribute to the loss — used to ignore padding tokens in instruction fine-tuning. |
| **Transfer Learning** | Reusing a pretrained model's knowledge for a new task via fine-tuning, instead of training from scratch. |
| **Classification Head** | A small linear layer added on top of a pretrained model that maps the last hidden state to class logits. |
| **LLM-as-Judge** | Using a powerful language model to evaluate another model's outputs, replacing expensive human annotation. |
| **Catastrophic Forgetting** | When fine-tuning overwrites pretrained knowledge. Mitigated by freezing most layers and using low learning rates. |
| **LR Warmup** | A training technique that linearly increases the learning rate from a small initial value to a peak over a fixed number of steps. Prevents early divergence when gradients are large and noisy at the start of training. |
| **Cosine Annealing** | A learning rate decay schedule that reduces LR from peak to minimum following a half-cosine curve: `η_t = η_min + 0.5*(η_max - η_min)*(1 + cos(t/T * π))`. Smooth decay widely used in LLM training. |
| **Gradient Clipping** | Capping the global gradient norm to a maximum value before the optimiser step. Prevents gradient explosion from destroying learned representations during training spikes. |
| **Gradient Norm** | The Euclidean norm of all gradients stacked into a single vector. Used by `clip_grad_norm_` to detect and correct oversized gradient updates. |
| **LoRA (Low-Rank Adaptation)** | A parameter-efficient fine-tuning method that injects small trainable low-rank matrices (A and B) alongside frozen pretrained weights. The adaptation is: `ΔW = alpha * A @ B` where rank << d_in. |
| **Low-Rank Decomposition** | Representing a large matrix as a product of two smaller matrices. For an (m×n) matrix, a rank-r decomposition uses an (m×r) and (r×n) matrix, with r << min(m,n). |
| **Rank (LoRA)** | The bottleneck dimension of the LoRA matrices. Controls the number of trainable parameters per layer. Typical values: 4, 8, 16, 64. |
| **Alpha (LoRA)** | A scaling factor for the LoRA output. Common to set alpha = rank so the effective LR of LoRA adapters is consistent across rank choices. |
| **Parameter-Efficient Fine-Tuning (PEFT)** | A family of techniques for adapting large pretrained models with far fewer trainable parameters than full fine-tuning. Includes LoRA, adapters, prefix tuning, and prompt tuning. |
| **Kaiming Uniform Initialization** | A weight initialisation strategy (`torch.nn.init.kaiming_uniform_`) designed for layers followed by ReLU-family activations. Preserves variance through the network. Used for LoRA's A matrix. |
| **Unicode Code Point** | A unique integer assigned to every character in the Unicode standard. Written as U+XXXX (e.g. U+0041 = 'A'). Python's `ord()` returns a code point; `chr()` converts back. |
| **UTF-8** | A variable-length byte encoding for Unicode. ASCII characters encode as 1 byte; non-ASCII as 2–4 bytes. The dominant encoding on the web and in modern tokenizers. |
| **Byte-Pair Encoding (BPE) — algorithm** | An iterative compression algorithm: find the most frequent adjacent pair of tokens, replace every occurrence with a new token, repeat. Used to build LLM vocabularies from raw bytes. |
| **`get_stats`** | A function that counts every adjacent pair of token IDs in a sequence. The core frequency-counting primitive of BPE. Returns a dict `{(a, b): count}`. |
| **`merge_pair`** | A function that replaces every occurrence of a target pair `(a, b)` in a token ID list with a new index. The core mutation primitive of BPE. |
| **Compression Ratio** | `len(tokens_before) / len(tokens_after)`. Measures how much BPE has compressed a sequence. GPT-2 achieves ~3–4× on English; more merges → higher compression. |
| **Pre-tokenization** | Splitting raw text into chunks with a regex *before* applying BPE, so merges never cross word or category boundaries. Used in GPT-2 (GPT-2 pattern) and GPT-4 (cl100k pattern). |
| **GPT-2 Regex Pattern** | `'s\|'t\|'re\|'ve\|'m\|'ll\|'d\| ?\p{L}+\| ?\p{N}+\| ?[^\s\p{L}\p{N}]+\|\s+(?!\S)\|\s+` — splits contractions, words, numbers, punctuation, and whitespace into separate pre-token chunks. |
| **Special Tokens** | Reserved token strings with fixed IDs that signal control information to the model (e.g. `<\|endoftext\|>` = index 50256 in GPT-2, marking end of a document). Must be explicitly allowed during encoding. |
| **SentencePiece** | A tokenizer library (used in Llama, Mistral, T5) that runs BPE on Unicode code points rather than bytes, with optional byte fallback for rare characters. |
| **`byte_fallback`** | A SentencePiece option: rare code points not covered by `character_coverage` are encoded as their raw UTF-8 bytes (each byte becomes a separate byte token) instead of mapping to `<unk>`. |
| **`character_coverage`** | A SentencePiece hyperparameter (0–1) controlling what fraction of the training corpus's character types are included directly in the vocabulary. Rare characters outside coverage fall back to bytes or UNK. |
| **`add_dummy_prefix`** | A SentencePiece option used by Llama: prepends a space to all input text so that the first word tokenises identically to mid-sentence occurrences of the same word. |
| **Grapheme Cluster** | A "user-perceived character" — one or more Unicode code points that render as a single visual glyph (e.g. a base letter plus diacritics). The natural unit for cursor movement and string-length UX. |
| **Undertrained Tokens** | Vocabulary entries that appear so rarely in training data that their embeddings receive very few gradient updates, resulting in poor-quality representations. A key risk of oversized vocabularies. |

---

## 🛠️ Skills Demonstrated

- **Python & Regex** — text preprocessing with `re.split`, filtering, pattern design, Unicode-aware `\p{L}` / `\p{N}` character classes
- **Data Structures** — building bidirectional vocab dictionaries from scratch
- **OOP** — clean class-based design across tokenizers, attention, full model, and dataset classes
- **PyTorch** — `Dataset`, `DataLoader`, `nn.Embedding`, `nn.Linear`, `nn.Parameter`, `register_buffer`, tensor operations, `.view()`, `.transpose()`, `.contiguous()`, `.flatten()`, `.multinomial()`
- **NLP Fundamentals** — tokenization pipeline, vocabulary construction, OOV handling
- **Unicode & Encoding** — UTF-8 byte encoding, code points vs characters, `ord()`/`chr()`, multi-byte sequences
- **BPE Algorithm** — `get_stats`, `merge_pair`, training loop, compression ratio, encode/decode round-trip
- **Tokenizer Architecture** — `BasicTokenizer` (raw BPE), `RegexTokenizer` (regex pre-tokenization), SentencePiece comparison
- **Tokenizer Design** — vocab size tradeoffs, compositionality, undertrained tokens, multilingual considerations
- **Linear Algebra** — dot products, matrix multiplication, tensor reshaping for multi-head attention
- **Attention Mechanism** — from raw dot-product scores through scaled, masked, multi-head attention
- **Transformer Architecture** — Layer Norm, GELU, FFN, residual connections, Pre-LN design
- **LLM Architecture** — complete GPT-2-class model from embeddings to vocabulary logits
- **BPE Tokenization** — using `tiktoken` and understanding subword encoding
- **Self-supervised Learning** — constructing next-token prediction training pairs
- **Model Analysis** — parameter counting, weight tying, model variant scaling
- **Training Loop Engineering** — loss tracking, evaluation scheduling, tokens-seen counters
- **Loss Functions** — cross-entropy from first principles, perplexity, ignore_index masking
- **Optimisation** — AdamW, weight decay, learning rate selection
- **Advanced LR Scheduling** — linear warmup, cosine annealing decay, multi-phase LR schedules
- **Gradient Clipping** — `torch.nn.utils.clip_grad_norm_`, gradient norm inspection, training stability
- **Model Persistence** — saving/loading weights and optimizer state for training resumption
- **Transfer Learning** — loading pretrained OpenAI GPT-2 checkpoint weights into custom architecture
- **Inference Optimisation** — temperature scaling, top-k filtering, greedy decoding
- **Fine-Tuning** — classification head replacement, selective layer freezing, parameter-efficient fine-tuning
- **Dataset Engineering** — class balancing, padding, custom collate functions, ignore_index masking
- **Instruction Tuning** — Alpaca format, InstructionDataset, full instruction fine-tuning pipeline
- **Model Evaluation** — LLM-as-Judge evaluation with Ollama, automated scoring pipelines
- **LoRA Implementation** — `LoRALayer`, `LinearWithLoRA`, recursive model surgery, low-rank adapter math
- **Parameter-Efficient Fine-Tuning** — freeze-then-inject pattern, rank/alpha tuning, 98% parameter reduction

---

## 🖥️ Environment

```
Python         3.10.19
PyTorch        (cuda-compatible build recommended)
tiktoken       0.12.0
Jupyter        Notebook / Lab
```

Install dependencies:

```bash
pip install torch tiktoken jupyter
```

---

> *This README is a living document. New chapters and notebooks will be appended as the project progresses through pretraining, fine-tuning, and RLHF.*
