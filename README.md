# Token & Positional Embeddings

A minimal NumPy/Matplotlib demonstration of how transformer models represent tokens and their positions as dense vectors.

---

## What Are Embeddings?

Neural networks cannot process raw text — they need numbers. **Embeddings** are the bridge: they map discrete symbols (words, subwords, characters) to continuous, dense vectors in a high-dimensional space where semantic relationships can be learned.

In transformer architectures (GPT, BERT, etc.), every input token receives **two** embeddings that are summed together before entering the attention layers:

```
Input Representation = Token Embedding + Positional Embedding
```

---

## Token Embeddings

### Concept

A token embedding answers: *"What does this token mean?"*

Each token in the vocabulary is assigned a unique row in a learnable **embedding matrix** of shape `(vocab_size, embedding_dim)`. Looking up a token is a simple integer index into that matrix.

```
vocab_size  = 100   # number of unique tokens
embedding_dim = 10  # vector size per token

token_embedding_table: shape (100, 10)
```

### Why Dense Vectors?

| Representation | Dimensionality | Captures Similarity? |
|---|---|---|
| One-hot encoding | vocab_size (sparse) | No |
| Token embedding | embedding_dim (dense) | Yes (after training) |

After training, tokens with similar meanings cluster together in the embedding space. "king" and "queen" land closer to each other than either does to "car".

### In the Code

```python
token_embedding_table = np.random.randn(vocab_size, embedding_dim)  # (100, 10)

tokens = np.array([5, 10, 7, 21])                    # 4 input tokens (by index)
token_embeddings = token_embedding_table[tokens]      # shape (4, 10)
```

Row index 5 → the 10-dimensional vector for token #5. No computation needed — just a table lookup.

---

## Positional Embeddings

### The Problem Attention Solves (and Creates)

Self-attention considers every token against every other token simultaneously. This is powerful, but it means the order of tokens is invisible to the model — `"dog bites man"` and `"man bites dog"` would produce identical attention inputs without additional information.

### Concept

A positional embedding answers: *"Where in the sequence does this token appear?"*

A second learnable matrix of shape `(max_seq_len, embedding_dim)` maps each position index (0, 1, 2, …) to a dense vector:

```
max_seq_len   = 16   # maximum number of tokens in a sequence
embedding_dim = 10   # same dimensionality as token embeddings

positional_embedding_table: shape (16, 10)
```

### Learned vs. Sinusoidal

| Approach | How | Used In |
|---|---|---|
| **Learned** (this demo) | Position vectors are trained weights | GPT-2, BERT |
| **Sinusoidal** | Fixed sin/cos functions of position | Original Transformer ("Attention Is All You Need") |
| **Rotary (RoPE)** | Rotates query/key vectors by position | LLaMA, GPT-NeoX |

This demo uses the learned approach — the table is initialized randomly (in a real model it is optimized during training).

### In the Code

```python
positional_embedding_table = np.random.randn(max_seq_len, embedding_dim)  # (16, 10)

positions = np.arange(len(tokens))          # [0, 1, 2, 3]
position_embeddings = positional_embedding_table[positions]  # shape (4, 10)
```

---

## Combining the Two

Once both embeddings share the same dimensionality, they are added element-wise:

```python
x = token_embeddings + position_embeddings   # shape (4, 10)
```

The resulting matrix `x` encodes **both identity and order** for each token. This is the tensor that flows into the first transformer block.

```
Token #5  at position 0  →  token_emb[5]  + pos_emb[0]
Token #10 at position 1  →  token_emb[10] + pos_emb[1]
Token #7  at position 2  →  token_emb[7]  + pos_emb[2]
Token #21 at position 3  →  token_emb[21] + pos_emb[3]
```

---

## Visualization

The script produces a side-by-side heatmap of the three matrices:

```
[Token Embeddings]  [Position Embeddings]  [Token + Position]
     (4 × 10)             (4 × 10)              (4 × 10)
```

Each row is one token in the sequence; each column is one embedding dimension. The final panel shows how the two sources of information fuse.

---

## Running the Demo

```bash
pip install numpy matplotlib
python embedding.py
```

---

## Hyperparameters

| Variable | Value | Meaning |
|---|---|---|
| `vocab_size` | 100 | Total number of unique tokens |
| `embedding_dim` | 10 | Dimensions per embedding vector |
| `max_seq_len` | 16 | Maximum tokens in one sequence |

---

## Key Takeaways

1. **Token embeddings** give each word/subword a dense identity vector.
2. **Positional embeddings** inject sequence-order information lost by parallel attention.
3. **Element-wise addition** is all that is needed to merge the two — they live in the same vector space by design.
4. Both tables are typically **learned jointly** during training, so the model discovers the most useful representations for the task.
