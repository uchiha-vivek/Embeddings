import numpy as np
import matplotlib.pyplot as plt


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
np.random.seed(42)

# hyperparameters
vocab_size=100
embedding_dim=10
max_seq_len=16

token_embedding_table = np.random.randn(
    vocab_size,
    embedding_dim
    )
    
positional_embedding_table = np.random.randn(
    max_seq_len,
    embedding_dim
    )

tokens = np.array([5,10,7,21])


# token embeddings
token_embeddings = token_embedding_table[tokens]


# positional embeddings
positions = np.arange(len(tokens))
position_embeddings = positional_embedding_table[positions]


print("Token Embeddings",token_embeddings.shape)
print("Position embeddings",position_embeddings.shape)

x = token_embeddings + position_embeddings
print(x.shape)
print(x)

# Token embeddings
axes[0].imshow(token_embeddings, aspect="auto")
axes[0].set_title("Token Embeddings")
axes[0].set_xlabel("Embedding Dimension")
axes[0].set_ylabel("Token Position")

# Position embeddings
axes[1].imshow(position_embeddings, aspect="auto")
axes[1].set_title("Position Embeddings")
axes[1].set_xlabel("Embedding Dimension")

# Final embeddings
axes[2].imshow(x, aspect="auto")
axes[2].set_title("Token + Position")
axes[2].set_xlabel("Embedding Dimension")

plt.tight_layout()
plt.show()

