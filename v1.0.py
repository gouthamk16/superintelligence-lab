import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data loading and preprocessing
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

## Extracting all the unique characters thay occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Creating a mapping from characters to integers (using the default mapping in the set of unique characters we made)
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% of the data
train_data = data[:n]
val_data = data[n:]

# Creating our real training data
# Mini Batch - the data is split into number of batches, with each batch containing chunks

batch_size = 64
block_size = 256
n_embed = 384
n_layers = 6
n_heads = 6
drop_rate = 0.2

def get_batch(split):
    # Generates a small batch of inputs x and targets y
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y



# Attention block
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        wei = q @ k.transpose(-2, -1) / (C ** 0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, C)
        return out

## Multi_head attention
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(drop_rate)
        )
            
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
            
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Residual connections 
        x = x + self.ffwd(self.ln2(x)) # Residual connections
        return x

# Bigram character level language model

class BLM(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads=n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, idx, targets=None):
        token_embeddings = self.token_embedding_table(idx) # (B, T, C)
        position_embeddings = self.positional_embedding_table(torch.arange(idx.shape[1], device=device)) # (T, C)
        x = token_embeddings + position_embeddings 
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Reshaping our logits to be compatible with the 'F.cross_entropy' function
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
            
        
model = BLM()
model = model.to(device)

# Hyperparams
epochs = 5000
eval_interval = 500
lr = 3e-4
eval_iters = 200
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lossi = []

@torch.no_grad() # Context manager - nothing that goes on inside this function calls the backward function
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# training loop
for epoch in range(epochs):

    if epoch % eval_interval == 0:
        losses = estimate_loss()
        lossi.append(losses['val'])
        print(f"Step {epoch}/{epochs} : Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
    
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate samples
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=50000)[0].tolist()))
plt.plot(lossi)
# plt.show()