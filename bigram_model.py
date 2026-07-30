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

batch_size = 32
block_size = 8

def get_batch(split):
    # Generates a small batch of inputs x and targets y
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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

# Bigram character level language model

class BLM(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
            
        
model = BLM()
model = model.to(device)

# Hyperparams
epochs = 10000
eval_interval = 300
lr = 1e-2
eval_iters = 200
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lossi = []

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
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
plt.plot(lossi)
# plt.show()