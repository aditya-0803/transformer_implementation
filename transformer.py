import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 64
block_size = 256
n_embd = 384
n_heads = 6
n_layers = 6
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Reading the file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#Obtaining vocab size, which is the number of unique tokens (characters) in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

#Building a table where each character corresponds to an integer
string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}

#Encoding function: Converts text to a corresponding list of integers
def encode(s):
    result = []
    for char in s:
        result.append(string_to_int[char])
    return result

#Decoding function: Converts a list of integers to corresponding text
def decode(nums):
    result = ''
    for i in nums:
        result += int_to_string[i]
    return result

#Training and validation splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#Getting `batch_size` random chunks of data of length `block_size`
def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = val_data
    
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = q @ k.transpose(-2, -1) / (k.shape[-1]**0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        output = weights @ v

        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.n_heads = nn.ModuleList([])
        for i in range(n_heads):
            self.n_heads.append(Head(head_size))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.n_heads], dim=-1)
        output = self.proj(output)
        output = self.dropout(output) 
        return output


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.model_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.model_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_end = idx[:, -block_size:]
            logits, loss = self(idx_end)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx      


model = GPTModel()
model = model.to(device)

@torch.no_grad()
def estimate_loss():
    result = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        result[split] = losses.mean()
    model.train()
    return result

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))