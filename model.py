import torch
import torch.nn as nn
import math
from tokenizer import CharTokenizer

class SelfAttention(nn.Module):
    def __init__(self,d_model):
        super(SelfAttention,self).__init__()
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model,d_model)
        self.W_K = nn.Linear(d_model,d_model)
        self.W_V = nn.Linear(d_model,d_model)
        self.W_O = nn.Linear(d_model,d_model)

    def forward(self,x):
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        B, T, C = q.shape
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        output = attention @ v
        return self.W_O(output)

class MLP(nn.Module):
    def __init__(self,d_model):
        super(MLP,self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model,d_model),
        )

    def forward(self,x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self,d_model):
        super(DecoderBlock,self).__init__()
        self.self_attention = SelfAttention(d_model=d_model)
        self.mlp = MLP(d_model=d_model)

    def forward(self,x):
        attn_output = self.self_attention(x)
        x = attn_output + x
        mlp_output = self.mlp(x)
        x = x + mlp_output
        return x


class MiniDecoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(MiniDecoder,self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.block = DecoderBlock(d_model=d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self,input_ids):
        x = self.embedding(input_ids)
        x = self.block(x)
        logits = self.lm_head(x)
        return logits

if __name__ == "__main__":
    tokenizer = CharTokenizer()
    input_ids = tokenizer.encode("hello cuda")
    model = MiniDecoder(vocab_size=tokenizer.vocab_size, d_model=64);
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(100):
        logits = model(input_ids)
        logits_for_next_token = logits[:,:-1,:]
        targets = input_ids[:,1:]
        loss = nn.functional.cross_entropy(logits_for_next_token.reshape(-1,tokenizer.vocab_size), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {i+1}, Loss: {loss.item()}")

    model.eval()
    generated = tokenizer.encode("h")
    for _ in range(9):
        logits = model(generated)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1,keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        print("Generated: ", generated)
        print(tokenizer.decode(generated))
