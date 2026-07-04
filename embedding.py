import torch
import torch.nn as nn
from tokenizer import CharTokenizer

tokens = CharTokenizer()
input_ids = tokens.encode("hello cuda")

embeddings = nn.Embedding(tokens.vocab_size,64)
x = embeddings(input_ids)
print("input ids: ", input_ids)
print("\n")
print("x: ", x)
print("\n")
print("x shape: ", x.shape)
print("\n")
print("embeddings: ", embeddings)
print("\n")


