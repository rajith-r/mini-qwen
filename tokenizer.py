import torch
import string
class CharTokenizer:
    def __init__(self):
        chars = (
            string.ascii_lowercase +
            string.ascii_uppercase +
            string.digits +
            " .,!?;:'\"-_\n"
        )

        # Add special unknown token
        self.unk_token = "<UNK>"

        self.chars = [self.unk_token] + list(chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.vocab_size = len(self.chars)

    def encode(self,text):
        ids = [self.stoi[c] for c in text]
        return torch.tensor([ids])
    
    def decode(self,ids):
        ids = ids[0].tolist()
        return ''.join([self.itos[i] for i in ids])


if __name__ == "__main__":
    tokenizer = CharTokenizer()
    text = "hello cuda"
    input_ids = tokenizer.encode(text)
    print(input_ids)
    print(tokenizer.decode(input_ids))