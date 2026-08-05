class Tokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        self.stoi = {
            ch:i for i,ch in enumerate(chars)
        }
        self.itos = {
            i:ch for i,ch in enumerate(chars)
        }

    def encode(self, text):
        ids = []
        for ch in text:
            ids.append(
                self.stoi[ch]
            )

        return ids

    def decode(self,ids):
        chars = []
        for id in ids:
            chars.append(
                self.itos[id]
            )

        return "".join(chars)