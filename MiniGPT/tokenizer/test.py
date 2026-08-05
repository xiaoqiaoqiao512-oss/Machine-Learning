from tokenizer import Tokenizer

text = "I love AI"
tokenizer = Tokenizer(text)

print(tokenizer.stoi)

ids = tokenizer.encode("I love")

print(ids)

print(tokenizer.decode(ids))