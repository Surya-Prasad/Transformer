import regex as re

# Token regex: 
token_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
INITIAL_VOCAB_SIZE = 256

def pre_tokenize(text):
    token_match_list = re.finditer(token_regex, text)
    return [token.group().encode("utf-8") for token in token_match_list]
    
def train_bpe(input_path, vocab_size, special_tokens):
    # Vocabulary
    vocab = {}
    for i in range(INITIAL_VOCAB_SIZE):
        vocab[bytes([i])] = i

    for i, token in enumerate(special_tokens):
        vocab[INITIAL_VOCAB_SIZE + i] = token.encode("utf-8")

    with open(input_path, "rb") as f:
        text = f.read()
    tokens = pre_tokenize(text)


