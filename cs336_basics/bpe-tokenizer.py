import regex as re
from collections import Counter
import pickle

# Token regex: 

INITIAL_VOCAB_SIZE = 256

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens


    """
    Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
    (in the same format that your BPE training code output) and (optionally) a list of special
    tokens. This method should accept the following additional parameters:
    vocab_filepath: str
    merges_filepath: str
    special_tokens: list[str] | None = None
    """

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        vocab_normalized = {}
        for token, idx in vocab.items():
            idx_int = int(idx)
            if isinstance(idx, str):
                idx = idx.encode("utf-8")
            vocab_normalized[idx_int] = idx

        merges_normalized = []
        for p1, p2 in merges:
            b1 = p1 if isinstance(p1, bytes) else p1.encode("utf-8")
            b2 = p2 if isinstance(p2, bytes) else p2.encode("utf-8")
            merges_normalized.append((b1, b2))

        return cls(vocab_normalized, merges_normalized, special_tokens)

    def pretokenize(self, text):
        token_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        matches = re.finditer(token_regex, text)
        pre_tokens = []

        for m in matches:
            pre_tokens.append(m.group().encode("utf-8"))

        return pre_tokens

    def encode(self, text):
        token_ids = []
        pre_tokens = self.pretokenize(text)
        
        for token_bytes in pre_tokens:
            word = [bytes(b) for b in token_bytes]

            for p1, p2 in self.merges:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == p1 and word[i+1] == p2:
                        new_word.append(p1 + p2)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word

            for chunk in word:
                token_ids.append(self.byte_to_id[chunk])

        return token_ids

    def encode_iterable(self, texts):
        for text in texts:
            yield self.encode(text)

    def decode(self, token_ids):
        encoded_stream = b"".join(self.vocab[idx] for idx in token_ids)

        return encoded_stream.decode("utf-8", errors="replace")

