import regex as re
import pickle
from typing import Iterable, Iterator, Optional
from collections import Counter, defaultdict

import regex as re
import pickle

def merge(chunks, pair, new_chunk):
    new_chunks = []
    i = 0
    while i < len(chunks):
        if i + 1 < len(chunks) and chunks[i] == pair[0] and chunks[i + 1] == pair[1]:
            new_chunks.append(new_chunk)
            i += 2
        else:
            new_chunks.append(chunks[i])
            i += 1
    return new_chunks

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.byte_to_id = {v: k for k, v in vocab.items()}
        
        self.merges_dict = {pair: i for i, pair in enumerate(self.merges)}

        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.byte_to_id:
                new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.vocab[new_id] = st_bytes
                self.byte_to_id[st_bytes] = new_id

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        vocab_normalized = {}
        for token, idx in vocab.items():
            idx_int = int(idx)
            token_bytes = token if isinstance(token, bytes) else token.encode("utf-8")
            vocab_normalized[idx_int] = token_bytes

        merges_normalized = []
        for p1, p2 in merges:
            b1 = p1 if isinstance(p1, bytes) else p1.encode("utf-8")
            b2 = p2 if isinstance(p2, bytes) else p2.encode("utf-8")
            merges_normalized.append((b1, b2))

        return cls(vocab_normalized, merges_normalized, special_tokens)

    def pretokenize(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
            
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped_specials = [re.escape(st) for st in sorted_specials]
            pattern = "(" + "|".join(escaped_specials) + ")"
            chunks = [c for c in re.split(pattern, text) if c]
        else:
            chunks = [text]

        token_regex = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        pre_tokens = []
        
        for chunk in chunks:
            if chunk in self.special_tokens:
                pre_tokens.append(chunk)
            else:
                for match in token_regex.finditer(chunk):
                    pre_tokens.append(match.group())
                    
        return pre_tokens

    def encode(self, text):
        token_ids = []
        pre_tokens = self.pretokenize(text)
        
        for p in pre_tokens:
            if p in self.special_tokens:
                token_bytes = p.encode('utf-8')
                token_ids.append(self.byte_to_id[token_bytes])
            else:
                # Convert string to a list of single-byte objects
                word = [bytes([b]) for b in p.encode("utf-8")]
                
                while len(word) >= 2:
                    best_pair = None
                    best_rank = float('inf')
                    
                    for i in range(len(word) - 1):
                        pair = (word[i], word[i+1])
                        rank = self.merges_dict.get(pair)
                        if rank is not None and rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                            
                    if best_pair is None:
                        break
                        
                    new_chunk = best_pair[0] + best_pair[1]
                    word = merge(word, best_pair, new_chunk)
                    
                for chunk in word:
                    token_ids.append(self.byte_to_id[chunk])
                    
        return token_ids

    def encode_iterable(self, texts):
        for text in texts:
            yield from self.encode(text)

    def decode(self, token_ids):
        bytes_list = []
        for idx in token_ids:
            if idx in self.vocab:
                bytes_list.append(self.vocab[idx])
        
        encoded_stream = b"".join(bytes_list)
        return encoded_stream.decode("utf-8", errors="replace")

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    base_vocab = {i: bytes([i]) for i in range(256)}
    
    tokenizer = Tokenizer(vocab=base_vocab, merges={}, special_tokens=special_tokens)
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    pre_tokens = tokenizer.pretokenize(text)
    
    word_counts = Counter()
    for p in pre_tokens:
        if p not in tokenizer.special_tokens:
            word_counts[p.encode("utf-8")] += 1
            
    splits = {word: [bytes([b]) for b in word] for word in word_counts}
    pair_counts = defaultdict(int)
    
    for word, count in word_counts.items():
        split = splits[word]
        for i in range(len(split) - 1):
            pair_counts[(split[i], split[i+1])] += count
            
    merges = []
    
    num_merges = vocab_size - len(tokenizer.vocab)
    
    working_vocab = tokenizer.vocab.copy()
    
    for _ in range(num_merges):
        if not pair_counts:
            break
            
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        merges.append(best_pair)
        
        new_token_bytes = best_pair[0] + best_pair[1]
        
        new_id = len(working_vocab)
        working_vocab[new_id] = new_token_bytes
        
        for word, count in word_counts.items():
            split = splits[word]
            
            if best_pair[0] not in split or best_pair[1] not in split:
                continue 
                
            new_split = merge(split, best_pair, new_token_bytes)
            
            if len(new_split) != len(split):
                for i in range(len(split) - 1):
                    old_pair = (split[i], split[i+1])
                    pair_counts[old_pair] -= count
                    if pair_counts[old_pair] <= 0:
                        del pair_counts[old_pair]
                        
                for i in range(len(new_split) - 1):
                    pair_counts[(new_split[i], new_split[i+1])] += count
                    
                splits[word] = new_split
                
    return working_vocab, merges