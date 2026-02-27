import numpy as np
import pickle
from cs336_basics.bpe_tokenizer import train_bpe, Tokenizer

def prepare_custom_dataset():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    print(f"Training BPE tokenizer on {input_path}...")
    print("(This might take some time depending on your BPE implementation!)")
    
    # 1. Train the BPE model
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    # 2. Save the vocab and merges so generate.py can load the exact same tokenizer later
    print("Saving tokenizer vocab and merges to data/custom_bpe.pkl...")
    with open("data/custom_bpe.pkl", "wb") as f:
        pickle.dump({
            "vocab": vocab, 
            "merges": merges, 
            "special_tokens": special_tokens
        }, f)
        
    # 3. Instantiate the tokenizer and encode the text
    print("Tokenizing the dataset...")
    tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens):,}")
    
    # 4. Save to .npy
    token_array = np.array(tokens, dtype=np.uint16)
    np.save("data/tinystories_tokenized.npy", token_array)
    print("Saved custom tokenized dataset to data/tinystories_tokenized.npy")

if __name__ == "__main__":
    prepare_custom_dataset()