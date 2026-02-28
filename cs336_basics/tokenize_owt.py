import pickle
from cs336_basics.bpe_tokenizer import train_bpe

def train_and_serialize_owt(input_path: str, vocab_path: str, merges_path: str) -> dict[int, bytes]:
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    
    print(f"Starting BPE training on {input_path}...")
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    print("Serializing vocabulary and merges...")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
        
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
        
    return vocab

def print_longest_token(vocab: dict[int, bytes]) -> None:
    sorted_tokens = sorted(vocab.values(), key=len, reverse=True)
    longest_token_bytes = sorted_tokens[0]
    longest_token_str = longest_token_bytes.decode('utf-8', errors='replace')
    
    print(f"\nLongest token: '{longest_token_str}'")
    print(f"Length in bytes: {len(longest_token_bytes)}")

def main():
    # Ensure this points to your extracted OpenWebText file
    input_data_path = "data/owt_train.txt" 
    
    vocab = train_and_serialize_owt(
        input_path=input_data_path, 
        vocab_path="owt_vocab.pkl", 
        merges_path="owt_merges.pkl"
    )
    
    print_longest_token(vocab)

if __name__ == "__main__":
    main()