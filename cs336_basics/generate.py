import torch
import pickle
from cs336_basics.modules import TransformerLM
from cs336_basics.bpe_tokenizer import Tokenizer
from tests.adapters import run_load_checkpoint

def generate(model, idx, max_new_tokens, context_length, temperature=1.0, top_k=None, top_p=None):
    model.eval()
    
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generating on device: {device}")

    vocab_size = 10000
    context_length = 256
    d_model = 256
    num_layers = 4
    num_heads = 8
    d_ff = 1024

    print("Loading custom BPE tokenizer...")
    with open("data/custom_bpe.pkl", "rb") as f:
        bpe_data = pickle.load(f)
        
    tokenizer = Tokenizer(
        vocab=bpe_data["vocab"], 
        merges=bpe_data["merges"], 
        special_tokens=bpe_data["special_tokens"]
    )

    model = TransformerLM(
        vocab_size=vocab_size, 
        context_length=context_length,
        d_model=d_model, 
        num_layers=num_layers,
        num_heads=num_heads, 
        d_ff=d_ff, 
        rope_theta=10000.0
    ).to(device)

    checkpoint_path = "checkpoints/ckpt_4000.pt" 
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    dummy_optimizer = torch.optim.AdamW(model.parameters()) 
    run_load_checkpoint(checkpoint_path, model, dummy_optimizer)

    prompt = "Once upon a time, there was a little"
    print(f"\nPrompt: '{prompt}'\n" + "-"*40)
    
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long).to(device)

    y = generate(
        model=model, 
        idx=x, 
        max_new_tokens=150, 
        context_length=context_length, 
        temperature=0.8,
        top_k=None,   
        top_p=0.9     
    )

    generated_text = tokenizer.decode(y[0].tolist())
    print(generated_text)
    print("-" * 40)

if __name__ == "__main__":
    main()