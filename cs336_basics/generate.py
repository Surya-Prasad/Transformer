import torch
import tiktoken
from cs336_basics.modules import TransformerLM
from tests.adapters import run_load_checkpoint

def generate(model, idx, max_new_tokens, context_length, temperature=1.0, top_k=None, top_p=None):
    """
    Takes a conditioning sequence of indices (idx) and completes the sequence 
    autoregressively, using temperature, top-k, and/or top-p (nucleus) sampling.
    """
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
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    vocab_size = 50257
    context_length = 256
    d_model = 256
    num_layers = 4
    num_heads = 8
    d_ff = 1024

    # 2. Initialize Model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff
    ).to(device)

    # 3. Load Checkpoint
    # Make sure this points to the latest checkpoint in your checkpoints/ folder
    checkpoint_path = "checkpoints/ckpt_4000.pt" 
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # We create a dummy optimizer just to satisfy your run_load_checkpoint signature
    dummy_optimizer = torch.optim.AdamW(model.parameters()) 
    run_load_checkpoint(checkpoint_path, model, dummy_optimizer)

    # 4. Setup Tokenizer & Prompt
    enc = tiktoken.get_encoding("gpt2")
    prompt = "Once upon a time"
    print(f"\nPrompt: '{prompt}'\n" + "-"*40)
    
    # Encode prompt and add batch dimension (1, seq_len)
    input_ids = enc.encode_ordinary(prompt)
    x = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 5. Generate Text
    y = generate(
        model=model, 
        idx=x, 
        max_new_tokens=100, # Number of tokens to generate
        context_length=context_length, 
        temperature=0.8,    # Lower = more deterministic, Higher = more creative
        top_k=10            # Only sample from the top 10 most likely next words
    )

    # 6. Decode and Print
    generated_text = enc.decode(y[0].tolist())
    print(generated_text)
    print("-" * 40)

if __name__ == "__main__":
    main()