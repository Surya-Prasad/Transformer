import torch
import numpy as np
import time
import os

# Import all your custom components
from cs336_basics.modules import TransformerLM
from cs336_basics.optimizers import AdamW, gradient_clipping, LR_cosine_schedule
from tests.adapters import run_get_batch, run_cross_entropy, run_save_checkpoint

def train():
    vocab_size = 10000       # Match this to your BPE tokenizer vocab size
    context_length = 256
    d_model = 256
    num_layers = 4
    num_heads = 8
    d_ff = 1024
    
    batch_size = 32
    max_iters = 5000
    learning_rate = 5e-4
    min_lr = 1e-5
    warmup_iters = 100
    lr_decay_iters = 5000
    max_grad_norm = 1.0
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    dataset = np.load("data/tinystories_tokenized.npy") 
    # dataset = np.random.randint(0, vocab_size, size=(100000,)) # Dummy data for testing the loop
    
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    
    os.makedirs("checkpoints", exist_ok=True)
    t0 = time.time()
    
    for it in range(max_iters):
        lr = LR_cosine_schedule(it, learning_rate, min_lr, warmup_iters, lr_decay_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        x, y = run_get_batch(dataset, batch_size, context_length, device)
    
        logits = model(x)
        loss = run_cross_entropy(
            logits.view(-1, vocab_size), 
            y.view(-1)
        )
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        gradient_clipping(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        if it % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"Iter {it} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Time/10-steps: {dt:.2f}s")
            
        if it > 0 and it % 1000 == 0:
            checkpoint_path = f"checkpoints/ckpt_{it}.pt"
            run_save_checkpoint(model, optimizer, it, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train()