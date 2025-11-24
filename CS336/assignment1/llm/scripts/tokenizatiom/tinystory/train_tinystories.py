import sys
import os
import pickle
from pathlib import Path

# Add the project root to sys.path so we can import cs336_basics
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from cs336_basics.bpe import train_bpe
# Try to import the byte encoder for pretty printing merges, if available
try:
    from tests.common import gpt2_bytes_to_unicode
    byte_encoder = gpt2_bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    USE_PRETTY_MERGES = True
except ImportError:
    USE_PRETTY_MERGES = False
    print("Could not import gpt2_bytes_to_unicode, saving merges as raw bytes repr.")

def save_merges(merges, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            if USE_PRETTY_MERGES:
                # Convert bytes to the GPT-2 unicode representation
                s1 = "".join([byte_encoder[b] for b in p1])
                s2 = "".join([byte_encoder[b] for b in p2])
                f.write(f"{s1} {s2}\n")
            else:
                f.write(f"{p1} {p2}\n")

def main():
    data_dir = project_root / "data"
    input_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
    vocab_path = "vocab_tinystories.pkl"
    merges_path = "merges_tinystories.txt"
    
    print(f"Training Tokenizer on {input_path}...")
    print("Target Vocab Size: 10,000")
    
    print("Starting BPE training (this may take a while)...")
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    
    print(f"Training complete. Vocab size: {len(vocab)}, Merges: {len(merges)}")
    
    # Save Vocab
    print(f"Saving vocab to {vocab_path}...")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
        
    # Save Merges
    print(f"Saving merges to {merges_path}...")
    save_merges(merges, merges_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
