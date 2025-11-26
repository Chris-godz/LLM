import sys
import os
import pickle
import json
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import array
from tqdm import tqdm

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from cs336_basics.tokenizer import Tokenizer
from tests.common import gpt2_bytes_to_unicode

# Global tokenizer for workers
worker_tokenizer = None

def load_tokenizer(vocab_path, merges_path, special_tokens):
    print(f"Loading tokenizer from {vocab_path} and {merges_path}...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    merges = []
    # Create reverse mapping for pretty merges
    byte_encoder = gpt2_bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    
    with open(merges_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line: continue
            parts = line.split(' ')
            if len(parts) != 2: continue
            
            # Decode pretty chars back to bytes
            try:
                p1 = bytes([byte_decoder[c] for c in parts[0]])
                p2 = bytes([byte_decoder[c] for c in parts[1]])
                merges.append((p1, p2))
            except KeyError:
                # Fallback if not pretty printed or unknown char
                # This might happen if the file was saved without pretty printing
                # But assuming it was saved with the training script which tries to use it
                print(f"Warning: Failed to decode merge line: {line}")
                continue
            
    return Tokenizer(vocab, merges, special_tokens)

def init_worker(vocab_path, merges_path, special_tokens):
    global worker_tokenizer
    worker_tokenizer = load_tokenizer(vocab_path, merges_path, special_tokens)

def process_chunk(text):
    return worker_tokenizer.encode(text)

def process_file(input_path, output_path, vocab_path, merges_path, special_tokens, num_workers=None):
    if num_workers is None:
        # Reduce workers to save memory (each worker loads a full tokenizer)
        num_workers = max(1, (os.cpu_count() or 1) // 2)
    
    print(f"Processing {input_path} -> {output_path}")
    print(f"Using {num_workers} workers")
    
    file_size = os.path.getsize(input_path)
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")
    
    # Reduce chunk size to 1MB to lower memory pressure
    def line_chunk_generator(file_path, chunk_size_bytes=1*1024*1024): 
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            chunk = []
            size = 0
            for line in f:
                chunk.append(line)
                size += len(line)
                if size >= chunk_size_bytes:
                    yield "".join(chunk)
                    chunk = []
                    size = 0
            if chunk:
                yield "".join(chunk)

    chunk_gen = line_chunk_generator(input_path)
    
    # Estimate total chunks for progress bar
    estimated_chunks = file_size // (1*1024*1024)
    
    from collections import deque
    
    # Open output file for binary writing (append mode)
    # We will write raw bytes directly to disk to keep memory usage low
    with open(output_path, 'wb') as out_f:
        pass

    # Actually, let's write to a .bin file first
    bin_path = output_path.with_suffix('.bin')
    total_tokens = 0
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(vocab_path, merges_path, special_tokens)) as executor:
        futures = deque()
        max_queue_size = num_workers * 2
        
        pbar = tqdm(total=estimated_chunks, desc=f"Tokenizing {input_path.name}", unit="chunk")
        
        with open(bin_path, 'wb') as f_out:
            for chunk in chunk_gen:
                future = executor.submit(process_chunk, chunk)
                futures.append(future)
                
                if len(futures) >= max_queue_size:
                    oldest_future = futures.popleft()
                    tokens = oldest_future.result()
                    
                    # Write to disk immediately
                    np_arr = np.array(tokens, dtype=np.uint16)
                    f_out.write(np_arr.tobytes())
                    total_tokens += len(tokens)
                    
                    pbar.update(1)
            
            # Drain remaining
            while futures:
                oldest_future = futures.popleft()
                tokens = oldest_future.result()
                np_arr = np.array(tokens, dtype=np.uint16)
                f_out.write(np_arr.tobytes())
                total_tokens += len(tokens)
                pbar.update(1)
                
        pbar.close()
            
    print(f"Total tokens: {total_tokens}")
    print(f"Raw binary saved to {bin_path}")
    
    # Convert .bin to .npy (add header)
    # We can do this efficiently by reading the bin file as a numpy array and saving it
    # Since we know the size now.
    # But loading 3GB into memory to save might spike memory again.
    # Better approach: Construct the header manually and prepend it?
    # Or just use np.save on the memmap?
    
    print(f"Converting to .npy format at {output_path}...")
    try:
        # Load as memmap (zero memory usage)
        arr = np.memmap(bin_path, dtype=np.uint16, mode='r', shape=(total_tokens,))
        # Save to .npy (this will read from disk and write to disk, efficient)
        np.save(output_path, arr)
        # Delete raw bin
        os.remove(bin_path)
        print("Saved .npy successfully.")
    except Exception as e:
        print(f"Error converting to .npy: {e}")
        print(f"Keeping raw binary file: {bin_path}")
        print("You can load it with: np.fromfile('...', dtype=np.uint16)")

def main():
    data_dir = project_root / "data"
    scripts_dir = project_root / "scripts" / "tokenization"
    
    # 1. TinyStories
    ts_vocab = scripts_dir / "tinystory" / "vocab_tinystories.pkl"
    ts_merges = scripts_dir / "tinystory" / "merges_tinystories.txt"
    ts_special = ["<|endoftext|>"]
    
    ts_train_in = data_dir / "TinyStoriesV2-GPT4-train.txt"
    ts_train_out = scripts_dir / "TinyStoriesV2-GPT4-train.npy"
    
    ts_valid_in = data_dir / "TinyStoriesV2-GPT4-valid.txt"
    ts_valid_out = scripts_dir / "TinyStoriesV2-GPT4-valid.npy"
    
    # if ts_train_in.exists():
    #     process_file(ts_train_in, ts_train_out, ts_vocab, ts_merges, ts_special)
    # if ts_valid_in.exists():
    #     process_file(ts_valid_in, ts_valid_out, ts_vocab, ts_merges, ts_special)

    # 2. OpenWebText
    owt_vocab = scripts_dir / "openweb" / "vocab_owt.pkl"
    owt_merges = scripts_dir / "openweb" / "merges_owt.txt"
    owt_special = ["<|endoftext|>"]
    
    owt_train_in = data_dir / "owt_train.txt"
    owt_train_out = scripts_dir / "owt_train.npy"
    
    owt_valid_in = data_dir / "owt_valid.txt"
    owt_valid_out = scripts_dir / "owt_valid.npy"
    
    if owt_train_in.exists():
        process_file(owt_train_in, owt_train_out, owt_vocab, owt_merges, owt_special)
    if owt_valid_in.exists():
        process_file(owt_valid_in, owt_valid_out, owt_vocab, owt_merges, owt_special)

if __name__ == "__main__":
    main()
