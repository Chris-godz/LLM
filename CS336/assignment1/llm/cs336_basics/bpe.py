"""
BPE (Byte Pair Encoding) Tokenizer Implementation

This module implements a BPE tokenizer from scratch, including:
1. Training: Learn merges from a corpus
2. Encoding: Convert text to token IDs
3. Decoding: Convert token IDs back to text
"""

from __future__ import annotations

import os
from typing import Iterator
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
import regex as re
from tqdm import tqdm
from .pretokenization import find_chunk_boundaries

# 全局编译 Regex Pattern (供多进程调用)
# 来源: github.com/openai/tiktoken/pull/234/files
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def _pre_tokenize_chunk(text_chunk: str) -> list[tuple[int, ...]]:
    """
    辅助函数：对单个文本块进行正则切分，并转换为字节元组。
    用于多进程并行处理。
    
    Args:
        text_chunk: 文本块
        
    Returns:
        字节元组列表，每个元组代表一个 pre-token
    """
    if not text_chunk:
        return []
    
    # 使用全局编译的 regex 进行切分
    words = GPT2_PAT.findall(text_chunk)
    
    # 将每个单词转为 byte tuple，例如 "dog" -> (100, 111, 103)
    return [tuple(w.encode("utf-8")) for w in words]


def _process_file_chunk(args):
    """
    Helper function to process a file chunk in a separate process.
    Reads the chunk, handles special tokens, and counts word frequencies.
    """
    input_path, start, end, special_tokens = args
    
    with open(input_path, 'rb') as f:
        f.seek(start)
        bytes_chunk = f.read(end - start)
        # Decode with ignore to handle potential boundary issues
        text_chunk = bytes_chunk.decode('utf-8', errors='ignore')
    
    # Handle special tokens splitting within the chunk
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        sub_chunks = re.split(pattern, text_chunk)
    else:
        sub_chunks = [text_chunk]
        
    local_freqs = Counter()
    for sub in sub_chunks:
        if sub:
            words = _pre_tokenize_chunk(sub)
            local_freqs.update(words)
    return local_freqs


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train BPE tokenizer with GPT-2 style pre-tokenization.
    
    Algorithm:
    1. Initialize vocabulary with all 256 byte values (0-255)
    2. Add special tokens to vocabulary
    3. Pre-tokenize corpus using GPT-2 regex pattern
    4. Build word frequency dictionary
    5. Iteratively find most frequent byte pair and merge them
    6. Repeat until vocabulary reaches desired size
    """
    special_tokens = special_tokens or []
    
    # Step 1: 初始化基础词汇表 (256 个字节)
    vocab = {i: bytes([i]) for i in range(256)}
    
    # Step 2: 注册特殊 token，但它们不参与 BPE merge 计算，只是占据 ID，以便后续 Tokenizer 使用
    next_token_id = 256
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode('utf-8')
        next_token_id += 1
    
    # Step 3 & 4: 读取语料并构建词频字典 (Memory Efficient)
    word_freqs = Counter()
    
    # Determine split token for chunking
    # If special tokens exist, use the first one to ensure we don't split it.
    # Otherwise use newline as a safe boundary.
    split_token = special_tokens[0].encode('utf-8') if special_tokens else b'\n'
    
    num_processes = max(1, cpu_count() - 1)
    
    with open(input_path, 'rb') as f:
        # Find safe boundaries to split the file
        # We ask for more chunks than processes to balance load
        boundaries = find_chunk_boundaries(f, num_processes * 4, split_token)
        
    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((input_path, start, end, special_tokens))
        
    # Parallel processing
    try:
        with Pool(processes=num_processes) as pool:
            results = pool.map(_process_file_chunk, chunk_args)
            for local_freqs in results:
                word_freqs.update(local_freqs)
    except Exception as e:
        print(f"Multiprocessing failed: {e}. Falling back to serial.")
        for args in chunk_args:
            local_freqs = _process_file_chunk(args)
            word_freqs.update(local_freqs)
    
    # Step 5 & 6: BPE 训练循环（在词频字典上操作，而不是整个语料）
    merges = []
    num_merges_needed = vocab_size - len(vocab)
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(range(num_merges_needed), desc="Training BPE", unit="merge")
    for i in pbar:
        # 统计 Pair 频率（加权）
        pair_counts = defaultdict(int)
        for word, freq in word_freqs.items():
            for j in range(len(word) - 1):
                pair_counts[(word[j], word[j + 1])] += freq
        
        if not pair_counts:
            break  # 没有更多可合并的 pair
        
        # 找到频率最高的 Pair
        # 当频率相同时，按字节表示的字典序排序（选择字典序较大的）
        max_count = max(pair_counts.values())
        candidates = [p for p in pair_counts if pair_counts[p] == max_count]
        # 按 (vocab[p[0]] + vocab[p[1]]) 排序，选择字典序最大的
        most_frequent_pair = max(candidates, key=lambda p: (vocab[p[0]], vocab[p[1]]))
        
        # 获取对应的 bytes
        token1_bytes = vocab[most_frequent_pair[0]]
        token2_bytes = vocab[most_frequent_pair[1]]
        
        # 记录合并
        merges.append((token1_bytes, token2_bytes))
        
        # 创建新 token
        vocab[next_token_id] = token1_bytes + token2_bytes
        
        # 更新词频字典：合并所有包含该 pair 的单词
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            # 在这个单词中合并 pair
            new_word = _merge_word(word, most_frequent_pair, next_token_id)
            new_word_freqs[new_word] = new_word_freqs.get(new_word, 0) + freq
        
        word_freqs = new_word_freqs
        next_token_id += 1
        
        # 更新进度条描述，显示当前词表大小
        if i % 100 == 0:
            pbar.set_postfix({"vocab_size": len(vocab)})
            
    pbar.close()
    
    return vocab, merges


def _merge_word(
    word: tuple[int, ...],
    pair: tuple[int, int],
    new_token_id: int
) -> tuple[int, ...]:
    """
    在单个单词（token tuple）中合并指定的 pair
    
    Args:
        word: 单词的 token ID 序列 (tuple)
        pair: 要合并的 pair
        new_token_id: 新 token 的 ID
    
    Returns:
        合并后的单词 (tuple)
    """
    if len(word) < 2:
        return word
    
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            new_word.append(new_token_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    
    return tuple(new_word)
