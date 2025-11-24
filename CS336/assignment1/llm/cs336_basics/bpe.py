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
        # Increase chunk count to reduce memory pressure per chunk
        boundaries = find_chunk_boundaries(f, num_processes * 20, split_token)
        
    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((input_path, start, end, special_tokens))
        
    # Parallel processing
    try:
        with Pool(processes=num_processes) as pool:
            # 使用 imap_unordered 并配合 tqdm 显示进度
            # imap_unordered 会在结果准备好时立即 yield，适合显示进度
            results_iter = pool.imap_unordered(_process_file_chunk, chunk_args)
            
            for local_freqs in tqdm(results_iter, total=len(chunk_args), desc="Pre-tokenizing & Counting", unit="chunk"):
                word_freqs.update(local_freqs)
    except Exception as e:
        print(f"Multiprocessing failed: {e}. Falling back to serial.")
        for args in chunk_args:
            local_freqs = _process_file_chunk(args)
            word_freqs.update(local_freqs)
    
    # Step 5 & 6: BPE 训练循环（使用倒排索引进行增量更新优化）
    merges = []
    num_merges_needed = vocab_size - len(vocab)
    
    # --- 优化数据结构初始化 ---
    # 将 word_freqs 转换为列表，方便索引
    # words: list[list[int]] (使用 list 因为需要原地修改)
    words_list = [list(w) for w in word_freqs.keys()]
    freqs_list = list(word_freqs.values())
    
    # 全局 Pair 计数
    pair_counts = defaultdict(int)
    # 倒排索引: pair -> set of word indices
    pair_to_word_indices = defaultdict(set)
    
    # 初始统计
    for word_idx, word in enumerate(words_list):
        freq = freqs_list[word_idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_word_indices[pair].add(word_idx)
            
    # 使用 tqdm 显示进度条
    pbar = tqdm(range(num_merges_needed), desc="Training BPE", unit="merge")
    for i in pbar:
        if not pair_counts:
            break
            
        # 1. 找到频率最高的 Pair
        # 优化：不需要每次都 max() 整个字典，但为了保持确定性 tie-breaking，
        # 我们还是需要遍历。对于几万个 pair，这通常不是瓶颈。
        # 瓶颈在于更新所有单词。
        
        max_count = max(pair_counts.values())
        candidates = [p for p in pair_counts if pair_counts[p] == max_count]
        most_frequent_pair = max(candidates, key=lambda p: (vocab[p[0]], vocab[p[1]]))
        
        # 获取对应的 bytes
        token1_bytes = vocab[most_frequent_pair[0]]
        token2_bytes = vocab[most_frequent_pair[1]]
        
        # 记录合并
        merges.append((token1_bytes, token2_bytes))
        
        # 创建新 token
        vocab[next_token_id] = token1_bytes + token2_bytes
        
        # 2. 增量更新：只处理包含该 Pair 的单词
        indices_to_update = pair_to_word_indices[most_frequent_pair]
        
        # 我们需要小心处理：一个单词中可能出现多次该 Pair
        # 例如 "A B A B" -> "AB AB"
        
        for word_idx in indices_to_update:
            word = words_list[word_idx]
            freq = freqs_list[word_idx]
            
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i+1]) == most_frequent_pair:
                    # 找到一个匹配！准备合并 word[i] 和 word[i+1]
                    
                    # --- 减少旧邻居的计数 ---
                    # 左邻居: (word[i-1], word[i])
                    if i > 0:
                        prev_pair = (word[i-1], word[i])
                        pair_counts[prev_pair] -= freq
                        if pair_counts[prev_pair] == 0:
                            del pair_counts[prev_pair]
                        # 注意：我们不急着从 pair_to_word_indices 删除，因为 set.remove 比较慢，
                        # 而且 lazy removal 也是可以的，只要 pair_counts 正确即可。
                        
                    # 右邻居: (word[i+1], word[i+2])
                    if i < len(word) - 2:
                        next_pair = (word[i+1], word[i+2])
                        pair_counts[next_pair] -= freq
                        if pair_counts[next_pair] == 0:
                            del pair_counts[next_pair]
                            
                    # --- 执行合并 ---
                    # 将 word[i] 替换为新 ID，删除 word[i+1]
                    word[i] = next_token_id
                    del word[i+1] # 这会改变后续索引，但我们不增加 i，所以下一次循环检查新的 word[i] (即新 token) 和它的新右邻居
                    
                    # --- 增加新邻居的计数 ---
                    # 新左邻居: (word[i-1], new_token)
                    if i > 0:
                        new_prev_pair = (word[i-1], word[i])
                        pair_counts[new_prev_pair] += freq
                        pair_to_word_indices[new_prev_pair].add(word_idx)
                        
                    # 新右邻居: (new_token, word[i+1])
                    # 注意：此时 word[i+1] 已经是原来的 word[i+2] 了
                    if i < len(word) - 1:
                        new_next_pair = (word[i], word[i+1])
                        pair_counts[new_next_pair] += freq
                        pair_to_word_indices[new_next_pair].add(word_idx)
                        
                else:
                    i += 1
                    
        # 清理：从索引中移除已合并的 pair
        del pair_counts[most_frequent_pair]
        del pair_to_word_indices[most_frequent_pair]
        
        next_token_id += 1
        
        # 更新进度条描述
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
    Legacy function, kept for compatibility if needed, but not used in optimized train_bpe.
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
