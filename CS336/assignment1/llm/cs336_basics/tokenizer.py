import regex as re
from typing import List, Dict, Tuple, Optional, Iterable, Iterator

# 复用训练时的正则模式
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class Tokenizer:
    def __init__(
        self, 
        vocab: Dict[int, bytes], 
        merges: List[Tuple[bytes, bytes]], 
        special_tokens: Optional[List[str]] = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # 构建反向词表：bytes -> id (用于 encode)
        self.vocab_inverse = {v: k for k, v in vocab.items()}
        
        # 为了加速 encode，我们可以把 merges 列表转为字典，记录 pair -> rank (优先级)
        # 优先级越小越先合并
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}
        
        # 预编译特殊 token 的正则，用于 encode 时切分
        self.special_token_pattern = None
        if self.special_tokens:
            # 按长度降序排序，确保最长匹配优先（解决重叠问题）
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern_str = "|".join(re.escape(tok) for tok in sorted_special_tokens)
            self.special_token_pattern = re.compile(f"({pattern_str})")

    def _encode_word(self, word_bytes: bytes) -> List[int]:
        """
        核心 BPE 编码逻辑：对单个单词应用合并规则
        """
        # 初始状态：每个字节都是一个单独的 token
        word_tokens = [bytes([b]) for b in word_bytes]
        
        if len(word_tokens) < 2:
            return [self.vocab_inverse[word_tokens[0]]]

        while len(word_tokens) >= 2:
            # 1. 找出当前单词中所有相邻的 pair
            pairs = [(word_tokens[i], word_tokens[i+1]) for i in range(len(word_tokens)-1)]
            
            # 2. 找出这些 pair 中，在 merges 列表中排名最靠前（优先级最高）的一个
            # 如果所有 pair 都不在 merges 里，说明没法再合并了
            bigram = min(pairs, key=lambda p: self.merges_rank.get(p, float('inf')))
            
            if bigram not in self.merges_rank:
                break # 没有可用的合并规则了
            
            # 3. 执行合并：将找到的 bigram 替换为合并后的 token
            first, second = bigram
            new_word_tokens = []
            i = 0
            while i < len(word_tokens):
                # 检查是否匹配我们要合并的 bigram
                if i < len(word_tokens) - 1 and word_tokens[i] == first and word_tokens[i+1] == second:
                    new_word_tokens.append(first + second)
                    i += 2
                else:
                    new_word_tokens.append(word_tokens[i])
                    i += 1
            word_tokens = new_word_tokens
            
        # 将最终的 bytes 列表转换为 ID
        return [self.vocab_inverse[token] for token in word_tokens]

    def encode(self, text: str) -> List[int]:
        ids = []
        
        # 1. 处理特殊 Token
        if self.special_token_pattern:
            chunks = self.special_token_pattern.split(text)
        else:
            chunks = [text]
            
        for chunk in chunks:
            if not chunk:
                continue
                
            # 如果是特殊 Token，直接查找 ID
            if chunk in self.special_tokens:
                if chunk.encode('utf-8') in self.vocab_inverse:
                    ids.append(self.vocab_inverse[chunk.encode('utf-8')])
                else:
                    # 理论上不应该发生，除非 vocab 里没加特殊 token
                    print(f"Warning: Special token {chunk} not in vocab")
                continue
                
            # 2. 普通文本：GPT-2 Pre-tokenization
            words = GPT2_PAT.findall(chunk)
            
            for word in words:
                # 3. 对每个单词进行 BPE 编码
                word_bytes = word.encode('utf-8')
                ids.extend(self._encode_word(word_bytes))
                
        return ids

    def decode(self, ids: List[int]) -> str:
        # 1. 将 ID 映射回 bytes
        # 遇到未知 ID (Robustness) 可以跳过或报错，这里假设都在 vocab 里
        byte_parts = []
        for i in ids:
            if i in self.vocab:
                byte_parts.append(self.vocab[i])
            else:
                print(f"Warning: ID {i} not in vocab during decode")
                pass
                
        # 2. 拼接所有 bytes
        full_bytes = b"".join(byte_parts)
        
        # 3. 解码为字符串，错误字符用 U+FFFD () 替换
        return full_bytes.decode('utf-8', errors='replace')

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)