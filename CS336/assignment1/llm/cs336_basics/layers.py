from curses import noecho
from turtle import forward
import torch
import torch.nn as nn
import einx
import math

class linear(nn.Module):
    def __init__(self, in_features, out_features, device= None , dtype = None):  
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features,
            device=device, dtype=dtype))
        self._init_weight()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einx.dot("d_out d_in , ... d_in -> ... d_out" \
            , self.weight , x)

    def _init_weight(self):
        # Linear weights: N(μ=0, σ²=2/(din+dout)) truncated at [−3σ, 3σ]
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

class embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None , dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim,
            device=device, dtype=dtype))
        self._init_weight()

    def forward(self, token_ids: torch.Tensor) ->torch.Tensor:
        return self.weight[token_ids]

    def _init_weight(self):
        # Embedding: N(μ=0, σ²=1) truncated at [−3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
class rmsnorm(nn.Module):
    def __init__( self, d_model: int, eps: float = 1e-5, device = None, dtype = None ):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model , dtype = torch.float32))

    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (einx.mean("... ([d_model])", x ** 2) + self.eps).sqrt()
        return  (x / rms * self.weight).to(in_dtype)

    def _init_weight(self):
        self.weight = nn.Parameter(torch.ones(self.d_model , dtype = torch.float32))


class swiglu(nn.Module):
    def __init__(self, d_model: int , d_ff: int) -> None:
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))      # gate projection
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))      # down projection (输出)
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))      # up projection
        self._init_weight()

    def _init_weight(self):
        # w1, w3: fan_in = d_model, fan_out = d_ff
        std_1 = math.sqrt(2.0 / (self.d_model + self.d_ff))
        nn.init.trunc_normal_(self.w1, mean=0.0, std=std_1, a=-3*std_1, b=3*std_1)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=std_1, a=-3*std_1, b=3*std_1)
        # w2: fan_in = d_ff, fan_out = d_model
        std_2 = math.sqrt(2.0 / (self.d_ff + self.d_model))
        nn.init.trunc_normal_(self.w2, mean=0.0, std=std_2, a=-3*std_2, b=3*std_2)

    def forward(self, in_features):
        gate = einx.dot("d_ff d_model , ... d_model -> ... d_ff", self.w1 , in_features)
        upper = einx.dot("d_ff d_model , ... d_model -> ... d_ff", self.w3 , in_features)
        silu = gate * torch.sigmoid(gate)
        return einx.dot("d_model d_ff , ... d_ff -> ... d_model", self.w2 , silu * upper)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # 预计算旋转频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device = device, dtype = torch.float32) / d_k)) # inv_freq: (d_k // 2,)

        # 预计算所有位置的旋转角度
        positions = torch.arange(max_seq_len, device = device, dtype = torch.float32) # (max_seq_len,)
        
        angles = einx.multiply("pos, freq -> pos freq", positions, inv_freq) # (max_seq_len, d_k // 2)

        # 预计算 cos 和 sin
        self.register_buffer("cos_cached", angles.cos(), persistent=False) # (max_seq_len, d_k // 2)
        self.register_buffer("sin_cached", angles.sin(), persistent=False) # (max_seq_len, d_k // 2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        
        # 索引预计算的 cos/sin
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k // 2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k // 2)

        # 相邻配对分离（课程测试要求的方式）
        x1 = x[..., 0::2]  # 偶数索引: (..., seq_len, d_k // 2)
        x2 = x[..., 1::2]  # 奇数索引: (..., seq_len, d_k // 2)
        
        # 应用旋转
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # 交错合并
        return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)  # (..., seq_len, d_k)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    数值稳定的 softmax 实现
    """
    # 减去最大值防止 exp 溢出
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,  # (..., queries, d_k)
    K: torch.Tensor,  # (..., keys, d_k)
    V: torch.Tensor,  # (..., keys, d_v)
    mask: torch.Tensor | None = None,  # (..., queries, keys), True=attend
) -> torch.Tensor:
    """
    Scaled Dot-Product Attention
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """
    d_k = Q.shape[-1]
    
    # 计算注意力分数: Q @ K^T / sqrt(d_k)
    # Q: (..., queries, d_k), K: (..., keys, d_k)
    # scores: (..., queries, keys)
    scores = einx.dot("... queries d_k, ... keys d_k -> ... queries keys", Q, K) / (d_k ** 0.5)
    
    # 应用 mask: False 位置设为 -inf
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # softmax 归一化
    attn_weights = softmax(scores, dim=-1)  # (..., queries, keys)
    
    # 加权求和
    output = einx.dot("... queries keys, ... keys d_v -> ... queries d_v", attn_weights, V)
    
    return output


class MultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention
    """
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # d_k = d_v
        
        # 投影层权重
        self.q_proj = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.o_proj = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self._init_weight()

    def _init_weight(self):
        # All projections: d_model -> d_model
        std = math.sqrt(2.0 / (self.d_model + self.d_model))
        nn.init.trunc_normal_(self.q_proj, mean=0.0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.k_proj, mean=0.0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.v_proj, mean=0.0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.o_proj, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(
        self, 
        x: torch.Tensor,  # (..., seq_len, d_model)
        rope: RoPE | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return multihead_self_attention(
            x=x,
            q_proj_weight=self.q_proj,
            k_proj_weight=self.k_proj,
            v_proj_weight=self.v_proj,
            o_proj_weight=self.o_proj,
            num_heads=self.num_heads,
            rope=rope,
            token_positions=token_positions,
        )


def multihead_self_attention(
    x: torch.Tensor,           # (..., seq_len, d_model)
    q_proj_weight: torch.Tensor,  # (h * d_k = d_model, d_model) 
    k_proj_weight: torch.Tensor,  # (h * d_k = d_model, d_model)
    v_proj_weight: torch.Tensor,  # (h * d_v = d_model, d_model)
    o_proj_weight: torch.Tensor,  # (d_model, h * d_v = d_model)
    num_heads: int,
    rope: RoPE | None = None,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Causal Multi-Head Self-Attention
    """
    seq_len = x.shape[-2]
    d_model = x.shape[-1]
    d_k = d_model // num_heads
    
    # 线性投影 Q, K, V
    Q = einx.dot("... seq d_model, hdk d_model -> ... seq hdk", x, q_proj_weight)
    K = einx.dot("... seq d_model, hdk d_model -> ... seq hdk", x, k_proj_weight)
    V = einx.dot("... seq d_model, hdv d_model -> ... seq hdv", x, v_proj_weight)
    
    # reshape 成多头
    Q = einx.rearrange("... seq (h d_k) -> ... h seq d_k", Q, h=num_heads)
    K = einx.rearrange("... seq (h d_k) -> ... h seq d_k", K, h=num_heads)
    V = einx.rearrange("... seq (h d_v) -> ... h seq d_v", V, h=num_heads)
    
    # 如果有 RoPE，对 Q 和 K 应用
    if rope is not None and token_positions is not None:
        Q = rope(Q, token_positions)
        K = rope(K, token_positions)
    
    # 构造 causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
    causal_mask = ~causal_mask
    
    # SDPA
    attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    
    # reshape 回来
    attn_output = einx.rearrange("... h seq d_k -> ... seq (h d_k)", attn_output)
    
    # 输出投影
    output = einx.dot("... seq hdv, d_model hdv -> ... seq d_model", attn_output, o_proj_weight)
    
    return output


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block
    forward:
      y = x + MHA(RMSNorm(x))   (RoPE 只作用在 Q/K)
      z = y + FFN(RMSNorm(y))   (FFN = SwiGLU)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.ln1 = rmsnorm(d_model=d_model, eps=1e-5)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, device=device)
        self.ln2 = rmsnorm(d_model=d_model, eps=1e-5)
        self.ffn = swiglu(d_model=d_model, d_ff=d_ff)

        # RoPE 没有可学习参数，但需要 buffer，所以保留为 Module 成员
        head_dim = d_model // num_heads
        self.rope = RoPE(theta=theta, d_k=head_dim, max_seq_len=max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        # token_positions: (..., seq_len)

        seq_len = x.shape[-2]

        # 如果没有提供 token_positions，使用默认的连续位置
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm, rope=self.rope, token_positions=token_positions)
        y = x + attn_out

        y_norm = self.ln2(y)
        z = y + self.ffn(y_norm)
        return z


class TransformerLM(nn.Module):
    """
    Transformer Language Model
    Architecture:
      1. Token embedding
      2. N x TransformerBlock
      3. ln_final (RMSNorm)
      4. lm_head (linear projection to vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # Token embedding
        self.token_embeddings = embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_final = rmsnorm(d_model=d_model, eps=1e-5, device=device)

        # Output projection (lm_head)
        self.lm_head = linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
        )

    def forward(
        self,
        in_indices: torch.Tensor,  # (batch, seq_len)
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Token embedding
        x = self.token_embeddings(in_indices)  # (batch, seq_len, d_model)

        # 2. Transformer blocks
        for block in self.layers:
            x = block(x, token_positions=token_positions)

        # 3. Final layer norm
        x = self.ln_final(x)

        # 4. Output projection
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        return logits
