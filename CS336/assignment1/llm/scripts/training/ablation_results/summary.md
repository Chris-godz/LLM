# Ablation Study Summary

| Experiment | Min Val Loss | Step | Final Val Loss | Description |
|------------|--------------|------|----------------|-------------|
| baseline | 1.2746 | 19000 | 1.2842 | Baseline: Pre-norm, RMSNorm, RoPE, SwiGLU |
| ablation_no_rmsnorm | 176387076740114932329086976.0000 | 500 | nan | Remove RMSNorm |
| ablation_post_norm | 1.3145 | 19500 | 1.3202 | Post-norm Transformer |
| ablation_no_pos_emb | 1.3353 | 19500 | 1.3449 | No Position Embeddings (NoPE) |
| ablation_silu | 1.2872 | 20000 | 1.2872 | SiLU FFN (d_ff adjusted to 2048) |