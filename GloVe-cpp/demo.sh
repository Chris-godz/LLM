#!/usr/bin/env bash
set -euo pipefail

# =============================================
# GloVe-cpp 一键流程脚本
# 支持通过环境变量覆盖默认参数：
#   VECTOR_SIZE=100 MAX_ITER=10 ./demo.sh
# =============================================

echo "=========================================="
echo "GloVe-cpp Demo Pipeline"
echo "=========================================="

# -------- 可配置变量（可用环境变量覆盖） --------
BUILDDIR=${BUILDDIR:-build}
CORPUS=${CORPUS:-text8}
VOCAB_FILE=${VOCAB_FILE:-vocab.txt}
COOCCURRENCE_FILE=${COOCCURRENCE_FILE:-cooccurrence.bin}
COOCCURRENCE_SHUF_FILE=${COOCCURRENCE_SHUF_FILE:-cooccurrence.shuf.bin}
SAVE_FILE=${SAVE_FILE:-vectors}

VERBOSE=${VERBOSE:-2}
MEMORY=${MEMORY:-4.0}              # GB 用于排序/打乱缓冲区
VOCAB_MIN_COUNT=${VOCAB_MIN_COUNT:-5}
VECTOR_SIZE=${VECTOR_SIZE:-50}
MAX_ITER=${MAX_ITER:-15}
WINDOW_SIZE=${WINDOW_SIZE:-15}
NUM_THREADS=${NUM_THREADS:-8}
X_MAX=${X_MAX:-10}
ALPHA=${ALPHA:-0.75}
MODEL=${MODEL:-2}                  # 0=全部参数,1=词向量,2=词+上下文向量和
GRAD_CLIP=${GRAD_CLIP:-100.0}

# -------- 工具检查 --------
command -v uv >/dev/null 2>&1 || echo "[提示] 未检测到 uv，可手动安装: pip install uv"
command -v wget >/dev/null 2>&1 || { echo "需要 wget"; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "需要 unzip"; exit 1; }

# -------- 步骤 1: 构建 --------
printf "\n[1/7] Building (CMake + Make)...\n"
mkdir -p "$BUILDDIR"
cmake -S . -B "$BUILDDIR" >/dev/null
cmake --build "$BUILDDIR" --target all -j"$(nproc)"

# -------- 步骤 2: 下载语料 --------
printf "\n[2/7] Downloading corpus (text8)...\n"
if [ ! -s "$CORPUS" ]; then
    wget -q http://mattmahoney.net/dc/text8.zip -O text8.zip
    unzip -q text8.zip && rm text8.zip
fi

# -------- 步骤 3: 构建词表 --------
printf "\n[3/7] Building vocabulary...\n"
"$BUILDDIR"/bin/vocab_count -c "$VOCAB_MIN_COUNT" -v "$VERBOSE" < "$CORPUS" > "$VOCAB_FILE"

# -------- 步骤 4: 共现矩阵 --------
printf "\n[4/7] Counting cooccurrences...\n"
"$BUILDDIR"/bin/cooccur -V "$VOCAB_FILE" -w "$WINDOW_SIZE" -m "$MEMORY" -v "$VERBOSE" < "$CORPUS" > "$COOCCURRENCE_FILE"

# -------- 步骤 5: 打乱 --------
printf "\n[5/7] Shuffling cooccurrences...\n"
"$BUILDDIR"/bin/shuffle -m "$MEMORY" -v "$VERBOSE" < "$COOCCURRENCE_FILE" > "$COOCCURRENCE_SHUF_FILE"

# -------- 步骤 6: 训练 --------
printf "\n[6/7] Training GloVe model...\n"
"$BUILDDIR"/bin/glove \
    --input-file "$COOCCURRENCE_SHUF_FILE" \
    --vocab-file "$VOCAB_FILE" \
    --save-file "$SAVE_FILE" \
    --vector-size "$VECTOR_SIZE" \
    --iter "$MAX_ITER" \
    --threads "$NUM_THREADS" \
    --x-max "$X_MAX" \
    --alpha "$ALPHA" \
    --model "$MODEL" \
    --grad-clip "$GRAD_CLIP" \
    --verbose "$VERBOSE"

# -------- 步骤 7: 评估 --------
printf "\n[7/7] Evaluating vectors...\n"
if [ -f pyproject.toml ]; then
    uv sync >/dev/null 2>&1 || true
    uv run python eval/python/evaluate.py --vocab_file "$VOCAB_FILE" --vectors_file "${SAVE_FILE}.txt" || echo "[警告] 评估失败，检查 Python 依赖"
else
    echo "[跳过] 未找到 pyproject.toml，跳过自动评估"
fi

printf "\n✅ 全流程完成。词向量输出: %s.txt\n" "$SAVE_FILE"
