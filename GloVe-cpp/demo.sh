#!/bin/bash
set -e

# GloVe-cpp 演示脚本

echo "=========================================="
echo "GloVe-cpp Demo"
echo "=========================================="

BUILDDIR=build
CORPUS=text8
VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
SAVE_FILE=vectors

# 参数
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=50
MAX_ITER=15
WINDOW_SIZE=15
NUM_THREADS=8
X_MAX=10

# 1. 构建
echo ""
echo "Step 1: Building..."
mkdir -p "$BUILDDIR" && cd "$BUILDDIR" && cmake .. && make -j4 && cd ..

# 2. 下载语料
echo ""
echo "Step 2: Downloading corpus..."
if [ ! -e "$CORPUS" ]; then
    wget -q http://mattmahoney.net/dc/text8.zip && unzip -q text8.zip && rm text8.zip
fi

# 3. 词汇统计
echo ""
echo "Step 3: Building vocabulary..."
$BUILDDIR/bin/vocab_count -c $VOCAB_MIN_COUNT -v $VERBOSE < $CORPUS > $VOCAB_FILE

# 4. 共现矩阵
echo ""
echo "Step 4: Counting cooccurrences..."
$BUILDDIR/bin/cooccur -V $VOCAB_FILE -w $WINDOW_SIZE -m $MEMORY -v $VERBOSE < $CORPUS > $COOCCURRENCE_FILE

# 5. 打乱
echo ""
echo "Step 5: Shuffling..."
$BUILDDIR/bin/shuffle -m $MEMORY -v $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

# 6. 训练
echo ""
echo "Step 6: Training..."
$BUILDDIR/bin/glove -i $COOCCURRENCE_SHUF_FILE -V $VOCAB_FILE -o $SAVE_FILE \
    -d $VECTOR_SIZE -n $MAX_ITER -t $NUM_THREADS -x $X_MAX -v $VERBOSE

echo ""
echo "Done!"
