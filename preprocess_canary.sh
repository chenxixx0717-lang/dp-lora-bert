#!/bin/bash

echo "============================================================"
echo "Preprocessing Canary Dataset with BPE Encoding"
echo "============================================================"

TRAIN_TSV="train_with_canary.tsv"
DEV_TSV="../AdaLoRA-main/NLU/glue_data/SST-2/dev.tsv"
TEST_TSV="../AdaLoRA-main/NLU/glue_data/SST-2/test.tsv"

PROCESSED_DIR="../glue_data/SST-2-canary-processed"
BIN_DIR="../glue_data/SST-2-canary-bin"
DICT_FILE="../glue_data/SST-2-bin/input0/dict.txt"

ENCODER_JSON="encoder.json"
VOCAB_BPE="vocab.bpe"

echo ""
echo "[0] Checking BPE files..."
if [ ! -f "$ENCODER_JSON" ]; then
    echo "   ✗ $ENCODER_JSON not found!"
    echo "   Run: wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'"
    exit 1
fi
if [ ! -f "$VOCAB_BPE" ]; then
    echo "   ✗ $VOCAB_BPE not found!"
    echo "   Run: wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'"
    exit 1
fi
echo "   ✓ BPE files exist"

echo ""
echo "[1] Creating processed directory..."
rm -rf "$PROCESSED_DIR"
mkdir -p "$PROCESSED_DIR"

echo ""
echo "[2] Extracting raw text..."
cut -f1 "$TRAIN_TSV" > "$PROCESSED_DIR/train.raw.input0"
tail -n +2 "$DEV_TSV" | cut -f1 > "$PROCESSED_DIR/dev.raw.input0"
tail -n +2 "$TEST_TSV" | cut -f1 > "$PROCESSED_DIR/test.raw.input0"

echo ""
echo "[3] BPE encoding train set..."
python multiprocessing_bpe_encoder.py \
  --encoder-json "$ENCODER_JSON" \
  --vocab-bpe "$VOCAB_BPE" \
  --inputs "$PROCESSED_DIR/train.raw.input0" \
  --outputs "$PROCESSED_DIR/train.input0" \
  --workers 20 \
  --keep-empty

if [ $? -ne 0 ]; then
    echo "   ✗ BPE encoding failed! Aborting."
    exit 1
fi

echo ""
echo "[4] BPE encoding dev set..."
python multiprocessing_bpe_encoder.py \
  --encoder-json "$ENCODER_JSON" \
  --vocab-bpe "$VOCAB_BPE" \
  --inputs "$PROCESSED_DIR/dev.raw.input0" \
  --outputs "$PROCESSED_DIR/dev.input0" \
  --workers 20 \
  --keep-empty

if [ $? -ne 0 ]; then
    echo "   ✗ BPE encoding failed! Aborting."
    exit 1
fi

echo ""
echo "[5] BPE encoding test set..."
python multiprocessing_bpe_encoder.py \
  --encoder-json "$ENCODER_JSON" \
  --vocab-bpe "$VOCAB_BPE" \
  --inputs "$PROCESSED_DIR/test.raw.input0" \
  --outputs "$PROCESSED_DIR/test.input0" \
  --workers 20 \
  --keep-empty

if [ $? -ne 0 ]; then
    echo "   ✗ BPE encoding failed! Aborting."
    exit 1
fi

echo ""
echo "[6] Verifying BPE encoding..."
first_raw=$(head -n 1 "$PROCESSED_DIR/train.raw.input0")
first_bpe=$(head -n 1 "$PROCESSED_DIR/train.input0")
echo "   Raw: $first_raw"
echo "   BPE: $first_bpe"

echo ""
echo "[7] Running fairseq preprocess for input0..."
rm -rf "$BIN_DIR/input0"
python preprocess.py \
  --only-source \
  --trainpref "$PROCESSED_DIR/train.input0" \
  --validpref "$PROCESSED_DIR/dev.input0" \
  --testpref "$PROCESSED_DIR/test.input0" \
  --destdir "$BIN_DIR/input0" \
  --srcdict "$DICT_FILE" \
  --dataset-impl mmap \
  --workers 60

echo ""
echo "[8] Extracting labels..."
cut -f2 "$TRAIN_TSV" > "$PROCESSED_DIR/train.label"
tail -n +2 "$DEV_TSV" | cut -f2 > "$PROCESSED_DIR/dev.label"
tail -n +2 "$TEST_TSV" | cut -f2 > "$PROCESSED_DIR/test.label"

echo ""
echo "[9] Verifying counts..."
train_text_count=$(wc -l < "$PROCESSED_DIR/train.input0")
train_label_count=$(wc -l < "$PROCESSED_DIR/train.label")
echo "   Train text: $train_text_count, Train labels: $train_label_count"
if [ "$train_text_count" -ne "$train_label_count" ]; then
    echo "   ✗ MISMATCH! Aborting."
    exit 1
fi
echo "   ✓ Counts match"

echo ""
echo "[10] Running fairseq preprocess for labels..."
rm -rf "$BIN_DIR/label"
mkdir -p "$BIN_DIR/label"

# 创建标签字典
cat > label_dict.txt << 'EOF'
0 1
1 1
EOF

python preprocess.py \
  --only-source \
  --trainpref "$PROCESSED_DIR/train.label" \
  --validpref "$PROCESSED_DIR/dev.label" \
  --testpref "$PROCESSED_DIR/test.label" \
  --destdir "$BIN_DIR/label" \
  --srcdict label_dict.txt \
  --dataset-impl mmap \
  --workers 60

echo ""
echo "============================================================"
echo "✓ Preprocessing complete!"
echo "   Data saved to: $BIN_DIR"
echo "============================================================"
