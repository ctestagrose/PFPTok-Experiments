#!/bin/bash
set -euo pipefail

TOKENIZER_TYPE="${1:?Usage: bash run_ablation.sh <unigram|bpe|pfptok> [quick|focused|comprehensive]}"
STUDY_SIZE="${2:-focused}"

if [[ ! "$TOKENIZER_TYPE" =~ ^(unigram|bpe|pfptok)$ ]]; then
    echo "Unknown tokenizer type: $TOKENIZER_TYPE  (choose between: unigram, bpe, pfptok)"
    exit 1
fi
if [[ ! "$STUDY_SIZE" =~ ^(quick|focused|comprehensive)$ ]]; then
    echo "Unknown study size: $STUDY_SIZE  (choose between: quick, focused, comprehensive)"
    exit 1
fi

# CONFIGURATION — edit these paths for your environment
SEQUENCE_DIR="./Sample_Data/Train"
TEST_SEQUENCE_DIR="./Sample_Data/Test"
TARGET_FILE="./Sample_Data/cryptic_targets_all.json"

OUTPUT_DIR="./ablation_results/${TOKENIZER_TYPE}_${STUDY_SIZE}_$(date +%Y%m%d_%H%M%S)"

NUM_SEQUENCES=""

case "${TOKENIZER_TYPE}_${STUDY_SIZE}" in

    # UNIGRAM
    unigram_quick)
        VOCAB_SIZES="10000 50000"
        MAX_LENGTHS="64 256"
        MODEL_TYPES="unigram"
        CHUNK_SIZES="50000"
        ;;
    unigram_focused)
        VOCAB_SIZES="5000 10000 50000 100000 200000 500000 1000000"
        MAX_LENGTHS="512"
        MODEL_TYPES="unigram"
        CHUNK_SIZES="1000 5000 10000 25000 50000"
        ;;
    unigram_comprehensive)
        VOCAB_SIZES="10000 50000 100000 200000 500000 1000000"
        MAX_LENGTHS="64 128 256 512"
        MODEL_TYPES="bpe unigram"
        CHUNK_SIZES="10000 20000 50000"
        ;;

    # BPE
    bpe_quick)
        VOCAB_SIZES="10000 50000"
        MIN_FREQUENCIES="1 5"
        MAX_TOTAL_CHARS="10000000 20000000"
        WINDOW_SIZES="10000 100000"
        STRIDE_SIZES="10000 100000"
        ;;
    bpe_focused)
        VOCAB_SIZES="5000 10000 50000 100000 200000 500000 1000000"
        MIN_FREQUENCIES="1 2 5 10 25"
        MAX_TOTAL_CHARS="20000000"
        WINDOW_SIZES="1000 5000 10000 25000 50000"
        STRIDE_SIZES="1000 5000 10000 25000 50000"
        ;;
    bpe_comprehensive)
        VOCAB_SIZES="5000 10000 50000 100000 200000 500000"
        MIN_FREQUENCIES="1 5 10 20"
        MAX_TOTAL_CHARS="10000000 20000000 50000000 100000000"
        WINDOW_SIZES="1000 5000 10000 50000 100000"
        STRIDE_SIZES="1000 5000 10000 50000 100000"
        NUM_SEQUENCES=10
        ;;

    # PFPTOK
    pfptok_quick)
        W_VALUES="5 10"
        D_VALUES="63 127 255"
        ;;
    pfptok_focused)
        W_VALUES="2000"
        D_VALUES="255 511 1021 4096"
        NUM_SEQUENCES=1000
        ;;
    pfptok_comprehensive)
        W_VALUES="3 5 10 20 50 75 100 250 500 750 1000 1500 2000"
        D_VALUES="31 63 127 255 511 1021 4096"
        NUM_SEQUENCES=10
        ;;
esac


CMD="python main_ablation.py \
    --tokenizer_type $TOKENIZER_TYPE \
    --sequence_dir \"$SEQUENCE_DIR\" \
    --test_sequence_dir \"$TEST_SEQUENCE_DIR\" \
    --target_file \"$TARGET_FILE\" \
    --output_dir \"$OUTPUT_DIR\" \
    --antibiotic RIF \
    --use_scaffolds \
    --Kmer_Size 31 \
    --stride 1 \
    --seed 42"

if [[ -n "$NUM_SEQUENCES" ]]; then
    CMD="$CMD --num_sequences $NUM_SEQUENCES"
fi

case "$TOKENIZER_TYPE" in
    unigram)
        CMD="$CMD \
            --vocab_sizes $VOCAB_SIZES \
            --max_sentencepiece_lengths $MAX_LENGTHS \
            --model_types $MODEL_TYPES \
            --chunk_sizes $CHUNK_SIZES"
        ;;
    bpe)
        CMD="$CMD \
            --vocab_sizes $VOCAB_SIZES \
            --min_frequencies $MIN_FREQUENCIES \
            --max_total_chars_list $MAX_TOTAL_CHARS \
            --window_sizes $WINDOW_SIZES \
            --stride_sizes $STRIDE_SIZES"
        ;;
    pfptok)
        CMD="$CMD \
            --w_values $W_VALUES \
            --d_values $D_VALUES"
        ;;
esac

# Run
echo "============================================================"
echo "  Ablation Study: ${TOKENIZER_TYPE} / ${STUDY_SIZE}"
echo "============================================================"
echo " * Started at: $(date)"
echo " * Output dir: $OUTPUT_DIR"
echo ""

eval "$CMD"
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo ""
    echo "${TOKENIZER_TYPE} ablation completed!"
    echo ""
    echo "Results:"
    echo " * JSON    -> $OUTPUT_DIR/ablation_results_${TOKENIZER_TYPE}.json"
    echo " * CSV     -> $OUTPUT_DIR/ablation_results_${TOKENIZER_TYPE}.csv"
    echo " * Summary -> $OUTPUT_DIR/summary_${TOKENIZER_TYPE}.txt"

    if [ -f "analyze_ablation_results.py" ]; then
        echo ""
        echo "Running analysis..."
        python analyze_ablation_results.py "$OUTPUT_DIR" || echo "Analysis failed (results still saved)"
    fi
else
    echo "ERROR: Ablation study failed"
    exit 1
fi

echo ""
echo "Finished at $(date)"