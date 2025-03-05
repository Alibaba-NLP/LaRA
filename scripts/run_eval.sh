cd ../evaluation
# DELAY=28800
# sleep $DELAY

MODELS=("qwen2.5-7b-instruct")

QUERY_TYPES=("hallu" "location" "reasoning" "comp")
CONTEXT_TYPES=("financial" "paper" "book")
CONTEXT_LENGTHS=("32k" "128k")
SCRIPTS=("eval_full.py" "eval_rag.py" )

for MODEL in "${MODELS[@]}"; do
    for QUERY_TYPE in "${QUERY_TYPES[@]}"; do
        for CONTEXT_TYPE in "${CONTEXT_TYPES[@]}"; do
            for CONTEXT_LENGTH in "${CONTEXT_LENGTHS[@]}"; do
                for SCRIPT in "${SCRIPTS[@]}"; do
                    python $SCRIPT --query_type $QUERY_TYPE --context_type $CONTEXT_TYPE --context_length $CONTEXT_LENGTH --eval_model $MODEL
                done
            done
        done
    done
done
