#!/bin/bash

INPUTS=("corpus/fomc_minutes" "corpus/press_conferences" "corpus/fed_speeches" )


for INPUT in "${INPUTS[@]}"
do
    # echo "Running classification on $INPUT"
    # python finbert_classification.py "$INPUT"

    echo "Running similarity on $INPUT"
    python finbert_similarity.py "$INPUT"
done