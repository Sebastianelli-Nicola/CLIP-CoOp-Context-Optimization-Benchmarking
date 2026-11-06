#!/bin/bash
# Script per eseguire l'addestramento CoOp su tutti i livelli di few-shot (1, 2, 4, 8, 16)

DATASET=${1:-caltech101} # caltech101 o eurosat
CFG=rn50_ep50
CTX_POS=${2:-end}        # end o mid

SHOT_LIST=(1 2 4 8 16)

echo "=== Avvio benchmark Few-Shot CoOp su ${DATASET} (Posizione classe: ${CTX_POS}) ==="

for SHOTS in "${SHOT_LIST[@]}"
do
    echo "--- Training CoOp con ${SHOTS} shots per classe ---"
    bash scripts/coop/main.sh ${DATASET} ${CFG} ${CTX_POS} ${SHOTS}
done

echo "Benchmark completato per tutte le configurazioni di shot!"
