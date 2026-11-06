#!/bin/bash
# Script per l'inferenza Zero-Shot di CLIP con prompt manuali

DATASET=${1:-caltech101} # caltech101 o eurosat
CFG=rn50_ep50

echo "=== Esecuzione CLIP Zero-Shot su ${DATASET} ==="

# Invoca lo script zeroshot nativo di CoOp/Dassl
bash scripts/coop/zeroshot.sh ${DATASET} ${CFG}
