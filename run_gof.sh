#!/bin/bash

TYPE=""
SIZE=""
OBS=""
VR=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            TYPE="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        --obs)
            OBS="$2"
            shift 2
            ;;
	--tag)
            TAG="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done


CARD="datacard_${TYPE}_combined_${SIZE}_${OBS}_${TAG}.txt"
WORKSPACE="workspace_${TYPE}_combined_${SIZE}_${OBS}_${TAG}.root"

# All channels active in the fit
FIT_MASKS="mask_SR_SR=0,mask_SR_CR1=0,mask_SR_CR2=0,mask_SR_CR3=0,mask_VR1_SR=0,mask_VR1_CR1=0,mask_VR1_CR2=0,mask_VR1_CR3=0"
#FIT_MASKS="mask_SR3_SR=0,mask_SR3_CR1=0,mask_SR3_CR2=0,mask_SR3_CR3=0,mask_SR2_SR=0,mask_SR2_CR1=0,mask_SR2_CR2=0,mask_SR2_CR3=0,mask_VR1_SR=0,mask_VR1_CR1=0,mask_VR1_CR2=0,mask_VR1_CR3=0"

# Only SR_SR and VR1_SR active in the GOF evaluation
EVAL_MASKS="mask_SR_SR=0,mask_SR_CR1=1,mask_SR_CR2=1,mask_SR_CR3=1,mask_VR1_SR=0,mask_VR1_CR1=1,mask_VR1_CR2=1,mask_VR1_CR3=1"
#EVAL_MASKS="mask_SR3_SR=0,mask_SR3_CR1=1,mask_SR3_CR2=1,mask_SR3_CR3=1,mask_SR2_SR=0,mask_SR2_CR1=1,mask_SR2_CR2=1,mask_SR2_CR3=1,mask_VR1_SR=0,mask_VR1_CR1=1,mask_VR1_CR2=1,mask_VR1_CR3=1"
# Build workspace with channel masks
text2workspace.py "${CARD}" \
    -m 125 \
    --channel-masks \
    -o "${WORKSPACE}" \
    -v 2 > text.txt

# -------------------------
# S+B GOF
# -------------------------

# Observed
combine -M GoodnessOfFit "${WORKSPACE}" \
    --algo saturated \
    -n .HHdijet_${TYPE}_gof_sb_${SIZE}_${OBS}_${TAG} \
    --mass 125.0 \
    --setParametersForFit "${FIT_MASKS}" \
    --setParametersForEval "${EVAL_MASKS}" \
    --cminDefaultMinimizerStrategy 0 \
    --cminDefaultMinimizerPrecision 1E-12

# Toys
combine -M GoodnessOfFit "${WORKSPACE}" \
    --algo saturated \
    -t 500 \
    -n .HHdijet_${TYPE}_gof_sb_toys_${SIZE}_${OBS}_${TAG} \
    -s 1234 \
    --mass 125.0 \
    --toysFrequentist \
    --setParametersForFit "${FIT_MASKS}" \
    --setParametersForEval "${EVAL_MASKS}" \
    --cminDefaultMinimizerStrategy 0 \
    --cminDefaultMinimizerPrecision 1E-12

# -------------------------
# B-only GOF
# -------------------------

# Observed
combine -M GoodnessOfFit "${WORKSPACE}" \
    --algo saturated \
    -n .HHdijet_${TYPE}_gof_b_${SIZE}_${OBS}_${TAG} \
    --mass 125.0 \
    --freezeParameters r \
    --setParameters r=0 \
    --setParametersForFit "${FIT_MASKS}" \
    --setParametersForEval "${EVAL_MASKS}" \
    --cminDefaultMinimizerStrategy 0 \
    --cminDefaultMinimizerPrecision 1E-12

# Toys
combine -M GoodnessOfFit "${WORKSPACE}" \
    --algo saturated \
    -t 5 \
    -n .HHdijet_${TYPE}_gof_b_toys_${SIZE}_${OBS}_${TAG} \
    -s 1234 \
    --mass 125.0 \
    --toysFrequentist \
    --freezeParameters r \
    --setParameters r=0 \
    --setParametersForFit "${FIT_MASKS}" \
    --setParametersForEval "${EVAL_MASKS}" \
    --cminDefaultMinimizerStrategy 0 \
    --cminDefaultMinimizerPrecision 1E-12

# -------------------------
# Collect results
# -------------------------

combineTool.py -M CollectGoodnessOfFit \
    --input \
    higgsCombine.HHdijet_${TYPE}_gof_sb_${SIZE}_${OBS}_${TAG}.GoodnessOfFit.mH125.root \
    higgsCombine.HHdijet_${TYPE}_gof_sb_toys_${SIZE}_${OBS}_${TAG}.GoodnessOfFit.mH125.1234.root \
    --mass 125.0 \
    -o gof_sig_${TYPE}_${SIZE}_${OBS}_${TAG}.json

combineTool.py -M CollectGoodnessOfFit \
    --input \
    higgsCombine.HHdijet_${TYPE}_gof_b_${SIZE}_${OBS}_${TAG}.GoodnessOfFit.mH125.root \
    higgsCombine.HHdijet_${TYPE}_gof_b_toys_${SIZE}_${OBS}_${TAG}.GoodnessOfFit.mH125.1234.root \
    --mass 125.0 \
    -o gof_bkg_${TYPE}_${SIZE}_${OBS}_${TAG}.json

# -------------------------
# Plot
# -------------------------

python3 plot_gof.py gof_sig_${TYPE}_${SIZE}_${OBS}_${TAG}.json \
    -o gof_sig_${TYPE}_${SIZE}_${OBS}_${TAG} \
    --title-right="signal+background"

python3 plot_gof.py gof_bkg_${TYPE}_${SIZE}_${OBS}_${TAG}.json \
    -o gof_bkg_${TYPE}_${SIZE}_${OBS}_${TAG} \
    --title-right="background-only"
