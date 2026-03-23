#!/bin/bash

TYPE=""
SIZE=""
OBS=""

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


text2workspace.py   "${CARD}"   -m 125     -o   "${WORKSPACE}"   -v 2 > text2workspace_log.txt

combine     -M Significance     -d "${WORKSPACE}"  -m 125 --X-rtd MINIMIZER_MaxCalls=999999999 --cminDefaultMinimizerPrecision 1E-12    --signif     -n .HHdijet_${TYPE}_sig_${SIZE}_${OBS}_${TAG}     2>&1 | tee significance_${TYPE}_${SIZE}_${OBS}_obs.log

combine     -M FitDiagnostics     -d "${WORKSPACE}"  --robustFit=1 -m 125 --setRobustFitTolerance 0.1 --X-rtd MINIMIZER_MaxCalls=999999999  --cminDefaultMinimizerPrecision 1E-12  -n .HHdijet_${TYPE}_combined_${SIZE}_${OBS}_${TAG} --rMin -50 --rMax 50  --cminDefaultMinimizerStrategy 0 --saveShapes --saveNormalizations --saveWithUncertainties 2>&1 | tee fitDiagnostics_${TYPE}_combined_${SIZE}_${OBS}_${TAG}.log
PostFitShapesFromWorkspace -w "${WORKSPACE}" --output postfit_${TYPE}_shapes_bkg_${SIZE}_${OBS}_${TAG}.root -f fitDiagnostics.HHdijet_${TYPE}_combined_${SIZE}_${OBS}_${TAG}.root:fit_b --postfit -d "${CARD}"
PostFitShapesFromWorkspace -w "${WORKSPACE}" --output postfit_${TYPE}_shapes_sig_${SIZE}_${OBS}_${TAG}.root -f fitDiagnostics.HHdijet_${TYPE}_combined_${SIZE}_${OBS}_${TAG}.root:fit_s --postfit -d "${CARD}"


