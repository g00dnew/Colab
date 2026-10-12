#!/bin/zsh
# 위치 라벨 순열(표집 평균) 8워커 + 20회 재표집 1워커. 인자: NPERM NDRAWS PART WORDCAP GROUP
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
NPERM=${1:-200}; ND=${2:-4}; PART=${3:-held_out}; WC=${4:-0}; GRP=${5:-trial}; W=8; PER=$((NPERM / W))
TAG="${PART}_wc${WC}_g${GRP}"
echo "POSPERM START $(date) $TAG" >> results/t1c_driver.log
$PY code/run_t1_controls.py resample --n-resample 20 --part $PART --wordcap $WC --group $GRP > results/t1c_rs_$TAG.log 2>&1 &
for i in $(seq 0 $((W-1))); do
  $PY code/run_t1_controls.py posperm --perm-start $((i*PER)) --perm-n $PER --n-draws $ND --part $PART --wordcap $WC --group $GRP > results/t1c_pp_${TAG}_$i.log 2>&1 &
done
wait
$PY code/run_t1_controls.py merge > results/t1c_merge.log 2>&1
echo "POSPERM_ALL_DONE $(date) $TAG merge=$?" >> results/t1c_driver.log
