#!/bin/zsh
# T1′ 통제 분석 병렬 드라이버. 로그 results/t1c_*.log, 완료 표식 results/t1c_done_*
set -u
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
NPERM=${1:-200}; NPOS=${2:-200}; NRES=${3:-20}
rm -f results/t1c_done_* results/t1c_position_perm_part*.csv
echo "START $(date)" > results/t1c_driver.log
$PY code/run_t1_controls.py features > results/t1c_features.log 2>&1 || { echo "FEATURES FAIL" >> results/t1c_driver.log; exit 1; }
echo "FEATURES OK $(date)" >> results/t1c_driver.log
# 워커 5개
$PY code/run_t1_controls.py runs --only main_heldout_trialcv_bal --perm $NPERM > results/t1c_w1.log 2>&1 &
$PY code/run_t1_controls.py runs --only main3_heldout_trialcv,heldout_trialcv_nobal,test_trialcv_bal,heldout_blockcv_bal,heldout_sessioncv_bal,heldout_trialcv_bal_wordcap20,heldout_wordcv_bal,heldout_trialcv_bal_center80,lar6_heldout_trialcv_bal > results/t1c_w2.log 2>&1 &
$PY code/run_t1_controls.py runs --only pooled_trialcv_bal,pooled3_trialcv,pooled_trialcv_bal_wordcap20,pooled_wordcv_bal,lar6_pooled_trialcv_bal,ref_allmod_pooled_randomcv_nobal,ref_allmod_pooled_trialcv_nobal > results/t1c_w3.log 2>&1 &
( $PY code/run_t1_controls.py resample --n-resample $NRES > results/t1c_w4.log 2>&1 ; $PY code/run_t1_controls.py posperm --perm-start 0 --perm-n $((NPOS/2)) >> results/t1c_w4.log 2>&1 ) &
$PY code/run_t1_controls.py posperm --perm-start $((NPOS/2)) --perm-n $((NPOS - NPOS/2)) > results/t1c_w5.log 2>&1 &
wait
echo "WORKERS DONE $(date)" >> results/t1c_driver.log
$PY code/run_t1_controls.py merge > results/t1c_merge.log 2>&1
echo "MERGE $? $(date)" >> results/t1c_driver.log
echo "T1C_ALL_DONE" >> results/t1c_driver.log
