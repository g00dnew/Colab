#!/bin/zsh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
PY=/Users/hojaechoi/Documents/research/speech-bci-confusion/.venv/bin/python
cd /Users/hojaechoi/Documents/research/speech-bci-confusion/t12
LOG=results/rerun_stage2.log; echo "START $(date '+%F %T')" > $LOG
$PY code/run_stage2_final.py 100 seg >> $LOG 2>&1 && echo "STAGE2 OK $(date '+%T')" >> $LOG || echo "STAGE2 FAIL" >> $LOG
$PY code/run_stage2_t3.py >> $LOG 2>&1 && echo "T3 OK $(date '+%T')" >> $LOG || echo "T3 FAIL" >> $LOG
$PY code/run_stage2_t3_pairs.py >> $LOG 2>&1 && echo "T3PAIRS OK $(date '+%T')" >> $LOG || echo "T3PAIRS FAIL" >> $LOG
echo "STAGE2_CHAIN_DONE $(date '+%F %T')" >> $LOG
