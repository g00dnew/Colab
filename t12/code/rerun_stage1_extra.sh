#!/bin/zsh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
PY=/Users/hojaechoi/Documents/research/speech-bci-confusion/.venv/bin/python
cd /Users/hojaechoi/Documents/research/speech-bci-confusion/t12
LOG=results/rerun_stage1_extra.log; echo "START $(date '+%T')" > $LOG
$PY code/run_stage1_split.py >> $LOG 2>&1 && echo "SPLIT OK $(date '+%T')" >> $LOG || echo "SPLIT FAIL" >> $LOG
$PY code/run_stage1_onset.py >> $LOG 2>&1 && echo "ONSET OK $(date '+%T')" >> $LOG || echo "ONSET FAIL" >> $LOG
echo "EXTRA_DONE $(date '+%T')" >> $LOG
