#!/bin/zsh
PY=/Users/hojaechoi/Documents/research/speech-bci-confusion/.venv/bin/python
cd /Users/hojaechoi/Documents/research/speech-bci-confusion/t12
LOG=results/step01.log; echo "START $(date '+%F %T')" > $LOG
$PY code/run_t1_stops_primary.py 100 >> $LOG 2>&1 && echo "T1STOPS OK $(date '+%T')" >> $LOG || echo "T1STOPS FAIL" >> $LOG
$PY code/run_stage1_timecourse.py >> $LOG 2>&1 && echo "TIMECOURSE OK $(date '+%T')" >> $LOG || echo "TIMECOURSE FAIL" >> $LOG
echo "STEP01_DONE $(date '+%F %T')" >> $LOG
