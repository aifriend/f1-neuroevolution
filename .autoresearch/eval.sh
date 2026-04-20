#!/bin/bash
# Autoresearch evaluation script for F1 neuroevolution
# Runs training 3 times and extracts metrics from best run
cd "$(dirname "$0")/.."

BEST_LAP=999
BEST_SCORE=0
AVG_PROGRESS=0
TOTAL_FINISHERS=0
TOTAL_TIME=0
RESTARTS=0

for i in 1 2 3; do
  OUTPUT=$(node train.js --gens 300 --cars 50 2>&1)

  # Extract metrics from final summary
  LAP=$(echo "$OUTPUT" | grep "Best lap:" | grep -oE '[0-9]+\.[0-9]+s' | head -1 | tr -d 's')
  SCORE=$(echo "$OUTPUT" | grep "Best score:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  TIME=$(echo "$OUTPUT" | grep "Done!" | grep -oE '[0-9]+\.[0-9]+s' | head -1 | tr -d 's')
  R=$(echo "$OUTPUT" | grep -c "RESTART")

  # Extract avg progress from last 50 gens
  AVG=$(echo "$OUTPUT" | grep "Gen " | tail -50 | grep -oE 'avg: *[0-9]+\.[0-9]+%' | sed 's/avg: *//;s/%//' | awk '{s+=$1; n++} END{if(n>0) print s/n; else print 0}')

  # Extract avg finishers from last 50 gens
  FIN=$(echo "$OUTPUT" | grep "Gen " | tail -50 | grep -oE 'fin:[0-9]+' | sed 's/fin://' | awk '{s+=$1; n++} END{if(n>0) print s/n; else print 0}')

  # Keep best across runs
  if [ -n "$SCORE" ] && (( $(echo "$SCORE > $BEST_SCORE" | bc -l) )); then
    BEST_SCORE=$SCORE
  fi
  if [ -n "$LAP" ] && (( $(echo "$LAP < $BEST_LAP" | bc -l) )); then
    BEST_LAP=$LAP
  fi

  RESTARTS=$((RESTARTS + R))
  if [ -n "$AVG" ]; then
    AVG_PROGRESS=$(echo "$AVG_PROGRESS + $AVG" | bc -l)
  fi
  if [ -n "$FIN" ]; then
    TOTAL_FINISHERS=$(echo "$TOTAL_FINISHERS + $FIN" | bc -l)
  fi
  if [ -n "$TIME" ]; then
    TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME" | bc -l)
  fi
done

# Average over 3 runs
AVG_PROGRESS=$(echo "$AVG_PROGRESS / 3" | bc -l)
AVG_FINISHERS=$(echo "$TOTAL_FINISHERS / 3" | bc -l)
AVG_TIME=$(echo "$TOTAL_TIME / 3" | bc -l)

echo "BEST_LAP=$BEST_LAP"
echo "BEST_SCORE=$BEST_SCORE"
echo "AVG_PROGRESS=$AVG_PROGRESS"
echo "AVG_FINISHERS=$AVG_FINISHERS"
echo "AVG_TIME=$AVG_TIME"
echo "RESTARTS=$RESTARTS"
