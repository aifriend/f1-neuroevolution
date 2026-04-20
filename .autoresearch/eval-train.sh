#!/bin/bash
# Autoresearch eval: run training + check 5 binary criteria
# Returns score as count of passed criteria (0-5)
set -e
cd "$(dirname "$0")/.."

PASS=0
FAIL=0

# Run training (1000 gens)
OUTPUT=$(node train.js --track monaco --cars 80 --gens 1000 --timeout 3000 2>&1)

# Criterion 1: First lap within 50 gens
FIRST_FIN=$(echo "$OUTPUT" | grep "fin:" | head -1 | grep -oE "Gen +[0-9]+" | grep -oE "[0-9]+")
if [ -n "$FIRST_FIN" ] && [ "$FIRST_FIN" -le 50 ] 2>/dev/null; then
  echo "C1 PASS: first lap at gen $FIRST_FIN (<=50)"
  PASS=$((PASS+1))
else
  echo "C1 FAIL: first lap at gen ${FIRST_FIN:-none}"
  FAIL=$((FAIL+1))
fi

# Criterion 2: Sub-2.0s lap in first 300 gens
# Check best lap at gen 300 (look at lines up to gen 300)
BEST_300=$(echo "$OUTPUT" | grep -E "Gen +[0-9]+" | while read line; do
  GEN=$(echo "$line" | grep -oE "Gen +[0-9]+" | grep -oE "[0-9]+")
  if [ "$GEN" -le 300 ] 2>/dev/null; then
    BEST=$(echo "$line" | grep -oE "best: +[0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
    if [ -n "$BEST" ]; then echo "$BEST"; fi
  fi
done | sort -n | head -1)

if [ -n "$BEST_300" ] && (( $(echo "$BEST_300 < 2.0" | bc -l 2>/dev/null || echo 0) )); then
  echo "C2 PASS: best lap ${BEST_300}s at gen <=300"
  PASS=$((PASS+1))
else
  echo "C2 FAIL: best lap at gen 300 = ${BEST_300:-none}s"
  FAIL=$((FAIL+1))
fi

# Criterion 3: Curriculum reaches Lv2 (at least 2 escalations)
ESC_COUNT=$(echo "$OUTPUT" | grep -c "ESCALATION" || true)
if [ "$ESC_COUNT" -ge 2 ]; then
  echo "C3 PASS: $ESC_COUNT escalations"
  PASS=$((PASS+1))
else
  echo "C3 FAIL: only $ESC_COUNT escalations"
  FAIL=$((FAIL+1))
fi

# Criterion 4: Healthy finisher rate (avg 3+ in last 100 gen lines with finishers)
AVG_FIN=$(echo "$OUTPUT" | grep "Gen " | tail -100 | grep -oE "fin:[0-9]+" | sed 's/fin://' | awk '{s+=$1; n++} END{if(n>0) printf "%.1f", s/n; else print "0"}')
if (( $(echo "$AVG_FIN >= 3.0" | bc -l 2>/dev/null || echo 0) )); then
  echo "C4 PASS: avg finishers $AVG_FIN"
  PASS=$((PASS+1))
else
  echo "C4 FAIL: avg finishers $AVG_FIN"
  FAIL=$((FAIL+1))
fi

# Criterion 5: Integration tests pass
if node test-integration.js > /dev/null 2>&1; then
  echo "C5 PASS: tests pass"
  PASS=$((PASS+1))
else
  echo "C5 FAIL: tests failed"
  FAIL=$((FAIL+1))
fi

echo ""
echo "SCORE: $PASS/5 (pass=$PASS fail=$FAIL)"
