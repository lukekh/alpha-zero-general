#!/bin/sh
# Run from the repository root, with the pinned smoke requirements installed.
set -eu
BENCH_PY=${BENCH_PY:-python3.11}
BENCH_ROOT=${BENCH_ROOT:-checkpoints/issue14-reproduction}
mkdir -p "$BENCH_ROOT/logs"
"$BENCH_PY" -m unittest discover -s intransitive/tests -v > "$BENCH_ROOT/logs/focused.log" 2>&1
"$BENCH_PY" -m unittest discover -s tests -v > "$BENCH_ROOT/logs/repository.log" 2>&1
# Alternate worker order across seeds. Fresh processes; never concurrent trials.
for trial in '1 140' '2 140' '2 141' '1 141' '1 142' '2 142'; do
    set -- $trial
    name="final-w$1-seed$2"
    "$BENCH_PY" -m intransitive.benchmark_training --output "$BENCH_ROOT/$name" \
        --workers "$1" --seed "$2" > "$BENCH_ROOT/logs/$name.log" 2>&1
    echo "Completed $name"
done
"$BENCH_PY" -m intransitive.benchmark_training --output "$BENCH_ROOT/calibration-w1-seed143" \
    --workers 1 --seed 143 --games 8 --arena-games 8 --simulations 32 \
    > "$BENCH_ROOT/logs/calibration-w1-seed143.log" 2>&1
echo 'Completed baseline-budget calibration'
