#!/bin/sh
set -eu
export ORT_DISABLE_TELEMETRY=1
SYMMETRY_PY=${SYMMETRY_PY:-python}
SYMMETRY_ROOT=${SYMMETRY_ROOT:-checkpoints/issue16-reproduction}
if [ -e "$SYMMETRY_ROOT" ]; then
    echo "Use a fresh SYMMETRY_ROOT: $SYMMETRY_ROOT" >&2
    exit 1
fi
mkdir -p "$SYMMETRY_ROOT/logs" "$SYMMETRY_ROOT/issue15"
"$SYMMETRY_PY" -m unittest discover -s intransitive/tests -v > "$SYMMETRY_ROOT/logs/tests.log" 2>&1
"$SYMMETRY_PY" -m unittest discover -s tests -v > "$SYMMETRY_ROOT/logs/repository-tests.log" 2>&1
tar -xzf intransitive/baselines/issue15/baseline-artifacts.tar.gz -C "$SYMMETRY_ROOT/issue15"
"$SYMMETRY_PY" -m intransitive.symmetry_efficiency run \
    --baseline-folder "$SYMMETRY_ROOT/issue15" --output "$SYMMETRY_ROOT/comparison" \
    > "$SYMMETRY_ROOT/logs/comparison.log" 2>&1
echo "Completed: $SYMMETRY_ROOT/comparison/comparison.json"
