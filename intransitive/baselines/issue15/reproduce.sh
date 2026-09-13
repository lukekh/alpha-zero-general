#!/bin/sh
set -eu
export ORT_DISABLE_TELEMETRY=1
BASELINE_PY=${BASELINE_PY:-python3}
BASELINE_ROOT=${BASELINE_ROOT:-checkpoints/issue15-reproduction}
# Invoke from repository root. Each command refuses existing output directories.
"$BASELINE_PY" -m unittest discover -s intransitive/tests
"$BASELINE_PY" -m unittest discover -s tests
"$BASELINE_PY" -m intransitive.baseline train --output "$BASELINE_ROOT"
"$BASELINE_PY" -m intransitive.baseline evaluate --model-folder "$BASELINE_ROOT" --output "$BASELINE_ROOT/evaluation"
"$BASELINE_PY" -m intransitive.baseline verify --model-folder "$BASELINE_ROOT" --output "$BASELINE_ROOT/verification"
