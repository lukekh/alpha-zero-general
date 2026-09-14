#!/bin/sh
set -eu
export ORT_DISABLE_TELEMETRY=1
if [ -z "${TEACHER_PY+x}" ]; then
    if [ -x .venv/bin/python ]; then
        TEACHER_PY=.venv/bin/python
    else
        TEACHER_PY=python
    fi
fi
TEACHER_ROOT=${TEACHER_ROOT:-checkpoints/issue33-reproduction}
if [ -e "$TEACHER_ROOT" ]; then
    echo "Use a fresh TEACHER_ROOT: $TEACHER_ROOT" >&2
    exit 1
fi
mkdir -p "$TEACHER_ROOT-logs"
"$TEACHER_PY" -m unittest intransitive.tests.test_teacher_learning -v \
    > "$TEACHER_ROOT-logs/focused-tests.log" 2>&1
set +e
"$TEACHER_PY" -m unittest discover -s intransitive/tests -v \
    > "$TEACHER_ROOT-logs/tests.log" 2>&1
test_status=$?
set -e
test "$test_status" -eq 1
test "$(grep -c '^FAIL: test_' "$TEACHER_ROOT-logs/tests.log")" -eq 5
grep -q 'FAILED (failures=5)' "$TEACHER_ROOT-logs/tests.log"
"$TEACHER_PY" -m unittest discover -s tests -v \
    > "$TEACHER_ROOT-logs/repository-tests.log" 2>&1
"$TEACHER_PY" -m intransitive.teacher_learning run --output "$TEACHER_ROOT" \
    > "$TEACHER_ROOT-logs/experiment.log" 2>&1
"$TEACHER_PY" -m intransitive.teacher_learning verify --output "$TEACHER_ROOT" \
    > "$TEACHER_ROOT-logs/verification.log" 2>&1
echo "Completed: $TEACHER_ROOT/report.json"
