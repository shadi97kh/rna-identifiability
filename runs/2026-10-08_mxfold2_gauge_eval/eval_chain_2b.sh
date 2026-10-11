#!/bin/bash
# Detached stage-2b chain, 2026-10-11 (amendment 5). Starts only if 2a's RECORDED status is
# "complete". 2b's workers and first reduction run; the V3 re-draw for 2b's direction-0
# misses follows (scripts/mxfold2_v3_redraw.py); then the amended reduction (--reduce-only).
# Completion is read from the artifact at every step. Launched with setsid.
cd /home/shadi/sirna-thermo || exit 1
OUT=runs/2026-10-08_mxfold2_gauge_eval
st() { jq -r ".\"$1\".status // \"none\"" "$OUT/metrics.json" 2>/dev/null; }
echo "chain-2b pid $$ sid $(ps -o sid= -p $$) start $(date '+%F %T'); 2a status: $(st 2a)"
if [ "$(st 2a)" != "complete" ]; then echo "2b NOT started: 2a is not complete"; exit 1; fi
.venv-mxfold2/bin/python scripts/mxfold2_gauge_eval.py --stage 2b --workers 4 --threads 2
echo "2b workers+reduce exit $?, status: $(st 2b), $(date '+%F %T')"
CUDA_VISIBLE_DEVICES= .venv-mxfold2/bin/python scripts/mxfold2_v3_redraw.py --stage 2b
rc=$?
echo "2b re-draw exit $rc, $(date '+%F %T')"
if [ "$rc" -ne 0 ]; then echo "2b amended reduction NOT run"; exit 1; fi
.venv-mxfold2/bin/python scripts/mxfold2_gauge_eval.py --stage 2b --workers 4 --threads 2 --reduce-only
echo "2b amended reduce exit $?, status: $(st 2b), $(date '+%F %T')"
echo "2b chain done"
