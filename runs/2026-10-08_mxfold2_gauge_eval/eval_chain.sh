#!/bin/bash
# Detached evaluation chain, 2026-10-10. Stage 2a, then stage 2b ONLY if 2a's RECORDED status
# is "complete" (completion is read from the artifact, never from a process ending). Launched
# with setsid so no terminal or agent session can end it.
cd /home/shadi/sirna-thermo || exit 1
OUT=runs/2026-10-08_mxfold2_gauge_eval
echo "chain pid $$ sid $(ps -o sid= -p $$) start $(date '+%F %T')"
.venv-mxfold2/bin/python scripts/mxfold2_gauge_eval.py --stage 2a --workers 4 --threads 2
rc=$?
st=$(jq -r '."2a".status // "none"' "$OUT/metrics.json" 2>/dev/null)
echo "2a exit $rc, recorded status: $st, $(date '+%F %T')"
if [ "$rc" -ne 0 ] || [ "$st" != "complete" ]; then
  echo "2b NOT started"
  exit 1
fi
.venv-mxfold2/bin/python scripts/mxfold2_gauge_eval.py --stage 2b --workers 4 --threads 2
rc=$?
st=$(jq -r '."2b".status // "none"' "$OUT/metrics.json" 2>/dev/null)
echo "2b exit $rc, recorded status: $st, $(date '+%F %T')"
