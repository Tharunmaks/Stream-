#!/usr/bin/env bash
# One-shot Step 3 test: build, make the dataset if needed, run es_run in
# four configurations, and save everything to step3_results.txt.
#
#   bash scripts/step3.sh            # dataset in ~/es_bench
#   bash scripts/step3.sh /path/dir  # somewhere else
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=${1:-$HOME/es_bench}
OUT=step3_results.txt

make -s

# es_run needs a dataset made by the current es_gen (finite fp16 scales).
if ! ./es_run -d "$DIR" -L 1 -n 1 -w 0 -a 0 -q >/dev/null 2>&1; then
    echo "creating dataset in $DIR (8 layers x 128 experts, 6.3 GB)..."
    rm -rf "$DIR"
    ./es_gen -d "$DIR" -L 8 -E 128
fi

{
    echo "### $(date)"
    echo; echo "### 1/4 defaults (reuse 0.3, predictor 0.6, 1 GB cache)"
    ./es_run -d "$DIR"
    echo; echo "### 2/4 worst case: no prefetch, no locality"
    ./es_run -d "$DIR" -q -p 0 -r 0
    echo; echo "### 3/4 optimistic: reuse 0.5, predictor 0.8, 1.5 GB cache"
    ./es_run -d "$DIR" -q -r 0.5 -p 0.8 -C 1536
    echo; echo "### 4/4 defaults without attention cost"
    ./es_run -d "$DIR" -q -a 0
} 2>&1 | tee "$OUT"

echo
echo "saved to $(pwd)/$OUT"
