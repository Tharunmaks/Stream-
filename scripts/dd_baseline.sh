#!/usr/bin/env bash
# Cross-check es_bench with plain dd: read every expert file in DIR once,
# split across N parallel dd readers, with O_DIRECT. Works without fio.
#
#   bash scripts/dd_baseline.sh ~/es_bench 4
set -euo pipefail
DIR=${1:?usage: dd_baseline.sh DIR [readers=4]}
N=${2:-4}

mapfile -t FILES < <(find "$DIR" -name '*.exp' | sort)
[ ${#FILES[@]} -gt 0 ] || { echo "no .exp files in $DIR"; exit 1; }
BYTES=$(( $(stat -c %s "${FILES[0]}") * ${#FILES[@]} ))

reader() { # reader INDEX: every Nth file
    local i
    for ((i = $1; i < ${#FILES[@]}; i += N)); do
        dd if="${FILES[$i]}" of=/dev/null bs=1M iflag=direct status=none
    done
}

start=$(date +%s%N)
for ((r = 0; r < N; r++)); do reader "$r" & done
wait
end=$(date +%s%N)

awk -v b="$BYTES" -v ns="$((end - start))" -v n="$N" -v f="${#FILES[@]}" \
    'BEGIN { printf "dd: %d files, %.2f GB, %d readers: %.2f GB/s\n", f, b/1e9, n, b/ns }'
