#!/usr/bin/env bash
# Download GGUF files from Hugging Face with resume support (curl only).
#
#   bash scripts/hf_get.sh REPO MATCH [OUTDIR]
#
# MATCH selects .gguf files whose path contains it, so a split model's parts
# all come down together. Examples:
#   bash scripts/hf_get.sh bartowski/OLMoE-1B-7B-0924-Instruct-GGUF Q2_K.gguf ~/models
#   bash scripts/hf_get.sh unsloth/Qwen3-30B-A3B-GGUF Q2_K_XL ~/models
# Gated models need a token: export HF_TOKEN=hf_...
set -euo pipefail
REPO=${1:?usage: hf_get.sh REPO MATCH [OUTDIR]}
MATCH=${2:?usage: hf_get.sh REPO MATCH [OUTDIR]}
OUT=${3:-$HOME/models}
AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")

mkdir -p "$OUT"
LIST=$(curl -sSf "${AUTH[@]}" "https://huggingface.co/api/models/$REPO/tree/main?recursive=1") || {
    echo "cannot list $REPO (wrong name, or gated: set HF_TOKEN)"; exit 1; }

# pull "path" and "size" pairs out of the JSON without needing jq/python
mapfile -t ROWS < <(printf '%s' "$LIST" | tr '{' '\n' |
    sed -n 's/.*"size":\([0-9]*\).*"path":"\([^"]*\.gguf\)".*/\2 \1/p; s/.*"path":"\([^"]*\.gguf\)".*"size":\([0-9]*\).*/\1 \2/p' |
    grep -F -- "$MATCH" | sort -u)
[ ${#ROWS[@]} -gt 0 ] || { echo "no .gguf file in $REPO matches '$MATCH'"; exit 1; }

TOTAL=0
for r in "${ROWS[@]}"; do TOTAL=$((TOTAL + ${r##* })); done
FREE=$(df -Pk "$OUT" | awk 'NR==2 {printf "%.0f", $4 * 1024}')
awk -v n=${#ROWS[@]} -v t="$TOTAL" -v f="$FREE" \
    'BEGIN { printf "%d file(s), %.2f GB; free space %.2f GB\n", n, t / 1e9, f / 1e9 }'
if [ "$TOTAL" -gt "$FREE" ]; then
    echo "not enough free space in $OUT"; exit 1
fi

for r in "${ROWS[@]}"; do
    f=${r% *}
    dest="$OUT/$(basename "$f")"
    echo "-> $dest"
    curl -fL --progress-bar -C - --retry 5 --retry-delay 3 "${AUTH[@]}" -o "$dest" \
        "https://huggingface.co/$REPO/resolve/main/$f"
done
echo "done. import with:  ./es_import -o ~/packs/NAME $OUT/<the .gguf file(s)>"
