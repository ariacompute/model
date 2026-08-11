#!/usr/bin/env bash
# Pack each model bundle directory under out/ into its own .tar (tar cvf).
#
# Usage (from model root):
#   ./scripts/pack_out_bundles.sh
#   ./scripts/pack_out_bundles.sh ./out
# Already inside out/:
#   ../scripts/pack_out_bundles.sh .
set -euo pipefail

OUT_DIR="${1:-./out}"

if [[ ! -d "$OUT_DIR" ]]; then
  echo "error: not a directory: $OUT_DIR" >&2
  exit 1
fi

cd "$OUT_DIR"

shopt -s nullglob
dirs=( */ )
if ((${#dirs[@]} == 0)); then
  echo "error: no subdirectories in $(pwd)" >&2
  exit 1
fi

for d in "${dirs[@]}"; do
  name="${d%/}"
  archive="${name}.tar"
  if [[ -e "$archive" ]]; then
    echo "skip (exists): $archive"
    continue
  fi
  echo "==> tar cvf $archive $name/"
  tar cvf "$archive" "$name"
done

echo "done."
