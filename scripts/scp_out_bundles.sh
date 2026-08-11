#!/usr/bin/env bash
# scp each *.tar under out/ to a remote host (companion to pack_out_bundles.sh).
#
# Usage (from model root):
#   ./scripts/scp_out_bundles.sh
#   ./scripts/scp_out_bundles.sh ./out
#   DEST=ubuntu@43.167.164.129:~/ ./scripts/scp_out_bundles.sh ./out
#   REMOTE_USER=ubuntu REMOTE_HOST=43.167.164.129 REMOTE_DIR=~/ \
#     ./scripts/scp_out_bundles.sh ./out
set -euo pipefail

OUT_DIR="${1:-./out}"

REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_HOST="${REMOTE_HOST:-43.167.164.129}"
REMOTE_DIR="${REMOTE_DIR:-~/}"
# Full scp destination override, e.g. ubuntu@43.167.164.129:~/
DEST="${DEST:-${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}}"

if [[ ! -d "$OUT_DIR" ]]; then
  echo "error: not a directory: $OUT_DIR" >&2
  exit 1
fi

cd "$OUT_DIR"

shopt -s nullglob
tars=( *.tar )
if ((${#tars[@]} == 0)); then
  echo "error: no *.tar files in $(pwd) (run pack_out_bundles.sh first)" >&2
  exit 1
fi

echo "destination: $DEST"
for archive in "${tars[@]}"; do
  echo "==> scp $archive $DEST"
  scp "$archive" "$DEST"
done

echo "done."
