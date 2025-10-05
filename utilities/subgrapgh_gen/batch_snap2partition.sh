#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/snap2partition"

usage() {
  echo "Usage: $(basename "$0") <input_dir> <num_partitions> <output_dir>" >&2
  echo "  input_dir: directory containing SNAP .txt files (searched recursively)" >&2
  echo "  num_partitions: positive integer (k)" >&2
  echo "  output_dir: directory where partitioned files are written" >&2
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -lt 3 ]]; then
  usage
  exit 1
fi

INPUT_DIR="$1"; shift
NPARTS="$1"; shift
OUT_DIR="$1"; shift || true

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[ERR] input_dir not found: $INPUT_DIR" >&2
  exit 1
fi
if ! [[ "$NPARTS" =~ ^[0-9]+$ ]] || [[ "$NPARTS" -le 0 ]]; then
  echo "[ERR] num_partitions must be a positive integer: $NPARTS" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

# Build tool if missing. Allow caller to override METIS paths via env.
if [[ ! -x "$BIN" ]]; then
  echo "[INFO] Building snap2partition ..." >&2
  make -C "$SCRIPT_DIR" ${INCLUDES_METIS:+INCLUDES_METIS="$INCLUDES_METIS"} \
                        ${LIBS_METIS:+LIBS_METIS="$LIBS_METIS"}
fi
if [[ ! -x "$BIN" ]]; then
  echo "[ERR] snap2partition binary not found after build: $BIN" >&2
  exit 1
fi

echo "[INFO] Scanning for .txt under: $INPUT_DIR" >&2
mapfile -t files < <(find "$INPUT_DIR" -type f -name '*.txt' | sort)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "[WARN] No .txt files found under $INPUT_DIR" >&2
  exit 0
fi

ok=0; fail=0
for f in "${files[@]}"; do
  echo "[RUN] $BIN \"$f\" $NPARTS \"$OUT_DIR\"" >&2
  if "$BIN" "$f" "$NPARTS" "$OUT_DIR"; then
    ((ok++)) || true
  else
    echo "[FAIL] $f" >&2
    ((fail++)) || true
  fi
done

echo "[DONE] Success: $ok, Failed: $fail" >&2

