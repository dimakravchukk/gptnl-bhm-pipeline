#!/bin/bash
# Install the patched datatrove 0.3.0 SlurmPipelineExecutor (parallel_tasks_per_job,
# tasks_per_node) into the data-curation-pipeline venv.
#   bash pipeline_patches/apply_datatrove_patch.sh [/path/to/data-curation-pipeline]
# Replaces executor/slurm.py only if it is the stock 0.3.0 file; keeps a backup.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PIPE="${1:-$HOME/projects/data-curation-pipeline}"
PY="$PIPE/.venv/bin/python"
VER=$("$PY" -c "from importlib.metadata import version; print(version('datatrove'))")
[ "$VER" = "0.3.0" ] || { echo "datatrove is $VER, patch is for 0.3.0"; exit 1; }
TARGET=$("$PY" -c "import datatrove.executor.slurm as s; print(s.__file__)")
SUM=$(sha256sum "$TARGET" | cut -d' ' -f1)
if [ "$SUM" = "a895b6d7ea8d063874262c3cf596a35987af5001a5cbecb8679a29360bf50014" ]; then echo "already patched: $TARGET"; exit 0; fi
[ "$SUM" = "22585b3c4c4b1b95f8808d85de1d4c9d44a08143bcd66788f88306f077e0d113" ] || { echo "$TARGET is not the stock 0.3.0 file (sha256 $SUM); not touching it"; exit 1; }
cp "$TARGET" "$TARGET.orig"
cp "$HERE/datatrove-0.3.0-executor-slurm.py" "$TARGET"
"$PY" -c "import inspect, datatrove.executor.slurm as s; assert 'parallel_tasks_per_job' in inspect.signature(s.SlurmPipelineExecutor).parameters"
echo "patched: $TARGET (backup: $TARGET.orig)"
