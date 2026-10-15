#!/usr/bin/env bash
# Run the aquarium-species comparison for each backbone (same images, seeds and aquarium splits):
#   1) feature analysis on inat_aquarium (100 species, aquarium scenes only, no training)
#   2) linear probe on inat_aquarium, split by aquarium
# Usage: bash run_backbones.sh [backbone ...]   (default: eva02_large siglip2_so400m siglip2_giant pe_core_g14)
set -u
cd "$(dirname "$0")"
PY=${PY:-venv/Scripts/python.exe}
BACKBONES=("$@")
[ ${#BACKBONES[@]} -eq 0 ] && BACKBONES=(eva02_large siglip2_so400m siglip2_giant pe_core_g14)
export PYTHONIOENCODING=utf-8

for bb in "${BACKBONES[@]}"; do
  # linear probe first: it extracts embeddings for all aquarium photos, which the analysis then reuses
  echo "===== $bb : linear probe (inat_aquarium) ====="
  "$PY" -u linear_probe.py --backbone "$bb" || echo "FAILED $bb linear_probe"
  echo "===== $bb : feature analysis (inat_aquarium) ====="
  "$PY" -u analyze_dinov2.py --backbone "$bb" --dataset inat_aquarium --aquarium-only || echo "FAILED $bb analysis"
  echo "===== $bb : done ====="
done
