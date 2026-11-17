#!/usr/bin/env bash
# Regenerate every sweep JSON in site/data/sweeps/ (fixed seeds). ~35-45 min on 4 CPUs.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p scratch/out/sweeps
for s in s1_parts s2_phase s3_alloc s4_diversity s5_dynamics s6_readout s7_tours; do
  echo "== $s"; PYTHONPATH=. uv run python -u -m hebb.sweeps.$s > scratch/out/sweeps/$s.log 2>&1
  tail -n 2 scratch/out/sweeps/$s.log
done
PYTHONPATH=. uv run python -m hebb.sweeps.make_index
