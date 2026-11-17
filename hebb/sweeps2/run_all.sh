#!/usr/bin/env bash
# Regenerate every v2 sweep JSON in site/data/sweeps2 (CPU only; each script parallelises over 4 processes).
# Approximate wall times on 4 vCPU: fusion ~8 min, resolution ~3 min, diversity ~6 min, eigen ~3 min,
# allocation ~6 min, readout ~2 min.
set -euo pipefail
cd "$(dirname "$0")/../.."
for name in fusion resolution diversity eigen allocation readout; do
    echo "=== $name"
    start=$(date +%s)
    uv run python -m "hebb.sweeps2.$name"
    echo "=== $name done in $(( $(date +%s) - start )) s"
done
