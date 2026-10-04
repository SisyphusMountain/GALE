#!/usr/bin/env bash
# Fit global DTL rates and draw 10 reconciliations per family (run from this folder).
set -euo pipefail
gpurec -f families.txt -s species.nwk -p output -g 10 --seed 42
