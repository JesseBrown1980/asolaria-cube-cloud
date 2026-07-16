#!/bin/bash
# Cloud build: compile the dep-free unified-Omega trainer and run one script corpus.
# Usage: bash build.sh <script-name>   (e.g. bash build.sh egyptian-hieroglyphs)
set -euo pipefail
S="${1:?need script name}"
# ensure rust
if ! command -v rustc >/dev/null 2>&1; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
  source "$HOME/.cargo/env"
fi
tr -d '\r' < unified_omega.rs > uo.rs
rustc --edition=2021 -O uo.rs -o uo
./uo build --source "scripts/$S" --output "out/$S" --seat 8467a937cba309f7
echo "=== RESULT HBP for $S ==="
cat "out/$S/UNIFIED-OMEGA-RESULT.hbp" | grep -E 'UNIFIEDHDR|SUBOMEGAS|UNIFIEDOMEGA|UNIFIEDFTR'
