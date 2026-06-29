#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: source submission/scripts/set_data_env.sh <LUMIERE_ROOT> <LUMIERE_PROCESSED> <LUMIERE_CACHE>"
  return 1 2>/dev/null || exit 1
fi

export LUMIERE_ROOT="$1"
export LUMIERE_PROCESSED="$2"
export LUMIERE_CACHE="$3"

echo "LUMIERE_ROOT=$LUMIERE_ROOT"
echo "LUMIERE_PROCESSED=$LUMIERE_PROCESSED"
echo "LUMIERE_CACHE=$LUMIERE_CACHE"
