#!/usr/bin/env bash
# Compatibility launcher; keep existing commands working.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_attention_experiments.sh" "$@"
