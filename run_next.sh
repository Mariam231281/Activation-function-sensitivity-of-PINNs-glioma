#!/bin/bash
# Runs the next N pending jobs (default 2) in parallel and waits for them.
cd "$(dirname "$0")"
./jobs_remaining.sh | head -n "${1:-2}" | xargs -P 2 -I{} sh -c '{} 2>/dev/null' >> ../results_p2/run.log
echo "pending: $(./jobs_remaining.sh | wc -l)"
