#!/usr/bin/env bash
# Launch the 15-lane net-of-cost frontier sweep across both GPUs (throwaway).
# cost in {flat10, base, high} x band in {free, 0.15, 0.25, 0.40, 0.60}.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR=artifacts/walk_forward/net_of_cost
mkdir -p "$LOGDIR"

COSTS=(flat10 base high)
BANDS=(free 0.15 0.25 0.40 0.60)
MAXPER=3            # concurrent lanes per GPU

declare -a PIDS
i=0
for cost in "${COSTS[@]}"; do
  for band in "${BANDS[@]}"; do
    gpu=$(( i % 2 ))
    # throttle: keep at most MAXPER*2 lanes in flight
    while [ "$(jobs -rp | wc -l)" -ge $(( MAXPER * 2 )) ]; do wait -n; done
    bstr="${band/./_}"
    log="$LOGDIR/lane_${cost}_${bstr}.log"
    echo "LAUNCH lane cost=$cost band=$band gpu=$gpu -> $log"
    CUDA_VISIBLE_DEVICES=$gpu python3 tb_cost_frontier.py --cost "$cost" --band "$band" \
        > "$log" 2>&1 &
    PIDS+=($!)
    i=$(( i + 1 ))
  done
done

fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then fail=$(( fail + 1 )); fi
done
echo "FRONTIER SWEEP DONE (failed lanes: $fail)"
