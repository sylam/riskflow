#!/usr/bin/env bash
# STREAMING LADDER — the arbiter of Solver.DiffV2_Streaming_Batches (throwaway; tb_ prefix).
#
# Baseline to beat: artifacts/walk_forward/reval48_final.csv — 48 monthly trades, garch,
# seeds 7/42/314, --batch 2048, --fit-iters 40, recal every 3 months, margin 8.0, corridor-free
# checkpoints + realized roll. VERIFIED from tb_reval48_launch.sh + the fixture: that campaign
# trained on Batch_Size 2048 x Simulation_Batches 1 = 2048 paths per seed (NOT 8192).
#
# Arm A — MATCHED-DATA head-to-head (the decisive arm). Streaming --batch 512 --streaming-batches 5
#   => 4 training batches x 512 = 2048 trained paths per seed (identical data volume to the
#   baseline) + a 512-path held-out batch for the verdict. Everything else identical to REVAL48,
#   including the realized ROLL (non-streaming, Batch_Size=1, 3-checkpoint ensemble, inner 256), so
#   the ONLY difference between the arms is HOW training consumed its paths: one fixed set of 2048
#   vs four fresh sets of 512 under a frame locked on the first.
#
# Arm B — WALLS-MOVE probe. Streaming --batch 2048 --streaming-batches 5 => 4 x 2048 = 8192 trained
#   paths per seed, 4x the baseline's data, at unchanged fork width (2048). project_lever_sweep_
#   optimum measured that MORE PATHS degrade OOS on a fixed set; if the fresh-data regime removes
#   that mechanism, this arm should not degrade — and should beat arm A. Run on the 12-trade 2020
#   window (which both other arms cover) to keep it affordable. NOTE: the T_Min wall the design
#   names cannot be probed — the shipping recipe already runs T_Min=0, the deepest possible window,
#   so "deeper" does not exist. The paths wall is the measurable one at this depth.
#
# Expect one WARNING per loaded checkpoint per trade in the roll logs: streaming-trained frames
# rolled by a non-streaming eval. That is the deliberate arrangement above, not a misconfiguration.
#
# ATTEMPT 2 (run dirs *2_*). Attempt 1 is kept alongside as evidence and is NOT comparable: it ran
# before the multi-batch burn-in rewind, so its batches walked away from the calibrated world
# (symlog c +94% by the held-out batch) and its arm-A lane lost 9 of 24 months to the X_0 guard.
# Both are fixed on this branch; every trade below is re-run from scratch on one arithmetic.
# Arm B stays on two parallel lanes: attempt 1's arm B did NOT die of memory — the host rebooted
# under it (both lanes stop within 1s at 12:43:11, the logs end in unflushed NUL padding, the
# launcher died with them, journald itself crashed and the block layer was erroring; no OOM
# killer, no Xid, uptime confirms the boot). A resource sampler runs alongside so the next
# failure has arithmetic attached.
set -u
cd /home/vretiel/PycharmProjects/riskflow || exit 1
WF=artifacts/walk_forward/streaming_ladder
mkdir -p "$WF"

( while true; do
    printf '%s %s | rss_MB=%s\n' "$(date -Is)" \
      "$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')" \
      "$(ps -o rss= -C python | awk '{s+=$1} END {printf "%d", s/1024}')" >> "$WF/resources.log"
    sleep 60
  done ) & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

echo "=== STREAMING LADDER START $(date -Is) | riskflow=$(python -c 'import riskflow;print(riskflow.__file__)') ==="
echo "=== HEAD: $(git log --oneline -1 | cut -c1-72) ==="

# ---- Arm A: 48 trades, two lanes (24 each), matched trained-path count --------------------
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u production_walk_forward.py \
    --spot-model garch --seeds 7 42 314 --batch 512 --streaming-batches 5 --fit-iters 40 \
    --start 2020-01 --months 24 --run-dir "$WF/armA2_gpu0" \
    > "$WF/armA2_gpu0.log" 2>&1 &
A0=$!
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u production_walk_forward.py \
    --spot-model garch --seeds 7 42 314 --batch 512 --streaming-batches 5 --fit-iters 40 \
    --start 2022-01 --months 24 --run-dir "$WF/armA2_gpu1" \
    > "$WF/armA2_gpu1.log" 2>&1 &
A1=$!
echo "=== arm A lanes launched: gpu0 pid=$A0  gpu1 pid=$A1 ==="
wait $A0; RA0=$?
wait $A1; RA1=$?
echo "=== arm A done $(date -Is): gpu0 rc=$RA0 gpu1 rc=$RA1 | trades gpu0=$(ls "$WF/armA2_gpu0"/row_*.json 2>/dev/null | wc -l)/24 gpu1=$(ls "$WF/armA2_gpu1"/row_*.json 2>/dev/null | wc -l)/24 ==="

# ---- Arm B: the 2020 window (12 trades), 4x the trained paths at unchanged fork width -----
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u production_walk_forward.py \
    --spot-model garch --seeds 7 42 314 --batch 2048 --streaming-batches 5 --fit-iters 40 \
    --start 2020-01 --months 6 --run-dir "$WF/armB2_gpu0" \
    > "$WF/armB2_gpu0.log" 2>&1 &
B0=$!
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u production_walk_forward.py \
    --spot-model garch --seeds 7 42 314 --batch 2048 --streaming-batches 5 --fit-iters 40 \
    --start 2020-07 --months 6 --run-dir "$WF/armB2_gpu1" \
    > "$WF/armB2_gpu1.log" 2>&1 &
B1=$!
echo "=== arm B lanes launched: gpu0 pid=$B0  gpu1 pid=$B1 ==="
wait $B0; RB0=$?
wait $B1; RB1=$?
echo "=== arm B done $(date -Is): gpu0 rc=$RB0 gpu1 rc=$RB1 ==="

# ---- Aggregate + the head-to-head table ---------------------------------------------------
python - <<'PY'
import os
import numpy as np
import pandas as pd

WF = 'artifacts/walk_forward/streaming_ladder'
BASE = 'artifacts/walk_forward/reval48_final.csv'


def lanes(*paths):
    got = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if not got:
        return None
    return pd.concat(got, ignore_index=True).sort_values('trade').reset_index(drop=True)


def stats(v):
    v = np.asarray(v, dtype=float)
    p5 = np.percentile(v, 5)
    return dict(n=len(v), mean=v.mean(), std=v.std(ddof=1) if len(v) > 1 else 0.0,
                p5=p5, cvar5=v[v <= p5].mean() if (v <= p5).any() else p5, worst=v.min())


def line(tag, s):
    print(f'  {tag:22} n={s["n"]:3d} mean {s["mean"]:+8.2f} std {s["std"]:7.2f} '
          f'p5 {s["p5"]:+8.2f} cvar5 {s["cvar5"]:+8.2f} worst {s["worst"]:+8.2f}')


base = pd.read_csv(BASE) if os.path.exists(BASE) else None
armA = lanes(f'{WF}/armA2_gpu0/trades.csv', f'{WF}/armA2_gpu1/trades.csv')
armB = lanes(f'{WF}/armB2_gpu0/trades.csv', f'{WF}/armB2_gpu1/trades.csv')
for name, df in (('armA', armA), ('armB', armB)):
    if df is not None:
        df.to_csv(f'{WF}/{name}_final.csv', index=False)

print('\n===== STREAMING LADDER: realized roll, $/oz per trade (greedy_usd_oz) =====')
if base is not None:
    line('REVAL48 (fixed 2048)', stats(base['greedy_usd_oz']))
    line('  nohedge', stats(base['nohedge_usd_oz']))
if armA is not None:
    line('arm A (4x512 fresh)', stats(armA['greedy_usd_oz']))
if armB is not None:
    line('arm B (4x2048 fresh)', stats(armB['greedy_usd_oz']))

for name, df in (('arm A', armA), ('arm B', armB)):
    if df is None or base is None:
        continue
    m = df.merge(base, on='trade', suffixes=('_s', '_b'))
    if m.empty:
        continue
    d = m['greedy_usd_oz_s'] - m['greedy_usd_oz_b']
    print(f'\n--- {name} vs REVAL48, PAIRED on {len(m)} shared trades ---')
    print(f'  delta $/oz: mean {d.mean():+.2f} median {d.median():+.2f} '
          f'p5 {np.percentile(d, 5):+.2f} worst {d.min():+.2f} best {d.max():+.2f} | '
          f'streaming better on {int((d > 0).sum())}/{len(d)} trades')
    line('  streaming', stats(m['greedy_usd_oz_s']))
    line('  baseline', stats(m['greedy_usd_oz_b']))
    print(f'  train_u mean  streaming {m["train_u_s"].mean():+.4f} vs baseline '
          f'{m["train_u_b"].mean():+.4f} | V_0 {m["V_0_s"].mean():+.4f} vs {m["V_0_b"].mean():+.4f}')
    print(f'  churn mean    streaming {m["churn_s"].mean():.1f} vs baseline {m["churn_b"].mean():.1f}'
          f' | bound-PASS {int(m["bound_pass_s"].sum())}/{len(m)} vs '
          f'{int(m["bound_pass_b"].sum())}/{len(m)}')
    # Same worlds check: the unhedged realized path is policy-independent, so it MUST match.
    gap = (m['nohedge_usd_oz_s'] - m['nohedge_usd_oz_b']).abs().max()
    print(f'  world check   max |nohedge_s - nohedge_b| = {gap:.4f} (0 => identical worlds)')

if armA is not None and armB is not None:
    m = armB.merge(armA, on='trade', suffixes=('_B', '_A'))
    if not m.empty:
        d = m['greedy_usd_oz_B'] - m['greedy_usd_oz_A']
        print(f'\n--- WALLS-MOVE: arm B (8192 trained) vs arm A (2048 trained), {len(m)} trades ---')
        print(f'  delta $/oz: mean {d.mean():+.2f} median {d.median():+.2f} | '
              f'B better on {int((d > 0).sum())}/{len(d)} trades')
        print('  (fixed-set precedent: more paths DEGRADE OOS. No degradation here => the wall '
              'moved; degradation => the wall is not about the path count.)')
PY
echo "=== STREAMING LADDER COMPLETE $(date -Is) ==="
