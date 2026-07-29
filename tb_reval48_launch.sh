#!/usr/bin/env bash
# 48-trade RE-VALIDATION campaign after the ObservedBasis redundancy wave (throwaway; tb_ prefix).
#
# Old checkpoints are dimensionally incompatible (market_dim 9->8 HMM / 7->6 GARCH after the
# CME_FLAT deletion), so this is a FULL per-trade-date retrain + realized roll, same
# seeds/architecture as the committed garch48 campaign (garch, seeds 7/42/314, batch 2048,
# fit_iters 40, recal every 3 months, margin 8.0) so the comparison is like-for-like.
#
# Phase 1: two training lanes in parallel, one per GPU (24 trades each) -> train + corridor-FREE
#          realized roll (the shipped recipe: corridor-free checkpoints + roll-time band).
# Phase 2: eval-only realized re-rolls of those checkpoints at bands 0.60 and 0.15.
# Phase 3: aggregate per-trade + per-band CSVs for the headline comparison.
set -u
cd /home/vretiel/PycharmProjects/riskflow || exit 1
WF=artifacts/walk_forward
mkdir -p "$WF"

echo "=== REVAL48 START $(date -Is) | riskflow=$(python -c 'import riskflow;print(riskflow.__file__)') ==="
echo "=== HEAD: $(git log --oneline -1 | cut -c1-72) ==="

# ---- Phase 1: training lanes (parallel, one GPU each) -------------------------------------
CUDA_VISIBLE_DEVICES=0 python -u production_walk_forward.py \
    --spot-model garch --seeds 7 42 314 --batch 2048 --fit-iters 40 \
    --start 2020-01 --months 24 --run-dir "$WF/reval48_gpu0" \
    > "$WF/reval48_gpu0.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=0 python -u production_walk_forward.py \
    --spot-model garch --seeds 7 42 314 --batch 2048 --fit-iters 40 \
    --start 2022-01 --months 24 --run-dir "$WF/reval48_gpu1" \
    > "$WF/reval48_gpu1.log" 2>&1 &
PID1=$!
echo "=== phase1 lanes launched: gpu0 pid=$PID0  gpu1 pid=$PID1 ==="
wait $PID0; RC0=$?
wait $PID1; RC1=$?
echo "=== phase1 done $(date -Is): gpu0 rc=$RC0 gpu1 rc=$RC1 ==="

# Both lanes must be complete: the band re-roll's month map asserts all 48 rows exist.
N0=$(ls "$WF/reval48_gpu0"/row_*.json 2>/dev/null | wc -l)
N1=$(ls "$WF/reval48_gpu1"/row_*.json 2>/dev/null | wc -l)
echo "=== completed trades: gpu0=$N0/24 gpu1=$N1/24 ==="
if [ "$N0" -ne 24 ] || [ "$N1" -ne 24 ]; then
    echo "=== ABORT phase2: incomplete training lanes (resume by re-running this script; ==="
    echo "===   both lanes are idempotent per trade and per seed checkpoint)             ==="
    exit 1
fi

# ---- Phase 2: eval-only realized re-rolls at both bands (parallel) ------------------------
CUDA_VISIBLE_DEVICES=0 python -u tb_reval48_corridor.py --band 0.60 \
    > "$WF/reval48_corridor_b060.log" 2>&1 &
PB0=$!
CUDA_VISIBLE_DEVICES=0 python -u tb_reval48_corridor.py --band 0.15 \
    > "$WF/reval48_corridor_b015.log" 2>&1 &
PB1=$!
wait $PB0; RB0=$?
wait $PB1; RB1=$?
echo "=== phase2 done $(date -Is): b060 rc=$RB0 b015 rc=$RB1 ==="

# ---- Phase 3: aggregate -------------------------------------------------------------------
python - <<'PY'
import glob, json, os
import pandas as pd
WF = 'artifacts/walk_forward'
tr = [pd.read_csv(f) for f in (f'{WF}/reval48_gpu0/trades.csv', f'{WF}/reval48_gpu1/trades.csv') if os.path.exists(f)]
if tr:
    df = pd.concat(tr, ignore_index=True).sort_values('trade')
    df.to_csv(f'{WF}/reval48_final.csv', index=False)
    g = df['greedy_usd_oz']
    print(f'reval48_final.csv: {len(df)} trades | greedy mean {g.mean():+.2f} std {g.std():.2f} '
          f'min {g.min():+.2f} | nohedge mean {df["nohedge_usd_oz"].mean():+.2f} | '
          f'bound-PASS {int(df["bound_pass"].sum())}/{len(df)}')
rows = [json.load(open(f)) for f in sorted(glob.glob(f'{WF}/reval48_corridor/tb_row_*.json'))]
if rows:
    cr = pd.DataFrame(rows).sort_values(['band', 'tag'])
    cr.to_csv(f'{WF}/reval48_corridor_rolls.csv', index=False)
    for b, sub in cr.groupby('band'):
        q = sub['greedy']
        print(f'  band {b:.2f}: n={len(sub)} greedy mean {q.mean():+.2f} std {q.std():.2f} '
              f'p5 {q.quantile(0.05):+.2f} cvar5 {q[q <= q.quantile(0.05)].mean():+.2f} '
              f'worst {q.min():+.2f} | bound-PASS {int(sub["pass"].sum())}/{len(sub)} '
              f'| breaches {int(sub["breaches"].sum())}')
PY
echo "=== REVAL48 COMPLETE $(date -Is) ==="
