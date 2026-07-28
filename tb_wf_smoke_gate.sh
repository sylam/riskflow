#!/usr/bin/env bash
# WALK-FORWARD SMOKE GATE — streaming recipe (throwaway; tb_ prefix).
#
# Replaces the retired non-streaming anchor (--batch 2048, Simulation_Batches=1), which commit
# "delete the inner-MC chunk loop" made unrunnable: 2048x64 single-pass needs ~23.5 GiB of a
# 23.6 GiB card on the production world. Streaming is the adopted mode, so the anchor moves with
# it — arm A of the streaming ladder, whose 48-trade result was parity with REVAL48.
#
# RECIPE (identical to ladder arm A): garch, --batch 512 --streaming-batches 5 (4 training
# batches x 512 = 2048 trained paths + a 512-path held-out batch), --fit-iters 40, trade 202001.
# Seed 7 only, so the gate costs one train + one roll.
#
# ANCHORS. Measured on this code, twice, bit-identical across the two runs:
#   train_u  -0.5006                 seed 7's training utility
#   V_0      -0.1082737073302269     seed 7's held-out value, full precision
#   greedy   -104.71                 realized roll, $/oz
#   churn     193.8                  realized roll turnover
#   nohedge  -194.35                 unhedged realized path  (policy-independent)
#   pf_bound  810.1                  perfect-foresight bound (policy-independent)
#
# WHY train_u/V_0 ARE NOT THE LADDER'S NUMBERS. The ladder row (armA2_gpu0/row_202001.json) has
# train_u -0.5146 and V_0 -0.13939759135246277. Those came from the PRE-deletion code, where the
# solver split each grad fork into sub-slices: at 512x64 the budget gave
# grad_chunk = (32768*64/2)/(64*64) = 256 < 512, so every grad fork ran as 2 slices of 256, each
# drawing its OWN Sobol stream. One 512-wide draw is statistically equivalent, not bitwise — the
# same "partitions are statistically equivalent" property the no-grad chunk loop always had.
# So the training side legitimately moved with the deletion and is re-pinned here; the realized
# roll (greedy/churn) and the policy-independent anchors did NOT move, which is the useful check
# that the deletion changed draw partitioning and nothing else.
set -u
cd /home/vretiel/PycharmProjects/riskflow || exit 1
RUN=${1:-/tmp/wf_smoke_gate}
rm -rf "$RUN"

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u production_walk_forward.py --spot-model garch --start 2020-01 --months 1 \
    --seeds 7 --batch 512 --streaming-batches 5 --fit-iters 40 --run-dir "$RUN" \
    > "$RUN.log" 2>&1
RC=$?
echo "driver rc=$RC  log=$RUN.log"

python - "$RUN" <<'PY'
import json, sys
run = sys.argv[1]
got = json.load(open(f'{run}/row_202001.json'))
PINNED = {'train_u': -0.5006, 'V_0': -0.1082737073302269,
          'greedy_usd_oz': -104.71, 'churn': 193.8,
          'nohedge_usd_oz': -194.35, 'pf_bound': 810.1, 'bound_pass': True}
# Pre-deletion values, kept as provenance (see the header for why they moved).
INFO = {'train_u': -0.5146, 'V_0': -0.13939759135246277}
bad = []
for k, want in PINNED.items():
    have = got.get(k)
    ok = (have == want)
    print(f'  {k:16} {have!r:24} anchor {want!r:24} {"OK" if ok else "MISMATCH"}')
    if not ok:
        bad.append(k)
for k, want in INFO.items():
    print(f'  {k:16} {got.get(k)!r:24} pre-deletion ladder value {want!r} (expected to differ)')
print('GATE ' + ('GREEN' if not bad else f'RED — {bad}'))
sys.exit(1 if bad else 0)
PY
