"""Run the textbook hedge over the revised simulator (no training).

Captures bundle + runtime via the simulate_only path, runs `rollout_textbook` from
diagnose_vs_textbook.py on each scenario, and dumps per-day P&L plus summary stats.

Output:
  - artifacts/textbook_hedge_paths.csv (per-day P&L, representative paths + means)
  - stdout: summary table + sanity checks
"""
import json as jsonlib
import re
import numpy as np
import pandas as pd
import torch

import riskflow as rf
from riskflow import torchrl_hedge

torch.set_default_dtype(torch.float32)


# --- Hijack execution to capture bundle + runtime; suppress training ---
_holder = {}
_orig = torchrl_hedge.run_torchrl_execution

def _capture(bundle, runtime):
    _holder['bundle'] = bundle
    _holder['runtime'] = runtime
    # Force the eval path (no training) regardless of what runtime says.
    runtime = {**runtime, 'execution_mode': 'simulate_only'}
    return torchrl_hedge.evaluate_torchrl_policy(bundle, runtime)

torchrl_hedge.run_torchrl_execution = _capture


# --- Pull JSON config from policy_test.py and force simulate_only ---
with open('policy_test.py') as f:
    src = f.read()
m = re.search(r"json\s*=\s*'''(.*?)'''", src, re.DOTALL)
data = jsonlib.loads(m.group(1))
calc = data['Calc']['Calculation']
calc['Execution_Mode'] = 'simulate_only'
calc['Simulation_Batches'] = 1
calc['Batch_Size'] = 1024
calc['Random_Seed'] = 42

print(f'Base_Date:           {calc["Base_Date"][".Timestamp"]}')
print(f'Simulation_Batches × Batch_Size: {calc["Simulation_Batches"]} × {calc["Batch_Size"]}')


# --- Build bundle ---
try:
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(data), 'tb.json'))
    cx.run_job()
finally:
    torchrl_hedge.run_torchrl_execution = _orig

bundle = _holder['bundle']
runtime = _holder['runtime']
print(f'\nBundle keys:                {sorted(bundle.keys())[:10]} ...')
print(f'Tradables (action space):   {list(runtime["names"]["tradables"])}')
print(f'Action instruments:         {list(runtime["names"]["action_instruments"])}')
print(f'Liabilities:                {list(runtime["names"].get("liabilities", []))}')


# --- Reuse the canonical textbook rollout from diagnose_vs_textbook.py ---
import importlib.util
spec = importlib.util.spec_from_file_location('dvt', 'diagnose_vs_textbook.py')
dvt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dvt)

# Pick the hedge instrument — diagnose_vs_textbook defaults to PL_JUL_2026 (the contract
# whose expiry brackets the JUL deal's averaging window).
INSTR = 'PL_JUL_2026'
print(f'\nHedge instrument:           {INSTR}')

rollout = dvt.rollout_textbook(bundle, runtime, instrument=INSTR)
print(f'\nTextbook rollout:')
print(f'  decision steps:           {len(rollout["times"])}')
print(f'  position shape:           {tuple(rollout["position"].shape)}    (T_decision, B)')
print(f'  trade shape:              {tuple(rollout["trade"].shape)}')
print(f'  pnl_excess shape:         {tuple(rollout["pnl_excess"].shape)}')
print(f'  liability shape:          {tuple(rollout["liability"].shape)}')
print(f'  net_pnl shape:            {tuple(rollout["net_pnl"].shape)}')


# --- Path summary stats ---
net = rollout['net_pnl'].numpy()
pnl_x = rollout['pnl_excess'].numpy()
liab = rollout['liability'].numpy()
print(f'\nTerminal P&L summary across {len(net)} paths:')
print(f'  net_pnl       mean={net.mean():,.2f}    std={net.std():,.2f}    min={net.min():,.2f}    p5={np.percentile(net,5):,.2f}    p95={np.percentile(net,95):,.2f}    max={net.max():,.2f}')
print(f'  hedge pnl_x   mean={pnl_x.mean():,.2f}    std={pnl_x.std():,.2f}')
print(f'  liability     mean={liab.mean():,.2f}    std={liab.std():,.2f}')

# How well does the textbook hedge offset the liability?
# Track liability dispersion vs hedged dispersion. If hedge works, std(net) << std(liab).
hedge_efficiency = 1.0 - net.std() / liab.std()
print(f'\nHedge efficiency (1 - std_net/std_liab): {hedge_efficiency:+.2%}    (positive = hedge reduces dispersion)')


# --- Per-day CSV via the canonical helper ---
import os
os.makedirs('artifacts', exist_ok=True)
csv_path = 'artifacts/textbook_hedge_paths.csv'

# write_pnl_trajectory_csv needs a "policy" object. The textbook rollout doesn't have
# one — but the helper only reads the spread/cost from runtime, not from policy. Pass None.
class _NullPolicy: pass
try:
    dvt.write_pnl_trajectory_csv(rollout, bundle, runtime, _NullPolicy(), csv_path, instrument=INSTR)
    print(f'\nWrote {csv_path}')
except Exception as e:
    print(f'\nwrite_pnl_trajectory_csv error: {e}')
    # Fallback: dump a minimal CSV manually.
    times = list(rollout['times'])
    dates = dvt._bundle_scenario_dates(bundle)
    decision_dates = [dates[t] for t in times]
    rows = []
    for k, t in enumerate(times):
        pos_t = rollout['position'][k].numpy()
        trd_t = rollout['trade'][k].numpy()
        prc_t = rollout['price'][k].numpy()
        rows.append({
            'sim_date': pd.Timestamp(decision_dates[k]).strftime('%Y-%m-%d'),
            'time_step': int(t),
            'position_mean': float(pos_t.mean()),
            'trade_mean': float(trd_t.mean()),
            'price_mean': float(prc_t.mean()),
            'price_std': float(prc_t.std()),
            'position_min': float(pos_t.min()),
            'position_max': float(pos_t.max()),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'  → wrote minimal fallback CSV at {csv_path}')
