"""Sense-check the platinum simulator against the archive.

Runs the canonical simulate-only fixture with Execution_Mode='simulate_only' (no
training) and a small batch size, then compares simulated factor paths against
empirical archive statistics.
"""
import json as jsonlib
import os
import numpy as np
import pandas as pd
import torch

import riskflow as rf

torch.set_default_dtype(torch.float32)


def main():
    # --- Load the canonical simulate-only config and tweak batch/seed ---
    fixture = os.path.join(os.path.dirname(__file__), 'tests', 'fixtures', 'policy_test_simulate_only.json')
    data = jsonlib.load(open(fixture))

    calc = data['Calc']['Calculation']
    calc['Execution_Mode'] = 'simulate_only'
    calc['Simulation_Batches'] = 1
    calc['Batch_Size'] = 1024
    calc['Random_Seed'] = 42

    print(f'Base_Date:           {calc["Base_Date"][".Timestamp"]}')
    print(f'Time_Grid:           {calc["Time_Grid"]}')
    print(f'Batch_Size:          {calc["Batch_Size"]}')
    print(f'Simulation_Batches:  {calc["Simulation_Batches"]}')

    # --- Run ---
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(data), 'sense_check.json'))
    calc_obj, result = cx.run_job()

    bundle = result.bundle
    runtime = result.runtime
    H = int(runtime.get('history_lookback_business_days', 0))

    def _factor(substr):
        """Return (string_key, numpy_array_with_history_prefix_stripped) for the first factor
        whose key contains `substr`. Bundle factors are torch tensors of shape (H+T_sim, ...);
        we slice off the history rows so the array matches the original (T_sim, ...) shape."""
        key = next(k for k in bundle['factors'] if substr in k)
        arr = bundle['factors'][key].detach().cpu().numpy()
        return key, arr[H:]

    time_grid = bundle['time_grid_days'].detach().cpu().numpy()[H:]
    sample_factor = next(iter(bundle['factors'].values()))
    T_sim = int(sample_factor.shape[0]) - H
    N = int(sample_factor.shape[-1])
    print(f'\nT = {T_sim} time steps    N = {N} paths')
    print(f'time_grid_days[:5]:  {time_grid[:5]}')
    print(f'time_grid_days[-5:]: {time_grid[-5:]}')
    print(f'\nFactors simulated:')
    for k, t in bundle['factors'].items():
        print(f'  {k}    shape={tuple(t.shape)}')

    # --- Archive ---
    print(f'\n\n{"="*70}\nArchive comparisons\n{"="*70}')
    df = pd.read_csv('data/plat_archive.csv', index_col=0)
    df.index = pd.to_datetime(df.index)

    # --- 1. CommodityPrice.PLATINUM_LME (HMM spot) ---
    key_lme, sim_lme = _factor('CommodityPrice.PLATINUM_LME')           # (T, N)
    print(f'\n[CommodityPrice.PLATINUM_LME] sim shape: {sim_lme.shape}')
    sim_lme_diffs = np.diff(sim_lme, axis=0)                           # (T-1, N) — daily diffs
    print(f'  sim mean diff:          {sim_lme_diffs.mean():+.4f}')
    print(f'  sim std diff:           {sim_lme_diffs.std():.4f}')
    print(f'  sim kurt diff:          {pd.Series(sim_lme_diffs.flatten()).kurtosis():.4f}')
    print(f'  sim t=0 spread:         min={sim_lme[0].min():.2f}  max={sim_lme[0].max():.2f}    (should be tight — deterministic init)')

    emp_lme = df['CommodityPrice.PLATINUM_LME'].dropna().diff().dropna()
    print(f'  emp mean diff:          {emp_lme.mean():+.4f}')
    print(f'  emp std diff:           {emp_lme.std():.4f}')
    print(f'  emp kurt diff:          {emp_lme.kurtosis():.4f}')

    # --- 2. CommodityBasis.LME_CME (lagged-AR(1) basis) ---
    key_basis, sim_basis = _factor('LME_CME')                          # (T, N)
    print(f'\n[CommodityBasis.LME_CME] sim shape: {sim_basis.shape}')
    print(f'  sim t=0:                min={sim_basis[0].min():.4f}  max={sim_basis[0].max():.4f}    (should be tight — deterministic init)')
    print(f'  sim std at horizon:     {sim_basis[-1].std():.4f}')
    print(f'  sim std (all paths/t):  {sim_basis.std():.4f}')

    emp_basis = df['CommodityBasis.LME_CME,PLATINUM_LME'].dropna()
    print(f'  emp std (all):          {emp_basis.std():.4f}')
    print(f'  emp mean:               {emp_basis.mean():+.4f}')
    print(f'  emp last value:         {emp_basis.iloc[-1]:+.4f}')

    # --- 3. ForwardRate.PLATINUM_CARRY (VAR(1) carry) ---
    key_carry, sim_carry = _factor('PLATINUM_CARRY')                   # (T, n_contracts, N)
    print(f'\n[ForwardRate.PLATINUM_CARRY] sim shape: {sim_carry.shape}')
    print(f'  sim t=0 per contract:')
    for j in range(sim_carry.shape[1]):
        print(f'    knot {j}:               mean={sim_carry[0, j].mean():.6f}    std={sim_carry[0, j].std():.6f}    (t=0 should match input curve, std~0)')
    print(f'  sim t=mid per contract (carry rate at evolving T):')
    mid = sim_carry.shape[0] // 2
    for j in range(sim_carry.shape[1]):
        pct_alive = (sim_carry[mid, j] != 0).mean() * 100
        nonzero = sim_carry[mid, j][sim_carry[mid, j] != 0]
        if len(nonzero):
            print(f'    knot {j}:               alive={pct_alive:.0f}%  mean(alive)={nonzero.mean():+.6f}    std(alive)={nonzero.std():.6f}')
        else:
            print(f'    knot {j}:               expired (alive=0%)')

    emp_carry = df[[c for c in df.columns if 'PLATINUM_CARRY' in c]].dropna()
    print(f'  emp carry stats (last 30 days):')
    for c in emp_carry.columns:
        last = emp_carry[c].iloc[-30:]
        print(f'    {c}:  mean={last.mean():+.6f}  std={last.std():.6f}')

    # --- 4. InterestRate.USD-SOFR (PCA) ---
    sofr_keys = [k for k in bundle['factors'] if 'SOFR' in k]
    if sofr_keys:
        key_sofr, sim_sofr = _factor('SOFR')                            # (T, n_tenors, N)
        print(f'\n[InterestRate.USD-SOFR] sim shape: {sim_sofr.shape}')
        print(f'  sim t=0 spread:         min={sim_sofr[0].min():.6f}  max={sim_sofr[0].max():.6f}')
        print(f'  sim t=mid:              mean={sim_sofr[mid].mean():.6f}  std={sim_sofr[mid].std():.6f}')
        print(f'  sim horizon:            mean={sim_sofr[-1].mean():.6f}  std={sim_sofr[-1].std():.6f}')

    print(f'\n\nDone.')


if __name__ == '__main__':
    main()
