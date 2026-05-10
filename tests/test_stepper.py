"""End-to-end test for the interactive `BundleStepper`.

Exercises three scenarios:
  1. No-trade rollout — terminal P&L should match the framework's no_trade baseline in
     `evaluation_summary.reference.no_trade.metrics`.
  2. Textbook hedge expressed as a stepper loop — short Volume/cs from t=0, linearly
     buy back over the averaging window. Verifies the stepper can express any
     deterministic policy without touching framework internals.
  3. Counterfactual via deepcopy — fork the stepper mid-rollout, run two different
     action streams from the same checkpoint, confirm independence.

JSON in, run_job, drive the stepper through `result.create_stepper()`. No internal
imports.
"""
import copy
import json as jsonlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import riskflow as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures',
                        'policy_test_simulate_only.json')


def _run_fixture():
    cfg = jsonlib.load(open(FIXTURE))
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'stepper_test.json'))
    _, result = cx.run_job()
    return result


def test_no_trade_stepper_matches_framework_baseline():
    result = _run_fixture()
    stepper = result.create_stepper()
    last = None
    while not stepper.done:
        last = stepper.step(None)  # zero trades every step
    # Terminal P&L from the stepper run
    pnl = (last['transition_pnl_excess'] + last['transition_liability_value']).to(dtype=torch.float64)
    stepper_mean = float(pnl.mean().item())
    stepper_worst = float(pnl.min().item())

    baseline = result.evaluation_summary['reference']['no_trade']['metrics']
    fw_mean = float(baseline['average_net_pnl'])
    fw_worst = float(baseline['worst_net_pnl'])

    assert abs(stepper_mean - fw_mean) < 1.0, f"stepper mean ${stepper_mean:,.0f} ≠ framework ${fw_mean:,.0f}"
    assert abs(stepper_worst - fw_worst) < 1.0, f"stepper worst ${stepper_worst:,.0f} ≠ framework ${fw_worst:,.0f}"
    print(f"test_no_trade_stepper_matches_framework_baseline: PASS  "
          f"(mean=${stepper_mean:+,.0f}, worst=${stepper_worst:+,.0f})")


def test_textbook_hedge_as_stepper_loop():
    """Express the textbook averaging hedge entirely in client code via the stepper:
    short Volume/cs from t=0, ramp linearly to 0 across the averaging window."""
    result = _run_fixture()
    runtime = result.runtime
    bundle = result.torchrl_bundle

    # Pull deal volume + averaging window from runtime/liabilities (problem-specific data
    # but read via public surface — the runtime dict is part of the public contract).
    liab = next(iter(runtime['liabilities'].values()))
    items = liab['params']['Payments']['Items']
    period_start = items[0]['Period_Start']
    period_end = items[0]['Period_End']
    deal_volume = float(items[0]['Volume'])
    hedge_instrument = 'PL_JUL_2026'
    contract_size = float(runtime['tradables'][hedge_instrument]['contract_size'])
    short_at_start = int(round(deal_volume / contract_size))

    # Find the averaging window's t-index range from scenario_dates
    import pandas as pd
    period_start = pd.Timestamp(period_start)
    period_end = pd.Timestamp(period_end)
    dates = bundle['scenario_dates']
    bday = bundle['meta']['business_day']
    last_trade = runtime['tradables'][hedge_instrument].get('last_trade_date')
    window_end = min(pd.Timestamp(last_trade), period_end) if last_trade else period_end
    buyback_indices = [i for i, d in enumerate(dates)
                       if period_start <= d <= window_end and bday.is_on_offset(d)]
    tn_idx = next(i for i, d in enumerate(dates) if d >= period_start)
    N = len(buyback_indices)
    target_at_idx = {idx: -short_at_start * (1.0 - (i + 1) / N) for i, idx in enumerate(buyback_indices)}

    stepper = result.create_stepper()
    last = None
    while not stepper.done:
        if not stepper.is_decision_step:
            last = stepper.step(None)
            continue
        t = stepper.time_index
        if t < tn_idx:
            target = -short_at_start
        elif t in target_at_idx:
            target = target_at_idx[t]
        else:
            target = 0.0
        cur_pos = float(stepper.observe()['positions'][hedge_instrument][0].item())
        delta = int(round(target)) - int(round(cur_pos))
        last = stepper.step({hedge_instrument: delta})

    pnl = (last['transition_pnl_excess'] + last['transition_liability_value']).to(dtype=torch.float64)

    # Drop the per-day per-instrument CSV + terminal summary (worst/p5/mean/p95/best),
    # same shape as result.write_diagnostic_csvs but for this textbook trajectory.
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_stepper_out')
    stepper.write_diagnostic_csvs(out_dir, label='textbook')
    paths_csv = os.path.join(out_dir, 'textbook_paths.csv')
    summary_csv = os.path.join(out_dir, 'textbook_summary.csv')
    assert os.path.exists(paths_csv) and os.path.exists(summary_csv)
    import pandas as pd
    summary = pd.read_csv(summary_csv)
    cases_present = set(summary['metric'].tolist())
    assert {'mean', 'std', 'min', 'p5', 'p95', 'max'}.issubset(cases_present), \
        f"summary missing rows: {cases_present}"
    paths = pd.read_csv(paths_csv)
    assert set(paths['case'].unique()) == {'worst', 'p5', 'p95', 'best', 'mean'}, \
        f"paths cases: {set(paths['case'].unique())}"
    print(f"test_textbook_hedge_as_stepper_loop: PASS  "
          f"(mean=${float(pnl.mean()):+,.0f}, std=${float(pnl.std()):,.0f}, "
          f"worst=${float(pnl.min()):+,.0f})  CSVs → {out_dir}/")


def test_counterfactual_branching_via_deepcopy():
    """Fork the stepper after some steps; run two different action streams from the
    same checkpoint; confirm the branches diverge."""
    result = _run_fixture()
    stepper = result.create_stepper()
    # Advance 30 steps with zero actions
    for _ in range(30):
        stepper.step(None)

    branch_a = copy.deepcopy(stepper)
    branch_b = copy.deepcopy(stepper)
    last_a = None; last_b = None
    while not branch_a.done:
        last_a = branch_a.step(None)  # do nothing
    while not branch_b.done:
        if branch_b.is_decision_step:
            last_b = branch_b.step({'PL_JUL_2026': -10})  # accumulate short
        else:
            last_b = branch_b.step(None)

    pnl_a = last_a['transition_pnl_excess'].to(dtype=torch.float64)
    pnl_b = last_b['transition_pnl_excess'].to(dtype=torch.float64)
    diff = float((pnl_a - pnl_b).abs().mean().item())
    assert diff > 1.0, f"branches did not diverge — got identical P&L (diff={diff})"
    print(f"test_counterfactual_branching_via_deepcopy: PASS  "
          f"(branch_a mean pnl=${float(pnl_a.mean()):+,.0f}, branch_b=${float(pnl_b.mean()):+,.0f})")


def test_write_diagnostic_csvs_with_unmirrored_symlog_runtime():
    """Regression: `result.write_diagnostic_csvs` invokes `_diag_rollout_policy`, which
    must mirror utility_scale onto runtime BEFORE calling reward_and_terminal_payoff.
    Earlier, that call site relied on training to have already mutated the runtime;
    in any pipeline that constructs a fresh runtime + bundle and goes straight to
    write_diagnostic_csvs (offline analysis, deserialized result, post-hoc CSV regen),
    `_require_utility_scale` would raise. Simulate that scenario by clearing
    `runtime['objective']['utility_scale']` and confirm the diagnostic path still works.
    """
    cfg = jsonlib.load(open(FIXTURE))
    obj = cfg['Calc']['Calculation']['Hedging_Problem']['Objective']
    obj.update({'Object': 'AsymmetricUtility_Symlog', 'Floor_Penalty': 10.0,
                'Surplus_Reward': 1.0, 'Power': 1.0})
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'unmirrored.json'))
    _, result = cx.run_job()

    # Simulate the unmirrored path: clear utility_scale from runtime; the bundle still
    # has it (set at bundle build), but the runtime is fresh as far as reward computation
    # is concerned.
    result.runtime['objective'].pop('utility_scale', None)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_stepper_out_unmirrored')
    result.write_diagnostic_csvs(out_dir)
    assert os.path.exists(os.path.join(out_dir, 'ml_paths.csv'))
    assert os.path.exists(os.path.join(out_dir, 'summary.csv'))
    print("test_write_diagnostic_csvs_with_unmirrored_symlog_runtime: PASS")


if __name__ == '__main__':
    test_no_trade_stepper_matches_framework_baseline()
    test_textbook_hedge_as_stepper_loop()
    test_counterfactual_branching_via_deepcopy()
    test_write_diagnostic_csvs_with_unmirrored_symlog_runtime()
    print("\nAll stepper tests passed.")
