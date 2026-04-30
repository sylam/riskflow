"""Average-price-swap benchmark hedges. Provides several rule-based policies for the JUL_2026
APS so we can bound the value an ML policy needs to add over a vanilla rule:

  no_trade           — unhedged baseline (lower bound on net P&L variance).
  quadratic_ramp     — quadratic ramp 0 → -50 by averaging start, linear unwind during averaging.
  delta1_immediate   — textbook APS replication: full -50 from day 1, linear unwind during avg.
  constant_short     — full -50 from day 1, no unwind (illustrates why unwinding matters).
"""
import json
import pandas as pd
import torch
import riskflow as rf
from riskflow import torchrl_hedge


def get_bundle_and_runtime():
    with open('policy_test.py') as f:
        src = f.read()
    js = src.split("json = '''", 1)[1].split("'''", 1)[0]
    cfg = json.loads(js)
    cfg['Calc']['Calculation']['Execution_Mode'] = 'evaluate_policy'
    cfg['Calc']['Calculation']['Hedging_Problem']['Optimizer'] = {
        'Object': 'PPO', 'Epochs': 1, 'Validation_Fraction': 0.0,
        'Decision_Interval_Curriculum': [{'Start_Epoch': 1, 'End_Epoch': 1, 'Interval_Business_Days': 1}],
        'Seed': 42,
    }
    holder = {}
    original = torchrl_hedge.run_torchrl_execution

    def stub(bundle, runtime):
        holder['bundle'] = bundle
        holder['runtime'] = runtime
        return None

    torchrl_hedge.run_torchrl_execution = stub
    try:
        cx = rf.Context()
        cx.load_json((json.dumps(cfg), 'aps_bench.json'))
        cx.run_job()
    finally:
        torchrl_hedge.run_torchrl_execution = original
    return holder['bundle'], holder['runtime']


def _print_metrics(label, net_pnl, final_pos):
    print(
        f'{label:<22}  '
        f'avg=${net_pnl.mean().item():>+12,.0f}  '
        f'med=${torch.quantile(net_pnl.double(), 0.5).item():>+12,.0f}  '
        f'worst=${net_pnl.min().item():>+12,.0f}  '
        f'best=${net_pnl.max().item():>+12,.0f}  '
        f'std=${net_pnl.std().item():>10,.0f}  '
        f'final_pos={final_pos:+.1f}'
    )


def run_target_policy(
    bundle, runtime, target_fn, *,
    instrument='PL_JUL_2026', label='policy',
):
    """Generic runner: at each decision step, computes target_fn(t, t0_idx, tn_idx) and trades the
    delta from current position to that target (clamped to ±10 per step)."""
    from riskflow.torchrl_hedge import (
        build_shared_state, step_runtime_state, reward_and_terminal_payoff,
        _decision_time_indices, _optimizer_settings, _last_time_index, _bundle_scenario_dates,
    )
    settings = _optimizer_settings(runtime)
    decision_indices = _decision_time_indices(bundle, settings, epoch=None, evaluation=True)
    decision_set = set(int(i) for i in decision_indices)
    instrument_order = tuple(runtime['names']['action_instruments'])
    dates = _bundle_scenario_dates(bundle)
    avg_start_ts = pd.Timestamp('2026-07-01')
    tn_idx = next(i for i, d in enumerate(dates) if d >= avg_start_ts)

    state = build_shared_state(bundle, runtime)
    t0_idx = int(state['time_index'])
    batch_size = int(next(iter(state['positions'].values())).shape[0])
    device = bundle['time_grid_days'].device
    last_idx = _last_time_index(bundle)

    transition = None
    while int(state['time_index']) < last_idx:
        t = int(state['time_index'])
        if t in decision_set:
            target = target_fn(t, t0_idx, tn_idx)
            target_int = int(round(target))
            current = int(round(state['positions'][instrument][0].item()))
            trade = max(min(target_int - current, 10), -10)
            action = {
                'trade_deltas': {
                    name: torch.full(
                        (batch_size,),
                        float(trade if name == instrument else 0),
                        dtype=torch.float32, device=device,
                    )
                    for name in instrument_order
                },
            }
        else:
            action = None
        next_state = step_runtime_state(state, action, bundle, runtime)
        transition = reward_and_terminal_payoff(state, next_state, bundle, runtime)
        state = next_state

    pnl_excess = transition['pnl_excess']
    liability = transition['liability_value']
    net_pnl = pnl_excess + liability
    final_pos = state['positions'][instrument].mean().item()
    _print_metrics(label, net_pnl, final_pos)
    return {
        'net_pnl': net_pnl.detach().cpu(),
        'pnl_excess': pnl_excess.detach().cpu(),
        'liability': liability.detach().cpu(),
        'tn_idx': tn_idx, 't0_idx': t0_idx,
    }


def no_trade_baseline(bundle, runtime, *, label='no_trade'):
    from riskflow.torchrl_hedge import (
        build_shared_state, step_runtime_state, reward_and_terminal_payoff, _last_time_index,
    )
    state = build_shared_state(bundle, runtime)
    last_idx = _last_time_index(bundle)
    transition = None
    while int(state['time_index']) < last_idx:
        next_state = step_runtime_state(state, None, bundle, runtime)
        transition = reward_and_terminal_payoff(state, next_state, bundle, runtime)
        state = next_state
    net_pnl = transition['pnl_excess'] + transition['liability_value']
    final_pos = state['positions']['PL_JUL_2026'].mean().item()
    _print_metrics(label, net_pnl, final_pos)
    return {'net_pnl': net_pnl.detach().cpu()}


def target_quadratic_ramp(t, t0_idx, tn_idx, *, short_at_start=50, buyback_days=21):
    if t <= tn_idx:
        u = (t - t0_idx) / max(tn_idx - t0_idx, 1)
        return -short_at_start * (u * u)
    if t < tn_idx + buyback_days:
        u = (t - tn_idx) / buyback_days
        return -short_at_start * (1.0 - u)
    return 0.0


def target_delta1_immediate(t, t0_idx, tn_idx, *, short_at_start=50, buyback_days=21):
    """Textbook APS delta hedge: full short from day 1, linear unwind through the averaging window."""
    if t < tn_idx:
        return -short_at_start
    if t < tn_idx + buyback_days:
        u = (t - tn_idx) / buyback_days
        return -short_at_start * (1.0 - u)
    return 0.0


def target_constant_short(t, t0_idx, tn_idx, *, short_at_start=50, buyback_days=21):
    """Full short throughout — no unwinding during averaging. Shows the cost of holding past
    fixings (you pay VM on shorts that no longer correspond to remaining exposure)."""
    # Hold full short until the very last step, then flatten so terminal accounting closes cleanly.
    return -short_at_start


def main():
    bundle, runtime = get_bundle_and_runtime()
    print()
    header = (
        f'{"policy":<22}  {"avg":>14}  {"med":>14}  {"worst":>14}  '
        f'{"best":>14}  {"std":>11}  final_pos'
    )
    print(header)
    print('-' * len(header))

    no_trade_baseline(bundle, runtime, label='no_trade')
    run_target_policy(bundle, runtime, target_quadratic_ramp,    label='quadratic_ramp')
    run_target_policy(bundle, runtime, target_delta1_immediate,  label='delta1_immediate')
    run_target_policy(bundle, runtime, target_constant_short,    label='constant_short')


if __name__ == '__main__':
    main()
