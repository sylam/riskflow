"""Cross-market DUAL-STRIP validation: hedge the pure-LME liability with BOTH the CME
futures strip (basis risk, Implied_Basis=LME_CME) AND an LME-native July future.

The verified baseline is already cross-market (3 CME futures vs an LME liability, stochastic
regime-keyed basis). The new question: given simultaneous access to a basis-FREE instrument
on the fixing index, does DiffSolverV2 route the fixing-month hedge into it, and what is
basis-free access worth in tail terms?

The LME leg is pure JSON: an identity basis (`CommodityBasis.LME_FLAT`, Spot=0, observing
PLATINUM_LME) + a degenerate `BasisLinkedSpotModel.LME_FLAT` (A=Phi=Mu=0, Sigma_By_State=0)
so F_LME = S_LME * exp(carry+repo) exactly. No framework code.

Variants: baseline (3 CME futures) vs dual (+ PL_LME_JUL_2026). Same seed/profile.
JSON-is-the-contract: load_json + run_job only.
"""
import copy
import csv
import datetime
import json
import logging
import os
import sys

import riskflow as rf

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'tests', 'fixtures', 'policy_test_simulate_only.json')

PROFILES = {
    'smoke': dict(batch=48, inner=8, t_min=100, iters=30),
    'full': dict(batch=192, inner=32, t_min=90, iters=50),
    'deep': dict(batch=256, inner=32, t_min=60, iters=60),
}

LME_NAME = 'PL_LME_JUL_2026'


def add_lme_leg(cfg):
    """Add the LME-native July future + identity basis. CME July settles 2047.07 with basis
    -32.5375 (CME = LME + basis), so the LME-side July future marks ~2079.61."""
    calc = cfg['Calc']['Calculation']
    hp = calc['Hedging_Problem']
    hp['Tradable_Instruments']['CommodityFutureDeal'][LME_NAME] = {
        'Maturity_Date': {'.Timestamp': '2026-07-29'},
        'Currency': 'USD', 'Carry': 'PLATINUM_CARRY', 'Repo_Rate': 'USD-SOFR',
        'Implied_Basis': 'LME_FLAT', 'Contract_Size': 50,
    }
    hp['Evaluator']['Position_Limits'][LME_NAME] = {'Min_Position': -50, 'Max_Position': 0}
    ps = hp['Portfolio_State']
    ps['Positions'][LME_NAME] = 0
    ps['Settlement_Prices'][LME_NAME] = 2079.6060
    ps['Initial_Margin'][LME_NAME] = {'Method': 'per_contract', 'Amount': 9000.0}
    emd = cfg['Calc']['MergeMarketData']['ExplicitMarketData']
    emd['Price Factors']['CommodityBasis.LME_FLAT'] = {
        'Spot': 0.0, 'Observed_Commodity': 'PLATINUM_LME'}
    emd.setdefault('Price Models', {})['BasisLinkedSpotModel.LME_FLAT'] = {
        'A': 0.0, 'Phi': 0.0, 'Nu': 5.0, 'Mu': 0.0,
        'Sigma_By_State': [0.0, 0.0, 0.0],
        'Calibration_DT_Years': 0.003968253968253968}
    return cfg


def build_cfg(template, dual, prof, seed):
    cfg = copy.deepcopy(template)
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    # a solve is a stream: fit batches, then a held-out one. (B, 1)+OOS 0.5 -> (B/2, 2)
    calc['Batch_Size'], calc['Simulation_Batches'] = prof['batch'] // 2, 2
    calc['Inner_Sub_Batch'] = prof['inner']
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Inner_Antithetic'] = 'Yes'
    calc['Random_Seed'] = seed
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes'
    hp['Solver'] = {
        'Object': 'DiffSolverV2',
        'Training_Action_Grid_Levels_Per_Axis': 5,
        'Training_Action_Chunk_Size': 64,
        'T_Min': prof['t_min'],
        'DiffV2_Fit_Iters': prof['iters'],
    }
    if dual:
        add_lme_leg(cfg)
    return cfg


def main():
    profile = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1234
    prof = PROFILES[profile]
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('artifacts', 'daily_runs', f'{stamp}_cross_market_{profile}_s{seed}')
    os.makedirs(run_dir, exist_ok=True)
    template = json.load(open(FIXTURE))

    rows = []
    for variant in ('baseline', 'dual'):
        cfg = build_cfg(template, variant == 'dual', prof, seed)
        json.dump(cfg, open(os.path.join(run_dir, variant + '.json'), 'w'), indent=1, default=str)
        hedge_names = list(cfg['Calc']['Calculation']['Hedging_Problem']
                           ['Tradable_Instruments']['CommodityFutureDeal'].keys())
        logging.info('=== %s (%s, seed=%d) hedges=%s ===', variant, profile, seed, hedge_names)
        cx = rf.Context()
        cx.load_json((json.dumps(cfg), variant + '.json'))
        _, result = cx.run_job()
        diag = (result.evaluation_summary or {}).get('diagnostics') or {}
        v = diag.get('verdict') or {}
        g = v.get('greedy') or {}
        rows.append({
            'variant': variant, 'V_0': diag.get('V_0'), 'bounded': diag.get('bounded'),
            'u_mean': g.get('u_mean'), 'wT_mean': g.get('wT_mean'),
            'wT_p5': g.get('wT_p5'), 'wT_cvar5': g.get('wT_cvar5'),
            'hedges': hedge_names, 'mean_abs_q': v.get('greedy_mean_abs_q'),
            'q_t0': v.get('greedy_q_first'), 'q_mid': v.get('greedy_q_mid'),
            'textbook_u': (v.get('textbook') or {}).get('u_mean'),
            'textbook_p5': (v.get('textbook') or {}).get('wT_p5'),
            'textbook_cvar5': (v.get('textbook') or {}).get('wT_cvar5'),
        })
        json.dump(diag, open(os.path.join(run_dir, f'diag_{variant}.json'), 'w'),
                  indent=1, default=str)

    print('\n===== CROSS-MARKET DUAL-STRIP (%s, seed=%d) =====' % (profile, seed))
    for r in rows:
        print(f"{r['variant']:<10} u={r['u_mean']:.4f} mean={r['wT_mean']:.0f} "
              f"p5={r['wT_p5']:.0f} cvar5={r['wT_cvar5']:.0f} V_0={r['V_0']:.4f}")
        print(f"           allocation mean|q|: "
              + ', '.join(f'{n}={q:.2f}' for n, q in zip(r['hedges'], r['mean_abs_q'])))
        print(f"           textbook: u={r['textbook_u']:.4f} p5={r['textbook_p5']:.0f} "
              f"cvar5={r['textbook_cvar5']:.0f}")

    with open(os.path.join(run_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('run dir:', run_dir)


if __name__ == '__main__':
    main()
