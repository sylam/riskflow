"""PHASE B follow-up (eval only, no training):
1) BIT-IDENTITY: regenerate the roll-clip roll (corridor-FREE full48 checkpoints rolled under the
   band-0.40 corridor — the cr_ re-roll recipe: copy checkpoints into the run dir so one_trade
   skips training) for --month, then assert element-exact identity of greedy_q_traj vs the
   corridor-TRAINED roll from tb_runs.
2) IN-SIM VERDICT: evaluate BOTH checkpoint sets frozen under the corridor on a fresh simulated
   verdict (batch 1024, same seed => same outer worlds, paired): u_mean / E[W_T] / p5 / cvar5.
"""
import os, sys, json, argparse, logging, copy, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import riskflow
assert 'PycharmProjects' in riskflow.__file__, f'wrong riskflow: {riskflow.__file__}'

from production_walk_forward import build_corrected_archive, build_deal_config, one_trade
from production_solver import apply_config, run
from tb_train_in_corridor import MD_MAP, trade_date_of

ROOT = os.path.dirname(os.path.abspath(__file__))
WF = os.path.join(ROOT, 'artifacts', 'walk_forward')
BASE = os.environ['TB_BASE']
SEEDS = [7, 42, 314]
BAND = 0.40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', required=True)
    args = ap.parse_args()
    month = args.month
    tag = month

    raw = pd.read_csv(os.path.join(ROOT, 'data', 'pl_exp.csv'), index_col=0, parse_dates=True)
    arch = build_corrected_archive(raw)
    template = json.load(open(os.path.join(ROOT, 'tests', 'fixtures', 'policy_test_simulate_only.json')))
    md = MD_MAP[month]
    trade_date = trade_date_of(month)

    free_ckpts = [f'{WF}/full48_gpu0/value_fn_{tag}_s{s}.pt' for s in SEEDS]
    trained_dir = os.path.join(BASE, f'run_{month}_b040')
    trained_ckpts = [os.path.join(trained_dir, f'value_fn_{tag}_s{s}.pt') for s in SEEDS]
    for p in free_ckpts + trained_ckpts:
        assert os.path.exists(p), p

    # ---- 1) regenerate roll-clip: copy corridor-free ckpts into a fresh run dir, one_trade skips
    #         training and rolls the ensemble under the corridor (identical recipe to cr_ re-roll).
    regen_dir = os.path.join(BASE, f'regen_clip_{month}_b040')
    os.makedirs(regen_dir, exist_ok=True)
    for src, s in zip(free_ckpts, SEEDS):
        dst = os.path.join(regen_dir, f'value_fn_{tag}_s{s}.pt')
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    aargs = argparse.Namespace(margin=8.0, volume=2500.0, batch=2048, fit_iters=40,
                               seeds=list(SEEDS), roll_inner=512, delta_corridor=BAND)
    logging.info('=== REGEN roll-clip %s b%.2f (corridor-FREE ckpts, roll only) ===', month, BAND)
    rec = one_trade(template, arch, trade_date, md, aargs, regen_dir, tag)
    logging.info('REGEN row: greedy=%s churn=%s PASS=%s', rec['greedy_usd_oz'], rec['churn'], rec['bound_pass'])

    d_clip = json.load(open(os.path.join(regen_dir, f'diag_{tag}.json')))
    d_ti = json.load(open(os.path.join(trained_dir, f'diag_{tag}.json')))
    q_clip = np.array(d_clip['stepper_verdict']['greedy_q_traj'], dtype=np.float64)
    q_ti = np.array(d_ti['stepper_verdict']['greedy_q_traj'], dtype=np.float64)
    t_clip = list(d_clip['stepper_verdict']['greedy_q_t'])
    t_ti = list(d_ti['stepper_verdict']['greedy_q_t'])
    same_shape = q_clip.shape == q_ti.shape
    l1 = float(np.abs(q_clip - q_ti).sum()) if same_shape else float('nan')
    exact = bool(same_shape and (q_clip == q_ti).all() and t_clip == t_ti)
    w_clip = d_clip['stepper_verdict']['greedy']['wT_mean']
    w_ti = d_ti['stepper_verdict']['greedy']['wT_mean']
    print(f'\nBITID {month} b{BAND}: shape {q_clip.shape} vs {q_ti.shape} | steps match={t_clip == t_ti} | '
          f'L1(q_clip - q_trained)={l1} | element-exact={exact} | wT clip={w_clip} trained={w_ti} '
          f'dW={w_ti - w_clip}')

    # ---- 2) in-sim verdict under the corridor, both checkpoint sets, same worlds (seed 7) -------
    cfg, _ = build_deal_config(template, arch, trade_date, md, 8.0, 2500.0, delta_corridor=BAND)
    results = {}
    for label, ckpts in (('corridor-FREE', free_ckpts), ('corridor-TRAINED', trained_ckpts)):
        ev = apply_config(copy.deepcopy(cfg), batch=1024, seed=7, load=ckpts,
                          randomize_initial_state=False)
        logging.info('=== INSIM EVAL %s %s (batch=1024, corridor active) ===', month, label)
        diag = run(ev, f'insim_{month}_{label}')
        v = (diag.get('verdict') or {}).get('greedy') or {}
        results[label] = {k: v.get(k) for k in ('u_mean', 'wT_mean', 'wT_p5', 'wT_cvar5')}
        results[label]['verdict_is_oos'] = diag.get('verdict_is_oos')
        tb = (diag.get('verdict') or {}).get('textbook') or {}
        nh = (diag.get('verdict') or {}).get('nohedge') or {}
        results[label]['textbook_u'] = tb.get('u_mean')
        results[label]['nohedge_u'] = nh.get('u_mean')
        print(f"INSIM {month} {label}: u={v.get('u_mean'):+.4f} E[W_T]={v.get('wT_mean'):+,.0f} "
              f"p5={v.get('wT_p5'):+,.0f} cvar5={v.get('wT_cvar5'):+,.0f} "
              f"(textbook u={tb.get('u_mean')}, nohedge u={nh.get('u_mean')})")

    out = {'month': month, 'band': BAND,
           'bit_identity': {'exact': exact, 'l1': l1, 'shape': list(q_clip.shape),
                            'steps_match': t_clip == t_ti,
                            'wT_clip': w_clip, 'wT_trained': w_ti,
                            'regen_greedy_usd_oz': rec['greedy_usd_oz'], 'regen_churn': rec['churn']},
           'insim': results}
    path = os.path.join(BASE, f'tb_insim_{month}_b040.json')
    json.dump(out, open(path + '.tmp', 'w'), indent=1, default=str)
    os.replace(path + '.tmp', path)
    logging.info('WROTE %s', path)


if __name__ == '__main__':
    main()
