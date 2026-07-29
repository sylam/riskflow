"""§6.7 end-to-end smoke for GARCHSpotModel — repo-root, untracked (tb_ convention) so it
survives the OS tmp-cleaner. GPU1 only. Four solves: GARCH train seed 7 (save), identical
re-run (determinism pair), frozen-checkpoint eval (provenance reload), HMM reference.
Asserts: bounded True everywhere, V_0 bit-match on the pair, market_dim 7 (GARCH) vs 9 (HMM)."""
import os, sys, json, logging

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'                          # GPU1 only — GPU0 has the user's job
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO); sys.path.insert(0, REPO)
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
import riskflow as rf

print('riskflow', rf.__file__, flush=True)
CKPT = os.path.join(REPO, 'tb_garch_smoke_vf.pt')
GARCH = './artifacts/MarketDataRF_platinum_garch.json'
HMM = './artifacts/MarketDataRF_platinum_calibrated_cme.json'


def cfg_for(market_file):
    base = json.load(open('artifacts/platinum_hedge_shipping.json'))
    calc = base['Calc']['Calculation']
    base['Calc']['MergeMarketData']['MarketDataFile'] = market_file
    calc['Batch_Size'] = 512
    calc['Simulation_Batches'] = 1
    calc['Inner_Sub_Batch'] = 32
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Hedging_Problem']['Solver']['DiffV2_Fit_Iters'] = 5
    calc['Hedging_Problem']['Randomize_Initial_State'] = 'No'
    return base


def run(market_file, seed, save=None, load=None):
    c = cfg_for(market_file)
    cc = c['Calc']['Calculation']; cc['Random_Seed'] = seed
    sol = cc['Hedging_Problem']['Solver']
    if save:
        sol['DiffV2_Save_Value_Fn'] = os.path.abspath(save)
    if load:
        sol['DiffV2_Load_Value_Fn'] = [os.path.abspath(load)]
    cx = rf.Context(); cx.load_json((json.dumps(c, default=str), 'smoke.json'))
    _, res = cx.run_job()
    return (res.evaluation_summary or {}).get('diagnostics') or {}


def show(tag, d):
    v = d.get('verdict') or {}; g = v.get('greedy') or {}; tb = v.get('textbook') or {}
    print('[%s] bounded=%s market_dim=%s V_0=%s' % (tag, d.get('bounded'), d.get('market_dim'), d.get('V_0')), flush=True)
    print('  greedy   u=%+.4f E[W_T]=%+.1f p5=%+.1f' % (g.get('u_mean', 0), g.get('wT_mean', 0), g.get('wT_p5', 0)), flush=True)
    print('  textbook u=%+.4f E[W_T]=%+.1f p5=%+.1f' % (tb.get('u_mean', 0), tb.get('wT_mean', 0), tb.get('wT_p5', 0)), flush=True)


print('=== GARCH TRAIN seed 7 (save checkpoint) ===', flush=True)
d1 = run(GARCH, 7, save=CKPT); show('garch-train7', d1)
print('=== GARCH TRAIN seed 7 again (determinism) ===', flush=True)
d1b = run(GARCH, 7, save=CKPT + '.b'); show('garch-train7b', d1b)
print('=== GARCH EVAL frozen checkpoint (provenance reload) ===', flush=True)
d2 = run(GARCH, 7, load=CKPT); show('garch-eval7', d2)
print('=== HMM TRAIN seed 7 (market_dim reference) ===', flush=True)
dh = run(HMM, 7); show('hmm-train7', dh)

print(flush=True)
print('checkpoint exists:', os.path.exists(CKPT), flush=True)
det = d1.get('V_0') == d1b.get('V_0')
print('determinism V_0 bit-match:', det, '(%s vs %s)' % (d1.get('V_0'), d1b.get('V_0')), flush=True)
print('market_dim GARCH=%s HMM=%s (delta=%s)' % (
    d1.get('market_dim'), dh.get('market_dim'),
    (dh.get('market_dim') - d1.get('market_dim')) if (dh.get('market_dim') and d1.get('market_dim')) else '?'), flush=True)

ok = (d1.get('bounded') is True and d1b.get('bounded') is True and d2.get('bounded') is True
      and dh.get('bounded') is True and det
      and d1.get('market_dim') == 7 and dh.get('market_dim') == 9)
print('SMOKE_RESULT:', 'PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
