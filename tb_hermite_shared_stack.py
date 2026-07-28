"""Base valuation + credit Monte Carlo over a HERMITE curve (throwaway; tb_ prefix).

The hedge stack is not the only consumer of `make_curve_tensor`: `Base_Revaluation` and
`Credit_Monte_Carlo` price through the same `Interpolation`, and their gathers cover EVERY
scenario row — the opposite of an inner-MC fork. Deferring the coefficient build must leave their
answers bitwise and must not cost them memory or wall.

The book is an equity option portfolio on a GBM equity discounted off a 12-tenor USD curve whose
interpolation is declared `Hermite` in `Price Factor Interpolation` — the factor-owned channel
that carries the interpolation type into `make_curve_tensor`. `HERMITE_REPO` selects which
checkout to import riskflow from, so a baseline and a candidate each run against their own code.

    HERMITE_REPO=<repo-or-worktree> python tb_hermite_shared_stack.py <out.pt>
    python tb_hermite_shared_stack.py cmp <before.pt> <after.pt>
"""
import logging, os, sys, time

REPO = os.environ.get('HERMITE_REPO', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import pandas as pd
import torch

BASE = pd.Timestamp('2024-06-28')
TENORS = [0.003, 0.083, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0]
RATES = [0.052, 0.051, 0.049, 0.047, 0.045, 0.043, 0.039, 0.037, 0.036, 0.037, 0.038, 0.039]


def snap(v):
    if torch.is_tensor(v):
        return v.detach().cpu().clone()
    if isinstance(v, np.ndarray):
        return torch.as_tensor(v.astype(np.float64) if v.dtype.kind == 'f' else v).clone()
    if isinstance(v, pd.DataFrame):
        return {str(c): snap(v[c].to_numpy()) if v[c].dtype.kind in 'fiub'
                else [repr(x) for x in v[c]] for c in v.columns}
    if isinstance(v, dict):
        return {str(k): snap(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [snap(x) for x in v]
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    return str(v)


def book(interp='Hermite'):
    """A vanilla equity book discounted off a USD curve carrying `interp` interpolation."""
    from riskflow import utils
    from riskflow.config import Config
    from riskflow.instruments import construct_instrument

    cfg = Config()
    cfg.params['System Parameters']['Base_Currency'] = 'USD'
    cfg.params['System Parameters']['Base_Date'] = BASE
    cfg.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD-OIS': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], list(map(list, zip(TENORS, RATES))))},
        'DiscountRate.USD-OIS': {'Interest_Rate': 'USD-OIS'},
        'EquityPrice.EQ': {'Spot': 100.0, 'Currency': 'USD', 'Interest_Rate': 'USD-OIS', 'Issuer': '',
                           'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.01, 0.01], [5.0, 0.01]])},
        'EquityPriceVol.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, 0.25] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.02, 5.0)])},
    }
    cfg.params['Price Models'] = {
        'GBMAssetPriceModel.EQ': {'Vol': 0.25, 'Drift': 0.02},
        # SIMULATED curve: the Hermite block is then (scen, n_tenors, batch), which is the shape
        # the fork restricts and the shape base valuation / credit MC read in full.
        'HullWhite1FactorInterestRateModel.USD-OIS': {
            'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Volatility': None,
            'Sigma': utils.Curve([], [[0.0, 0.008], [30.0, 0.008]])}}
    cfg.params['Model Configuration'].append('EquityPrice', (), 'GBMAssetPriceModel')
    cfg.params['Model Configuration'].append('InterestRate', (), 'HullWhite1FactorInterestRateModel')
    # THE CHANNEL: the factor declares its interpolation here; `construct_factor` maps it onto the
    # factor's own `Interpolation`, `update_tenors` carries the TYPE into the CurveTenor, and
    # `make_curve_tensor` reads it off `curve_component[FACTOR_INDEX_Tenor_Index].type`.
    cfg.params['Price Factor Interpolation'].append('InterestRate', (), interp)
    cfg.params['Valuation Configuration'] = {}

    deals = [construct_instrument({
        'Object': 'EquityOptionDeal', 'Reference': f'EQOPT_{k}', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD-OIS',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy' if k % 2 else 'Sell',
        'Option_Style': 'European', 'Option_Type': 'Call' if k % 3 else 'Put',
        'Strike_Price': 80.0 + 10.0 * k, 'Units': 10.0 + k,
        'Expiry_Date': BASE + pd.DateOffset(months=6 * (k + 1)),
    }, {}) for k in range(6)]
    cfg.deals = {'Attributes': {'Reference': 'shared_stack', 'Tag_Titles': ''},
                 'Deals': {'Children': [{'Instrument': d} for d in deals]},
                 'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return cfg


def instrument():
    """Count Hermite coefficient builds + rows so the gate cannot pass vacuously on a book that
    never reaches the changed branch."""
    from riskflow import utils
    tally = {'builds': 0, 'rows': 0}
    orig = utils.hermite_interpolation_tensor

    def counted(t, rate_tensor):
        tally['builds'] += 1
        tally['rows'] += int(rate_tensor.shape[0])
        return orig(t, rate_tensor)

    utils.hermite_interpolation_tensor = counted
    return tally


def measure(label, fn):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return {'label': label, 'out': snap(out), 'wall_s': round(time.perf_counter() - t0, 2),
            'peak_alloc_MiB': round(torch.cuda.max_memory_allocated() / 2**20, 1),
            'peak_reserved_MiB': round(torch.cuda.max_memory_reserved() / 2**20, 1)}


# `Dynamic_Scenario_Dates` makes scenario dates == mtm dates, so the SPARSE arm gathers 3 scenario
# rows under heavy time interpolation (alpha non-null everywhere) and the DENSE arm gathers ~157
# rows with alpha null — the two opposite ends of the row-span/route logic, on the same book.
ARMS = [(kind, dense) for kind in ('Hermite', 'HermiteRT', 'Linear') for dense in (False, True)]


def run(out_path):
    import riskflow as rf
    tally = instrument()
    out, builds = {}, 0
    for kind, dense in ARMS:
        if not dense:                                   # base valuation has no scenario grid to vary
            tally.update(builds=0, rows=0)
            bv = measure(f'BaseVal/{kind}', lambda: rf.run_baseval(
                book(kind), overrides={'MCMC_Simulations': 8192, 'Random_Seed': 5})[1])
            bv['hermite'] = dict(tally)
            out[f'bv/{kind}'] = bv
        tally.update(builds=0, rows=0)
        cmc = measure(f'CMC/{kind}/{"dense" if dense else "sparse"}', lambda: rf.run_cmc(
            book(kind), overrides={
                'Time_grid': '0d 1w(3y)', 'Batch_Size': 8192, 'Simulation_Batches': 4,
                'Random_Seed': 5, 'Percentile': '95',
                'Dynamic_Scenario_Dates': 'Yes' if dense else 'No'})[1]['Results'])
        cmc['hermite'] = dict(tally)
        out[f'cmc/{kind}/{dense}'] = cmc
        builds += cmc['hermite']['builds'] if kind.startswith('Hermite') else 0
    assert builds, 'the Hermite branch never ran'
    torch.save(out, out_path)
    for r in out.values():
        print(f"{r['label']:22} wall {r['wall_s']:7.2f}s  peak alloc {r['peak_alloc_MiB']:9.1f} MiB"
              f"  reserved {r['peak_reserved_MiB']:9.1f} MiB  hermite {r['hermite']}")
    print('->', out_path)


# Wall-clock stats and object reprs are run-to-run noise, not results.
IGNORE = ('.Stats.', '.Netting')


def walk(a, b, path, bad):
    if any(k in path for k in IGNORE):
        return
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.shape != b.shape:
            bad.append(f'{path}: shape {tuple(a.shape)} != {tuple(b.shape)}')
        elif a.contiguous().numpy().tobytes() != b.contiguous().numpy().tobytes():
            # byte comparison, so NaN == NaN (an unpriceable cell is NaN in both arms)
            d = float((a.to(torch.float64) - b.to(torch.float64)).abs().nan_to_num().max()) \
                if a.is_floating_point() else -1
            bad.append(f'{path}: NOT BITWISE (max|d|={d:.3g})')
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            bad.append(f'{path}: keys {sorted(set(a) ^ set(b))}')
        for k in sorted(set(a) & set(b)):
            walk(a[k], b[k], f'{path}.{k}', bad)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            bad.append(f'{path}: len {len(a)} != {len(b)}')
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f'{path}[{i}]', bad)
    elif isinstance(a, float) and isinstance(b, float):
        if repr(a) != repr(b):
            bad.append(f'{path}: {a!r} != {b!r}')
    elif a != b:
        bad.append(f'{path}: {a!r} != {b!r}')


def leaves(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from leaves(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from leaves(x)
    else:
        yield v


def cmp(f1, f2):
    a, b = torch.load(f1, weights_only=False), torch.load(f2, weights_only=False)
    bad = []
    for k in a:
        walk(a[k]['out'], b[k]['out'], k, bad)
        print(f"{a[k]['label']:22} wall {a[k]['wall_s']:7.2f} -> {b[k]['wall_s']:7.2f}s   "
              f"alloc {a[k]['peak_alloc_MiB']:9.1f} -> {b[k]['peak_alloc_MiB']:9.1f} MiB   "
              f"reserved {a[k]['peak_reserved_MiB']:9.1f} -> {b[k]['peak_reserved_MiB']:9.1f} MiB   "
              f"hermite {a[k]['hermite']} -> {b[k]['hermite']}")
    n = sum(1 for _ in leaves({k: a[k]['out'] for k in a}))
    if bad:
        print(f'MISMATCH ({len(bad)}) over {n} leaves:')
        for m in bad[:40]:
            print('  ' + m)
        sys.exit(1)
    print(f'BITWISE IDENTICAL over {n} leaves')


if __name__ == '__main__':
    logging.disable(logging.WARNING)
    if sys.argv[1] == 'cmp':
        cmp(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1])
