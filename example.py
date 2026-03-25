########################################################################
# Copyright (C)  Shuaib Osman (vretiel@gmail.com)
# This file is part of RiskFlow.
#
# RiskFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# RiskFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RiskFlow.  If not, see <http://www.gnu.org/licenses/>.
########################################################################

import os
import logging
import numpy as np
import pandas as pd


def diag_ir(out, calc, factor, tenor_point, aa=None):
    ir = out['Results']['scenarios'][factor]
    size = ir.shape[0]
    perc = np.array([50, 0.50, 2.50, 5.00, 95.00, 97.50, 99.50])
    ten_ir = calc.all_factors[factor].factor.tenors
    all = [pd.DataFrame(np.percentile(i, 100.0 - perc, axis=1), index=perc, columns=ten_ir).T for i in ir]
    comp = pd.DataFrame([x.iloc[ten_ir.searchsorted(tenor_point)] for x in all],
                        index=calc.time_grid.time_grid_years[:size])
    if aa is not None:
        return comp.reindex(comp.index.union(aa['Time (Years)'])).interpolate('index').reindex(aa['Time (Years)'])
    else:
        return comp


def check_correlation(scenarios, ir_factor, fx_factor):
    timepoints, _, sims = scenarios[ir_factor].shape
    f = [np.array([np.corrcoef(scenarios[ir_factor][:, 10 * j, i],
                               np.log(scenarios[fx_factor][:timepoints, i]))[0][1] for i in range(sims)])
         for j in range(20)]
    return np.array(f)


def bootstrap(path, rundate, reuse_cal=True):
    from riskflow.config import Config

    cfg = Config()
    cal_file = 'CVAMarketData_Calibrated_New.json'
    if reuse_cal and os.path.isfile(os.path.join(path, rundate, cal_file)):
        cfg.parse_json(os.path.join(path, rundate, 'CVAMarketData_TST.json'))
        context_tmp = Config()
        context_tmp.parse_json(os.path.join(path, rundate, cal_file))
        for factor in [x for x in context_tmp.params['Price Factors'].keys()
                       if x.startswith('HullWhite2FactorModelParameters') or
                          x.startswith('GBMAssetPriceTSModelParameters')]:
            # override it
            cfg.params['Price Factors'][factor] = context_tmp.params['Price Factors'][factor]
    else:
        # context.parse_json(os.path.join(path, rundate, cal_file))
        cfg.parse_json(os.path.join(path, '', cal_file))

    cfg.params['System Parameters']['Base_Date'] = pd.Timestamp(rundate)
    cfg.params['System Parameters'][
        'Swaption_Premiums'] = 'T:\\OTHER\\ICE\\Archive\\IR_Volatility_Swaption_{}_0000_LON.csv'.format(rundate)

    cfg.parse_calendar_file(os.path.join(path, 'calendars.cal'))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

    cfg.bootstrap()
    # cfg.write_marketdata_json(os.path.join(path, rundate, cal_file))
    # context.write_market_file(os.path.join(path, rundate, 'MarketDataCal.dat'))


def plot_matrix(df, title='Full Gamma'):
    f = plt.figure(figsize=(15, 15))
    plt.matshow(df, fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns.get_level_values(1), fontsize=3, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns.get_level_values(1), fontsize=3)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=5)
    plt.title(title, fontsize=10)


def checkdate(outputs, inputs, date):
    import glob
    loaded = [os.path.split(x)[1].replace('.json', '') for x in glob.glob(os.path.join(inputs, date, 'CrB*.json'))]
    calced = [os.path.split(x)[1].replace('.csv', '')[15:] for x in glob.glob(
        os.path.join(outputs, 'CVA_{}*.csv'.format(date)))]
    return set(loaded).difference(calced)


def get_missing_rates(aa, model_factor, from_date, to_date, MDR=500):
    '''
    Simple function to determine what can be calibrated and what's missing in the .ada file.
    Needs a crstal context and a list of all potential price factors to calibrate between the
    provided from_date and to_date.

    MDR is an acronym for Minumum Data Required (default is 500)
    '''
    from riskflow.utils import filter_data_frame

    to_calibrate = {}
    missing = {}
    data_window = filter_data_frame(aa.archive, from_date, to_date)
    for factor_name, factor_model in sorted(model_factor.items()):
        if factor_model.archive_name not in aa.archive_columns:
            missing[factor_name] = -1
        else:
            data = data_window[aa.archive_columns[factor_model.archive_name]]
            valid = (data[data.columns[1:]].count() > MDR).all() if data.shape[1] > 1 else (data.count() > MDR).all()
            if valid:
                to_calibrate[factor_name] = factor_model
            else:
                missing[factor_name] = data.count().values[0]
    return missing, to_calibrate


def calibrate_PFE(arena_path, rundate, scratch="Z:\\"):
    from riskflow.utils import calc_statistics, excel_offset, filter_data_frame
    from riskflow import AdaptivContext

    aa = AdaptivContext()
    aa.parse_json(os.path.join(arena_path, rundate, 'MarketData.json'))
    # aa.ParseCalibrationfile(os.path.join(scratch, 'calibration_prd_linux.config'))
    # aa.ParseCalibrationfile(os.path.join(scratch, 'calibration_uat_linux.config'))
    aa.parse_calibration_file(os.path.join(scratch, 'calibration.config'))

    end_date = excel_offset + pd.offsets.Day(aa.archive.index.max())
    # now work out 3 years prior for the start date
    start_date = end_date - pd.DateOffset(years=3)
    # the zar-swap curve needs to include a downturn - The earliest data we have is from 2007-01-01
    swap_start_date = pd.Timestamp('2007-01-01')

    # print out the dates used
    print('Calibrating from', start_date, 'to', end_date)

    # now check if all rates are present
    all_factors = aa.fetch_all_calibration_factors(override={})
    model_factor = all_factors['present']
    model_factor.update(all_factors['absent'])

    # report which rates are missing or have less than MDR data points and which have enough to calibrate
    missing, to_calibrate = get_missing_rates(aa, model_factor, start_date, end_date, 600)
    # EquityPrice.ZAR_SHAR
    # EquityPrice.ZAR_TFGP

    #for i in ['EquityPrice.ZAR_SHAR', 'EquityPrice.ZAR_TFGP']:
    #    del to_calibrate[i]

    smoothing_std = 2.15
    # perform an actual calibration - first, clear out the old parameters
    aa.params['Price Models'] = {}
    # if there are any issues with the data, it will be logged here
    aa.calibrate_factors(
        start_date, end_date, to_calibrate, smooth=smoothing_std, correlation_cuttoff=0.1, overwrite_correlations=True)

# PyTorch implementation of a Geared TARF one-step-survival Monte Carlo pricer
# with autograd-friendly Greeks.
#
# This implements:
# - Pathwise GBM dynamics for FX spot
# - One-step survival / truncated-normal sampling against the "PnL barrier"
#   B_i^{PnL} = K + R_{i-1} / (N_itm * gear * conv)
# - Analytic KO-in-the-step contribution: (1 - p_i) * R_{i-1}
# - Survival branch sampling using Z ~ N(0,1) truncated at Z < z_max
# - Full period cashflows:
#     * If S_i >= K: profit = (S_i - K) * N_itm * gear  (counts to target)
#     * If S_i <  K: loss   = (S_i - K) * N_otm         (does NOT count to target)
#   Target accumulation uses only ITM profit (capped by remaining target)
# - Early termination when target is hit
#
# Greeks (Delta, Vega, Rho_d) are computed with autograd (reverse-mode).
#
# Notes:
# - Units: all cashflows are in domestic currency.
# - Rates: r_d (domestic), r_f (foreign). Drift mu = r_d - r_f (under domestic measure).
# - conv can be set to 1.0 if payoff already in domestic units; otherwise include conversion.
import math
import torch
from torch.distributions.normal import Normal

normal = Normal(torch.tensor(0.0), torch.tensor(1.0))

def test_tarf():
    def price_geared_tarf_autograd(
        S0: float,
        K: float,
        sigma: float,
        r_d: float,
        r_f: float,
        T_years: float,
        n_fixings: int,
        N_itm: float,
        N_otm: float,
        gear: float = 1.0,
        target: float = 1.0,
        conv: float = 1.0,         # FX conversion to domestic (usually 1.0 if quoting domestic/foreign)
        n_paths: int = 20000,
        device: str = "cpu",
        seed: int = 1234,
        requires_grads=("S0","sigma","r_d"),
    ):
        """
        Returns:
            price (torch.Scalar): MC price estimate (domestic currency)
            grads (dict): autograd Greeks for requested inputs
            debug (dict): misc diagnostics
        """
        torch.manual_seed(seed)
        dt = T_years / n_fixings
        t_grid = torch.arange(1, n_fixings+1, device=device) * dt
        disc = torch.exp(-r_d * t_grid)  # discount factors per fixing (vector, used index-wise)

        # Parameters as tensors (enable autograd where requested)
        S0_t = torch.tensor(float(S0), device=device, requires_grad=("S0" in requires_grads))
        K_t  = torch.tensor(float(K),  device=device, requires_grad=("K" in requires_grads))
        sig_t= torch.tensor(float(sigma), device=device, requires_grad=("sigma" in requires_grads))
        rd_t = torch.tensor(float(r_d), device=device, requires_grad=("r_d" in requires_grads))
        rf_t = torch.tensor(float(r_f), device=device, requires_grad=("r_f" in requires_grads))
        N_itm_t = torch.tensor(float(N_itm), device=device)
        N_otm_t = torch.tensor(float(N_otm), device=device)
        gear_t = torch.tensor(float(gear), device=device)
        target_t = torch.tensor(float(target), device=device)
        conv_t = torch.tensor(float(conv), device=device)

        # Precompute GBM constants
        mu = rd_t - rf_t
        drift = (mu - 0.5 * sig_t**2) * dt
        vol_step = sig_t * math.sqrt(dt)

        # Random uniforms for truncated draws (fixed, no grads)
        U = torch.rand(n_paths, n_fixings, device=device)

        # Initialize path state
        S = S0_t.repeat(n_paths)
        R = target_t.repeat(n_paths)  # remaining target
        alive = torch.ones(n_paths, dtype=torch.bool, device=device)  # path still running
        L = torch.ones(n_paths, device=device)  # continuation likelihood weight product
        pv = torch.zeros(n_paths, device=device)  # accumulated PV

        # Diagnostics
        ko_step = torch.zeros(n_paths, dtype=torch.int32, device=device)  # 0 if not KO, else step index (1..n_fixings)

        for i in range(n_fixings):
            # Skip finished paths
            if not alive.any():
                break

            # Compute PnL barrier for this step (finite where alive, inf where not alive)
            # B = K + R / (N_itm*gear*conv)
            denom = (N_itm_t * gear_t * conv_t)
            # Avoid division by zero (if N_itm=0, there is no profit accumulation; treat barrier as +inf)
            safe_denom = torch.where(denom>0, denom, torch.tensor(1e16, device=device))
            B = K_t + torch.where(denom>0, R / safe_denom, torch.tensor(float('inf'), device=device))

            # Map to z_max under GBM: log(S_i/S_{i-1}) < log(B/S_{i-1})
            # z_max = [ln(B/S) - drift] / vol_step
            # Handle cases where B<=0 or B<=S (-> z_max possibly negative/finite), and if B is inf -> z_max = +inf
            with torch.no_grad():
                # For non-alive paths, zmax irrelevant; set to +inf to get p=1
                B_eff = torch.where(alive, B, torch.tensor(float('inf'), device=device))
            log_ratio = torch.log(B_eff.clamp_min(1e-300)) - torch.log(S.clamp_min(1e-300))
            # If B is inf, log_ratio inf, zmax inf; that's fine.
            z_max = (log_ratio - drift) / (vol_step + 1e-16)

            # One-step survival probability p_i = Phi(z_max), clipped to [0,1]
            p_i = normal.cdf(z_max).clamp(0.0, 1.0)

            # KO in this step: contribute R (remaining target) discounted at this fixing, weighted by (1 - p_i)
            # Only for alive paths; others contribute zero.
            ko_weight = (1 - p_i) * L * alive
            pv = pv + ko_weight * disc[i] * R

            # Survive branch weight update
            L = L * (p_i + (~alive).float())  # keep L unchanged for dead paths

            # Sample truncated Z for survivors: Z = Phi^{-1}(U * p_i)
            u = U[:, i]
            # To avoid nan in icdf when p_i=0, clamp multiplier
            u_scaled = (u * p_i).clamp(min=1e-12, max=1 - 1e-12)
            Z = normal.icdf(u_scaled)

            # GBM step for survivors
            S_next = S * torch.exp(drift + vol_step * Z)

            # Period cashflow for survivors only
            itm = (S_next >= K_t) & alive
            otm = (S_next <  K_t) & alive

            # ITM profit (counts to target)
            profit = (S_next - K_t).clamp_min(0.0) * N_itm_t * gear_t * conv_t
            # OTM loss (doesn't count to target but affects PV)
            loss   = (S_next - K_t).clamp_max(0.0) * N_otm_t * conv_t

            # Add survival-branch PV contribution (discount at this fixing)
            pv = pv + disc[i] * (profit + loss) * (alive.float()) * 1.0  # already weighted via truncated draw & L in measure

            # Update remaining target: subtract only the profit actually realized, but cap at R (can't go negative via survival branch)
            # By survival construction, profit < R for ITM survivors; OTM survivors earn no profit
            R = torch.where(itm, R - profit, R)

            # Determine which paths actually KO after this survival step (should be none by construction);
            # but we mark KO when ko_weight>0 or when R==0 exactly due to numerical rounding.
            just_ko = (ko_weight > 0) & alive
            # For numerical edge cases, also KO if R <= tiny after survival ITM
            just_ko = just_ko | ((R <= 1e-10) & alive)
            ko_step = torch.where(just_ko & (ko_step==0), torch.tensor(i+1, dtype=torch.int32, device=device), ko_step)

            # Update S and alive flags
            S = torch.where(alive, S_next, S)
            alive = alive & (~just_ko)

        # For any paths still alive at final fixing, nothing special (we already booked the last period CF).

        # Monte Carlo average (L is already embedded through truncated sampling + weights in pv)
        price = pv.mean()

        grads = {}
        # Compute Greeks for requested inputs
        if "S0" in requires_grads:
            (dS0,) = torch.autograd.grad(price, S0_t, retain_graph=True, allow_unused=True)
            grads["Delta_dS0"] = float(dS0) if dS0 is not None else None
        if "sigma" in requires_grads:
            (dSigma,) = torch.autograd.grad(price, sig_t, retain_graph=True, allow_unused=True)
            grads["Vega_dSigma"] = float(dSigma) if dSigma is not None else None
        if "r_d" in requires_grads:
            (dRd,) = torch.autograd.grad(price, rd_t, retain_graph=True, allow_unused=True)
            grads["Rho_domestic"] = float(dRd) if dRd is not None else None

        debug = {
            "alive_ratio": float(alive.float().mean()),
            "avg_ko_step": float((ko_step.float() * (ko_step>0).float()).sum() / ((ko_step>0).float().sum() + 1e-9)),
        }
        return price.detach().item(), grads, debug


    # --- Quick demo using the example flavor from the document ---
    S0_demo = 17.30    # spot
    K_demo  = 17.15    # strike
    sigma_demo = 0.16  # 16% annualized vol (illustrative)
    r_d_demo = 0.09    # domestic ZAR rate (illustrative)
    r_f_demo = 0.05    # USD rate (illustrative)
    T_years_demo = 0.5 # 6 months
    n_fix_demo = 6
    N_itm_demo = 1_000_000.0   # 1m ITM leg
    N_otm_demo = 2_000_000.0   # 2m OTM leg
    gear_demo  = 1.0           # "1m/2m" gearing already modeled via dual notional
    target_demo = 5_000_000.0  # R5 per USD * 1m? In doc it's R5 per USD across notionals; here we take absolute target in ZAR.

    price, greeks, dbg = price_geared_tarf_autograd(
        S0=S0_demo, K=K_demo, sigma=sigma_demo, r_d=r_d_demo, r_f=r_f_demo,
        T_years=T_years_demo, n_fixings=n_fix_demo,
        N_itm=N_itm_demo, N_otm=N_otm_demo, gear=gear_demo, target=target_demo,
        conv=1.0, n_paths=20000, device="cpu", seed=42,
        requires_grads=("S0","sigma","r_d"),
    )

    return price, greeks, dbg


def compress(data):
    import joblib
    # t=data['Calc']['Deals']['Deals']['Children'][0]['Children'][1121]['Instrument'].field
    # print (data['Children'][0]['Children'][1121]['Instrument'].field)
    # deals = data['Calc']['Deals']['Deals']['Children'][0]['Children']
    deals = joblib.load('C:\\temp\\gs.obj')
    equity_swaps = [x for x in deals if x['Instrument'].field['Object'] == 'EquitySwapletListDeal']
    if equity_swaps:
        eq_swap_ref = {x['Instrument'].field['Reference']: x['Instrument'].field['Equity'] for x in equity_swaps}
        all_other = [y for y in deals if y['Instrument'].field['Reference'] not in eq_swap_ref.keys()]
        all_eq_swap = [y for y in deals if y['Instrument'].field['Reference'] in eq_swap_ref.keys()]
        eq_unders = {}
        ir_unders = {}
        # first load all compressable deals
        for k in all_eq_swap:
            key = tuple([k['Instrument'].field[x] for x in k['Instrument'].field.keys()
                         if x not in ['Tags', 'Reference', 'Cashflows']])
            key_tag = key + tuple(k['Instrument'].field['Tags'])
            if k['Instrument'].field['Object'] == 'EquitySwapletListDeal':
                eq_unders.setdefault(key_tag, []).append(k)
            else:
                under_eq = eq_swap_ref[k['Instrument'].field['Reference']]
                ir_unders.setdefault(key_tag + (under_eq,), []).append(k)

        # now compress
        eq_compressed = {}
        for k, unders in eq_unders.items():
            cf_list = {}
            for deal in unders:
                for cf in deal['Instrument'].field['Cashflows']['Items']:
                    key = tuple([(k, v) for k, v in cf.items() if k != 'Amount'])
                    cf_list[key] = cf_list.setdefault(key, 0.0) + cf['Amount']

            # edit the last deal
            deal['Instrument'].field['Cashflows']['Items'] = [dict(k + (('Amount', v),)) for k, v in cf_list.items()]
            deal['Instrument'].field['Reference'] = 'COMPRESSED_{}_{}'.format(
                deal['Instrument'].field['Buy_Sell'], deal['Instrument'].field['Equity'])
            eq_compressed.setdefault(deal['Instrument'].field['Equity'], []).append(deal)

        ir_compressed = {}
        for k, unders in ir_unders.items():
            margin_list = {}
            notional_list = {}
            for deal in unders:
                for cf in deal['Instrument'].field['Cashflows']['Items']:
                    cf_key = tuple([(k, v) for k, v in cf.items() if k not in ['Notional', 'Resets', 'Margin']])
                    reset_key = tuple([tuple(x) for x in cf['Resets']])
                    key = (cf_key, reset_key)
                    margin_list[key] = margin_list.setdefault(key, 0.0) + cf['Margin'].amount * cf['Notional']
                    notional_list[key] = notional_list.setdefault(key, 0.0) + cf['Notional']

            # finish this off
            final = []
            for key, val in margin_list.items():
                notional = notional_list[key]
                cashflow = dict(key[0])
                cashflow['Notional'] = notional
                cashflow['Resets'] = [list(x) for x in list(key[1])]
                cashflow['Margin'] = rf.utils.Basis(10000.0 * val / notional)
                final.append(cashflow)

            # edit the last deal
            deal['Instrument'].field['Cashflows']['Items'] = final
            deal['Instrument'].field['Reference'] = 'COMPRESSED_{}_{}'.format(
                deal['Instrument'].field['Buy_Sell'], k[-1])
            ir_compressed.setdefault(k[-1], []).append(deal)

        for k, v in eq_compressed.items():
            all_other.extend(v)
            all_other.extend(ir_compressed[k])

        return all_other
    else:
        return deals


if __name__ == '__main__':
    import glob
    import matplotlib.pyplot as plt
    from conf import PROD_MARKETDATA, UAT_MARKETDATA

    plt.interactive(True)
    # make pandas pretty print
    pd.options.display.float_format = '{:,.5f}'.format
    pd.set_option("display.max_rows", 500, "display.max_columns", 20, 'display.width', 255)
    # test_tarf()
    # set the visible GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # set the log level
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    import riskflow as rf

    env = ''
    paths = {}
    for folder in ['JSON', 'COLLVA', 'CVA_SARB', 'Input_JSON', 'CVA_JSON',
                   'FVA_Merged', 'FVA', 'CVA', 'PFE', 'PFE_UAT',
                   'Upgrade', 'lch_munetzi']:
        paths[folder] = rf.getpath(
            [os.path.join('R:\\Riskflow', folder),
             os.path.join('N:\\Archive', folder),
             os.path.join('R:\\Riskflow\PFE_Credit', folder),
             os.path.join('Z:\\', folder),
             # os.path.join('S:\\CCR_PFE_EE_NetCollateral', folder),
             # os.path.join('S:\\Riskflow', folder),
             os.path.join('N:\\Archive', folder)])

    # path_json = paths['COLLVA']
    path_json = paths['FVA']
    # path_json = paths['FVA_Merged']
    # path = paths['CVA_UAT']
    # path_json = paths['CVA']
    path = paths['PFE']

    # rundate = '2024-03-26'
    # rundate = '2024-06-14'
    rundate = '2026-02-12'
    # rundate = '2024-09-10'

    # calibrate_PFE(path, rundate)
    # bootstrap(path_json, '', reuse_cal=True)
    # bootstrap('Z:\\', rundate, reuse_cal=False)

    # empty context
    cx = rf.StressedContext(
        path_transform={
            PROD_MARKETDATA: UAT_MARKETDATA,
        },
        file_transform={
            'CVAMarketData_Calibrated.dat': 'CVAMarketData_TST_New.json',
            'CVAMarketData_Calibrated_New.json': 'CVAMarketData_TST_New.json',
            'MarketData.dat': 'MarketData.json'
        })

    # md = rf.load_market_data(rundate, path, json_name=os.path.join(env, 'MarketData.json'))

    spreads = {
        'USD': {'FVA@Equity': {'collateral': 0, 'funding': 10}, 'FVA@Income': {'collateral': 0, 'funding': 65}},
        'EUR': {'FVA@Equity': {'collateral': 0, 'funding': 10}, 'FVA@Income': {'collateral': 0, 'funding': 65}},
        'GBP': {'FVA@Equity': {'collateral': 0, 'funding': 10}, 'FVA@Income': {'collateral': 0, 'funding': 65}},
        'ZAR': {'FVA@Equity': {'collateral': -10, 'funding': 15}, 'FVA@Income': {'collateral': -10, 'funding': 15}}}

    curves = {'USD': {'collateral': 'USD-OIS-STATIC-OLD', 'funding': 'USD-SOFR-CAS'},
              'EUR': {'collateral': 'EUR-ESTR', 'funding': 'EUR-EURIBOR-3M'},
              'GBP': {'collateral': 'GBP-SONIA', 'funding': 'GBP-SONIA'},
              'ZAR': {'collateral': 'ZAR-SWAP', 'funding': 'ZAR-SWAP'}}

    curves_rfr = {
        'USD': {'collateral': 'USD-SOFR', 'funding': 'USD-LIBOR-3M'},
        'EUR': {'collateral': 'EUR-ESTR', 'funding': 'EUR-EURIBOR-3M'}}

    # for json in glob.glob(os.path.join(path_json, rundate, 'Combination*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Soc_Gen_Paris_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_ACWA_Power_SolarReserve_Redstone_So_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_CS_Int_London_*.json')):
    # for json in glob.glob(os.path.join('C:\\Users\\shuaib.osman\\Downloads\\autocalls', '153899419.json')):
    # for json in glob.glob(os.path.join('C:\\Users\\shuaib.osman\\Downloads', 'InputAAJ_CrB_UBS_AG_Zurich_*.json')):
    # for json in glob.glob(os.path.join('Z:\\', 'baesval_158795766.json')):

    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Goldman_Sachs_Int_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_JPMorgan_Chase_NYK_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_NatWest_Markets_Plc_ISDA*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_M_Stanley___Co_Int_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_StanChart_Ldn_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_M_Lynch_Int_Ldn_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Nedbank_Ltd_ISDA*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Vescom_Twelve_NonISDA*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_ABSA_Bank_Jhb_ISDA2*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_ACWA_Power_SolarReserve_Redstone_So_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_UBS_AG_Zurich_ISDA*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Citibank_NA_NY_ISDA*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_BNP_Paribas__Paris_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'USD_FVA_ALL_NoWF_Speci*.json')):

    # for json in glob.glob(os.path.join(path_json, rundate, 'InputJSON_USD_CrB_ED_FIN_PROD_SPECIAL*')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputJSON_USD_CrB_Tikim_Ve_Tikim_ISDA*')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'ZAR_FVA_ALL_FULL_NotSpecial*')):
    # for json in glob.glob(os.path.join('Z:\\tmp','fx_asian.json')):
    # for json in glob.glob(os.path.join('Z:\\tmp','InputAAJ_CrB_SAPPI_SA_ISDA.json')):
    for json in glob.glob(os.path.join('Z:\\tmp','discrete_barrier_option_2.json')):
    # for json in glob.glob(os.path.join('Z:\\tmp','11silver_asian.json')):

    # for json in glob.glob(os.path.join('Z:\\tmp', 'jal2.json')):
    # for json in glob.glob(os.path.join('Z:\\tmp', 'range_accrual_capfloor_no_shift_2025-11-05.json')):
    # for json in glob.glob(os.path.join('Z:\\tmp', 'ILB_forward.json')):

    # for json in glob.glob(os.path.join('Z:\\tmp','W_Funding_10_CPY.json')):
    # for json in glob.glob(os.path.join('Z:\\tmp', 'zaronia_test_2.json')):
    # for json in glob.glob(os.path.join('Z:\\tmp', 'gold_forward.json')):

    # for json in glob.glob(os.path.join(path_json, rundate, 'InputJSON_ZAR_CrB_IBL_W_Funding_8_CPY*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Citigroup_G_Mkt_Ldn_*.json')):

    # for json in glob.glob(os.path.join('Z:\\tmp','InputAAJ_CrB_Credit_Agri_Corp_Inv_ISDA.json')):
    # for json in glob.glob(os.path.join('Z:\\FVA_JSON\\2026-01-05', 'ZAR_FVA_ALL_FULL_NotSpecial.json')):

    # for json in glob.glob(os.path.join('Z:\\tmp','InputAAJ_CrB_Avon_Peaking_Power_ISDA.json')):

    # for json in glob.glob(os.path.join('Z:\\tmp', 'InputAAJ_CrB_Just_Retirement_Life_SA_ISDA.json')):
    # for json in glob.glob(os.path.join('R:\\RiskFlow\\PFE\\2025-12-02', 'InputAAJ_CrB_SAPPI_SA_ISDA.json')):
    # for json in glob.glob(os.path.join('R:\\RiskFlow\\FVA_Merged\\2025-12-11', 'ZAR_FVA_ALL_NoWF_NotSpecial.json')):
    # for json in glob.glob(os.path.join('Z:\\tmp','commodity_asian_passed_dates.json')):
    # for json in glob.glob(os.path.join('Z:\\tmp','InputJSON_USD_CrB_IBL_ICIB_Bal_Sheet_Fund_Eurodollar_CPY.json')):

    # for json in glob.glob(os.path.join('Z:\\tmp', 'american.json')):
    # for json in glob.glob(os.path.join('Z:\\tmp','InputJSON_USD_CrB_ED_FIN_PROD_SPECIAL.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'USD_FVA_ALL_FULL_Special*')):

    #for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Citibank_NA_NY_ISDA*')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_African_Infrastructure_Investment_*')):

    # for json in glob.glob(os.path.join(path_json, rundate, 'json_combo_test_162595785_v2*')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputJSON_ZAR_CrB_ACWA_Power_SolarReserve_Redstone_So_NonISDA*')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Barclays_Plc_Ldn_ISDA*.json')):

    # for json in glob.glob(os.path.join('Z:\\', 'InputAAJ_CrB_BNP_Paribas__Paris__ISDA_*.json')):
    # for json in glob.glob(os.path.join('Z:\\', 'InputAAJ_CrB_M_Lynch_Int_Ldn_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Sygnia_Asset_Management_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Barclays_Plc_Ldn_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputJSON_FVA_CrB_BOA_Clearing_*.json')):
    # for json in glob.glob(os.path.join('Z:\\', 'USD_COLLVA_ALL_{}.json'.format(rundate))):
    # for json in glob.glob(os.path.join('Z:\\', 'citibank_pfe_after_{}.json'.format(rundate))):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_National_Home_Builders_Registration_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_M_Lynch_Int_Ldn_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Citibank_NA_NY_*.json')):
    # for json in glob.glob(os.path.join('C:\\Users\\shuaib.osman\\Downloads', 'InputAAJ_CrB_Citibank_NA_NY_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_FirstRand_Bank_Ltd_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputJSON_FVA_CrB_IBL_ICIB_BSF_ED_HQLA_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Redefine_Properties_Limited*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, 'InputAAJ_CrB_Standard_Bank_*.json')):
    # for json in glob.glob(os.path.join(path_json, rundate, '*otus*.json')):

        cx.load_json(json, compress=True)
        short_date = ''.join(rundate[2:].split('-')[::-1])
        cx.stressed_config_file = os.path.join(UAT_MARKETDATA, "CVAMarketDataBackup\\CVAMarketData_Calibrated_Vega_{}.json".format(short_date))

        # if 'HullWhite2FactorModelParameters.USD-OIS' not in cx.current_cfg.params['Price Factors']:
        #     cx.current_cfg.params['Price Factors']['HullWhite2FactorModelParameters.USD-OIS'] = cx.current_cfg.params[
        #         'Price Factors']['HullWhite2FactorModelParameters.USD-SOFR']

        # if 'HullWhite2FactorModelParameters.ZAR-OIS' not in cx.current_cfg.params['Price Factors']:
        #     cx.current_cfg.params['Price Factors']['HullWhite2FactorModelParameters.ZAR-OIS'] = cx.current_cfg.params[
        #         'Price Factors']['HullWhite2FactorModelParameters.ZAR-SWAP']

        if not cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            print('no children for crb {} - skipping'.format(json))
            continue

        # test hwhazardratemodel
        cx.current_cfg.params['Model Configuration'].modeldefaults['SurvivalProb'] = 'HWHazardRateModel'
        cx.current_cfg.params['Price Models']['HWHazardRateModel.ITRAXX_MAIN_S44_ZAR_MR_MMR'] = {
            'Alpha': 0.89, 'Sigma': 0.005, 'Lambda': 0.0}

        # grab the netting set
        ns = cx.current_cfg.deals['Deals']['Children'][0]['Instrument']
        agreement_currency = ns.field.get('Agreement_Currency', 'ZAR')
        factor = rf.utils.Factor('InterestRate', ('ZAR-SWAP',))

        # ns.field['Collateral_Assets']['Cash_Collateral'][0]['Funding_Rate'] = 'USD-LIBOR-3M.FUNDING'
        # ns.field['Collateral_Assets']['Cash_Collateral'][0]['Collateral_Rate'] = 'USD-OIS'
        # ns.field['Collateral_Assets']['Cash_Collateral'] = []
        # ns.field['Collateral_Call_Frequency']=pd.DateOffset(weeks=1)
        # ns.field['Collateralized'] = 'False'

        overrides = {
            'Calc_Scenarios': 'No',
            # 'Run_Date': '2025-07-02',
            # 'Tenor_Offset': 2.0,
            # 'Time_grid': '0d 2d 1w',
            'Random_Seed': 1,
            'Generate_Cashflows': 'Yes',
            # 'Greeks': 'First',
            # 'Currency': 'ZAR',
            'Percentile': '95',
            # 'Antithetic': 'Yes',
            # 'Deflation_Interest_Rate': 'ZAR-SWAP',
            'MCMC_Simulations': 2048,
            'Batch_Size': 256,
            'Simulation_Batches': 1,
            # 'Collateral_Valuation_Adjustment': {'Calculate': 'Yes', 'Gradient': 'Yes'},
            'Initial_Margin':
                {'Calculate': 'No',
                 'Liquidity_Weights': 'Z:\\IMMCalc\\liquidity_weights2.csv',
                 'IRS_Weights': 'Z:\\IMMCalc\\ZARIRS.csv',
                 'Local_Currency': 'ZAR',
                 'Delta_Factor': 71.4,
                 'IM_Currency': 'GBP',
                 'Gradient': 'No'},
            'Funding_Valuation_Adjustment':
                {'Calculate': 'No',
                 'Gradient': 'Yes'},
            'Credit_Valuation_Adjustment':
                {'Calculate': 'No', 'Gradient': 'Yes', 'CDS_Tenors': [0.5, 1, 3, 5, 10], 'Hessian': 'No'}
        }

        if ns.field['Collateralized'] == 'True':
            overrides['Dynamic_Scenario_Dates'] = 'Yes'
        else:
            overrides['Dynamic_Scenario_Dates'] = 'No'
        overrides['Dynamic_Scenario_Dates'] = 'Yes'
        if False:
            for i in cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            # for i in cx.current_cfg.deals['Deals']['Children']:
                # if False and not str(i['Instrument'].field['Reference']) in ['141573158']:#, '154727676', '158079450']:
                # if False and not str(i['Instrument'].field['Object']) in ['FXNonDeliverableForward']:

                # 141784934, 153489256
                # if False and str(i['Instrument'].field['Reference']) in ['169962788', '169962789', '169962790', '169962792']:  # , '154727676', '158079450']:
                if not str(i['Instrument'].field['Reference']) in ['211837146']:
                # if False and not str(i['Instrument'].field['Reference']) in ['CrB_BNP_Paribas__Paris__ISDA', 'CrB_Citibank_NA_NY_ISDA']:
                    i['Ignore'] = 'True'
                else:
                    i['Ignore'] = 'False'
                    # for si in i['Children']:
                    #     if not str(si['Instrument'].field['Object']) in ['QEDI_CustomAutoCallSwap_V2']:
                    #         si['Ignore'] = 'True'

        # ns.field['Opening_Balance'] = 0.0
        # if 'Collateral_Assets' in ns.field:
        #    ns.field['Collateral_Assets']['Cash_Collateral'][0]['Amount'] = 1.0
        # cx.current_cfg.params['Valuation Configuration']['FXDiscreteExplicitAsianOption']={'Valuation':'Full'}

        logging.getLogger().setLevel(logging.DEBUG)

        for x in []:#['InterestRate.ZAR-JIBAR-3M', 'InterestRate.ZAR-SWAP']:
        # for x in ['InterestRate.ZAR-JIBAR-3M']:
            rates = cx.current_cfg.params['Price Factors'][x]['Curve'].array[:, 1]
            # cx.current_cfg.params['Price Factors'][x]['Curve'].array[:, 1] = np.log(np.exp(rates) + 0.02)
            cx.current_cfg.params['Price Factors'][x]['Curve'].array[:, 1] = rates + 0.0195

        # cx.current_cfg.params['Price Factors']['InflationRate.ZAR-BOND-CPI']['Curve'].array[:, 0]-=31/365.0

        # calc, out = cx.Base_Valuation(overrides=overrides)
        calc, out = cx.Base_Valuation()
        if 1:
            calc, out = cx.Credit_Monte_Carlo(overrides=overrides)
            print(out['Results'].keys())

        if 0:
            for i in range(2):
                calc, out = cx.Credit_Monte_Carlo(overrides=overrides)
                print(i, 'CVA', out['Results']['cva'])

                # cx.stress_config(['InterestRate', 'InflationRate'])
                cds_spread = rf.utils.check_scope_name(
                    rf.utils.Factor('SurvivalProb', (calc.params['Credit_Valuation_Adjustment']['Counterparty'],)))
                delta_surv = out['Results']['grad_cva'].loc[cds_spread].reset_index()[
                    ['Tenor', 'Gradient']].set_index('Tenor')

                test = out['Results']['CS01'].groupby(
                    delta_surv.index[np.searchsorted(
                        delta_surv.index, out['Results']['CS01'].index, side='right') - 1]).mean()

                cx.stress_config(['ForwardPrice'])
                calc2, out2 = cx.Credit_Monte_Carlo(overrides=overrides)
                print(i, 'CVA', out2['Results']['cva'])
                # delta = test.min() * delta_surv.loc[test.min().index]['Gradient'].values.reshape(-1, 1)
                cx.restore_config()

            # cx.stress_config(['ForwardPrice'])
            # del params['CVA']
            calc, params = rf.run_cmc(cx.current_cfg, overrides=overrides, LegacyFVA=True)
            partial_FVA = calc.calc_individual_FVA(
                params, spreads=spreads[calculation_currency], discount_curves=curves[calculation_currency])

            output = json.replace('json', 'csv')
            filename = os.path.split(output)[1]
            outfile = os.path.join('C:\\temp', filename)

            try:
                assets = list(ns.field['Collateral_Assets'].keys()).pop()
                if assets is None and ns['Collateralized'] == 'True':
                    assets = 'Cash_Collateral'
            except:
                assets = 'Cash_Collateral' if ns.field['Collateralized'] == 'True' else 'None'

            collva_sect = cx.current_cfg.deals['Calculation'].get(
                'Collateral_Valuation_Adjustment', {'Calculate': 'Yes' if assets == 'Cash_Collateral' else 'No'})

            # sensible defaults
            collva_sect['Collateral_Curve'] = collva_sect.get(
                'Collateral_Curve', curves[agreement_currency]['collateral'])
            collva_sect['Funding_Curve'] = collva_sect.get(
                'Funding_Curve', curves[agreement_currency]['funding'])
            collva_sect['Collateral_Spread'] = collva_sect.get(
                'Collateral_Spread', spreads[agreement_currency]['FVA@Income']['collateral'])
            collva_sect['Funding_Spread'] = collva_sect.get(
                'Funding_Spread', spreads[agreement_currency]['FVA@Income']['funding'])
            cx.current_cfg.deals['Calculation']['Collateral_Valuation_Adjustment'] = collva_sect

            if 'Funding_Valuation_Adjustment' in cx.current_cfg.deals['Calculation']:
                del cx.current_cfg.deals['Calculation']['Funding_Valuation_Adjustment']

            # calc, out = cx.Base_Valuation()
            # calc, out = cx.run_job(overrides)
            # out['Results']['collateral_profile'].to_csv(outfile)
            # calc, out = cx.Base_Valuation()
            # out['Results']['mtm'].to_csv(outfile)

            mike_filename = 'N:\\Archive\\PFE\\{}\\{}'.format(rundate, filename.replace('InputAAJ_', ''))

            if False and os.path.exists(mike_filename):
                mike = pd.read_csv(mike_filename, index_col=0)
                me = pd.read_csv(outfile, index_col=0)
                adaptiv = float(mike.head(1)['PFE'])
                if 0:
                    myval = float(me.head(1)['Value'])

                    if adaptiv != 0 and np.abs((adaptiv - myval) / adaptiv).max() > 0.01:
                        print('Mismatch - Collateralized is {}, '.format(ns.field['Collateralized']), outfile, adaptiv,
                              myval)
                        if ns.field['Collateralized'] != 'True':
                            print('investigate', json)
                else:
                    common_index = mike.index.intersection(me.index)
                    adaptiv_pfe = mike.reindex(common_index)
                    my_pfe = me.reindex(common_index)
                    abs_error = (my_pfe - adaptiv_pfe).abs()
                    error = ((my_pfe - adaptiv_pfe) / adaptiv_pfe)['PFE'].dropna().abs().max()
                    if error > 0.02:
                        print('{0:06.2f} Error Mismatch - Crb {1} Collateralized is {2}'.format(
                            error, json, ns.field['Collateralized']))
                    else:
                        print('Crb {} is within 2%'.format(json))
            else:
                print('skipping', mike_filename)
