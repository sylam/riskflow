########################################################################
# Copyright (C)  Shuaib Osman (sosman@investec.co.za)
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

# import standard libraries
import itertools
import numpy as np
import pandas as pd

from . import utils
from collections import OrderedDict
from scipy.interpolate import RectBivariateSpline

# map the names of various factor interpolations to something simpler
factor_interp_map = {
    'CubicSplineCurveInterpolation': 'Hermite',
    'HermiteInterpolationCurveGetValue': 'Hermite',
    'LinearInterFlatExtrapCurveGetValue': 'Linear',
    'CubicSplineOnXTimesYCurveInterpolation': 'HermiteRT',
    'HermiteRTInterpolationCurveGetValue': 'HermiteRT'
}

class Factor0D(object):
    """Represents an instantaneous Rate (0D) risk factor"""

    def __init__(self, param):
        self.param = param
        self.delta = 0.0

    def bump(self, amount, relative=True):
        self.delta = self.param['Spot'] * amount if relative else amount

    def get_delta(self):
        return self.delta

    def get_tenor_indices(self):
        '''
        Method to extract the indices of this factor - for spot rates, it's always 0.0 .
        In general, whatever this method returns should be passed to the tenors parameter
        of current_value and be defined.
        '''

        return np.array([[0.0]])

    def current_value(self, tenors=None, offset=0.0):
        return np.array([self.param['Spot'] + self.delta])


class Factor1D(object):
    """Represents a risk factor with a term structure (1D)"""

    def __init__(self, param):
        self.param = param
        self.tenors = self.get_tenor()
        self.delta = np.zeros_like(self.tenors)
        self.interpolation = self.check_interpolation()

    def get_tenor(self):
        """Gets the tenor points stored in the Curve attribute"""
        if self.param['Curve'] is None or not isinstance(self.param['Curve'], utils.Curve):
            self.param['Curve'] = utils.Curve([], [(0.0, 0.0)])

        # make sure there are no duplicate tenors    
        tenors = np.unique(self.param['Curve'].array[:, 0])
        rates = np.interp(tenors, *self.param['Curve'].array.T)
        self.param['Curve'].array = np.vstack((tenors, rates)).T
        return self.param['Curve'].array[:, 0]

    def get_tenor_indices(self):
        '''
        Method to extract the indices of this factor - for curves, it's always the first tenor index.
        Whatever this method returns should be passed to the tenors parameter of current_value
        and be interpolated if necessary.
        '''
        return self.tenors.reshape(-1, 1)

    def bump(self, amount, relative=True):
        self.delta = self.param['Curve'].array[:, 1] * amount if relative else np.ones_like(self.tenors) * amount

    def get_delta(self):
        return self.delta.mean()

    def check_interpolation(self):
        if self.param.get('Interpolation') == 'HermiteRT':
            g, c = utils.hermite_interpolation(self.tenors, self.param['Curve'].array[:, 1] * self.tenors)
            return ('HermiteRT', g, c)
        elif self.param.get('Interpolation') == 'Hermite':
            g, c = utils.hermite_interpolation(self.tenors, self.param['Curve'].array[:, 1])
            return ('Hermite', g, c)
        else:
            return ('Linear',)

    def current_value(self, tenor_index=None, offset=0.0):
        """Returns the value of the rate at each tenor point (if set) else returns what's
        stored in the Curve parameter"""
        bumped_val = self.param['Curve'].array[:, 1] + self.delta
        # get the tenors - make sure it's in range
        tenors = ((np.array(tenor_index) if tenor_index is not None else self.tenors) + offset).clip(
            self.tenors.min(), self.tenors.max())

        if self.interpolation[0] != 'Linear':
            index = np.searchsorted(self.tenors, tenors, side='right') - 1
            index_next = np.clip(index + 1, 0, self.tenors.size - 1)
            dt = np.clip(self.tenors[index_next] - self.tenors[index], 1 / 365.0, np.inf)
            m = np.clip((tenors - self.tenors[index]) / dt, 0.0, 1.0)
            g, c = self.interpolation[1:]
            rate, denom = (bumped_val * self.tenors, tenors) if self.interpolation[0] == 'HermiteRT' else (
            bumped_val, 1.0)
            val = rate[index] * (1.0 - m) + m * rate[index_next] + m * (
                    1.0 - m) * g[index] + m * m * (1.0 - m) * c[index]
            return val / denom
        else:
            return np.interp(tenors, self.tenors, bumped_val)


class Factor2D(object):
    """Represents a risk factor that's a surface (2D) - Currently this is only vol surfaces"""

    def __init__(self, param):
        # default empty surfaces to 1%
        if not param['Surface'].array.any():
            param['Surface'].array = np.array([[0, 0, 0.01]])

        self.param = param
        self.flat = None
        self.index_map = OrderedDict()
        self.update()

    def update(self):
        self.moneyness = self.get_moneyness()
        self.expiry = self.get_expiry()
        self.vols = self.get_vols()
        self.tenor_ofs = np.array([len(x) for x in self.index_map.values()])

    def get_moneyness(self):
        """Gets the moneyness points stored in the Surface attribute"""
        return np.unique(self.param['Surface'].array[:, 0])

    def get_expiry(self):
        """Gets the expiry points stored in the Surface attribute"""
        return np.unique(self.param['Surface'].array[:, 1])

    def get_vols(self):
        """Uses flat extrapolation along moneyness and then linear interpolation along expiry"""
        surface = self.param['Surface'].array
        # Sort by moneyness, then expiry
        self.sorted_vol = surface[np.lexsort((surface[:, 0], surface[:, 1]))]
        # store an index for each expiry
        self.index_map.clear()
        for element in self.sorted_vol:
            self.index_map.setdefault(element[1], []).append(element[0])
        self.flat = self.sorted_vol[:, 2]
        # interpolate the full surface
        return np.array(
            [np.interp(self.moneyness, surface[surface[:, 1] == x][:, 0], surface[surface[:, 1] == x][:, 2])
             for x in self.expiry])

    def get_all_tenors(self):
        return np.hstack((self.moneyness, self.expiry))

    def get_tenor_indices(self):
        '''
        returns the shortened sorted vol surface indices - Moneyness and Expiry
        use np.array(list(itertools.product(self.expiry,self.moneyness))) for all possible indices
        
        The return value of this method again needs to be defined when called with current_value
        '''
        return self.sorted_vol[:, :2]

    def current_value(self, tenors=None, offset=0.0):
        """Returns the value of the vol surface"""
        if tenors is not None and self.expiry.size > 1 and self.moneyness.size > 1:
            interpolator = RectBivariateSpline(self.expiry, self.moneyness, self.vols, kx=1, ky=1)
            return np.array([interpolator(y + offset, x)[0][0] for x, y in tenors])

        return self.flat


class Factor3D(object):
    """Represents a risk factor that's a space (3D) - Things like swaption volatility spaces"""
    MONEYNESS_INDEX = 0
    EXPIRY_INDEX = 1
    TENOR_INDEX = 2

    def __init__(self, param):
        self.param = param
        self.update()

    def update(self):
        self.moneyness = self.get_moneyness()
        self.expiry = self.get_expiry()
        self.tenor = self.get_tenor()
        self.vols = self.get_vols()
        self.tenor_ofs = np.array([0, self.moneyness.size, self.moneyness.size + self.expiry.size])

    def get_moneyness(self):
        """Gets the moneyness points stored in the Surface attribute"""
        return np.unique(self.param['Surface'].array[:, self.MONEYNESS_INDEX])

    def get_expiry(self):
        """Gets the expiry points stored in the Surface attribute"""
        return np.unique(self.param['Surface'].array[:, self.EXPIRY_INDEX])

    def get_tenor(self):
        """Gets the tenor points stored in the Surface attribute"""
        return np.unique(self.param['Surface'].array[:, self.TENOR_INDEX])

    def get_vols(self):
        """Uses flat extrapolation along moneyness and then linear interpolation along expiry"""
        vols = []
        for tenor in self.tenor:
            surface = self.param['Surface'].array[self.param['Surface'].array[:, self.TENOR_INDEX] == tenor]
            for x in self.expiry:
                exp_surface = surface[surface[:, self.EXPIRY_INDEX] == x]
                if not exp_surface.any():
                    expiries = np.unique(surface[:, self.EXPIRY_INDEX])
                    nearest_expiry = expiries[expiries.searchsorted(x).clip(0, expiries.size - 1)]
                    exp_surface = surface[surface[:, self.EXPIRY_INDEX] == nearest_expiry]
                sigma = exp_surface[:, 3]
                mns = exp_surface[:, self.MONEYNESS_INDEX]
                vol = np.interp(self.moneyness, mns, sigma)
                vols.append(vol)

        return np.array(vols)

    def get_all_tenors(self):
        return np.hstack((self.moneyness, self.expiry, self.tenor))

    def get_tenor_indices(self):
        return np.array(list(itertools.product(self.tenor, self.expiry, self.moneyness)))

    def current_value(self, tenors=None, offset=0.0):
        """
        Returns the value of the Vol space
        Again, this is symmetric wih get_tenor_indices i.e. self.current_value(self.get_tenor_indices()) should be
        just the list of corresponding vols.
        """
        if tenors is not None and self.expiry.size > 1 and self.moneyness.size > 1:
            interpolator = [RectBivariateSpline(
                self.expiry, self.moneyness, vol.reshape(self.expiry.size, -1), kx=1, ky=1)
                for vol in self.vols.reshape(self.tenor.size, -1)]
            interp_vols = []
            for tenor in tenors:
                index = np.clip(self.moneyness.searchsorted(
                    tenor[self.TENOR_INDEX], side='right') - 1, 0, self.tenor.size - 1)
                index_p1 = np.clip(index + 1, 0, self.tenor.size - 1)
                val = np.interp(
                    tenor[self.TENOR_INDEX], self.tenor[[index, index_p1]],
                    np.dstack([interpolator[index](tenor[self.EXPIRY_INDEX], tenor[self.MONEYNESS_INDEX]),
                               interpolator[index_p1](tenor[self.EXPIRY_INDEX],
                                                      tenor[self.MONEYNESS_INDEX])]).flatten())
                interp_vols.append(val)
            return np.array(interp_vols)

        return self.vols.ravel()


class FxRate(Factor0D):
    """
    Represents the price of a currency relative to the base currency (snapped at end of day).
    """
    field_desc = ('FX and Equity',
                  ['- **Interest_Rate**: String. Associated interest rate curve name.',
                   '- **Spot**:Float. Spot rate in base currency.'
                   ])

    def __init__(self, param):
        super(FxRate, self).__init__(param)

    def get_repo_curve_name(self, default):
        return utils.check_rate_name(self.param['Interest_Rate'] if self.param['Interest_Rate'] else default)

    def get_domestic_currency(self, default):
        return utils.check_rate_name(self.param['Domestic_Currency'] if self.param['Domestic_Currency'] else default)


class FuturesPrice(Factor0D):
    def __init__(self, param):
        super(FuturesPrice, self).__init__(param)

    def current_value(self, tenors=None, offset=0.0):
        return np.array([self.param['Price']])


class PriceIndex(Factor0D):
    """
    Used to represent things like CPI/Stock Indices etc.
    """
    field_desc = ('Inflation', [
        '- **Index**: A *Curve* object representing a series of (date, value) pairs where the date is an excel integer',
        '- **Next_Publication_Date**: *TimeStamp* object',
        '- **Last_Period_Start**: *TimeStamp* object',
        '- **Publication_Period**: String. Either **Monthly** or **Quarterly**'
    ])

    def __init__(self, param):
        super(PriceIndex, self).__init__(param)
        # the start date for excel's date offset
        self.start_date = utils.excel_offset
        # the offset to the latest index value
        self.last_period = self.param['Last_Period_Start'] - self.start_date
        self.latest_index = np.where(self.param['Index'].array[:, 0] >= self.last_period.days)[0]

    def current_value(self, tenors=None, offset=0.0):
        return np.array([self.param['Index'].array[self.latest_index[0]][1]] if self.latest_index.any()
                        else [self.param['Index'].array[-1][1]])

    def get_reference_value(self, ref_date):
        query = float((ref_date - self.start_date).days)
        return np.interp(query, *self.param['Index'].array.T)

    def get_last_publication_dates(self, base_date, time_grid):
        roll_period = pd.DateOffset(months=3) \
            if self.param['Publication_Period'] == 'Quarterly' else pd.DateOffset(months=1)
        last_date = base_date + pd.DateOffset(days=time_grid[-1])
        publication = pd.date_range(self.param['Last_Period_Start'], last_date, freq=roll_period)
        next_publication = pd.date_range(self.param['Next_Publication_Date'], last_date + roll_period, freq=roll_period)
        eval_dates = [(base_date + pd.DateOffset(days=t)).to_datetime64() for t in time_grid]
        return publication[np.searchsorted(next_publication.tolist(), eval_dates, side='right')]


class EquityPrice(Factor0D):
    """
    This is just the equity price on a particular end of day
    """
    field_desc = ('FX and Equity',
                  ['- **Currency**: String',
                   '- **Interest_Rate**: String representing the equity repo curve',
                   '- **Spot**:Spot rate in the specified *Currency*'])

    def __init__(self, param):
        super(EquityPrice, self).__init__(param)

    def get_repo_curve_name(self):
        return utils.check_rate_name(
            self.param['Interest_Rate'] if self.param['Interest_Rate'] else self.param['Currency'])

    def get_currency(self):
        return utils.check_rate_name(self.param['Currency'])


class ForwardPriceSample(Factor0D):
    """
    This is just the sampling method for Forward Prices
    """
    field_desc = ('Energy',
                  ['- **Offset**: Integer specifying a calendar day offset',
                   '- **Holiday_Calendar**: String specifying the name of the calendar to use in the calendar xml file',
                   '- **Sampling_Convention**: String. Either **ForwardPriceSampleDaily** or ',
                   '**ForwardPriceSampleBullet**'])

    def __init__(self, param):
        super(ForwardPriceSample, self).__init__(param)

    def current_value(self, tenors=None, offset=0.0):
        return np.array([self.param['Offset']])

    def get_holiday_calendar(self):
        return self.param.get('Holiday_Calendar')

    def get_sampling_convention(self):
        return self.param.get('Sampling_Convention')


class DiscountRate(object):
    """
    A wrapper for interest rate price factors.
    """
    field_desc = (
        'Interest Rates',
        ['- **Interest_Rate**: String. Name of the *Interest Rate* price factor for discounting']
    )

    def __init__(self, param):
        self.param = param

    def get_interest_rate(self):
        return utils.check_rate_name(self.param['Interest_Rate'])


class ReferenceVol(object):
    field_desc = ('Energy',
                  ['- **ForwardPriceVol**: String. Name of the *ForwardPriceVol* price factor to use',
                   '- **ReferencePrice**: String: Name of the *ReferencePrice* price factor that defines the reference',
                   'lookup'
                   ])

    def __init__(self, param):
        self.param = param

    def get_forwardprice_vol(self):
        return utils.check_rate_name(self.param['ForwardPriceVol'])

    def get_forwardprice(self):
        return utils.check_rate_name(self.param['ReferencePrice'])


class InterestRateJacobian(object):
    def __init__(self, param):
        self.param = param
        self.tenors = None
        self.instruments = sorted(self.param.keys())
        self.jacobian = None
        self.inverse_jacobian = None

    def update(self, ir_curve):
        self.tenors = ir_curve.tenors
        j = []
        for bench in self.instruments:
            zero = np.zeros_like(self.tenors)
            sense = self.param[bench].array
            zero[np.searchsorted(self.tenors, sense[:, 0])] = sense[:, 1]
            j.append(zero)
        self.jacobian = np.array(j)
        self.inverse_jacobian = np.linalg.pinv(self.jacobian)

    def benchmark_tenors(self):
        return np.sort([x.array[-1, 0] for x in self.param.values()])

    def reduce_curve(self, factor, price_factors, interpolation='Hermite'):
        tenors = self.benchmark_tenors()
        factor_name = utils.check_tuple_name(factor)
        # fetch the actual rates
        rates = np.interp(tenors, *price_factors[factor_name]['Curve'].array.T)
        # overwrite the 'Curve'
        price_factors[factor_name]['Curve'] = utils.Curve(
            [], np.dstack((tenors, rates))[-1])
        # set the interpolation
        price_factors[factor_name]['Interpolation'] = interpolation

    def current_value(self):
        """Returns the value of the vol surface"""

        def approx_benchmark(ir_gradients):
            sensitivities = ir_gradients.reshape(1, -1).dot(self.inverse_jacobian)
            bench = OrderedDict(list(zip(self.instruments, sensitivities[0])))
            errors = sensitivities.dot(self.jacobian)[0] - ir_gradients
            return bench, errors

        return approx_benchmark


class Correlation(Factor0D):
    field_desc = ('General',
                  ['- **Value**: Float. The market implied correlation between rates. Specified as '
                   '"pricefactor1/pricefactor2" e.g. Correlation.FxRate.USD.ZAR/ReferencePrice.BRENT_OIL-IPE.USD']
                  )

    def __init__(self, param):
        super(Correlation, self).__init__(param)

    def current_value(self, tenors=None, offset=0.0):
        return np.array([self.param.get('Value', 0.0) + self.delta])


class DividendRate(Factor1D):
    """
    Represents the Dividend Yield risk factor
    """
    field_desc = ('FX and Equity',
                  ['- **Currency**: String.',
                   '- **Curve**: *Curve* object specifying the continuous dividend yield'])

    def __init__(self, param):
        super(DividendRate, self).__init__(param)
        tenor_delta = (1.0 / np.array(self.tenors[:-1]).clip(1e-5, np.inf)) - \
                      (1.0 / np.array(self.tenors[1:]).clip(1e-5, np.inf))
        self.tenor_delta = np.hstack((tenor_delta, [1.0]))
        self.min_tenor = max(1e-5, self.tenors.min())
        self.max_tenor = max(1e-5, self.tenors.max())

    def get_currency(self):
        return utils.check_rate_name(self.param['Currency'])

    @staticmethod
    def get_day_count():
        """hardcode the daycount for dividend rates to act/365"""
        return utils.DAYCOUNT_ACT365

    def current_value(self, tenor_index=None, offset=0):
        """Returns the value of the rate at each tenor point (if set) else returns what's
        stored in the Curve parameter"""
        bumped_val = self.param['Curve'].array[:, 1] + self.delta
        # get the tenors
        ten = (np.array(tenor_index) if tenor_index is not None else self.tenors) + offset
        max_tenor = max(self.tenors.size - 1, 0)
        index = np.clip(
            np.searchsorted(self.tenors, ten, side='right') - 1,
            0, max_tenor)
        index_next = np.clip(index + 1, 0, max_tenor)

        alpha = (1.0 / self.tenors[index].clip(1e-5, np.inf) -
                 1.0 / ten.clip(self.min_tenor, self.max_tenor)) / self.tenor_delta[index]

        return bumped_val[index] * (1.0 - alpha) + alpha * bumped_val[index_next]


class SurvivalProb(Factor1D):
    """
    Represents the Probability of Survival risk factor
    """
    field_desc = ('Credit',
                  ['- **Recovery_Rate**: Float. The assumed recovery amount. Enter 0.4 for 40%.',
                   '- **Curve**: *Curve* object specifying the negative log survival probability'])

    def __init__(self, param):
        super(SurvivalProb, self).__init__(param)

    def get_day_count(self):
        """hardcode the daycount for Survival Probability rates to act/365"""
        return utils.DAYCOUNT_ACT365

    def get_day_count_accrual(self, ref_date, time_in_days):
        return utils.get_day_count_accrual(ref_date, time_in_days, self.get_day_count())

    def recovery_rate(self):
        return self.param.get('Recovery_Rate')


class InterestRate(Factor1D):
    """
    Represents an Interest Rate risk factor - basically a time indexed array of rates
    Remember that the tenors are normally expressed as year fractions - not days.
    """
    field_desc = ('Interest Rates',
                  ['- **Currency**: String. The associated currency for this curve',
                   '- **Curve**: *Curve* object specifying the continuously compounded interest rate',
                   '- **Day_Count**: String. Either ACT_365 or ACT_360. The daycount convention for this curve',
                   '- **Sub_Type**: Optional String can be null or set to **BasisSpread** if this curve is a spread',
                   'over its parent'])

    def __init__(self, param):
        super(InterestRate, self).__init__(param)

    def get_currency(self):
        return utils.check_rate_name(self.param['Currency'])

    def get_day_count(self):
        return utils.get_day_count(self.param['Day_Count'])

    def get_subtype(self):
        return 'InterestRate' + (self.param['Sub_Type'] if self.param['Sub_Type'] else '')

    def get_day_count_accrual(self, ref_date, time_in_days):
        return utils.get_day_count_accrual(ref_date, time_in_days, self.get_day_count())


class InflationRate(Factor1D):
    """
    Represents an Interest Rate (1D) risk factor - basically a time indexed array of rates
    Remember that the tenors are normally expressed as year fractions - not days.
    """
    field_desc = ('Inflation',
                  ['- **Currency**: String. The associated currency for this curve',
                   '- **Reference_Name**: String. allowed values listed [here]'
                   '(../Theory/Inflation/#price-index-references)',
                   '- **Curve**: *Curve* object specifying the continuously compounded inflation growth rate',
                   '- **Day_Count**: String. Either ACT_365 or ACT_360. The daycount convention for this curve',
                   '- **Price_Index**: String. Name of associated PriceIndex factor'])

    def __init__(self, param):
        super(InflationRate, self).__init__(param)

    def get_reference_name(self):
        return self.param['Reference_Name']

    def get_day_count(self):
        return utils.get_day_count(self.param['Day_Count'])

    def get_day_count_accrual(self, ref_date, time_in_days):
        return utils.get_day_count_accrual(ref_date, time_in_days, self.get_day_count())


class ForwardPrice(Factor1D):
    """
    Used to represent things like Futures prices on OIL/GOLD/Platinum etc.
    """
    field_desc = ('Energy',
                  ['- **Currency**: String. The associated currency for this curve',
                   '- **Curve**: *Curve* object of date, rate pairs specifying forward price at the corresponding',
                   'excel date'
                   ])

    def __init__(self, param):
        super(ForwardPrice, self).__init__(param)

    def get_currency(self):
        return utils.check_rate_name(self.param['Currency'])

    def get_relative_tenor(self, reference_date):
        reference_date_excel = (reference_date - utils.excel_offset).days
        return self.get_tenor() - reference_date_excel

    def get_day_count(self):
        return utils.DAYCOUNT_None


class ReferencePrice(Factor1D):
    """
    Used to represent how lookups on the Forward/Futures curve are performed.
    """
    field_desc = ('Energy',
                  ['- **Currency**: String. The associated currency for this curve.',
                   '- **Fixing_Curve**: *Curve* object of date, reference date pairs specifying the delivery date for',
                   'a particular date. Both dates are in excel format.',
                   '- **ForwardPrice**: String. The name of associated ForwardPrice factor'
                   ])

    def __init__(self, param):
        super(ReferencePrice, self).__init__(param)
        # the start date for excel's date offset
        self.start_date = utils.excel_offset

    # the offset to the latest index value

    def get_forwardprice(self):
        return utils.check_rate_name(self.param['ForwardPrice'])

    def get_fixings(self):
        return self.param['Fixing_Curve']

    def get_tenor(self):
        """Gets the tenor points stored in the Curve attribute"""
        return self.param['Fixing_Curve'].array[:, 0]

    def current_value(self, tenors=None, offset=0.0):
        """Returns the value of the rate at each tenor point (if set) else returns what's
        stored in the Curve parameter"""
        return np.interp(tenors, *self.param['Fixing_Curve'].array.T) if tenors is not None else \
            self.param['Fixing_Curve'].array[:, 1]


class GBMAssetPriceTSModelParameters(Factor1D):
    """
    Represents the Bootstrapped TS implied parameters for a risk neutral process
    """

    def __init__(self, param):
        super(GBMAssetPriceTSModelParameters, self).__init__(param)

    def get_tenor(self):
        """Gets the tenor points stored in the Curve attribute"""
        return self.param['Vol'].array[:, 0]

    def get_tenor_indices(self):
        return {'Vol': self.param['Vol'].array[:, 0].reshape(-1, 1)}

    def current_value(self, tenors=None, offset=0.0):
        """Returns the parameters of the GBM factor model as a dictionary"""
        return {'Vol': self.param['Vol'].array[:, 1]}


class HullWhite2FactorModelParametersJacobian(Factor1D):
    """
    Represents the Bootstrapped implied parameters for a hull-white 2 factor model
    """

    def __init__(self, param):
        super(HullWhite2FactorModelParametersJacobian, self).__init__(param)

    def calculate_components(self, factor, gradients, currency):
        contrib = {}
        gradient_index = gradients.index.get_level_values(0)
        for instrument, value in self.param.items():
            numerator = []
            denominator = []
            values = []
            for param, grad_param in value.items():
                if param == 'Curve':
                    param_name = utils.Factor('InterestRate', factor.name)
                elif param == 'Quanto_FX_Volatility':
                    param_name = utils.Factor('GBMAssetPriceTSModelParameters', currency + ('Vol',))
                else:
                    param_name = utils.Factor(factor.type, factor.name + (param,))
                full_param_name = utils.check_tuple_name(param_name)
                if full_param_name in gradient_index:
                    if isinstance(grad_param, utils.Curve):
                        slice = gradients.loc(axis=0)[full_param_name, grad_param.array[:, 0]]
                        # have to make sure the denominator and numerator match
                        denominator.extend(np.interp(slice.index.get_level_values(1), *grad_param.array.T))
                        numerator.extend(slice['Gradient'].values)
                        values.extend(slice['Value'].values)
                    else:
                        denominator.append(grad_param)
                        numerator.append(gradients.loc[(full_param_name,)]['Gradient'].iloc[0])
                        values.append(gradients.loc[(full_param_name,)]['Value'].iloc[0])

            contrib[instrument] = {
                'Gradient': np.dot(numerator, values) / np.dot(denominator, values),
                'Premium': value['Premium']
            }

        return pd.DataFrame(contrib).T

    def get_tenor(self):
        """Gets the tenor points stored in the Curve attribute"""
        return np.array([0.0])

    def get_vol_tenors(self):
        return [self.param['Sigma_1'].array[:, 0], self.param['Sigma_2'].array[:, 0]]

    def get_tenor_indices(self):
        zero = np.array([[0.0]])
        sig1, sig2 = self.get_vol_tenors()
        return {'Alpha_1': zero,
                'Alpha_2': zero,
                'Correlation': zero,
                'Sigma_1': sig1.reshape(-1, 1),
                'Sigma_2': sig2.reshape(-1, 1),
                'Quanto_FX_Correlation_1': zero,
                'Quanto_FX_Correlation_2': zero}

    def current_value(self, tenors=None, offset=0.0, include_quanto=False):
        """Returns the parameters of the HW2 factor model as a dictionary"""

        params = OrderedDict([('Alpha_1', np.array([self.param['Alpha_1']])),
                              ('Alpha_2', np.array([self.param['Alpha_2']])),
                              ('Correlation', np.array([self.param['Correlation']])),
                              ('Sigma_1', self.param['Sigma_1'].array[:, 1]),
                              ('Sigma_2', self.param['Sigma_2'].array[:, 1])])

        if self.get_instantaneous_correlation() is None and (
                'Quanto_FX_Correlation_1' in self.param and 'Quanto_FX_Correlation_2' in self.param):
            # needs to be looked up if there's not instantaneous correlation - otherwise it's calculated
            params['Quanto_FX_Correlation_1'] = np.array([self.param['Quanto_FX_Correlation_1']])
            params['Quanto_FX_Correlation_2'] = np.array([self.param['Quanto_FX_Correlation_2']])

            if include_quanto:
                params['Quanto_FX_Volatility'] = self.param['Quanto_FX_Volatility'].array[:, 1]

        return params


class HullWhite2FactorModelParameters(Factor1D):
    """
    Represents the Bootstrapped implied parameters for a hull-white 2 factor model
    """

    def __init__(self, param):
        super(HullWhite2FactorModelParameters, self).__init__(param)

    def get_instantaneous_correlation(self):
        return self.param.get('short_rate_fx_correlation')

    def get_quanto_correlation(self, corr, vols):
        C = self.get_instantaneous_correlation()
        if C is not None:
            s1, s2, p = vols[0][-1], vols[1][-1], corr[0]
            scale = C / (s1 ** 2 + s2 ** 2 + 2.0 * p * s1 * s2) ** .5
            return [scale * (s1 + p * s2), scale * (p * s1 + s2)]
        else:
            return self.param.get('Quanto_FX_Correlation_1'), self.param.get('Quanto_FX_Correlation_2')

    def get_tenor(self):
        """Gets the tenor points stored in the Curve attribute"""
        if self.param['Quanto_FX_Volatility'] is None:
            self.param['Quanto_FX_Volatility'] = utils.Curve([], [(0.0, 0.0)])
        return self.param['Quanto_FX_Volatility'].array[:, 0]

    def get_vol_tenors(self):
        return [self.param['Sigma_1'].array[:, 0], self.param['Sigma_2'].array[:, 0]]

    def get_tenor_indices(self):
        zero = np.array([[0.0]])
        sig1, sig2 = self.get_vol_tenors()
        return {'Alpha_1': zero,
                'Alpha_2': zero,
                'Correlation': zero,
                'Sigma_1': sig1.reshape(-1, 1),
                'Sigma_2': sig2.reshape(-1, 1),
                'Quanto_FX_Correlation_1': zero,
                'Quanto_FX_Correlation_2': zero}

    def get_quanto_fx(self):
        if self.param.get('Quanto_FX_Volatility') is not None and self.param[
            'Quanto_FX_Volatility'].array.any():
            return self.param['Quanto_FX_Volatility'].array[:, 1]
        else:
            return None

    def current_value(self, tenors=None, offset=0.0, include_quanto=False):
        """Returns the parameters of the HW2 factor model as a dictionary"""

        params = OrderedDict([('Alpha_1', np.array([self.param['Alpha_1']])),
                              ('Alpha_2', np.array([self.param['Alpha_2']])),
                              ('Correlation', np.array([self.param['Correlation']])),
                              ('Sigma_1', self.param['Sigma_1'].array[:, 1]),
                              ('Sigma_2', self.param['Sigma_2'].array[:, 1])])

        if self.get_instantaneous_correlation() is None and (
                'Quanto_FX_Correlation_1' in self.param and 'Quanto_FX_Correlation_2' in self.param):
            # needs to be looked up if there's not instantaneous correlation - otherwise it's calculated
            params['Quanto_FX_Correlation_1'] = np.array([self.param['Quanto_FX_Correlation_1']])
            params['Quanto_FX_Correlation_2'] = np.array([self.param['Quanto_FX_Correlation_2']])

            if include_quanto:
                params['Quanto_FX_Volatility'] = self.param['Quanto_FX_Volatility'].array[:, 1]

        return params


class PCAMixedFactorModelParameters(Factor1D):
    """
    Represents the Bootstrapped implied paramters for a hull-white 2 factor model
    """

    def __init__(self, param):
        super(PCAMixedFactorModelParameters, self).__init__(param)

    def get_tenor(self):
        """Gets the tenor points stored in the Curve attribute"""
        if self.param['Quanto_FX_Volatility'] is None:
            self.param['Quanto_FX_Volatility'] = utils.Curve([], [(0.0, 0.0)])
        return self.param['Quanto_FX_Volatility'].array[:, 0]

    def get_tenor_indices(self):
        return {'Reversion_Speed': np.array([[0.0]]),
                'Yield_Volatility': self.param['Yield_Volatility'].array[:, 0].reshape(-1, 1)}

    def get_vol_tenor(self):
        return self.param['Yield_Volatility'].array[:, 0]

    def current_value(self, tenors=None, offset=0.0):
        """Returns the parameters of the mixed factor model as a dictionary"""
        return OrderedDict([('Reversion_Speed', np.array([self.param['Reversion_Speed']])),
                            ('Yield_Volatility', self.param['Yield_Volatility'].array[:, 1])])


class CommodityPriceVol(Factor2D):
    field_desc = ('Commodities',
                  ['- **Surface**: *Curve* object consisting of (moneyness, expiry, volatility) triples. Flat',
                   'extrapolated and linearly interpolated. All Floats.'
                   ])

    def __init__(self, param):
        super(CommodityPriceVol, self).__init__(param)


class FXVol(Factor2D):
    field_desc = ('FX and Equity',
                  ['- **Surface**: *Curve* object consisting of (moneyness, expiry, volatility) triples. Flat',
                   'extrapolated and linearly interpolated. All Floats.'
                   ])

    def __init__(self, param):
        super(FXVol, self).__init__(param)


class EquityPriceVol(Factor2D):
    field_desc = ('FX and Equity',
                  ['- **Surface**: *Curve* object consisting of (moneyness, expiry, volatility) triples. Flat',
                   'extrapolated and linearly interpolated. All Floats.'
                   ])

    def __init__(self, param):
        super(EquityPriceVol, self).__init__(param)


class InterestYieldVol(Factor3D):
    field_desc = ('Interest Rates',
                  ['- **Surface**: *Curve* object consisting of (moneyness, expiry, tenor, volatility) quads. Flat',
                   'extrapolated and linearly interpolated. All Floats.',
                   '- **Property_Aliases**: list of key value pairs specifying additional options e.g. Specification',
                   'of a shifted black scholes value via BlackScholesDisplacedShiftValue'
                   ])

    def __init__(self, param):
        super(InterestYieldVol, self).__init__(param)
        self.delta = 0.0
        self.atm_surface = None
        self.premiums = None

    def set_premiums(self, df, currency):
        if df is not None:
            self.premiums = df[df['Currency'] == currency[0]]

    def get_premium(self, expiry, tenor):
        prem = self.premiums[(self.premiums['UnderlyingTenor'] == tenor) &
                             (self.premiums['Expiry'] == expiry)]['Payer']
        return prem.values[0] / 10000.0

    @property
    def BlackScholesDisplacedShiftValue(self):
        shift_value = 0.0
        Property_Aliases = self.param.get('Property_Aliases')
        if Property_Aliases is not None:
            for property_alias in Property_Aliases:
                if 'BlackScholesDisplacedShiftValue' in property_alias:
                    return property_alias['BlackScholesDisplacedShiftValue']
        elif self.premiums is not None:
            return self.premiums['Shift'].apply(lambda x: float(x.replace('%', ''))).unique()[0]
        return shift_value

    @property
    def ATM(self):
        if self.atm_surface is None:
            mn_ix = np.searchsorted(self.moneyness, 0.0)
            atm_vol = np.array([np.interp(1.0, self.moneyness[mn_ix - 1:mn_ix + 1], y)
                                for y in self.get_vols()[:, mn_ix - 1:mn_ix + 1]])
            self.atm_surface = RectBivariateSpline(
                self.tenor, self.expiry, atm_vol.reshape(self.tenor.size, self.expiry.size))
        return self.atm_surface


class InterestRateVol(Factor3D):
    field_desc = ('Interest Rates',
                  ['- **Surface**: *Curve* object consisting of (moneyness, expiry, tenor, volatility) quads. Flat'
                   'extrapolated and linearly interpolated. All Floats.'
                   ])

    def __init__(self, param):
        super(InterestRateVol, self).__init__(param)


class ForwardPriceVol(Factor3D):
    TENOR_INDEX = 0
    EXPIRY_INDEX = 1
    MONEYNESS_INDEX = 2

    field_desc = ('Energy',
                  ['- **Surface**: *Curve* object consisting of (moneyness, expiry, delivery, volatility) quads. Flat',
                   'extrapolated and linearly interpolated. All Floats.'
                   ])

    def __init__(self, param):
        self.flat = None
        self.index_map = OrderedDict()
        super(ForwardPriceVol, self).__init__(param)

    def get_vols(self):
        """Uses flat extrapolation along moneyness and then linear interpolation along expiry"""
        vols = []
        # get the surface
        surface = self.param['Surface'].array
        # Sort by moneyness, then expiry
        self.sorted_vol = surface[np.lexsort((surface[:, self.TENOR_INDEX],
                                              surface[:, self.EXPIRY_INDEX], surface[:, self.MONEYNESS_INDEX]))]
        # store an index for each expiry
        self.index_map.clear()
        for element in self.sorted_vol:
            self.index_map.setdefault(element[self.MONEYNESS_INDEX], OrderedDict()).setdefault(
                element[self.EXPIRY_INDEX], []).append(element[self.TENOR_INDEX])
        self.flat = self.sorted_vol[:, 3]

        for moneyness in self.moneyness:
            surface = self.param['Surface'].array[self.param['Surface'].array[:, self.MONEYNESS_INDEX] == moneyness]
            for x in self.expiry:
                sigma = surface[surface[:, self.EXPIRY_INDEX] == x, 3]
                tenor = surface[surface[:, self.EXPIRY_INDEX] == x, self.TENOR_INDEX]
                vols.append(np.interp(self.tenor, tenor, sigma) if sigma.any() else np.zeros_like(self.tenor))

        return np.array(vols)

    def current_value(self, tenors=None, offset=0.0):
        """
        Returns the value of the Vol space
        Again, this is symmetric wih get_tenor_indices i.e. self.current_value(self.get_tenor_indices()) should be
        just the list of corresponding vols.
        """
        if tenors is not None and self.expiry.size > 1 and self.moneyness.size > 1:
            interpolator = [RectBivariateSpline(self.expiry, self.tenor, vol.reshape(self.expiry.size, -1), kx=1, ky=1)
                            for vol in self.vols.reshape(self.moneyness.size, -1)]
            interp_vols = []
            for tenor in tenors:
                index = np.clip(self.moneyness.searchsorted(
                    tenor[self.MONEYNESS_INDEX], side='right') - 1, 0, self.moneyness.size - 1)
                index_p1 = np.clip(index + 1, 0, self.moneyness.size - 1)
                val = np.interp(
                    tenor[self.MONEYNESS_INDEX], self.moneyness[[index, index_p1]],
                    np.dstack([interpolator[index](tenor[self.EXPIRY_INDEX], tenor[self.TENOR_INDEX]),
                               interpolator[index_p1](tenor[self.EXPIRY_INDEX], tenor[self.TENOR_INDEX])]).flatten())
                interp_vols.append(val)
            return np.array(interp_vols)
        return self.flat

    def get_tenor_indices(self):
        return self.sorted_vol[:, :3]


def construct_factor(factor, price_factors, factor_interp):
    # now lookup the params of the factor
    price_factor = price_factors[utils.check_tuple_name(factor)]
    # check the interpolation on interest Rates - can add more methods/price factors as desired
    if factor.type == 'InterestRate':
        interp_method = factor_interp.search(factor, price_factor, True)
        price_factor['Interpolation'] = factor_interp_map.get(interp_method, 'Linear')

    return globals().get(factor.type)(price_factor)


if __name__ == '__main__':
    pass
