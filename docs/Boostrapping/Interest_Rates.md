

---


## HullWhite2FactorModelParameters

A set of parameters $\sigma_1, \sigma_2, \alpha_1, \alpha_2, \rho$ are estimated from ATM
swaption volatilities. Swaption volatilities are preferred to caplets to better estimate $\rho$.Although assuming that $\sigma_1, \sigma_2$ are constant makes the calibration of this model
considerably easier, in general, $\sigma_1, \sigma_2$ should be allowed a piecewise linear term
structure dependent on the underlying swaptions.

For a set of $J$ ATM swaptions, we need to minimize:

$$E=\sum_{j\in J} \omega_j (V_j(\sigma_1, \sigma_2, \alpha_1, \alpha_2, \rho)-V_j)^2$$

Where $V_j(\sigma_1, \sigma_2, \alpha_1, \alpha_2, \rho)$ is the price of the $j^{th}$ swaption
under the model, $V_j$ is the market value of the $j^{th}$ swaption and $ \omega_j$ is the corresponding
weight. The market value is calculated using the standard pricing functions

To find a good minimum of the model value, basin hopping as implemented [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html) as well as
least squares [optimization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html) are used.

The error $E$ is algorithmically differentiated and then solved via brute-force monte carlo
using tensorflow and scipy.

If the currency of the interest rate is not the same as the base currency, then a quanto correction needs
to be made. Assume $C$ is the value of the interest rate/FX correlation price factor (can be estimated from
historical data), then the FX rate follows:

$$d(log X)(t)=(r_0(t)-r(t)-\frac{1}{2}v(t)^2)dt+v(t)dW(t)$$

with $r(t)$ the short rate and $r_0(t)$ the short rate in base currency. The short rate with a quanto
correction is:

$$dr(t)=r_T(0,t)dt+\sum_{i=1}^2 (\theta_i(t)-\alpha_i x_i(t)- \bar\rho_i\sigma_i v(t))dt+\sigma_i dW_i(t)$$

where $W_1(t),W_2(t)$ and $W(t)$ are standard Wiener processes under the rate currency's risk neutral measure
and $r_T(t,T)$ is the partial derivative of the instantaneous forward rate r(t,T) with respect to the maturity.
date $T$.
Define:
$$F(u,v)=\frac{\sigma_1u+\sigma_2v}{\sqrt{\sigma_1^2+\sigma_2^2+2\rho\sigma_1\sigma_2}}$$

Then $\bar\rho_1, \bar\rho_2$ are assigned:

$$\bar\rho_1=F(1,\rho)C$$
$$\bar\rho_2=F(\rho,1)C$$

This is simply assumed to work