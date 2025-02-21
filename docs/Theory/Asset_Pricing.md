## Spot

The spot price of an asset $S(t)$ (assuming immediate delivery) is expressed in the *asset currency*.
The asset currency is specified by the corresponding property:

- **Currency** for equity and commodity prices
- **Domestic Currency** for FX rates

The initial price $S(0)$ is given by its **Spot** property.

### FX Rates

There is just one FX rate price factor for each currency pair (including the base currency). FX rate
price factors always have a currency $C$ and a parent currency $p(C)$ which, currently, will always be
the base currency. If $f(C)$ is the *foreign* currency, and $d(C)$ is the domestic currency, then let
$X_{C/A}(t)$ be the spot price of currency $C$ in currency $A$ at time $t$. The asset factor models
evolve the FX rate $S(t)=X_{f(C)/d(C)}(t)$, effectively making the domestic currency of all FX rates
the same as the base currency

## Forward rates

For equity and commodity prices, the forward price at time $t$ for delivery at $T$ is the usual
no-arbitrage formula:
$$F(t,T)=S(t)\frac{Q(t,T)}{D(t,T)}$$

Here $D(t,T)$ is the discount rate from the asset's interest rate price factor (i.e. its repo rate
specified by the **Interest Rate** price factor) and $Q(t,T)$ is the discount rate from the dividend
rate (for equities) or convenience yield (for commodities).

### FX Rates

Forward FX rates for currency $C$ in currency $A$ is given by:

$$F_{C/A}(t,T)=X_{C/A}\frac{D_C(t,T)}{D_A(t,T)}$$

where $D_C(t;T)$ is the discount rate from the interest rate price factor specified by the
**Interest Rate** property on the FX rate price factor for currency $C$. This can be extended to handle
the case where a given equity/commodity price $S(t)$ with asset currency $C$ is required in another
currency $A$ as follows

$$F_{S/A}(t,T)=S(t)X_{C/A}\frac{Q(t,T)}{D_A(t,T)}$$

### Dividend rate interpolation

A initial dividend rate curve $q(t)$ can be derived from discrete dividends using the no-arbitrage
relationship between spot and forward prices. The spot price is the present-value forward price plus
the dividends the forward purchaser does not get (but the spot purchaser does):

$$S(0)=F(0,t)D(0,t)+\sum_{0 \lt s_i \le t} H_i D(0,t_i),$$

where $H_i$ is the projected/known dividend paid at time $t_i$ with ex-dividend date $s_i$
(with $s_i \le t_i$). With $Q(0,t)=e^{-q(t)t}$,
the implied dividend rate is:

$$q(t)=\frac{1}{t} \log \Biggl( \frac{S(0)}{S(0)-\sum_{0 \lt s_i \le t} H_i D(0,t_i)} \Biggl)$$.

Since the curve $q(t)t$ is constant on each interval $[s_i,s_{i+1}]$, and is a piecewise linear
function of $1/t$, interpolation should also be linear in $1/t$ with flat extrapolation. If $t_1$ and
$t_2$ are points on the curve with $t_1 \le t \le t_2$, then

$$q(t)=q(t_1)+\Biggl(\frac{1/t_1 - 1/t}{1/t_1 - 1/t_2}\Biggl)(q(t_2)-q(t_1))$$

### Interest and Inflation rate interpolation

#### Hermite Interpolation

The Hermite method addresses the drawbacks of cubic splines while maintaining smoothness. It begins 
by fixing the slope at each point using a second-degree polynomial between the target point and its 
two neighbours. Once the slope is established, a third-degree polynomial is adjusted to ensure 
continuity up to the second degree. This approach allows changes in one section of the curve to affect 
only its immediate neighbours. However, when using default extrapolation at the long end, a slight 
drop in forward rates may occur before they become flat.

The formulas for the Hermite method are provided below:

$$r'(t) = (r'_1,, r'_i,, r'_n)$$
$$r(t)=r_i + m_i(t)\big( r_{i+1} - r_i\big) + m_i(t)\big(1-m_i(t)\big)g_i+m_i^2(t)\big(1-m_i(t)\big)c_i$$

where
$$m_i(t) = \frac{t-t_i}{t_{i+1}-t_i} $$
$$g_i = (t_{i+1}-t_i)r'_i - (r_{i+1}-r_i)$$
$$c_i = 2(r_{i+1}-r_i) -(t_{i+1}-t_i)(r'_i+r'_{i+1})$$

The vector $r'$ is calculated as

$$ r'_i = \frac{1}{t_{i+1} - t_i} \left[ \frac{(r_i - r_{i-1}) (t_{i+1} - t_i)}{(t_i - t_{i-1})} + 
\frac{(r_{i+1} - r_i) (t_i - t_{i-1})}{t_{i+1} - t_i} \right]$$

Boundry Conditions

$$ r'_1 = \frac{1}{t_3 - t_1} \left[ \frac{(r_2 - r_1) (t_3 + t_2 - 2t_1)}{(t_2 - t_1)} - 
\frac{(r_3 - r_2) (t_2 - t_1)}{t_3 - t_2} \right]$$
$$ r'_n = \frac{-1}{t_n-t_{n-2}} \left[ \frac{(r_{n-1} - r_{n-2})(t_n - t_{n-1})}{(t_{n-1} - t_{n-2})} - 
\frac{(r_n - r_{n-1})(2t_n-t_{n-1}-t_{n-2})}{t_n - t_{n-1}} \right]$$

#### Linear Interpolation

Linear interpolation involves drawing a straight line between two points surrounding the desired date.
However, linear interpolation is not continuous in its differentials, which means that implied forward
rates can be discontinuous when the curve is interpolated between spot rates.

$$r(t) = \alpha r_{i + 1} + (1-\alpha)r_i$$

where $t_i < t < t_{i+1}$ and

$$ \alpha = \frac{t - t_i}{t_{i + 1} - t_i} $$

#### RT interpolation

Both Linear and Hermite interpolation can be applied to the rate multiplied by the tenor instead of 
just the rate. The variations are HermiteRT and LinearRT and they are exactly as described above except 
instead of interpolating just the rate, we interpolate the rate x tenor

i.e. instead of interpolating:$$r = (r_1,, r_i,, r_n)$$

we interpolate:
$$r = (r_1t_1,, r_it_i,, r_nt_n)$$


---


## GBMAssetPriceModel

The spot price of an equity or FX rate can be modelled as Geometric Brownian Motion (GBM).
The model is specified as follows:

$$ dS = \mu S dt + \sigma S dZ$$

Its final form is:

$$ S = exp \Big( (\mu-\frac{1}{2}\sigma^2)t + \sigma dW(t)  \Big ) $$

Where:

- $S$ is the spot price of the asset
- $dZ$ is the standard Brownian motion
- $\mu$ is the constant drift of the asset
- $\sigma$ is the constant volatility of the asset
- $dW(t)$ is a standard Wiener Process

## GBMAssetPriceTSModelImplied

GBM with constant drift and vol may not be suited to model risk-neutral asset prices. A generalization that
allows this would be to modify the volatility $\sigma(t)$ and $\mu(t)$ to be functions of time $t$.
This can be specified as follows:

$$ \frac{dS(t)}{S(t)} = (r(t)-q(t)-v(t)\sigma(t)\rho) dt + \sigma(t) dW(t)$$

Note that no risk premium curve is captured. For Equity factors, its final form is:

$$ S(t+\delta) = F(t,t+\delta)exp \Big(\rho(C(t+\delta)-C(t)) -\frac{1}{2}(V(t+\delta)) - V(t))         + \sqrt{V(t+\delta) - V(t)}Z  \Big) $$

Where:

- $\sigma(t)$ is the volatility of the asset at time $t$
- $v(t)$ is the *Quanto FX Volatility* of the asset at time $t$. $\rho$ is then the *Quanto FX Correlation*
- $V(t) = \int_{0}^{t} \sigma(s)^2 ds$
- $C(t) = \int_{0}^{t} v(s)\sigma(s) ds$
- $r$ is the interest rate in the asset currency
- $q$ is the yield on the asset (If S is a foreign exchange rate, q is the foreign interest rate)
- $F(t,t+\delta)$ is the forward asset price at time t
- $S$ is the spot price of the asset
- $Z$ is a sample from the standard normal distribution
- $\delta$ is the increment in timestep between samples

In the case that the $S(t)$ represents an FX rate, this can be further simplified to:

$$S(t)=S(0)\beta(t)exp\Big(\frac{1}{2}\bar\sigma(t)^2t+\int_0^t\sigma(s)dW(s)\Big)$$

Here $C(t)=\bar\sigma(t)^2t, \beta(t)=exp\Big(\int_0^t(r(s)-q(s))ds\Big), \rho=-1$ and $v(t)=\sigma(t)$