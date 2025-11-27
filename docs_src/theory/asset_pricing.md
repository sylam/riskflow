## Spot

The spot price of an asset $S(t)$ (assuming immediate delivery) is expressed in the *asset currency*.
The asset currency is specified by the corresponding property:

- **Currency** for equity and commodity prices
- **Domestic Currency** for FX rates

The initial price $S(0)$ is given by its **Spot** property.

## FX Rates

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

## Dividend rate interpolation

A initial dividend rate curve $q(t)$ can be derived from discrete dividends using the no-arbitrage
relationship between spot and forward prices. The spot price is the present-value forward price plus
the dividends the forward purchaser does not get (but the spot purchaser does):

$$S(0)=F(0,t)D(0,t)+\sum_{0 \lt s_i \le t} H_i D(0,t_i),$$

where $H_i$ is the projected/known dividend paid at time $t_i$ with ex-dividend date $s_i$
(with $s_i \le t_i$). With $Q(0,t)=e^{-q(t)t}$,
the implied dividend rate is:

$$q(t)=\frac{1}{t} \log \Biggl( \frac{S(0)}{S(0)-\sum_{0 \lt s_i \le t} H_i D(0,t_i)} \Biggl)$$

Since the curve $q(t)t$ is constant on each interval $[s_i,s_{i+1}]$, and is a piecewise linear
function of $1/t$, interpolation should also be linear in $1/t$ with flat extrapolation. If $t_1$ and
$t_2$ are points on the curve with $t_1 \le t \le t_2$, then

$$q(t)=q(t_1)+\Biggl(\frac{1/t_1 - 1/t}{1/t_1 - 1/t_2}\Biggl)(q(t_2)-q(t_1))$$

## Interest and Inflation rate interpolation

### Hermite Interpolation

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

### Linear Interpolation

Linear interpolation involves drawing a straight line between two points surrounding the desired rate.
However, linear interpolation is not continuous in its differentials, which means that implied forward
rates can be discontinuous when the curve is interpolated between spot rates.

$$r(t) = \alpha r_{i + 1} + (1-\alpha)r_i$$

where $t_i < t < t_{i+1}$ and

$$ \alpha = \frac{t - t_i}{t_{i + 1} - t_i} $$

### RT interpolation

Both Linear and Hermite interpolation can be applied to the rate multiplied by the tenor instead of just the rate. The variations are HermiteRT and LinearRT and they are exactly as described above except instead of interpolating just the rate, we interpolate the rate x tenor

i.e. instead of interpolating:

$$r = (r_1,, r_i,, r_n)$$

we interpolate:

$$r = (r_1t_1,, r_it_i,, r_nt_n)$$

## Volatility rate interpolation

Interpolation across expiry is linear in volatility across all surfaces defined below. This may change to being linear in variance in future.
### Sticky rule

The Sticky Rule dictates the fixed volatility parameters that influence the retrieved volatility. The volatility value $v(x)$ for Skews and SVI surfaces is based on log - moneyness, which is defined differently depending on the sticky rule applied. These are:

 - **Sticky Strike** calculates log-moneyness as
 - 
$$ x=ln(\frac{K}{ATM_{ref}})$$

$K$ is the option strike price

 - **Sticky Moneyness** calculates log-moneyness as
 - 
$$ x=ln(\frac{K}{F})$$

$K$ is the option strike price and $F$ is the underlying forward price

### Shifts

A shift value may be defined on Volatility surfaces used to price Caplets/Floorlets and Swaptions. The default value is 0 but should it be defined, it is ignored during volatility interpolation and only used when pricing the interest rate option by adding the shift value to both the Forward and Strike values in the usual Black Formula.
i.e.

Instead of $\mathcal B_\delta(F,K,\sigma\sqrt{t_0-t})$, we calculate $\mathcal B_\delta(F+s,K+s,\sigma\sqrt{t_0-t})$ 

where $s$ is the shift value. Note that this applies to lognormal Surfaces.


### Skew Surfaces

Skews are built up by nine parameters. These parameters are:

 - ATM Volatility $(v_0>0)$
 - Underlying ATM reference quote
 - the slope $s$
 - the Left curvature $L$
 - the Right curvature $R$
 - the Left cuttoff $C, C<0$
 - the Right cuttoff $D, 0<D$
 - the left wing $\lambda>0$
 - the right wing $\rho>0$

These parameters are used to calculate the volatility $v(x)$ in terms of log-moneyness $x$ as

| Region      | Limits on strike     | Volatility Value    |
| -----------:|:--------------------:| -------------------:|
| I           | $x \lt (1+\lambda)C$ | $\alpha+\beta(1+\lambda)C+\gamma(1+\lambda)^2C^2$ |
| II          | $(1+\lambda)C \lt x \lt C$  | $\alpha+\beta x + \gamma x^2$ |
| III         | $C \lt x \lt 0$    | $v_0+sx+Lx^2$ |
| IV          | $0 \lt x \lt D$    | $v_0+sx+Rx^2$ |
| V           | $D \lt x \lt (1+\rho) D$ | $\alpha'+\beta' x + \gamma' x^2$ |
| VI          | $(1+\rho)D \lt x$         | $\alpha'+\beta'(1+\rho)D+\gamma'(1+\rho)^2D^2$ |

The constants $\alpha, \beta, \gamma$ are derived by solving the following equations:

$$v_0+sC+LC^2=\alpha+\beta C+\gamma C^2$$

$$ s+2LC=\beta + 2\gamma C $$

$$0=\beta + 2\gamma(1+\lambda)C$$

The right side of the curve with coefficients $\alpha', \beta', \gamma'$ are solved analogously but using $D, \rho$ instead of $C, \lambda$

### SVI Surfaces

The SVI volatility skew is defined by the five parameters $a, b, \rho, m, \sigma$

$$ \sigma_{SVI}^2(x)=a+b\big(\rho(x-m)+\sqrt{(x-m)^2+\sigma^2}\big)$$

Arbitrage constraints are not checked (can be implemented in future)

$$0 \leq b \leq \frac{2}{(1+|\rho|)T}$$

and

$$|\rho| \leq 1$$

### Explicit Surfaces

These are moneyness surfaces (defined as either Spot over strike for equities or Forward over strike for FX). These moneyness surfaces must have the moneyness axis strictly greater than 1.0. Volatility is then linearly interpolated across moneyness.
