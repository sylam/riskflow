To use the models and valuations in the previous sections, a **Calculation** object needs to be
constructed and correctly setup. Note that all calculations need a **calc_name** field as a
description.

---


## Base_Revaluation

This applies the valuation models mentioned earlier to the portfolio per deal.

The only inputs are:

- **Currency** of the output.
- **Run_Date** at which the marketdata should be applied (i.e. $t_0$)

The output is a dictionary containing the DealStructure and the calculation computation
statistics.

## Credit_Monte_Carlo

A profile is a curve $V(t)$ with values specified at a discrete set of future dates $0=t_0<t_1<...<t_m$ with
values at other dates  obtained via linear interpolation or zero extrapolation i.e. if $t_{i-1}<t<t_i$ then
$V(t)$ is a linear interpolation of $V(t_{i-1})$ and $V(t_i)$; otherwise $V(t)=0$.

The valuation models described earlier are used to construct the profile. The profile dates $t_1,...,t_m$ are
obtained by taking the following union:

- The deal's maturity date.
- The dates in the **Base Time Grid** up the the deal's maturity date.
- Deal specific dates such as payment and exercise dates.

Deal specific dates improve the accuracy of the profile by showing the effect of cashflows, exercises etc.

### Aggregation

If $U$ and $V$ are profiles, then the set $U+V$ is the union of profile dates $U$ and $V$. If $E$ is the
credit exposure profile in reporting
currency (**Currency**), then:

$$E = \sum_{d} V_d$$

where $V_d$ is the valuation profile of the $d^{th}$ deal. Note that Netting is always assumed to be
**True**.

#### Peak Exposure

This is the simulated exposure at percentile $q$ where $0<q<1$ (typically q=.95 or .99).

#### Expected Exposure

This is the profile defined by taking the average of the positive simulated exposures i.e. for each profile
date $t$,

$$\bar E(t)=\frac{1}{N}\sum_{k=1}^N \max(E(t)(k),0).$$
#### Exposure Deflation

Exposure at time $t$ is simulated in units of the time $t$ reporting currency. Exposure deflation converts
this to time $0$ reporting currency i.e.

$$V^*(t)=\frac{V(t)}{\beta(t)}$$

where

$$\beta(t)=\exp\Big(\int_0^t r(s)ds\Big).$$

This can be approximated by:

$$\beta(t)=\prod_{i=0}^n\frac{1}{D(s_{i-1},s_i)}$$

where $0=s_0<...<s_n=t$. The discrete set of dates $s_1,...,s_{n-1}$ are model-dependent.
### Credit Valuation Adjustment

This represents the adjustment to the market value of the portfolio accounting for the risk of default. Only
unilateral CVA (i.e accounting for the counterparty risk of default but ignoring the potential default of
the investor) is calculated. It is given by:

$$C=\Bbb E(L(\tau)),$$

where the expectation is taken with respect to the risk-neutral measure, and 

$$L(t)=(1-R)\max(E^*(t),0),$$

with:

- $R$ the counterparty recovery rate
- $\tau$ the counterparty time to default
- $E^*(t)$ the exposure at time $t$ deflated by the money market account.

If **Deflate Stochastically** is **No** then the deflated expected exposure is assumed to be deterministic
i.e. $E^*(t)=E(t)D(0,t)$. Note that if $T$ is the end date of the portfolio exposure then $E^*(t)=0$ for
$t>T$.

Now,

$$\Bbb E(L(\tau))=\Bbb E\Big(\int_0^T L(t)(-dH(t))\Big),$$
$$H(t)=\exp\Big(-\int_0^t h(u)du\Big)$$

where $h(t)$ is the stochastic hazard rate. There are two ways to calculate the expectation:

If **Stochastic Hazard** is **No** then $H(t)=\Bbb P(\tau > t)=S(0,t)$, the risk neutral survival
probability to time $t$ and

$$C=\int_0^T \Bbb E(L(t))(-dH(t))\approx\sum_{i=1}^m C_i,$$

with

$$C_i=\Big(\frac{\Bbb E(L(t_{i-1}))+\Bbb E(L(t_i))}{2}\Big)(S(0,t_{i-1})-S(0,t_i))$$

and $0=t_0<...<t_m=T$ are the time points on the exposure profile. Note that the factor models used should
be risk-neutral to give risk neutral simulations of $\Bbb E^*(t)$.

If **Stochastic Hazard** is **Yes** then $S(t,u)$ is the simulated survival probability at time $t$ for
maturity $u$ and is related to $H$ by

$$S(t,u)=\Bbb E(\frac{H(u)}{H(t)}\vert\mathcal F(t)).$$

where $\mathcal F$ is the filtration given by the risk factor processes. For small $u-t$, the approximation
$H(u)\approx H(t)S(t,u)$ is accurate so that

$$C\approx\sum_{i=1}^m C_i$$

and

$$C_i=\Bbb E\Big[\Big(\frac{L(t_{i-1}))+L(t_i)}{2}\Big)(H_{i-1}-H_i)\Big],$$

again, $0=t_0<...<t_m=T$ are the time points on the exposure profile and

$$H_i=S(0,t_1)S(t_1,t_2)...S(t_{i-1},t_i).$$

### Funding Valuation Adjustment

Posting (or recieving) collateral can imply a funding cost (or benefit) when there is a spread between a
party's interal cost of funding and the contractual interest rate paid on the collateral balance. The
discounted expectation of this cost (or benefit) summed across all time horizons and scenarios constitutes
a funding value adjustment and can be expressed as:

$$\frac{1}{m}\sum_{j=1}^m \sum_{k=0}^{T-1} B_j(t_k)S_j(t_k)\Big(\frac{D_j^c(t_k,t_{k+1})}{D_j^f(t_k,t_{k+1})}-1\Big)D_j^c(0,t_k),$$

where

- $B_j(t)$ is the number of units of the collateral portfolio for scenario $j$ at time $t$
- $S_j(t)$ is the base currecy value of one unit of the collateral asset for scenario $j$ at time $t$
- $D_j^c(t)$ is the discount rate for the collateral rate at time t for scenario $j$
- $D_j^f(t)$ is the discount rate for the funding rate at time t for scenario $j$

Note that only cash collateral is supported presently although this can be extended.