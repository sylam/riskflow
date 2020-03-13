## Single Barrier Options

A single barrier option has an underlying European option that is either knocked in ot knocked out when
the underlying asset price touches the barrier level $H$. At the expiry date, if the option has not
been knocked in then a rebate is paid.

Consider a generalized European option that pays

$$(AS(T)+B)[\delta(S-K)>0],$$

where $A$ and $B$ are constants, $K$ is the strike and $\delta=+1$ for call options ($\delta=-1$ for
put options). Standard options have $A=\delta$ and $B=-\delta K$ with binary options having $A=0$
and $B$ the payoff amount.

Single barrier options are priced with the formulas from Merton, Reiner and Rubinstein. They are a
combination of the following formulas:

$$ V_1(t)=AS(t)e^{(b-r)\tau}\Phi(\delta x_1)+Be^{-rt}\Phi(\delta x_1-\delta\sigma\sqrt\tau)$$
$$ V_2(t)=AS(T)e^{(b-r)\tau}\Phi(\delta x_2)+Be^{-rt}\Phi(\delta x_2-\delta\sigma\sqrt\tau)$$
$$ V_3(t)=AS(T)e^{(b-r)\tau}\Big(\frac{H}{S(t)}\Big)^{2(\mu+1)} \Phi(\eta y_1)+Be^{-r\tau}
\Big(\frac{H}{S(t)}\Big)^{2\mu}\Phi(\eta y_1 - \eta\sigma\sqrt\tau)$$
$$ V_4(t)=AS(T)e^{(b-r)\tau}\Big(\frac{H}{S(t)}\Big)^{2(\mu+1)} \Phi(\eta y_2)+Be^{-r\tau}
\Big(\frac{H}{S(t)}\Big)^{2\mu}\Phi(\eta y_2 - \eta\sigma\sqrt\tau)$$

| Barrier Type      | Condition     | Parameters          | Option Value                 |
| -----------------:|:-------------:| -------------------:| ---------------------------: |
| Down-and-Out Call | $K \gt H$     | $\eta=1,\delta =1$  |$V_1 - V_3$                |
| Down-and-Out Call | $K \le H$     | $\eta=1,\delta=1$   |$V_2 - V_4$                |
| Up-and-Out Call   | $K \gt H$     | $\eta=-1,\delta=1$  |$0$                        |
| Up-and-Out Call   | $K \le H$     | $\eta=-1,\delta=1$  |$V_1 - V_2 + V_3 - V_4$    |
| Down-and-Out Put  | $K \gt H$     | $\eta=1,\delta=-1$  |$V_1 - V_2 + V_3 - V_4$    |
| Down-and-Out Put  | $K \le H$     | $\eta=1,\delta=-1$  |$0$                        |
| Up-and-Out Put    | $K \gt H$     | $\eta=-1,\delta=-1$ |$V_2 - V_4$                |
| Up-and-Out Put    | $K \le H$     | $\eta=-1,\delta=-1$ |$V_1 - V_3$                |

Where $\mu= {(b-\sigma}^2/2)/\sigma^2, \eta=+1$ for down options or $\eta=-1$ for up options, and

$$ x_1=\frac{1}{\sigma\sqrt\tau}\log\Big(\frac{S(t)}{K}\Big)+(1+\mu)\sigma\sqrt\tau$$
$$ x_2=\frac{1}{\sigma\sqrt\tau}\log\Big(\frac{S(t)}{H}\Big)+(1+\mu)\sigma\sqrt\tau$$
$$ y_1=\frac{1}{\sigma\sqrt\tau}\log\Big(\frac{H^2}{S(t)K}\Big)+(1+\mu)\sigma\sqrt\tau$$
$$ y_2=\frac{1}{\sigma\sqrt\tau}\log\Big(\frac{H}{S(t)}\Big)+(1+\mu)\sigma\sqrt\tau$$

### One touch and No touch Binary Options and Rebates

A one touch binary option pays a fixed amount if the barrier is touched during the life of the option,
otherwise nothing. The two types are:

- Pay the fixed amount when the barrier is touched
- Pay the fixed amount at expiry

A no-touch binary option pays a fixed amount if the barrier is not touched during its life (equivalent
to a fixed amount less a one touch binary option with payment at expiry).

The value of a no touch option is

$$e^{-r\tau}\Big[ \Phi(\eta x_2-\eta\sigma\sqrt\tau)-\Big(\frac{H}{S(t)}\Big)^{2\mu}
\Phi(\eta y_2-\eta\sigma\sqrt\tau) \Big] $$

where $\eta=+1$ for down options or $\eta=-1$ for up options

The value of the one touch option that pays 1 at expiry if S touches the barrier is $e^{-r\tau}$
minus the value of the no-touch option. If it pays 1 when $S$ touches the barrier, then its value is

$$\Big(\frac{H}{S(t)}\Big)^{\mu+\lambda}\Phi(\eta z)+\Big(\frac{H}{S(t)}\Big)^{\mu-\lambda}
\Phi(\eta z-2\eta\lambda\sigma\sqrt\tau))$$
\Phi(\eta z-2\eta\lambda\sigma\sqrt\tau))$$

with

$$z=\frac{1}{\sigma\sqrt\tau}\log\Big(\frac{H}{S(t)}\Big)+\lambda\sigma\sqrt\tau$$
$$\lambda=\sqrt{\mu^2+\frac{2r}{\sigma^2}}$$

Note that $r$ should be floored at $\frac{-(b/\sigma-\sigma/2)^2}{2}$ for $\lambda$ to be defined.
One touch and no touch options have strike $H$.

### Discontinuous Barrier Sampling

Assuming that barriers are not monitored continuously, an adjustment needs to be made to compensate
for discrete sampling. The adjustment is that of Broadie, Glasserman and Kou.

If the barrier is above the asset, the adjusted barrier rate is $He^{\beta\sigma\sqrt\delta}$
where $\delta$ is the period between observations and $\beta$ is the constant defined by:

$$\beta=\frac{\xi(1/2)}{\sqrt{2\pi}}\approx 0.5826$$

where $\xi$ is the Riemann zeta function. If the barrier is below the asset, the adjusted barrier is
$He^{-\beta\sigma\sqrt\delta}$

## Discrete Asian Options

The payoff of a discrete asian option is max$(\delta(\bar S-K),0)$ at expiry $T$ with:

$$\bar S=\frac{1}{D}\sum_{i=1}^n d_iS(t_i),$$

$t_1,...,t_n$ are the dates defined **Sampling Data**, $d_i$ is the weight assigned to $t_i$,
$D=\sum_{i=1}^n d_i$, $K$ is the strike, and $\delta$ is either $+1$ for a call or $-1$ for a put.

If $t_i \le 0$, then $S(t_i)$ is the known price assigned to $t_i$ in the sampling data list otherwise
it's the initial price factor value.

For valuation date $t$, let $m$ be the smallest index for which $t_{m+1} \gt t$ ($t_{n+1}=\infty$).
Then $\bar S-K=A-\bar K$, with

$$A=\sum_{i=m+1}^n \omega_i S(t_i),$$

$\omega_i=d_i/D$, and $\bar K$ is the adjusted strike:

$$\bar K=K-\sum_{i=1}^m \omega_i S(t_i).$$

The expectation of $S(t_i)$ under the risk-neutral measure is the forward price $F(t,t_i)$ with the
expectation of $A$ given by:

$$F=\Bbb{E}_t(A)=\sum_{i=m+1}^n \omega_i F(t,t_i).$$

Define:

- $\tau_i=t_i-t$
- $i\wedge j$ to mean $\min(i,j)$
- $cov_t(\log S(t_i),\log S(t_j))=\sigma_{i\wedge j}^2 \tau_{i\wedge j}$
- $var_t(\log S(t_i))=\sigma_i^2 \tau_i$

Assuming the moment matching approach (the idea that a sum of lognormal variables may be represented by
another lognormal variable with the same first and second moments as the sum), $A$ is lognormal and
$var_t(\log A)=v^2$ where $v$ is given by:

$$ F^2 e^{v^2} = \sum_{i,j=m+1}^n \omega_i\omega_j F(t,t_i)F(t,t_j)e^{\sigma^2_{i\wedge j}
\tau_{i\wedge j}},$$

and the option value is

$$\mathcal B_\delta(F,\bar K,v)D(t,T).$$

Assuming a constant carry rate and constant volatility, we can approximate $F(t,t_i)=S(t)e^{b\tau_i}$
and $\sigma_i=\sigma$ where
$\sigma$ is the implied volatility at time $t$ for expiry $T$ and strike $K'$ and,

$$K'=\frac{\bar K}{\sum_{i=m+1}^n \omega_i}$$

## Equity Swaps

Equity swap legs consist of a series of equity swaplets.

### Equity Swaplet

An equity swaplet has:

- a cashflow $A$
- start date $t_0$
- end date $t_1$
- payment date $T$
- a start multiplier $a_0$
- an end multiplier $a_1$
- a dividend multiplier $b$

For cashflow dates $t_0\le t_1\le T$, the equity swaplet pays:

$$A\Big( \frac{a_1 S(t_1)-a_0 S(t_0)+bH(t_0,t_1)}{S(t_0)^a}\Big)$$

at time $T$. where

- $S(t)$ is the equity price
- $H(t_0,t_1)$ is the time-$t_1$ value of dividends per share with ex-dividend date after $t_0$ but on
or before $t_1$
- $a=1$ if the amount type is **Principle** and $a=0$ if the amount type is **Shares**.

The swaplets have $a_0=a_1=1$ and either $b=1$ if **Include Dividends** is **Yes** or $b=0$ otherwise.

The formula used to value the swaplet at $t\le t_0$ is

$$A\Big((a_1-b)\frac{F(t,t_1)}{F(t,t_0)^a}+ \Big( b\frac{D_r(t,t_0)}{D_r(t,t_1)}-a_0\Big)
F(t,t_0)^{1-a}\Big)D(t,T)$$

where:

- $F$ is the forward equity price
- $D$ is the usual discount factor
- $D_r$ is the discount factor from the equity's repo rate

The value of the swaplet at $t_0\le t\lt t_1$ is

$$A\Big(\frac{(a_1-b)F(t,t_1)+ \frac{b(S(t)+H(t_0,t))}{D_r(t,t_1)}-a_0S(t_0)}{S(t_0)^a}\Big)
D(t,T)$$

The value of the swaplet for $t_1\le t\le T$ is the payoff multiplied by $D(t,T)$

When $t\ge t_0$, the dividend payoff $H(t_0,t\wedge t_1)$ is calculated from **Known Dividends**
and/or the simulated spot and forward prices along the current scenario path

**Dividend Timing** is assumed to be **Terminal** i.e. the swaplet pays $\frac{AbH(t_0,t_1)}{S(t_0)^a}$
at time $T$. Dividend timing could also be continuous (i.e. the dividends are settled on their dividend
dates) but that is not currently implemented.

---


## EquityBarrierOption

A path dependent option described [here](#single-barrier-options)

## EquityDiscreteExplicitAsianOption

A path independent option described [here](#discrete-asian-options)

## EquityForwardDeal

Described [here](Definitions#forwards)

## EquityOptionDeal

A vanilla option described [here](Definitions#european-options)

## EquitySwapLeg

Described [here](#equity-swaps)

## EquitySwapletListDeal

Described [here](#equity-swaps)

## FXBarrierOption

A path dependent FX Option described [here](#single-barrier-options)

## FXDiscreteExplicitAsianOption

A path independent option described [here](#discrete-asian-options)

## FXForwardDeal

An FX forward is an agreement to buy an amount $A$ of one currency in exchange for an amount $B$ of another currency at settlement date $T$.
The value of the deal in base currency at time $t$, ($t \le T$), is

$$A \tilde D(t,T) \tilde X(t)-BD(t,T)X(t)$$

where

- $D$ is the sell currency discount factor
- $\tilde D$ is the buy currency discount factor
- $X$ is the price of the sell currency in base currency
- $\tilde X$ is the price of the buy currency in base currency

## FXNonDeliverableForward

An FX non-deliverable forward effectively an FX Forward deal that is cash settled in a
(potentially) third currency. The deal pays

$$A \tilde X(t)-BX(t)$$

in **settlement currency** at the **settlement date** $T$ where:

- $A$ is the buy currency
- $B$ is the sell currency
- $\tilde X(t)$ is the price of the buy currency in settlement currency
- $X$ is the price of the sell currency in settlement currency

The value of the deal in settlement currency at time $t$, ($t \le T$), is
$$ \Big(A \tilde F(t,T)-BF(t,T)\Big)D(t,T),$$

where:

- $\tilde F(t,T)$ is the forward price of the buy currency in settlement currency
- $F(t,T)$ is the forward price of the sell currency in settlement currency

## FXOneTouchOption

A path dependent FX Option described [here](#one-touch-and-no-touch-binary-options-and-rebates)

## FXOptionDeal

A path independent vanilla FX Option described [here](Definitions#european-options)

## FXSwapDeal

An FX swap is a combination of an FX forward deal with near settlement date $t_1$ and
an FX forward deal in the opposite direction with far settlement date $t_2$, where $t_1 < t_2$.
The base-currency value of the FX swap at time $t$, $t \le t_1$, is

$$A_1 \tilde D(t,t_1) \tilde X(t)-B_1 D(t,t_1)X(t) + B_2 \tilde D(t,t_2) \tilde X(t)-A_2 D(t,t_2) \tilde X(t)$$

where $A_1$ ($A_2$) is amount of the buy currency bought (sold) at $t_1$, ($t_2$), and $B_1$ ($B_2$)
is amount of the sell currency sold (bought) at $t_1$ ($t_2$). Typically, $t_1$ is the spot
settlement date and $A_2 = A_1$