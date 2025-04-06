## Discount and zero rates

The discount rate $D(t,T)$ is the price at time $t$ of a unit cashflow paid at $T (T \ge t)$. The zero
rate $r(t,T)$ at time $t$ for maturity $T$ is the continuously compounded interest rate between $t$ and
$T$ where:

$$D(t,T)=exp(-r(t,T)(T-t))$$

Let $r_\tau (t)$ denote the $\tau$-tenor zero rate $r(t,t+\tau)$

## Spread Rates

A *spread rate* is an interest rate price factor that is defined relative to a *parent rate*. A
*base rate* is an interest rate price factor without a parent. The parent of a spread rate can be
another spread rate or a base rate.

A spread rate's ID has the form $R.S$, where $S$ is the spread name and $R$ is the ID of the parent rate.
The discount rate of $R.S$ is given by

$$D_{R.S}(t,T)=D_R(t,T)D_S(t,T)$$

where $D_R$ is the discount rate of the parent and $D_S$ is the spread. Examples of spreads are
ZAR-SWAP.ZAR-USD-BASIS (ZAR-SWAP is the parent and ZAR-USD-BASIS is the spread).

