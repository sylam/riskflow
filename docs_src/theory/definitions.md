## Price Factors

A financial instrument derives its value from several inherently random market variables (e.g. Equity,
commodity, FX, interest rates, etc.). These variables (referred to as *price factors*) will either have
a single value at time $t$ (i.e. $P(t)$,  e.g. Spot Equity or FX rates), or multiple values $P(t;p)$
parametrized by $p$ which represents term, moneyness etc. (e.g. interest rate curves and volatility
 surfaces).

Interest rates have values $D(t,T)$ representing discount factors parametrized by maturity $T$. 
Equity/FX volatility price factors has values $v(t,T,S,K)$ representing implied volatilities for expiry
$T$, spot price $S$ and strike $K$. Valuing a deal at time $t$ involves requesting the set of
corresponding price factor values $\{P(s;p)\}$ for times $s$, where $s \le t$ (in the general case),
and parameters $p$. Note that if there is no path dependency then $s=t$.

## Factor models

A *factor model* calculates future values of a price factor under monte carlo simulation. The factor
model attached to a price factor $P$ has a vector of random processes $R_1(t),R_2(t),...,R_l(t)$, which
generates the underlying ($l$-factor) source of randomness. All generated random numbers are gaussian
and then transformed to simulate other (potentially correlated) stochastic processes as necessary.

Under a particular factor model, the corresponding price factor value $P(t;p)$ at time $t$ may depend
on other price factor values, $Q(t;q)$. Typically, an FX rate at time $t$ would depend on the foreign
and domestic interest rates at time $t$.

## Simulation of price factors with no model

At $t=0$, all price factors are read directly from their marketdata file. If no factor model is
attached and a value is requested at time $t$, then the price factor at time $t=0$ is used (i.e. it
is assumed **Constant**). This can be extended for some risk factors to make it more **Risk Neutral**
but is not implemented yet.

---
