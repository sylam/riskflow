## Conventions

For a given valuation date $t$, expiry date $T$, and time remaining to expiry $\tau$ (i.e. $(T-t)$),
the interest rate and carry are defined as

$$r=\frac{1}{\tau}\log(D(t,T))$$

$$b=\frac{1}{\tau}\log\Big(\frac{F(t,T)}{S(t)}\Big)$$

Asset prices are assumed to be log-normally distributed with interest rates and asset yields assumed
deterministic. Under these assumptions, forwards and european options are given by closed-form formulas.

## Forwards

A forward deal pays $S(T)-K$ at the maturity date $T$. where $K$ is the forward price. The value of the
forward contract at time $t$ is

$$(F(t,T)-K)D(t,T)$$

## European Options

A European option pays $max(\delta(S(T)-K),0)$ at the expiry date $T$, where either $\delta=+1$ for a
call option or $\delta=-1$ for a put option. The value of the European option for annualized implied
volatility $\sigma$ is given by Black's formula:

$$\mathcal B_\delta(F(t,T),K,\sigma\sqrt\tau)D(t,T)$$

where $\mathcal B_\delta$ is given by:

$$\mathcal B_\delta(F,K,v)=\delta(F\Phi(\delta d_1)-K\Phi(\delta d_2))$$

and (for $F>0, K>0, v>0$):

- $d_1=\frac{\log(F/K)}{v}+\frac{v}{2}$
- $d_2=d_1-v$
- $\Phi$ is the standard normal cumulative function
