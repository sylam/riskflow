## Forward Prices

Physical commodities that encumber difficulties in storage or a lack of speculators that make the
arbitrage relationship between spot and forward prices at different tenors weak are (unlike equities
or FX) not suited to having the forward price treated as a deterministic function of the spot and
carry. Instead, the underlying forward prices themselves must be simulated.

Forward prices are stored as curves (the forward price at time $t$ for delivery at time $T$ is
denoted $F(t,T)$) with the initial forward price curve $F(0,T)$ entered in the market data file.

## Reference Prices

Most derivatives contracts are not written directly on daily energy prices but usually on averages of
future prices. Reference prices allow this by being a deterministic function of a Forward price curve,
a sampling period delimited by start and end dates ($t_s^s$ and $t_e^s$) and a set of sampling dates
$\mathcal S$. It is assumed that sampling periods are contiguous. Reference prices are simulated
indirectly via their forward prices and are denoted $S^R(t,t_s^s,t_e^s,\mathcal S)$.

### Fixings

Reference prices are represented by a fixing curve i.e. a mapping of reference dates to the underlying
forward price dates. In general, it is possible to contruct reference prices that sample the forward
price curve more than once by simply averaging all prices within a reference price window - this is
not implemented. Currently, each reference price simply looks up a single forward price using the
fixing curve mapping.

### Expectations

The risk-neutral expectation of a sampled price at time $t$ when the sampling date is before the
forward price date is simply:

$$\Bbb{E}_t (F(t^s,T^f))=F(t,T^f)$$

After the sampling date, we need to take the sample value into account and the expectation of a fixing
price with $K$ samples in the past at time $t$ but with $M$ samples still in the future is:

$$F^f(t,T^f)=\frac{1}{K+M}\Big(M F(t,T^f)+ \sum_{i=1}^K F(t_i^s,T^f)\Big)$$

A reference price with $N$ samples is a weighted sum:

$$ S^R(t,t_s^s,t_e^s,\mathcal S)=\frac{1}{K}\sum_{i=1}^K K_i F_i^f(t,T_i^f), $$

where the weight $K_i$ of the $i^{th}$ fixing is given by the number of samples within the fixing
period and $K$ is the normalization term $\sum_{i=1}^N K_i$.

### Sample Dates

The set of dates $\mathcal S$ used to compute the reference price is defined by the deal by using a
**forward price sample** which can then specify business days according to a given calendar.

### Realized Averages

The realized average price is prorated according to the number of sample dates within the realized
period (exactly as with unrealized samples). The currency depends on whether **FX Averaging** is
selected or if the deal is a compo. If the deal is a compo, the realized average is in the
 **payoff currency**, otherwise, it's in the currency of the forward price.

### FX Averaging

When an energy deal is specified in a currency other than the price factor currency, each price
sample must be converted to the native (deal) currency

Currently, **Average FX** can only be set to **No**, meaning that each price sample is converted to
deal currency at the prevailing spot FX rate. A fixing price with $N$ samples is given in deal
currency as:

$$F^f(t,T^f)=\frac{1}{N}\sum_{i=1}^N F(t_i^s,T^f)X(t_i^s) $$

Where $F(t,T)$ is the forward price in price factor currency and $X(t)$ is the price of one unit of
price factor currency in deal currency. Here, the realized average of any historical price samples must
be in deal currency.

Note that forward FX rates $X(t,T)$ are calculated under the usual risk-neutral measure.

$$X(t,T)=X(t)\frac{D_p(t,T)}{D_d(t,T)}$$

where $D_p,D_d$ are the price factor and deal currency discount factors respectively.

### Reference volatility

The volatility of a reference price $S^R(t,t_s^s,t_e^s,\mathcal S)$ with strike price $K$ can be
estimated using the moment matching technique mentioned earlier (assuming that future reference prices
are log-normally distributed).

For valuation date $t$, define $n$ to be the least index for which $t_{n+1}^s\gt t$ (here
$t_{N+1}^s=\infty$). Then $S^R(t,t_s^s,t_e^s,\mathcal S)-K=A-\bar K$, where

$$A=\frac{1}{N}\sum_{i=n+1}^N F(t_i^s,T^f)$$

$$\bar K=K-\frac{1}{N}\sum_{i=1}^n F(t_i^s,T^f)$$

If $F_i$ denotes $F(t_i^s,T_i^f)$ ($t_i^s$ is the $i^{th}$ sampling date in the reference period and
$T_i^f$ is the price date of fixing in which $t_i^s$ falls), then the correlation between $\log F_i$
and $\log F_j$ (for $t_i^s \le t_j^s$) is assumed to be

$$\rho_{ij}=\frac{\sigma_i\sqrt{t_i^s-t}}{\sigma_j\sqrt{t_j^s-t}}$$

and $\sigma_i^2 (t_i^s-t)$ is the standard deviation of $\log F_i$.

The standard deviation of $A$ at time $t$ (with $t\le T$) is then given by

$$w(t,T,t_s^s,t_e^s,\mathcal S)^2=\log\Big(\frac{M_2}{M_1^2}\Big)$$

with $M_1$ and $M_2$ being the first and second moments given by:

$$M_1=\Bbb E_t(A)=\frac{1}{N}\sum_{i=n+1}^N F(t,T_i^f)$$
$$M_2=\frac{1}{N^2}\sum_{i,j=n+1}^N F(t,T_i^f)F(t,T_j^f)\exp\Big(v(t,T_{i\wedge j}^f,
u_{i\wedge j},m)^2(u_{i\wedge j}-t)\Big)$$

where

- $t_1^s,...,t_N^s$ are the sample dates in $\mathcal S$ between $t_s^s$ and $t_e^s$
- $u_i=\min(T,t_i^s)$
- $i\wedge j$ denotes $\min(i,j)$
- $m=\frac{M_1}{\bar K}$ is the moneyness
- $v(t,T,u,m)$ is the forward price volatility at time $t$ for delivery at date $T$ with expiry $u$ and
moneyness $m$ (and $v=0$ for $u\lt t$).

Spreads on top of reference prices and volatilities are not currently implemented.

### Composite Deals

Pricing energy composite (compo) deals requires the forward price in payoff currency $F(t,T)X(t,T)$ 
with the compo-adjusted volatility $\sqrt{\sigma_S^2+\sigma_X^2+2\rho\sigma_S\sigma_x}$ used at
each sampling date for both the reference price and reference price volatility (during moment matching)
respectively. The deal currency and the payoff currency must be the same with the **Realized Average**
expressed in payoff currency.