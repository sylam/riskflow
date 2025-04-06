## Price Index

The Price Index can be monthly or quarterly. It is a flat-right interpolated curve $I(t)$ that contains
historical values. The curve is defined on a discrete set of dates $\mathcal T$ where for
$\tau\in\mathcal T, I(\tau)$ is the published price index for period start date $\tau$. If
$\tau_0=$max$\mathcal T$ is the latest historical start period, then for $\tau>\tau_0$,

$$I(\tau)=I(\tau_0)F(\tau-\tau_0)$$

Note that no seasonal factor adjustments are made (but can be included here). Also note that
**Last Period Start** should be set to $\tau_0$.

If $p(\tau)$ is the publication date for period starting $\tau$ (typically $p(\tau)\gt\tau$) and
$s(t)$ is the greatest $\tau$ for a given date $t$ for which $p(\tau) \le t$, then $p(s(t))\le t$
and $s(p(\tau))=\tau$. The value of the Index at $t$ is $I(t)=I(s(t))$.

## Price Index References

If an inflation contract needs the value of a price index sampled at time $T$, the sampled value is
called a **reference** value $I_R(T)$. Generally, $I_R(T)$ will be a function of several previously
published price index values.

For deal valuation, the price index reference convention needs to be known. The following conventions
have been implemented:

- **IndexReference$\mathscr l$M** (for $\mathscr l$=2,3) gives the published price index on the first
day of the month that is $\mathscr l$ calendar months prior to the month of $T$:

$$I_R(T)=I((T-\mathscr {l} m)^{(1)})$$

where $T^{(i)}$ denotes the $i^{th}$ day in month $T$ and $T-\mathscr {l} m$ is result of subtracting
$\mathscr l$ calendar months from T.

- **IndexReferenceInterpolated$\mathscr l$M** (for $\mathscr l$=3 or 4) gives the following
interpolation of $I((T-\mathscr {l} m)^{(1)})$ and $I((T-(\mathscr {l}-1) m)^{(1)})$:

$$I_R(T)=I((T-\mathscr {l} m)^{(1)})+\Biggl( \frac{T-T^{(1)}}{(T^{(1)}+1m)-T^{(1)}} \Biggl)\Big(
I((T-(\mathscr {l}-1) m)^{(1)})-I((T-\mathscr {l}m)^{(1)})\Big)$$

## Inflation Rates

Inflation rate price factors are similar to interest rate price factors but have an associated price
index factor.
