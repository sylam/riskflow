## Hazard Rates

A discrete forward hazard rate $H$ in some time interval $\delta$ given survival at the start of the
interval is:

$$H(t,T,T+\delta)=\frac{1}{\delta}\Big(\frac{S(t,T)-S(t,T+\delta)}{S(t,T)}\Big).$$

The instantaneous hazard rate (the limit as $\delta\to 0$) is:

$$h(t,T)=-\frac{1}{S(t,T)}\frac{\partial S(t,T)}{\partial T}$$

Which allows us to write:

$$S(t,T)=\exp\Big(-\int_t^T h(t,u)du\Big)$$

### Value of payments on default

Consider a cashflow paying $g(u)$ at time $u$. If default occurs at time $u$, and $t_1\le t\le t_2$,
then the value at $t\le t_1$ is:

$$V(t)=\int_{t_1}^{t_2} D(t,u)S(t,u)h(t,u)g(u)du.$$

This is approximated by assuming a constant forward hazard rate $\bar h$ and forward rate $f$ between
$t_1$ and $t_2$ so that

$$S(t,u)=S(t,t_1)e^{\bar h(u-t_1)}$$
$$D(t,u)=D(t,t_1)e^{f(u-t_1)}$$

where

$$\bar h=\frac{1}{t_2-t_1}\log\Big(\frac{S(t,t_1)}{S(t,t_2)}\Big)$$
$$f=\frac{1}{t_2-t_1}\log\Big(\frac{D(t,t_1)}{D(t,t_2)}\Big)$$

For a unit cashflow paid on default, $g(u)=1$ and

$$\begin{align}V(t) & = D(t,t_1)S(t,t_1)\bar h\Big(\frac{1-e^{(f+\bar h)(t_2-t_1)}}{f+\bar h}\Big)\\
 & = \frac{\bar h}{f+\bar h}\Big((D(t,t_1)S(t,t_1)-D(t,t_2)S(t,t_2)\Big).\end{align}$$

For credit derivatives, define the following:

- $t_0$ is the deal effective date
- $t_1<...<t_n$ are the accrual period end dates
- $T_1<...<T_n$ are the coupon payment dates
- $P_i$ is the principal for the period $t_{i-1}$ to $t_i$
- $\alpha_i$ is the accrual year fraction for the period $t_{i-1}$ to $t_i$
- $C_i$ is coupon paid at time $T_i$
- $\tilde t=\max(t_i,t)$ and $t_i^\prime=(\tilde t_{i-1}+\tilde t_i)/2$, $t$ is the valuation date
- $R$ is the recovery rate value on the survival probability price factor and $D(t,T)$ is $0$ for $t>T$

---


## DealDefaultSwap

This is a bilateral agreement where the buyer purchases protection from the seller against
default of a reference entity with period fixed payments. Should default of the reference
entity occur, the seller pays the buyer the default amount and payment ceases. The default
amount is $P(1-R)$, where $P$ is the principal amount. Note that there could also be an
accrued fee (but is currently ignored).

Assuming the default payment does not occur prior to the effective date of the swap, the
value of this deal at $t$ is

$$\sum_{i=1}^n P_i(1-R)V_i(t)-\sum_{i=1}^n P_i c\alpha_i D(t,T_i)S(t,t_i)$$

where $c$ is the fixed payment rate and

$$V_i(t)=\frac{\bar h_i}{f_i+\bar h_i}\Big((D(t,\tilde t_{i-1})S(t,\tilde t_{i-1})-D(t,\tilde t_i)S(t,\tilde t_i)\Big)$$
$$\bar h_i=\frac{1}{\tilde t_i-\tilde t_{i-1}}\log\Big(\frac{S(t,\tilde t_{i-1})}{S(t,\tilde t_i)}\Big)$$
$$f_i=\frac{1}{\tilde t_i-\tilde t_{i-1}}\log\Big(\frac{D(t,\tilde t_{i-1})}{D(t,\tilde t_i)}\Big)$$

Note that this is further approximated by

$$V_i(t)=\frac{D(t,\tilde t_{i-1})+D(t,\tilde t_i)}{2}\Big(S(t,\tilde t_{i-1})-S(t,\tilde t_i)\Big)$$

which is accurate for low to moderate rates of default.