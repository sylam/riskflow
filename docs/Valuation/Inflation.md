The payoff of inflation linked cashflows involves the ratio of reference values $I_R(T_2)$ and $I_R(T_1)$
with $T_1<T_2$. The following approximation is used:

$$\Bbb E_t^T\Big(\phi\Big(\frac{I_R(T_2)}{I_R(T_1)}\Big)\Big)\approx \phi\Big(\frac{I_R(T_2)}{I_R(T_1)}\Big) $$

where $\phi$ is an approximately linear function, $T$ is the cashflow payment date and $\Bbb E_t^T$ is
a $T$ forward measure conditional at time $t$. This approximation ignores the convexity correction which
is dependent on the underlying inflation model.

---


## YieldInflationCashflowListDeal

This pays a fixed coupon on an inflation indexed principal. Define the following: 

- $P$ the principal amount
- $T_b$ the base reference date
- $T_f$ the final reference date
- $t_1$ the accrual start date
- $t_2$ the accrual end date
- $\alpha$ the accrual daycount from $t_1$ to $t_2$
- $r$ the fixed yield
- $A\ne 0$ is the rate multiplier

The cashflow payoff is

$$P\Big(A\frac{I_R(t,T_f)}{I_R(t,T_b)}\Big)r\alpha$$
