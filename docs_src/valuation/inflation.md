## Inflation Linked Instruments

The payoff of inflation linked cashflows involves the ratio of reference values $I_R(T_2)$ and $I_R(T_1)$
with $T_1<T_2$. The following approximation is used:

$$\Bbb E_t^T\Big(\phi\Big(\frac{I_R(T_2)}{I_R(T_1)}\Big)\Big)\approx \phi\Big(\frac{I_R(T_2)}{I_R(T_1)}\Big) $$

where $\phi$ is an approximately linear function, $T$ is the cashflow payment date and $\Bbb E_t^T$ is
a $T$ forward measure conditional at time $t$. This approximation ignores the convexity correction which
is dependent on the underlying inflation model.
