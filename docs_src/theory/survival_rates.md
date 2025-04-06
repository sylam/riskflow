## Hazard Rates

The relationship between the hazard rate $h(t)$, survival probability $S(t,T)$ and the
forward hazard rate $h(t,T)$ is
$$exp\Big(-\int_t^T h(t,u)du\Big)=S(t,T)=\Bbb{E}_t\Bigl(exp\bigl(-\int_t^T h(s)ds\bigr)\Bigr) $$

Where $\Bbb{E}_t$ represents the risk-neutral expectation conditional on information at time $t$ and
$h(t)=h(t,t)$.

Initial survival probabilities are represented as a log survival probability curve $I(t)$ defined as

$$I(t)=\int_0^t h(0,s)ds=-\log S(0,t)$$

where $I(0)=0$ and $I(u)\ge I(t)$ for $u \ge t$.
