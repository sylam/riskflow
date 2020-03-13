## Hazard Rates

The relationship between the hazard rate $h(t)$, survival probability $S(t,T)$ and the
forward hazard rate $h(t,T)$ is
$$exp\Big(-\int_t^T h(t,u)du\Big)=S(t,T)=\Bbb{E}_t\Bigl(exp\bigl(-\int_t^T h(s)ds\bigr)\Bigr) $$

Where $\Bbb{E}_t$ represents the risk-neutral expectation conditional on information at time $t$ and
$h(t)=h(t,t)$.

Initial survival probabilities are represented as a log survival probability curve $I(t)$ defined as
$$I(t)=\int_0^t h(0,s)ds=-\log S(0,t)$$
where $I(0)=0$ and $I(u)\ge I(t)$ for $u \ge t$.

---


## HWHazardRateModel

The Hull-White instantaneous hazard rate process is modeled as:

$$ dh(t) = (\theta (t)-\alpha h(t))dt + \sigma dW^*(t)$$

All symbols defined as per Hull White 1 factor for interest rates.
The final form of the model is

$$ S(t,T) = \frac{S(0,T)}{S(0,t)}exp\Big(-\frac{1}{2}A(t,T)-\sigma B(T-t)(Y(t) + B(t)\lambda)\Big)$$

Where:

- $B(t) = \frac{1-e^{-\alpha t}}{\alpha}$, $Y(t) \sim N(0, \frac{1-e^{-2 \alpha t}}{2\alpha})$
- $A(t,T)=\sigma^2 B(T-t)\Big(B(T-t)\frac{B(2t)}{2}+B(t)^2\Big)$