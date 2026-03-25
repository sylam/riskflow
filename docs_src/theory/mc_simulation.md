## One-Step Survival (OSS) Monte Carlo

Standard Monte Carlo path simulation checks barrier crossings by evaluating a smooth
indicator at each discrete observation date. Near the barrier, this produces high variance
and — for gradient-based calculations — discontinuous or unstable sensitivities.

The **One-Step Survival** (OSS) technique eliminates this by analytically computing, at
each barrier date, the probability that the GBM step would cross the barrier. Surviving paths
are drawn from the conditional (truncated) distribution; the non-surviving weight contributes
its payoff analytically. The result is lower variance, gradient-friendly prices, and exact
(not smoothed) barrier treatment.

### GBM Step Decomposition

Under a GBM process, the log-return over one step $[t_{j-1},t_j]$ is:

$$\log\frac{S_j}{S_{j-1}} = r_j + \sigma_j Z, \qquad Z \sim \mathcal{N}(0,1)$$

where $r_j = \mu_j\,\Delta t_j - \tfrac{1}{2}\sigma_j^2\,\Delta t_j$ is the risk-neutral
drift and $\sigma_j = \bar\sigma\sqrt{\Delta t_j}$ is the step volatility. The barrier
crossing threshold in standard-normal space is:

$$z^* = \frac{\log(H/S_{j-1}) - r_j}{\sigma_j}$$

### Survival Probability and Truncated Draw

For an **up barrier** (barrier crossed when $S_j > H$, i.e. $Z > z^*$), the
probability of surviving (staying below $H$) is:

$$p_j = \Phi(z^*)$$

A draw conditional on surviving is obtained by mapping a uniform variate $u \sim U(0,1)$
through the truncated inverse CDF:

$$Z\;\big|\;\text{survive} = \Phi^{-1}\!\bigl(u\cdot p_j\bigr)$$

For a **down barrier** (barrier crossed when $S_j < H$, i.e. $Z < z^*$):

$$p_j = 1 - \Phi(z^*)$$

$$Z\;\big|\;\text{survive} = \Phi^{-1}\!\bigl((1-p_j) + u\cdot p_j\bigr)$$

Antithetic draws $u$ and $1-u$ are generated jointly to halve the effective variance.

### Survival Weight Track

A scalar weight $L_j$ tracks the probability that path $\omega$ has survived all
barrier dates up to and including $t_j$, starting at $L_0 = 1$:

$$L_j = p_j \cdot L_{j-1}$$

After all observation dates, $L_T$ is the probability of never hitting the barrier on
that path.

### Knock-Out Options

At each barrier observation date $t_j$, the weight $(1 - p_j) L_{j-1}$ represents the
fraction of paths that hit the barrier precisely at step $j$.  For a knock-out option with
cash rebate $R$ paid at the hitting date:

$$\Delta P = (1 - p_j)\,L_{j-1}\,R\,D_j$$

where $D_j$ is the stochastic discount factor to $t_j$. At expiry, surviving paths receive
the vanilla payoff:

$$P_{\text{KO}} = \sum_{j \in \mathcal{B}} (1-p_j)L_{j-1} R D_j + L_T\,g(S_T)\,D_T$$

where $g(S_T) = \phi(S_T - K)^+$ (or the digital equivalent) and $\mathcal{B}$ is the set
of barrier observation dates.

### Knock-In Options via In-Out Parity

For European payoffs with discrete observation dates, the exact identity

$$V_{\text{KI}} = V_{\text{vanilla}} - V_{\text{KO,pure}} + R \cdot E[L_T\,D_T]$$

holds regardless of the barrier schedule, where $V_{\text{KO,pure}}$ is the knock-out price
with *no rebate*. The OSS simulation computes $V_{\text{KO,pure}}$ directly as the
survival-weighted terminal payoff:

$$V_{\text{KO,pure}} = E\!\left[L_T\,g(S_T)\,D_T\right]$$

Applying the parity identity, the per-path estimate for the knock-in price is:

$$P = V_{\text{vanilla}} - L_T\,D_T\bigl(g(S_T) - R\bigr)$$

$V_{\text{vanilla}}$ is computed analytically via Black-Scholes at the start of each step
and is the same for all inner paths of a given outer scenario, so no extra MC noise is
introduced. The clamp $P \geq 0$ is applied to the sample mean to enforce no-arbitrage
(individual path realisations of the control-variate estimator can be negative, but the
expectation cannot).

### Remaining Forward and Variance

For the BARRIER_IN case, the forward and integrated variance from the current date to
expiry are needed for the Black-Scholes vanilla. These are precomputed in $O(N_{\text{fix}})$
from the cumulative sums of per-step contributions:

$$\log F(t_j,T) = \sum_{k>j} r_k + \tfrac{1}{2}\sigma_k^2$$

$$\Sigma^2(t_j,T) = \sum_{k>j} \sigma_k^2$$


