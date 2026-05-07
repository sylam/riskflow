## Reinforcement Learning for Hedging

Dynamic hedging is naturally framed as a sequential decision problem: at each business day
the hedger observes the state of the portfolio and the market, chooses a trade in one or
more hedge instruments, and accrues a profit-and-loss until the deal terminates. RiskFlow
treats this as a finite-horizon Markov Decision Process and learns the trading policy by
policy-gradient reinforcement learning.

### MDP Formulation

At each decision date $t_k$ ($k=0,1,\ldots,K-1$) the agent observes a state
$s_k\in\mathcal{S}$ summarising the portfolio, market, and any auxiliary signals; chooses
an action $a_k\in\mathcal{A}$; and receives an immediate reward $r_k$. The episode ends at
$t_K$ with a terminal reward $r_K$ that captures the asymmetric profit-and-loss objective.
Trajectories $\tau=(s_0,a_0,r_1,\ldots,s_{K-1},a_{K-1},r_K)$ are sampled from the
simulator and used to optimise a stochastic policy $\pi_\theta(a\mid s)$.

The discounted return from time $k$ is

$$G_k = \sum_{j=k}^{K} \gamma^{j-k}\,r_j$$

with $\gamma\le 1$. For finite-horizon episodic problems with a known terminal time
$\gamma=1$ is the natural choice — each early decision sees the full undiscounted terminal
payoff in its return target.

### Proximal Policy Optimisation

Policy gradient methods optimise $J(\theta)=\mathbb{E}[G_0]$ via the score-function
identity

$$\nabla_\theta J = \mathbb{E}\bigl[\nabla_\theta \log\pi_\theta(a_k\mid s_k)\,A_k\bigr],$$

where $A_k=G_k - V(s_k)$ is the advantage estimated against a learned value baseline
$V_\phi$. PPO (Schulman et al., 2017) replaces the score-function gradient with a clipped
surrogate that prevents large policy updates from a single rollout:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\Bigl[\min\bigl(\rho_k(\theta) A_k,\;
  \operatorname{clip}(\rho_k(\theta), 1-\varepsilon, 1+\varepsilon) A_k\bigr)\Bigr],$$

with importance ratio $\rho_k(\theta) = \pi_\theta(a_k\mid s_k)/\pi_{\theta_{\text{old}}}(a_k\mid s_k)$.
The full optimisation loss adds a value-regression term and an entropy bonus:

$$L(\theta,\phi) = -L^{\text{CLIP}}(\theta) + c_v\,\bigl(V_\phi(s_k) - G_k\bigr)^2
   - c_e\,\mathcal{H}\bigl[\pi_\theta(\cdot\mid s_k)\bigr].$$

### Generalised Advantage Estimation

Advantages are estimated by GAE (Schulman et al., 2016), an exponentially-weighted average
of multi-step temporal-difference residuals
$\delta_k = r_{k+1} + \gamma V(s_{k+1}) - V(s_k)$:

$$A^{\text{GAE}(\gamma,\lambda)}_k = \sum_{l=0}^{\infty} (\gamma\lambda)^l\,\delta_{k+l}.$$

The hyperparameter $\lambda\in[0,1]$ trades off bias for variance: $\lambda=0$ recovers
the one-step TD residual, $\lambda=1$ recovers the full Monte Carlo return.

### Asymmetric Terminal Utility

The terminal reward implements a Sortino-style hinge utility on the realised
profit-and-loss $X_K$:

$$r_K \;=\;
\begin{cases}
\rho_+ \cdot X_K^p,        & X_K\ge 0\\[2pt]
-\rho_- \cdot |X_K|^p,     & X_K<0
\end{cases}$$

with floor penalty $\rho_->0$ heavier than the surplus reward $\rho_+$, typically
$\rho_-/\rho_+=10$. The asymmetry encodes the production preference for avoiding losses
over capturing surplus and creates a directly-asymmetric gradient on the actor: actions
that drag the portfolio into loss territory are punished more steeply than equally-sized
actions on the upside are rewarded.

### CVaR Path-Level Advantage Weighting

Mean-only PPO can converge to a policy that performs well on the bulk of paths but accepts
a heavy tail, because rare crash paths contribute little gradient against the mean
aggregator. Conditional Value-at-Risk weighting amplifies the gradient signal from those
paths. With a tail fraction $\alpha\in(0,1)$ and a multiplier $\lambda>0$, define the
path-level realised return

$$R^{(b)} = \sum_{k=0}^K r_k^{(b)},$$

set the tail threshold $\tau_\alpha = Q_\alpha\bigl(\{R^{(b)}\}_{b}\bigr)$ to the
$\alpha$-quantile across paths in the rollout, and rescale the per-path advantages by

$$w^{(b)} = 1 + \lambda \cdot \mathbf{1}\bigl[R^{(b)} \le \tau_\alpha\bigr].$$

Tail paths receive $1+\lambda$ times the gradient weight of healthy paths. Standard
mean/variance advantage normalisation is *skipped* on CVaR-active runs — the
standardisation term partially undoes the intentional re-weighting.

### Asymmetric Actor-Critic with Privileged Factors

The simulator has access to information that the production hedger does not — the
realised regime path of a hidden Markov-switching process, the per-state OU parameters,
the latent factor draws driving each scenario. Following Pinto et al. (2017, *asymmetric
actor-critic*), this *privileged* information is exposed only to the value head:

$$\pi_\theta(a\mid s_{\text{obs}}),\qquad
  V_\phi\bigl(s_{\text{obs}}, s_{\text{priv}}\bigr).$$

A privileged critic produces lower-variance advantage estimates without leaking
calibration-only data into the actor. At deployment time only $s_{\text{obs}}$ is needed;
the value head is discarded.

### Discrete Categorical Actions

Hedge sizes are integer numbers of contracts. RiskFlow uses a per-instrument categorical
distribution over the integer trade-delta range $[a_{\min}, a_{\max}]$ rather than a
clipped Gaussian over a continuous trade size. Two motivations:

- The natural action space is integer; a categorical avoids the saturation pathology of
  $\tanh$-Gaussian heads near the support boundary, and avoids the bistability that comes
  from learning a continuous parameter with a hard clipping non-linearity.
- Position limits and contract-size constraints are most naturally enforced as a feasibility
  mask over discrete bins. Infeasible bins receive $-\infty$ logits before softmax, so
  $\pi_\theta$ assigns them zero probability without any gradient pressure to regions the
  environment will reject anyway.

The cost is that the categorical does not interpolate between bins — but for $a_{\max} -
a_{\min} \le 20$ this is operationally a non-issue, and the explicit ordinal structure can
be re-introduced via a Gaussian-over-bins prior (used in the optional textbook KL-anchor).

### References

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage
  Estimation*, 2016.
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), 2018 — chapters 3,
  9, 13 cover MDPs, function approximation, and policy gradients.
- Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning*, 2017.
- Rockafellar & Uryasev, *Optimization of Conditional Value-at-Risk*, 2000 — for the CVaR
  formulation underlying the path-level reweighting above.
