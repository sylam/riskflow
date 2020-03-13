## Discount and zero rates

The discount rate $D(t,T)$ is the price at time $t$ of a unit cashflow paid at $T (T \ge t)$. The zero
rate $r(t,T)$ at time $t$ for maturity $T$ is the continuously compounded interest rate between $t$ and
$T$ where:

$$D(t,T)=exp(-r(t,T)(T-t))$$

Let $r_\tau (t)$ denote the $\tau$-tenor zero rate $r(t,t+\tau)$

## Spread Rates

A *spread rate* is an interest rate price factor that is defined relative to a *parent rate*. A
*base rate* is an interest rate price factor without a parent. The parent of a spread rate can be
another spread rate or a base rate.

A spread rate's ID has the form $R.S$, where $S$ is the spread name and $R$ is the ID of the parent rate.
The discount rate of $R.S$ is given by

$$D_{R.S}(t,T)=D_R(t,T)D_S(t,T)$$

where $D_R$ is the discount rate of the parent and $D_S$ is the spread. Examples of spreads are
ZAR-SWAP.ZAR-USD-BASIS (ZAR-SWAP is the parent and ZAR-USD-BASIS is the spread).

---


## HullWhite1FactorInterestRateModel

The instantaneous spot rate (or short rate) which governs the evolution of the yield curve is modeled as:

$$ dr(t) = (\theta (t)-\alpha r(t) - v(t)\sigma(t)\rho)dt + \sigma(t) dW^*(t)$$

Where:

- $\sigma (t)$ is a deterministic volatility curve
- $\alpha$ is a constant mean reversion speed
- $\theta (t)$ is a deterministic curve derived from the vol, mean reversion and initial discount factors
- $v(t)$ is the quanto FX volatility and $\rho$ is the quanto FX correlation
- $dW^*(t)$ is the risk neutral Wiener process related to the real-world Wiener Process $dW(t)$
by $dW^*(t)=dW(t)+\lambda dt$ where $\lambda$ is the market price of risk (assumed to be constant)

Final form of the model is:
$$ D(t,T) = \frac{D(0,T)}{D(0,t)}exp\Big(-\frac{1}{2}A(t,T)-B(T-t)e^{-\alpha t}(Y(t) -\rho K(t) + \lambda H(t))\Big)$$

Where:

- $B(t) = \frac{(1-e^{-\alpha t})}{\alpha}, Y(t)=\int\limits_0^t e^{\alpha s}\sigma (s) dW$
- $A(t,T)=\frac{B(T-t)e^{-\alpha T}}{\alpha}(2I(t)-(e^{-\alpha t}+e^{-\alpha T})J(t))$
- $H(t) = \int\limits_0^t e^{\alpha s}\sigma (s)ds $
- $I(t) = \int\limits_0^t e^{\alpha s}{\sigma (s)}^2ds $
- $J(t) = \int\limits_0^t e^{2\alpha s}{\sigma (s)}^2ds $
- $K(t) = \int\limits_0^t e^{\alpha s}v(s){\sigma (s)}ds $

The simulation of the random increment $Y(t_{k+1})-Y(t_k)$ (where $0=t_0,t_1,t_2,...$
represents the simulation grid) is normal with zero mean and variance $J(t_{k+1})-J(t_k)$

## HullWhite2FactorImpliedInterestRateModel

This is a generalization of the 1 factor Hull White model. There are 2 correlated risk 
factors where the $i^{th}$ factor has a volatility curve $\sigma_i(t)$, constant reversion
speed $\alpha_i$ and market price of risk $\lambda_i$.

Final form of the model is:
$$ D(t,T) = \frac{D(0,T)}{D(0,t)}exp\Big(-\frac{1}{2}\sum_{i,j=1}^2\rho_{ij}A_{ij}(t,T)-\sum_{i=1}^2B_i(T-t)e^{-\alpha_it}(Y_i(t) -\tilde\rho_i K_i(t) + \lambda_i H_i(t))\Big) $$

Where:

- $B_i(t) = \frac{(1-e^{-\alpha_i t})}{\alpha_i}$, $Y_i(t)=\int\limits_0^t e^{\alpha_i s}\sigma_i (s) dW_i(s)$, $W_1$ and $W_2$ are correlated Weiner Processes with correlation $\rho$ ($\rho_{ij}=\rho$ if $i \neq j$ else 1)
- $A_{ij}(t,T)=B_i(T-t)B_j(T-t)e^{-(\alpha_i+\alpha_j)}J_{ij}(t)+\frac{B_i(T-t)}{\alpha_j}(e^{-\alpha_it}I_{ij}(t)-e^{-(\alpha_i+\alpha_j)t}J_{ij}(t))+\frac{B_j(T-t)}{\alpha_i}(e^{-\alpha_jt}I_{ji}(t)-e^{-(\alpha_i+\alpha_j)t}J_{ji}(t))$
- $H_i(t)=\int\limits_0^t e^{\alpha_is}\sigma_i(s)ds$
- $I_{ij}(t)=\int\limits_0^t e^{\alpha_is}\sigma_i(s)\sigma_j(s)ds$
- $J_{ij}(t)=\int\limits_0^t e^{(\alpha_i+\alpha_j)s}\sigma_i(s)\sigma_j(s)ds$
- $K_i(t)=\int\limits_0^t e^{\alpha_is}\sigma_i(s)v(s)ds$

If the rate and base currencies match, $v(t)=0$ and $\tilde\rho_i=0$. Otherwise, $v(t)$ is
the volatility of the rate currency (in base currency) and $\tilde\rho_i$ is the correlation
between the FX rate and the $i^{th}$ factor. The increment $Y(t_{k+1})-Y(t_k)$ (where
$0=t_0,t_1,t_2...$ corresponds to the simulation grid) is gaussian with zero mean and covariance
Matrix $C_{ij}=\rho_{ij}(J_{ij}(t_{k+1})-J_{ij}(t_k))$.

The cholesky decomposition of $C$ is $$L=\begin{pmatrix} \sqrt C_{11} & 0 \\ \frac{C_{12}}{\sqrt C_{11}} & \sqrt {C_{22}-\frac{C_{12}^2}{C_{11}} } \\ \end{pmatrix}$$
The increment is simulated using $LZ$ where $Z$ is a 2D vector of independent normals at
time step $k$.

## PCAInterestRateModel

The parameters of the model are:
- a volatility curve $\sigma_\tau$ for each tenor $\tau$ of the zero curve $r_\tau$
- a mean reversion parameter $\alpha$
- eigenvalues $\lambda_1,\lambda_2,..,\lambda_m$ and corresponding eigenvectors $Q_1(\tau),Q_2(\tau),...,Q_m(\tau)$
- optionally a historical yield curve $\Theta(\tau)$ for the long run mean of $r_\tau$

The stochastic process for the rate at each tenor on the interest rate curve is specified as:

$$ dr_\tau = r_\tau ( u_\tau  dt + \sigma_\tau dY )$$
$$ dY_t = -\alpha Ydt + dZ$$

with $dY$  a standard Ornstein-Uhlenbeck process and $dZ$ a Brownian motion. It can be shown that:
$$ Y(t) \sim N(0, \frac{1-e^{-2 \alpha t}}{2 \alpha})$$ 

Currently, only the covarience matrix is used to define the eigenvectors with corresponding weight curves
$w_k(\tau)=Q_k(\tau)\frac{\sqrt\lambda_k}{\sigma_\tau}$ and normalized weight curve$$B_k(\tau)=\frac{w_k(\tau)}{\sqrt{\sum_{l=1}^m w_l(\tau)^2}}$$
Final form of the model is

$$ r_\tau(t) = R_\tau(t) exp \Big( -\frac{1}{2} \sigma_\tau^2 (\frac{1-e^{-2 \alpha t}}{2 \alpha}) + \sigma_\tau \sum_{k=1}^{3} B_k(\tau) Y_k(t) \Big)$$

Where:

- $r_\tau(t)$ is the zero rate with a tenor $\tau$  at time $t$  ($t = 0$ denotes the current rate at tenor $\tau$)
- $\alpha$ is the mean reversion level of zero rates
- $Y_k(t)$ is the OU process associated with Principle component $k$

To simulate the mean rate $R_\tau(t)$ (note that $R_\tau(0)=r_\tau(0)$ ), there are 2 choices:

**Drift To Forward** where the mean rate is the inital forward rate from $t$ to $t+\tau$ so that
$$\frac{D(0,t+\tau)}{D(0,t)}=e^{R_\tau(t)\tau} $$
**Drift To Blend** is a weighted average function of the current rate and a mean reversion level
$\Theta_\tau$ $$R_\tau(t)=[e^{-\alpha t}r_\tau (0) + (1-e^{-\alpha t})\Theta_\tau]$$