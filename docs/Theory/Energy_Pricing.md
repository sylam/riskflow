## Forward Curve

Energy prices do not necessarily follow the same behaviour as other financial assets. Let $F(t,T)$
 denote the forward price at time t for settlement at time $T$. The (initial) forward price curve
$F(0,T)$ is specified at discrete settlement dates $T_1,...,T_m$. Linear interpolation is used for
other settlement dates T and:

$$F(t,T)=F(t,T_i)$$,

where $i$ is either the least index for which $T_i \ge T$, or $i=m$ if $T \gt T_m$.

---


## CSForwardPriceModel

For commodity/Energy deals, the Forward price is modeled directly. For each settlement date T,
the SDE for the forward price is:

$$ dF(t,T) = \mu F(t,T)dt + \sigma e^{-\alpha(T-t)}F(t,T)dW(t)$$

Where:

- $\mu$ is the drift rate
- $\sigma$ is the volatility
- $\alpha$ is the mean reversion speed
- $W(t)$ is the standard Weiner Process

Final form of the model is

$$ F(t,T) = F(0,T)exp\Big(\mu t-\frac{1}{2}\sigma^2e^{-2\alpha(T-t)}v(t)+\sigma e^{-\alpha(T-t)}Y(t)\Big)$$

Where $Y$ is a standard Ornstein-Uhlenbeck Process with variance:
$$v(t) = \frac{1-e^{-2\alpha t}}{2\alpha}$$

The spot rate is given by $$S(t)=F(t,t)=F(0,t)exp\Big(\mu t-\frac{1}{2}\sigma^2v(t)+\sigma Y(t)\Big)$$