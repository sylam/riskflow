

---


## GBMTSImpliedParameters

For Risk Neutral simulation, an integrated curve $\bar{\sigma}(t)$ needs to be specified and is
interpreted as the average volatility at time $t$. This is typically obtained from the corresponding
ATM volatility. This is then used to construct a new variance curve $V(t)$ which is defined as
$V(0)=0, V(t_i)=\bar{\sigma}(t_i)^2 t_i$ and $V(t)=\bar{\sigma}(t_n)^2 t$ for $t>t_n$ where
$t_1,...,t_n$ are discrete points on the ATM volatility curve.

Points on the curve that imply a decrease in variance (i.e. $V(t_i)<V(t_{i-1})$) are adjusted to
$V(t_i)=\bar\sigma(t_i)^2t_i=V(t_{i-1})$. This curve is then used to construct *instantaneous* curves
that are then input to the corresponding stochastic process.

The relationship between integrated $F(t)=\int_0^t f_1(s)f_2(s)ds$ and instantaneous curves $f_1, f_2$
where the instantaneous curves are defined on discrete points $P={t_0,t_1,..,t_n}$ with $t_0=0$ is defined
on $P$ by Simpson's rule:

$$F(t_i)=F(t_{i-1})+\frac{t_i-t_{i-1}}{6}\Big(f(t_i)+4f(\frac{t_i+t_{i-1}}{2})+f(t_i)\Big)$$

and $f(t)=f_1(t)f_2(t)$. Integrated curves are flat extrapolated and linearly interpolated.