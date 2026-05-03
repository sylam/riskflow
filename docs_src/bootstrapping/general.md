# Observed Market Data 

Market data needs to be calibrated to their corresponding price models in order to construct
a risk neutral calibration. Bootstrapping is the general term used to fit models to data via
*optimizers*. It is a form of calibration that typically only looks at current market data with no
reference to any historical data.

Generally, if $S$ is the price factor that needs to be calculated from $n$ benchmark deals, then the
$i^{th}$ deal has:

 - an observed quoted market value $Q_i$
 - a net value $V_i(S,Q_i)$
 - surface points $p_i$

The points should satisfy $p_1<p_2<...<p_n$ and the process of bootstrapping results in a price
factor $S$ such that $V_i(S, Q_i)=0$ (or at least minimized) for all $i=1,..,n$.
