## Forward Curve

Energy prices do not necessarily follow the same behaviour as other financial assets. Let $F(t,T)$
 denote the forward price at time t for settlement at time $T$. The (initial) forward price curve
$F(0,T)$ is specified at discrete settlement dates $T_1,...,T_m$. Linear interpolation is used for
other settlement dates T and:

$$F(t,T)=F(t,T_i)$$

where $i$ is either the least index for which $T_i \ge T$, or $i=m$ if $T \gt T_m$.
