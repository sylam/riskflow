Let $D(t,T)$ denote the discount factor for the discount rate and $D_f(t,T)$ the discount factor for
the forecast rate. The currency of the discount rate must be the same as the settlement currency
(**Currency**). Note that the currency of the forecast rate may in future be different to the
settlement currency but is currently not implemented.

The **Distribution Type** on the volatility price factor can only be set to **Lognormal** currently.
This assumes that the price factors are log-normally distributed (hence have implied *Black*
volatilities). Note that this can be extended to **Normal** resulting in the price factor having
implied *Bachelier* volatilities.

## Cashflows

### Fixed Interest Cashflows

A cashflow with

- principal $P$
- fixed interest rate $r$
- accrual start date $T_1$
- accrual end date $T_2$
- accrual day count convention with $\alpha$ the accrual year fraction from $T_1$ to $T_2$
- payment date $T$
- **Fixed Amount** $C$

has a standard payoff:

$$G(r)=Pr\alpha+C$$

with value at time $t$ of $G(r)D(t,T)$.

### Floating Interest Cashflows

In addtion to the Fixed Interest Cashflow, a Floating Interest Cashflow also has

- reset date $t_0$
- reset start $t_1$
- reset end $t_2$
- margin rate $m$

The cashflow dates must satisfy $T_1\le T_2, t_0\le t_1\lt t_2$ and $t_0\le T$ and the payoff is
$P(L(t_0)+m)\alpha$ with $L(t)$ is the simply-compounded forward rate at time $t$ given by

$$L(t)=\frac{1}{\alpha_2}\Big(\frac{D_f(t,t_1)}{D_f(t,t_2)}-1\Big)$$

where $\alpha_2$ is the accrual year fraction from $t_1$ to $t_2$ using the rate day count convention.

The value of a standard floating interest rate cashflow at time $t\lt t_0$ is,

$$P(L(t)+m)\alpha D(t,T_2)$$

If $T=t_2, \alpha=\alpha_2$ and the discount and forecast rate are the same, then the value is

$$P(D(t,t_1)-D(t,t_2))+Pm\alpha D(t,t_2)$$

For $T\neq t_2$, the valuation needs a *convexity correction* but this is yet to be implemented. The
standard payoff is:

$$G(r)=P(\eta r+\kappa\max((r-K_c),0)+\lambda\max((K_f-r),0)+m)\alpha+C$$

where

- $r$ is a simply-compounded forward rate
- $\eta$ is the swaplet multiplier
- $\kappa$ is a caplet multiplier, with $K_c$ the caplet strike
- $\lambda$ is a floorlet multiplier, with $K_f$ the floorlet strike

### Caplets/Floorlets

A caplet/floorlet is a call/put option on a simply compounded rate. The option payoff at time $T$ is:

$$P\max(\delta(L(t_0)-K),0)\alpha$$

where $K$ is the strike and $\delta$ is either $+1$ for caplets and $-1$ for floorlets. If $T=t_2$ then
the option value at time $t\lt t_0$ is

$$P\mathcal B_\delta(L(t),K,\sigma\sqrt{t_0-t})\alpha D(t,T)$$

where $\mathcal B_\delta(F,K,v)$ is the Black function and $\sigma$ is the volatility of the forecast
rate at time $t$ for expiry $t_0$, tenor $t_2-t_1$ and strike $K$. Note that if $T\neq t_2$ then the
above formula is still used as no covexity correction has been applied.

#### Averaging

A cashflow with averaging depends on a sequence of simply compounded rates $r_1,...,r_m$ with the same
nominal tenor $\tau$. Each rate $r_k$ jas a positive weight $\omega_k$. Let $t_{0,k}$ be the reset
date of $r_k$. The average rate $R$ at time $T$ is

$$R=\frac{\sum_{k=1}^m \omega_k r_k(t_{0,k})}{\sum_{k=1}^m \omega_k}$$

### Cashflow Lists

Consider a fixed or floating cashflow list with payment dates $t_1\le ... \le t_n$ and notional
principal amounts $P_1,...,P_n$. If $U_i(t)$ denotes the value of the $i^{th}$ cashflow at time $t$,
then the value of the cashflow list is $U(t)=\sum_{i=1}^n [t_i\ge t]U_i(t)$

There may also be an optional **Settlement Date** $T$ and **Settlement Amount** $C$. If not specified,
then $T=-\infty$. Cashflow payment dates must be after the settlement date ($T\lt t_1$).

The time $t(>T)$ value of the deal is

$$V(t)=U(t)\delta$$

where either $\delta=1$ for a **Buy**, else $\delta=-1$ for a **Sell**. If $t\le T$, the deal is
treated as a forward contract on the underling cashflow list. If $A$ is the accrued interest up to 
$T$, then $K=C+A$ if **Settlement Amount Is Clean** else $K=A$. The value at time $t\le T$ is

$$V(t)=\Big(\frac{U(t)}{D(t,T)}-K\Big)D_r(t,T)\delta,$$

For cash settled deals, the valuation profile terminates at $T$ with a corresponding cashflow
$V(T)=(U(t)-K)\delta$. If physically settled, then the cashflow is $-K\delta$ at $T$ and the profile
continues until $t_n$.

#### Fixed Compounding Cashflow Lists

For cashflow lists with the interest frequency greater than its payment frequency with payment dates
$T_1\lt ...\lt T_c$,let $n(k)$ be the index of the last cashflow with payment date $T_k$ (with
$n(0)=0$). For groups of cashflows with the same payment date, interest is compounded as follows: the
$i^{th}$ cashflow at time $T_k$ pays $K_i+C_i$ with

$$K_i=P_i I_i(1+I_{i+1})...(1+I_{n(k)})$$

where

- $I_i=r_i\alpha_i$
- $P_i$ is the principal amount$
- $r_i$ is the fixed rate
- $\alpha_i$ is the accrual year fraction
- $C_i$ is the **Fixed Amount** of the $i^{th}$ cashflow.

The final value of the cashflows with payment date $T_k$ is

$$\Big(\sum_{i=n(k-1)+1}^{n(k)}K_i+C_i\Big)D(t,T_k)$$

#### Floating Compounding Cashflow Lists

Similar to the Fixed Compounding Cashflow Lists, $I_i=G_i(r_i(u_i))\alpha_i+m_i\alpha_i$ where $u_i$
being the reset date. Let $V_i(t)$ be the value at time $t$ of an amount $I_i$ paid at the accrual end
date $t_i$ (as opposed to the actual payment date $T_k$). The *estimated interest* is
$I_i=\frac{V_i(t)}{D(t,t_i)}$ when $t<u_i$ otherwise $I_i(t)=I_i$ .

The compounding method can be:

**Include Margin** where the $i^{th}$ cashflow pays $P_i I_i(1+I_{i+1})...(1+I_{n(k)})+C_i$ at time
$T_k$ with a value at $t\le T_k$ of

$$(P_i I_i(t)(1+I_{i+1}(t))...(1+I_{n(k)}(t))+C_i)D(t,T_k)$$

**Flat** where the $i^{th}$ cashflow pays $P_i I_i(1+J_{i+1})...(1+J_{n(k)})+C_i$ at time $T_k$ with
$J_i=I_i-m_i\alpha_i$. Its value at $t\le T_k$ is

$$(P_i I_i(t)(1+J_{i+1}(t))...(1+J_{n(k)}(t))+C_i)D(t,T_k)$$

**None** in which case the $i^{th}$ cashflow pays $P_i I_i+C_i$ at time $T_k$.
