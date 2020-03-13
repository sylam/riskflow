Let $D(t,T)$ denote the discount factor for the discount rate and $D_f(t,T)$ the discount factor for
the forecast rate. The currency of the discount rate must be the same as the settlement currency
(**Currency**). Note that the currency of the forecast rate may in future be different to the
settelment currency but is currently not implemented.

The **Distrubution Type** on the volatility price factor can only be set to **Lognormal** currently.
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

$$L(t)=\frac{1}{\alpha_2}\Big(\frac{D_f(t,t_1)}{D_f(t,t_2)}-1\Big),$$

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

$$V(t)=U(t)\delta$$,

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

$$ \Big(\sum_{i=n(k-1)+1}^{n(k)}K_i+C_i\Big)D(t,T_k)$$

#### Floating Compounding Cashflow Lists

Similar to the Fixed Compounding Cashflow Lists, $I_i=G_i(r_i(u_i))\alpha_i+m_i\alpha_i$ where $u_i$
being the reset date. Let $V_i(t)$ be the value at time $t$ of an amount $I_i$ paid at the accrual end
date $t_i$ (as opposed to the actual payment date $T_k$). The *estimated interest* is
$I_i=\frac{V_i(t)}{D(t,t_i)}$ when $t<u_i$ otherwise $I_i(t)=I_i$ .

The compounding method can be:

**Include Margin** where the $i^{th}$ cashflow pays $P_i I_i(1+I_{i+1})...(1+I_{n(k)})+C_i$ at time
$T_k$ with a value at $t\le T_k$ of

$$ (P_i I_i(t)(1+I_{i+1}(t))...(1+I_{n(k)}(t))+C_i)D(t,T_k) $$

**Flat** where the $i^{th}$ cashflow pays $P_i I_i(1+J_{i+1})...(1+J_{n(k)})+C_i$ at time $T_k$ with
$J_i=I_i-m_i\alpha_i$. Its value at $t\le T_k$ is

$$ (P_i I_i(t)(1+J_{i+1}(t))...(1+J_{n(k)}(t))+C_i)D(t,T_k) $$

**None** in which case the $i^{th}$ cashflow pays $P_i I_i+C_i$ at time $T_k$.


---


## CFFixedInterestListDeal

A series of fixed interest cashflows as described [here](#fixed-interest-cashflows)

## CFFloatingInterestListDeal

A series of floating interest cashflows as described [here](#floating-interest-cashflows)

## FixedCashflowDeal

The time $t$ value of a fixed cashflow amount $C$ paid at time $T$ is $D(t,T)C$.

## MtMCrossCurrencySwapDeal

This currency swap adjusts the notional of one leg to capture any changes in the FX Spot rate
since the last reset. At each reset, the principal of the adjusted leg is set to the principal
of the unadjusted leg multiplied by the spot FX rate. MtM cross currency swaps are path
dependent.

The unadjusted leg is either a fixed or floating interest rate list and is valued as such,
however, the floating adjusted leg is valued as

$$\sum_{i=1}^n(P_i(t)(L_i(t)+m)\alpha_i+A_i(t))D(t,t_i),$$

where

- $A_i(t)=P_i(t)-P_{i+1}(t)$ for $i\lt n$ and $A_n(t)=P_n(t)$
- $P_i(t)$ is the expected principal $P_i(t)=F(t,t_{i-1})\tilde P_i$,
- $\tilde P_i$ is the unadjusted leg principal for the $i^{th}$ period.
- $F(t,T)$ is the forward FX rate for settlement at time $T$.

## SwaptionDeal

Let $t_0, T_1$ and $T_2$ be the **Option Expiry Date**, **Swap Effective Date** and **Swap Maturity Date** respectively of the swaption deal ($t_0 \le T_1 \lt T2$). If the deal is
cash settled, then let $T$ be the **Settlement Date**.

The value of the underlying swap is

$$U(t)=\delta(V_{float}(t)-V_{fixed}(t))$$

where $V_{float}(t)$ is the value of floating interest rate cashflows, $V_{fixed}(t)$ the
value of fixed interest cashflows and $\delta$ is either $+1$ for payer swaptions and $-1$
for receiver swaptions.

If the fixed leg has payments at times $t_2,...,t_n$, then the Present value of a Basis Point is

$$F(t)=\sum_{i=2}^n P_i \alpha_i D(t,t_i)$$

where $P_i$ is the principal amount and $\alpha_i$ is the accrual year fraction for the
$i^{th}$ fixed interest cashflow. The forward swap rate is

$$s(t)=\frac{V_{float}(t)}{F(t)}.$$

Define the *effective* strike rate as

$$K(t)=\frac{V_{fixed}(t)}{F(t)}$$

Note that presently only zero-margin floating cashflow lists are supported (but this can be
extended). The value of the underlying swap is given by $U(t)=\delta(s(t)-K(t))F(t)$. If
both fixed and floating cashflows have the same payment and accrual dates, then $K(t)=r$
where $r$ is the constant fixed rate on the fixed interest cashflow list.

#### Physically Settled Swaptions

If the **Settlement Style** is **Physical** and $U(t_0)\ge 0$, then the option holder
receives the underlying swap and the value of the deal for $t\ge t_0$ is $U(t)$. Note that
physical settlement has significant path dependency.

#### Cash Settled Swaptions

If the **Settlement Style** is **Cash**, then the option holder receives $\max(U(t_0),0)$
on settlement date $T$. The value of the deal at $t\lt t_0$ is
$F(t)\mathcal B_\delta(s(r),K(0),\sigma\sqrt{(t_0-t)})D(t_0,T)$. Note that this assumes a
lognormal distribution of the forecast rate and uses the Black Model as usual.

#### Swap Rate Volatility

Forward starting (where the effective date of the underlying swap is several months or years
after the option expiry) and  amortizing swaptions are not currently supported. This can be
extended as needed. Otherwise, $\sigma$ is the volatility of the underlying rate at time $t$
for expiry $t_0$, tenor $\tau=T_2-T_1$ and strike $K(0)$