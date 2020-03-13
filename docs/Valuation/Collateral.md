Collateral agreements (CSA's) are represented using a container instrument called a netting collateral
set. The idea is to first model the effect of an uncollateralized net portfolio $V(t)$ and then, per
scenario, transform this to a collateralized portfolio $\hat V(t)$.

In addition to posting and recieving collateral there are still two residual risks viz.

### Settlement Risk

This arises when counterparty default occurs unexpectantly. Potentially, a party may make payment (or 
post collateral) to the counterparty without receiving the corresponding collateral (or payment) in
return.

### Liquidity risk

This refers to the basis risk between the market cost of closing out or replacing the counterparty
portfolio against the realized market value of the collateral held.

---


## NettingCollateralSet

The general approach to simulating collateral is as follows:

Define:

- $t$ the simulation time
- $V(t)$ a realization of the uncollateralized portfolio for a scenario
- $\hat V(t)$ a realization of collateralized portfolio for a scenario
- $B(t)$ the number of units of the collateral portfolio that should be held if the collateral agreement was honoured by both parties. This includes minimum transfer amounts and is piecewise constant between collateral call dates
- $S(t)$ is the value of one unit of the collateral portfolio in base currency for a scenario
- $\delta_s$ the length of the settlement period
- $\delta_l$ the length of the liquidation period
- $t_s$ the start of the settlement period
- $t_l$ the start of the liquidation period
- $t_e$ the end of the liquidation period
- $C(t_1,t_2)$ the value in base currency of cash in all currencies accumulated by the portfolio over the interval $[t_1,t_2]$

The closeout period (when the counterparty becomes insolvent) starts at $t_s$ and ends at $t_e$
(when the position and collateral have been liquidated). This period is further divided into a
settlement period $\delta_s$ followed by a liquidation period $\delta_l$ such that the
liquidation begins at $t_l$ when the settlement period ends. After the settlement period,
neither party pays cashflows or transfers collateral but the market risk on the portfolio and
collateral continue until the end of liquidation period.

The collateralized portfolio at time $t$ is the difference between the sum of the liquidated
portfolio value and the cash accumulated during the closeout period $C(t_s,t_e)$ and the
liquidated value of the $B$ units of the collateral portfolio $S(t_e)$ held:
$$\hat V(t)=V(t_e)+C(t_s,t_e)-\min\{B(u):t_s\le u\le t_l\}S(t_e)$$

The calculation of $C(t_s,t_e), B(t)$ and $S(t)$ is described below.
### Standard and Forward looking closeout

Usually exposure is reported at the end of the closeout period i.e. :

- $t_s=t-\delta_l-\delta_s$
- $t_l=t-\delta_l$
- $t_e=t$

However, it is also possible to model the settlement and liquidation periods to come after $t$.
This forward looking closeout implies:

- $t_s=t$
- $t_l=t+\delta_s$
- $t_e=t+\delta_s+\delta_l$

Note that Standard closeout causes exposure to be reported one closeout period after portfolio
maturity (as the mechanics of default are still present).

### Cashflow Accounting

Define (for time $t$):

- $C_r(t)$ the unsigned cumulative base currency amount of cash received per scenario
- $C_p(t)$ the unsigned cumulative base currency amount of cash paid per scenario
- $C_i(t)$ the signed net cash amount paid per scenario per currency $i$ (positive if received, negative if paid)
- $X_i(t)$ the exchange rate from currency $i$ to base currency

Interest over the closeout period is assumed negligible and hence not calculated. If **Exclude Paid Today** is **No**,
then the portfolio value at $t$ includes cash paid on $t$ and hence:

$$C_r(t)=\sum_{t_j<t}\sum_i X_i(t_j)\max(0,C^i(t_j))$$
$$C_p(t)=\sum_{t_j<t}\sum_i X_i(t_j)\max(0,-C^i(t_j))$$

Otherwise (when **Exclude Paid Today** is **Yes**):

$$C_r(t)=\sum_{t_j\le t}\sum_i X_i(t_j)\max(0,C^i(t_j))$$
$$C_p(t)=\sum_{t_j\le t}\sum_i X_i(t_j)\max(0,-C^i(t_j))$$

#### Settlement risk mechanics

The **Cash Settlement Risk** mechanics are determined depending on the order of cash payments verses collateral.

- **Received Only** assumes that collateral is transfered before cash. All cash is retained during the settlement period.

$$C(t_s,t_e)=C_r(t_e)-C_p(t_e)-C_r(t_s)+C_p(t_s)$$

- **Paid Only** assumes that cash is transferred before collateral. Cash is paid by both sides
and no cash is retained during the settlement period.

$$C(t_s,t_e)=C_r(t_e)-C_p(t_e)-C_r(t_l)+C_p(t_l)$$

- **All** assumes that the ordering of cash and collateral is not fixed (i.e. paid and received cash are at risk). Only received cash is retained during the settlement period.

$$C(t_s,t_e)=C_r(t_e)-C_p(t_e)-C_r(t_s)+C_p(t_l)$$

### Collateral Accounting

Define:

- $A(t_i)$ the agreed value in base currency of collateral that should be held per scenario if
 the CSA was honoured by both parties *ignoring minimum transfer amounts*. Collateral is not
held simultaneously by both parties. Received collateral is positive and posted is negative.
- $A(0)$ the initial value of collateral in base currency specified by the **Opening Balance**.
- $X(t)$ the exchange from the CSA currency to the base currency. All amounts on the CSA
(Thresholds, minimum transfers etc.) are expressed in the **Agreement Currency**.
- $I$ the independent amount of collateral in agreement currency that is either posted (negative) or received (positive).
- $H(t)$ the received threshold of collateral: if the portfolio value is above this, the agreed collateral value must be increased by the difference.
- $G(t)$ the posted threshold of collateral: if the portfolio value is below this, the agreed collateral value must be decreased by the difference.
- $M_r(t)$ the minimum received transfer amount. The collateral held will not increase unless the increase is at least this amount.
- $M_p(t)$ the minimum posted transfer amount. The collateral posted will not increase unless the increase is at least this amount.
- $S_h(t)$ is the value in base currency of one unit of the collateral portfolio after haircuts.
- $t_i, i>0$ are collateral call dates. Note that $t_0=0$ and that in general, $t_0$ need not be a collateral call date.

We then have the following relationships:

$$A(t_i)=X(t_i)I+\begin{cases} V(t_i)-X(t_i)H(t_i), &\text{if } V(t_i)>X(t_i)H(t_i)\\ V(t_i)-X(t_i)G(t_i), &\text{if } V(t_i)<X(t_i)G(t_i) \end{cases}$$

In general, the presence of minimum transfer amounts introduce a path dependency on $B(t)$ and,
as such, is not a simple function of $A(t_i)$. Instead, it can be expressed via the following
recurrence:

$$B(t_i)=\begin{cases} \frac{A(0)}{S_h(0)}, &\text{if } i=0\\
\frac{A(t_i)}{S_h(t_i)}, &\text{if } A(t_i)-S(t_i)B(t_{i-1})\ge M_r(t_i)X(t_i)\text{ and } i>0,\\
&\text{or if } S(t_i)B(t_{i-1})-A(t_i)\ge M_p(t_i)X(t_i)\text{ and } i>0\\
B(t_{i-1}), &\text{otherwise}\end{cases}$$

Since $B(t_i)$ is constant between call dates, $B(t)=B(t_{i^*})$, where $t_{i^*}$ is the closest
call date on or before $t$. Since $B(t)$ is path dependent, it requires calculation on all
collateral call dates. Due to this being fairly prohibitive, the recurrence is only evaluated
at collateral call dates associated with a particular simulation time grid. This approximation
can be improved by using a finer simulation grid (also note that the path dependency dissipates
as the minimum transfer amounts reduce to zero).

### Collateral Portfolio

Define:

- $a_i$ the number of units of the $i^{th}$ asset in the collateral portfolio.
- $S_i(t)$ the base currency value per unit of the $i^{th}$ asset.

Currently, the collateral portfolio may only consist of equities and cash and therefore will be
sensitive to interest rates, FX and equity prices over the closeout period. The relative
amounts of each collateral asset in the collateral portfolio is held constant. The percent of
the collateral portfolio represented by a given asset may change as its price relative to other
assets change. The value of the portfolio is therefore:

$$S(t)=\sum_i a_i S_i(t)$$

Haircuts may be defined for each asset class, including cash. The value of each asset allowing
for haircuts is as follows:

$$S_{h,i}=(1-h)S_i(t)$$

Haircuts must be strictly less than one.