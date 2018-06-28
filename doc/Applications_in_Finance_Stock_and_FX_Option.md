Applications in Finance : Stock and FX Option
=================================

## Financial Derivatives and Arbitage
- Value at maturity of European call and put option
$$
\begin{align}
C(T) &= \left( S(T) - K \right)^+ = \max (0, S(T) - K) \\
P(T) &= \left( K - S(T) \right)^+ = \max (0, K - S(T)) 
\end{align}
$$

- Exotic Option
   - Look back Option
     - Look back Call : $X = \left( S(T) - S_{min} \right)^+$
     - Look back Put  : $X = \left( S_{max} - S(T) \right)^+$ 
   - Barrier Option
     - Call Option은 가격이 어떤 Level $H$ 밑으로 떨어지면 knock-out 된다. (돈 한푼 못 받는다.)
     $$
     X = \left( S(T) - K \right)^+ \cdot I(\min_{0 \leq t \leq T} S(t) \geq T),\; S(0) > H, K > H
     $$
   - Asian Option
     - $T$에서 평균 옵션 가격 $\bar{S} = \frac{1}{T} \int_0^T S(u) du$ 에 대하여 Average Call pays $X = (\bar{S} - K)^+$, Average Put  $X = (S(T) - K)^+$. Random Strike Option pays $X=(S(T) - \bar{S})^+$ .

<<<<<<< HEAD
=======

## Finite Market Model
Stock price at $t$ is $S(t)$, and a risklessinvestment (bond, cash 등 계좌에 있는..) with price $\beta(t)$ at time $t$. If the riskless rate of investment is a constant $r > 1$ then $\beta(t) = r^t \beta(0)$. 그리고 포트폴리오 $(a(t), b(t))$ is the number of shares of stock and bond units held during [t, t-1). 고로, $(a(t), b(t))$ 는 $\mathcal{F}_{t-1}$ measurable.

**The value of portpolio $V(t)$** is represented just before time $t$ transactiobns after time $t$ price was observed such that
$$
V(t) = a(t)S(t) + b(t)\beta(t)
$$
- **Self Financing**: Portpolio의 변화는 오직 투자 자체에서만 일어나지 외부로 부터 In/out에 영향 받지 않는다. 따라서 Self-financing이면 다음이 성립한다.
$$
a(t)S(t) + b(t)\beta(t) = a(t+1)S(t) + b(t+1)\beta(t)
$$
   - 즉, $S(t), \beta(t)$ 가 동일하면 $a(t), b(t)$는 시간이 변해도 동일하다.
- **EMM** : Equivalent Martingale Measure
- **Admissiable** : if a stratage is self-financing and the corresponding value process is non-negative.
- **Contingent claim** : a non negative random variable $X$ on $(\Omega, \mathcal{F}_T)$.
- **Attainable** : a claim $X$ is attainable if there exists an addmissible stratagy replicating the claim. i.e.
$$
V(t) = V(0) + \sum_{i=1}^t \left( a(i) \Delta S(i) + b(i) \Delta \beta(i) \right)
$$
일떄, $V(t) \geq 0$ and $V(T) = X$
- ** Arbitrage opportunity** : an admissible trading stratagy such that $V(0) = 0$, but $EV(T) > 0$.
  - $V(0) = 0$, but $EV(T) > 0$은 $P(V(T) > 0) > 0$ 과 같다.

### Theorem : 
Suppose there is probablity measure $Q$ , such that the discounted stock process $Z(t) = S(t)/\beta(t)$ is a $Q$-martingale. Then for any admissible trading stratage the discounted value process $V(t)/\beta(t)$ is also a $Q$-martingale.
#### proof
$$
\begin{align}
E_Q \left( \frac{V(t+1)}{\beta(t+1)} | \mathcal{F}_t\right) &= E_Q (a(t+1)Z(t+1) + b(t+1) | \mathcal{F}_t) \\
&= a(t+1)E_Q(Z(t+1)| \mathcal{F}_t) + b(t+1) \;\;\;\textit{since } a(t), b(t) \textit{is predictable}\\
&= a(t+1)Z(t) + b(t+1) \;\;\;\textit{since }Z(t)\textit{is martingale}\\
&= a(t)Z(t) + b(t) \;\;\;\textit{since }a(t), b(t) \textit{is self-financing} \\
&= \frac{V(t)}{\beta(t)}
\end{align}
$$
... 크게 정리할 것이 없다.

## Semimartingale market model
### Arbitrage in Continuous Time Model
Continuous Time 일떄 **Self Financing**
#### Definition : Self Financing
A portfolio $(a(t), b(t)), \; 0 \leq t \leq T$ is called **Self-Financing**
If **the change in value comes only from the change in priceof the assets**,
$$
\begin{align}
dV(t) &= a(t)dS(t) + b(t)d\beta(t) \\
V(t) &= V(0) + \int_0^t a(u) dS(u) + \int_0^t b(u) d\beta(u)
\end{align}
$$
##### Assumptions
- $S(t)$, $\beta{t}$ are semimartingales.
- The process $a(t), b(t)$ are predictable .i.e. integrable (즉, 적분값이 존재한다는 의미)

#### Theorem 11.11
$(a(t), b(t))$ is self-financing if and only if the discounted value process $\frac{V(t)}{\beta(t)}$ is a stochastic integral with respect to the discounted price process
$$
\frac{V(t)}{\beta(t)}= V(0) + \int_0^t a(u) dZ(u)
$$
where $Z(t) = S(t)/\beta(t)$. 

##### proof
$$
\begin{align}
d\frac{V(t)}{\beta(t)} &= \frac{1}{\beta(t-)}dV(t) + V(t-) d\left( \frac{1}{\beta(t)}\right) + d \left[ V, \frac{1}{\beta} \right](t) \\
&= \frac{1}{\beta(t)}dV(t) + V(t-) d\left( \frac{1}{\beta(t)}\right)
\end{align}
$$
=======
>>>>>>> 880d98a0633278e465530a6961aead1238d8ec5c
