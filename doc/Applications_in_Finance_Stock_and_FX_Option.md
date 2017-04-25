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

