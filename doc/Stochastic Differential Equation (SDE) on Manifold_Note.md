Stochastic Differential Equation (SDE) on Manifold : Note
========================

## Chebyshev 부등식을 사용한 Convergence with Probability 증명 방법
Suppose that **$\sigma$ is globally Lipschitz** and $X_0$ is square integrable. Then $SDE(\sigma, Z, X_0)$has a unique solution $X=\{ X_t, t \geq 0 \}$. 

이와 같이 문제가 주어지고 다음과 같이 Semimartingale이 정의된다고 가정하자 (by $X_t^0 = X_0$ )
$$
X_t^n = X_0 + \int_0^t \sigma(X_s^{n-1}) dZ_s
$$

- SDE 분석을 통헤 $|X^n - X^{n-1}|^2_{\infty, \eta_T}$ 의 **Expectation 상한**을 최대한 Cauchy Sequence에 가깝게 유도해낸다. 예를 들어 다음과 같이..
	- 이렇게 나오는 이유는 Chevyshev 부등식이 Variance에 대한 부등식이기 때문이다.
$$
\mathbb{E}|X^n - X^{n-1}|^2_{\infty, \eta_T} \leq \frac{(C_4 T)^n}{n!}\{\mathbb{E}|X_0|^2 + 1 \}
$$
    - 이 식을 보면 $n > C_4 T$ 일때 부터 수렴한다는 것을 알수 있다. 
- Chebyshev 부등식을 사용하여 

$$
\mathbf{P} \left\{ |X^n - X^{n-1}|_{\infty, \eta_T} \geq \frac{1}{2^n} \right\} \leq \frac{(4 C_4 T)^n}{n!}\{\mathbb{E}|X_0|^2 + 1 \}
$$

- Borel Cantelli lemma를 추가하여 확실하게 증명이 끝난다.

$$
\mathbf{P} \left\{ |X^n - X^{n-1}|_{\infty, \eta_T} \leq \frac{1}{2^n} \text{for almost all }n \right\} = 1
$$

- 위 증명에서 $\sigma$ is globally Lipschitz 이다. 그렇지 않으면 explode 가능성이 있다.
- $\sigma$ is globally Lipschitz .에서 다음의 추가 조건을 달면 explode 하지 않는다., (Proposition 1.1.11)

$$
|\sigma(x)| \leq C(1 + |x|)
$$


## Chebyshev 부등식의 간단한 증명 (by Wikipedia)

$$
\begin{aligned}
\mathbf{P}(|X - \mathbb{E}(X)| \geq k \sigma) &= \mathbb{E}(I_{|X - \mathbb{E}(X)| \geq k \sigma} ) \\
&= \mathbb{E} \left( I_{ \left( \frac{X - \mathbb{E}(X)}{k \sigma} \right) \geq 1} \right) \\
&\leq \mathbb{E} \left( \left( \frac{X - \mathbb{E}(X)}{k \sigma} \right)^2 \right) \\
& = \frac{1}{k^2} \frac{\mathbb{E}(X - \mathbb{E}(X))^2}{\sigma^2} \\
& = \frac{1}{k^2}
\end{aligned}
$$


### Note
- **Markov 부등식**은 **평균값**, **Chevyshev 부등식**은 **Variance**에 대한 부등식 이다.

#### Markov 부등식

$$
\mathbf{P}(X \geq a) \leq \frac{\mathbb{E}(X)}{a}
$$

#### Chevyshev 부등식

$$
\mathbf{P}(|X - \mathbb{E}(X)| \geq a) \leq \frac{ \text{Var} (X)}{a^2} \\
\mathbf{P}(|X - \mathbb{E}(X)| \geq k \sigma) \leq \frac{ 1}{k^2} \;\;\; \because \text{Var} (X)= \sigma^2
$$
