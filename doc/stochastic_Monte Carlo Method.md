Monte Carlo Method
==================

## Rejection Sampling
일반적인 Monte Carlo 방법은 $\mathbb{R}[0,1]$에서 Uniform 하게 변수를 뽑고 원하는 확률집적함수 (CDF)의 역수를 통해 원하는 결과를 얻는 방법이다. 즉,

- If $U \sim U(0,1)$ is a uniform r.v., then $F^{-1} \sim F$ 
- 원주율 구하기를 생각해 보면 자명하다. 
  - $\mathbb{R}^2[0,1]$ 에서 Uniform하게 뽑은 다음 원 안에 들어갔는지 않들어갔는지 Check하는 것이 CDF의 역수를 찾는 과정이다.

그런데, **CDF의 역수를 찾기 어려울 때**, 사용하는 방법이 Rejection Sampling이다.
- 어떤 Proposal Distrinution $q(x)$가 있고 $Mq(x) \geq \tilde{p}(x)$를 만족한다.
- 구하고자 하는 것은 $\tilde{p}(x)$ 인데 잘 모른다. 거기에 Unnormalization 이다.
- 임의의 변수 $u \sim U(0,1)$로 뽑는다.
- 다음을 만족하면  해당 변수를 **Reject** 한다.
$$
u > \frac{\tilde{p}(x)}{Mq(x)}
$$
$$
p(Accept) = \int \frac{\tilde p(x)}{M q(x)} q(x) dx = \frac{1}{M} \int \tilde{p}(x) dx
$$
$$
\frac{\sum^{accept}}{\sum^{total}} \cdot M \approx \int^x \tilde{p}(x)dx  = P(X \leq x)
$$

## Importance Sampling
원래 Monte Carlo 방법에서 무작위로 Sample을 뽑는 것이 아니라, 좀 더 **중요한 영역에서 더 많은 Sample**을 뽑고자 $p(x)$를 변형시킨 것을 말한다.
- 1990년대 후반, Particle Filter에 적용되면서 이론적으로 더욱 발전하였다.
- Particle Filter 적용 사례는 ANN에도 직접 적용이 가능하다.

### Basic Method
-Set the importance density $q(x)$ which is similar to $\pi(x)$,($\pi(x)$가 원래 확률밀도 함수)
$$
\begin{align}
I = \int f(x) \pi(x) dx = \int f(x) \frac{\pi(x)}{q(x)} q(x)dx \\
I_N = \frac{1}{N} \sum_{i=1}^N f(x^i) \tilde{w}(x^i), \;\;\; \tilde{w}(x^i) = \frac{\pi(x^i)}{q(x^i)}
\end{align}
$$

- 그런데, 실제적으로는 $\pi(x)$도 잘 알 수 없다. 그런데, 중요도는 정할 수 있다. 그래서...
$$
I_N = \frac{\frac{1}{N} \sum_{i=1}^N f(x^i) \tilde{w}(x^i)}{\frac{1}{N} \sum_{i=1}^N \tilde{w}(x^i)}
= \frac{1}{N} \sum_{i=1}^N f(x^i) w(x^i)
$$
where normalized importance weights are given by $w(x^i) = \frac{\tilde{w}(x^i)}{\sum_{j=1}^N \tilde{w}(x^j)}$

### Sequential Importance Sampling
Monte Carlo 방법에 의한 Recursive Baysian Filter
Let $X_k = \{x_j, j \in [0, k] \}$ ($x_k$는 Filter state, 즉, weight로 생각하고 $z_k$ 는 데이터, 따라서 $Z_k$는 데이터의 $\sigma$-algebra 이다.) 
- The joint **posterior density** at $k$ can be approximated as follows
$$
p(X_k |{Z}_k) \approx \sum_{i=1}^N w_k^i \delta(X_k - X_k^i)
$$
- 여기에 Importance density $q(X_k|Z_k)$를 생각하면 
$$
w_k^i \propto \frac{p(X_k^i|Z_k)}{q(X_k|Z_k)}   \tag{1}
$$
**즉, 이런 것이 가능하므로 만일, A priori 및 Likelihood를 정하고 여기에 맞춰 Importance Density를 맞추면 Sampling에 의한 Filtering이 가능해진다. **
- 이를 위해 Impotance Density를 이렇게 Factorization 해본다.
$$
q(X_k|Z_k) \triangleq q(x_k|X_{k-1}, Z_k)q(X_{k-1}|Z_{k-1})  \tag{2}
$$
- 그러면 정통적인 방법으로 **Posteriori = (Likwlihood) $\cdot$ (Apriori/Evidence)** 로 분해해 보면
$$
\begin{align}
p(X_k|Z_k) &= p(z_k|X_k, Z_{k-1})\frac{p(X_k|Z_{k-1})}{p(z_k|Z_{k-1})} \\
&= p(z_k|X_k, Z_{k-1})\frac{p(x_k|X_{k-1}, Z_{k-1})p(X_{k-1}|Z_{k-1})}{p(z_k|Z_{k-1})} \\
&= \frac{p(z_k|x_k)p(x_k|x_{k-1})}{p(z_k|Z_{k-1})}p(X_{k-1}|Z_{k-1}) \\
&\propto p(z_k|x_k)p(x_k|x_{k-1})p(X_{k-1}|Z_{k-1})   \tag{3}
\end{align}
$$
식 (2).(3)을 식 (1)에 대입하면
$$
\begin{align}
w_k^i &\propto \frac{p(z_k|x_k)p(x_k|x_{k-1})p(X_{k-1}|Z_{k-1})}{q(x_k|X_{k-1}, Z_k)q(X_{k-1}|Z_{k-1})} \\
&= w_{k-1}^{i}\frac{p(z_k|x_k)p(x_k|x_{k-1})}{q(x_k|X_{k-1}, Z_k)}
\end{align}
$$
- 만일, $q(x_k|X_{k-1}, Z_k) = q(x_k|x_{k-1}, z_k)$ 이면, 전체 State $X_{k-1}$ 을 사용할 필요가 없으므로 실용적으로 유용하다.
- 따라서 다음의 **Importance Weight** 를 사용할수 있다.
$$
w_k^i \propto w_{k-1}^{i} \frac{p(z_k|x_k)p(x_k|x_{k-1})}{q(x_k|x_{k-1}, z_k)}
$$
이렇게 되면 전체 State 데이터를 알 필요 없이 현재의 데이터 만으로 Posteriori Probability를 유추할 수 있다.
$$
p(x_k|Z_k) \approx \sum_{i=1}^N w_k^i \delta(x_k - x_k^i) 
$$
- 당연히 $N \uparrow \infty$ 이면 Sampling 결과는 실제 Posterior Probability에 근접한다.
- 구하고자 하는 것은 $\hat{x}_k = \int x_k p(x_k|Z_k) dx_k$, 결국
$$
\hat{x}_k \approx \frac{1}{M} \sum_{j=1}^M x_{k}^j w_k(x_{k}^j)
$$