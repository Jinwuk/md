Kullback-Leibler Divergence
================
KL Divergence를 최소화 하는 문제는 결국 Maximum Likelihood를 최대화 하는 문제로 귀결됨을 보인다. (뭐 당연하지만..)
Assume that two probability distribution $p$ and $q$.

##Definition KL Divergence
### Discrete Version
$$
\begin{align}
\mathbb{KL}(p||q) &\triangleq \sum_{k=1}^K p_k \log \frac{p_k}{q_k} = \sum_{k=1}^K p_k \log p_k - \sum_{k=1}^K p_k \log q_k \\
&= -\mathbb{H(p)} + \mathbb{H(p,q)}
\end{align}
$$
where $\mathbb{H(p)}$ is the entropy of $p$ and $\mathbb{H(p,q)}$ is called the **cross entropy** such that
$$
\mathbb{H}(X) \triangleq -\sum_{k=1}^Kp(X=k) \log_2 p(X=k), \;\; \mathbb{H}(p,q) \triangleq -\sum_k p_k \log q_k
$$

### Continuous Version
$$
\mathbb{KL}(p||q) \triangleq \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx
$$
If $p(x), q(X)$ are probablity measure over a set $X$ and $p(x)$ is absolutely continuous with repect to $q(x)$ then
$$
\mathbb{KL}(p||q) = \int_X \log \frac{dP}{dQ}dP  \tag{1}
$$

### Example : Same variance
Let $p(x)$ be a Gaussian p.d.f with the mean $m_p$, and $q(x)$ be a Gaussian p.d.f with the mean $m_q$. In addition, the variance of two p.d.f.is same to $\sigma$, such that
$$
p(x)= \frac{1}{Z} \exp \left( \frac{-(x - m_p)^2}{2 \sigma^2} \right), \;\; q(x)= \frac{1}{Z} \exp \left( \frac{-(x - m_q)^2}{2 \sigma^2} \right)
$$
$\ln(x)$를 써야 원래 아래 방정식이 성립하겠지만... 뭐 그렇다고 치고 (어차피 $\log2$ 에 동시 비례할 거니까..)

$$
\log p(x) =\left( \frac{-(x - m_p)^2}{2 \sigma^2} \right) - \log Z, \;\; \log q(x) =\left( \frac{-(x - m_q)^2}{2 \sigma^2} \right) - \log Z,
$$
$$
\begin{align}
\mathbb{KL}(p(x)||q(x)) &= \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx = \int_{-\infty}^{\infty} \left( p(x) \log p(x) - p(x) \log q(x) \right)dx \\
&= \int_{-\infty}^{\infty} \left( p(x) \left( \frac{-(x - m_p)^2}{2 \sigma^2} \right) - p(x)\log Z - p(x) \left( \frac{-(x - m_q)^2}{2 \sigma^2} \right) + p(x) \log Z \right)dx \\
&= \frac{1}{2 \sigma^2} \int_{-\infty}^{\infty} p(x) \left((x - m_q)^2 - (x - m_p)^2 \right)dx \\
&= \frac{1}{2 \sigma^2} \int_{-\infty}^{\infty} p(x) \left( 2 (m_p - m_q) x + (m_q^2 - m_p^2) \right)dx \\
&= \frac{1}{2 \sigma^2} \left( 2 (m_p - m_q) \int_{-\infty}^{\infty} x p(x) dx + (m_q^2 - m_p^2) \right) \\
&= \frac{1}{2 \sigma^2} \left( 2 m_p^2 - 2 m_p m_q + m_q^2 - m_p^2 \right) = \frac{1}{2 \sigma^2} (m_p - m_q)^2
\end{align}
$$
즉, 같은 Variance를 가지고 평균이 다른 두 Gaussian p.d.f 의 **KL Divergence**는 **평균 차이의 제곱에 비례하는 어떤 값**이다.
- 그래서 마치 거리 처럼 보인다. (평균 사이의 거리)

### Example : different variance
위 문제와 같이 두 개의 Gaussian p.d.f 이다. 그런데, Variance가 다르다. 이를 반영하여 다음과 같이 p.d.f를 쓴다.
$$
p(x)= \frac{1}{Z_1} \exp \left( \frac{-(x - m_p)^2}{2 \sigma_1^2} \right), \;\; q(x)= \frac{1}{Z_2} \exp \left( \frac{-(x - m_q)^2}{2 \sigma_2^2} \right)

$$
중간단계를 생략하며 전개하면
$$
\begin{align}
\mathbb{KL}(p(x)||q(x)) &= \frac{1}{2}\int_{-\infty}^{\infty} p(x) \left( \left( \frac{(x - m_q)^2}{\sigma_2^2} \right) - \left( \frac{(x - m_p)^2}{\sigma_1^2} \right) + 2(\log Z_2 - \log Z_1) \right)dx \\
&= \frac{1}{2 \sigma_1^2 \sigma_2^2 }\int_{-\infty}^{\infty} p(x) \left( \sigma_1^2 (x - m_q)^2  - \sigma_2^2(x - m_p)^2 \right)dx + \log \frac{Z_2}{Z_1}  \\
&= \frac{1}{2 \sigma_1^2 \sigma_2^2 }\int_{-\infty}^{\infty} p(x) \left( (\sigma_1^2 - \sigma_2^2) x^2 - 2 x (\sigma_1^2 m_q - \sigma_2^2 m_p) + \sigma_1^2 m_q^2  - \sigma_2^2 m_p^2 \right)dx + \log \frac{Z_2}{Z_1}  \\
&= \frac{1}{2 \sigma_1^2 \sigma_2^2 } \left[ (\sigma_1^2 - \sigma_2^2) \int_{-\infty}^{\infty} x^2 p(x) dx  - 2 (\sigma_1^2 m_q - \sigma_2^2 m_p) \int_{-\infty}^{\infty} x p(x) dx  + \left( \sigma_1^2 m_q^2  - \sigma_2^2 m_p^2 \right) \right] + \log \frac{Z_2}{Z_1}  \\
&= \frac{1}{2 \sigma_1^2 \sigma_2^2 } \left[ (\sigma_1^2 - \sigma_2^2) (\sigma_1^2 + m_p^2)  - 2 (\sigma_1^2 m_q - \sigma_2^2 m_p) m_p  + \left( \sigma_1^2 m_q^2  - \sigma_2^2 m_p^2 \right) \right] + \log \frac{Z_2}{Z_1}  \\
&= \frac{1}{2 \sigma_1^2 \sigma_2^2 } \left[ (\sigma_1^2 - \sigma_2^2) (\sigma_1^2 + m_p^2)  - 2 (\sigma_1^2 m_q - \sigma_2^2 m_p) m_p  + \left( \sigma_1^2 m_q^2  - \sigma_2^2 m_p^2 \right) \right] + \log \frac{Z_2}{Z_1}  \\
&= \frac{1}{2 \sigma_2^2 } \left[ (\sigma_1^2 - \sigma_2^2)  + (m_p - m_q)^2 \right] + \log \frac{\sigma_1}{\sigma_2}  
\end{align}
$$
**KL Divergence**는 **평균 차이의 제곱 더하기 분산의  차이에 비례하는 어떤 값**이다.


## Correspondence to Radon-Nykodym Derivation
Remind the equation (1), where $\frac{dP}{dQ}$ is the **Radon=Nikodym derivative** of $P$ with $Q$, and the equation (1) can be rewritten as
$$
\mathbb{KL} = \int_X \log \left( \frac{dP}{dQ}  \right) \frac{dP}{dQ} dQ
$$
which we recognize the entropy of $P$ relative to $Q$. 

Assume that there exist the value of Radon-Nykodym Derivation such that $\mu = \frac{dP}{dQ} $ then
$$
\mathbb{KL} = \int_X \log \mu \cdot \mu dQ = \int_X \mu \log \mu dQ = -\mathbb{H}(\mu)
$$

### Correspondence to Girsanov Theorem
Assume that the following SDE
$$
dW_t = dB_t + H_t dt
$$
where $B_t$ is the brownian motion measured by $Q$ and $W_t$ is the brownian motion measured by $P$. Thus,
the Radon-Nykodym Derivative is evaluated by the Girsanov Theorem
$$
\frac{dP}{dQ} = \exp \left( -\int_0^t H_s dB_s - \frac{1}{2} \int_0^t H_s^2 ds \right) = \Lambda
$$
Then
$$
\mathbb{KL} = \int_X \left( -\int_0^t H_s dB_s - \frac{1}{2} \int_0^t H_s^2 ds \right) \log \Lambda dQ = \mathbb{E}_{X, Q} \left( -\int_0^t H_s dB_s - \frac{1}{2} \int_0^t H_s^2 ds \right)^2 > 0
$$
If we want to **minimize the KL Divergence**, it **maximize the Radon-Nykodym derivative**.
Therefore, it is a same problem of **maximum likelihood** problem.

#### Example : Estimation in Ornstein-Uhlenbeck Model
$$
dX_t = -\alpha X_t dt + \sigma dB_t \;\;\; t \in \mathbb{R}[0, T]
$$
$X_t$ 에 대한 Probality는 P, $σB_t$의 Probablity는 $Q$라고 할 때, 방정식 (1)에 따라 Likelihood 함수는 다음과 같다.
$$
\Lambda(X)_T = \frac{dP}{dQ} = \exp \left( \int_0^T \frac{-\alpha X_t}{\sigma^2} dX_t - \frac{1}{2} \int_0^T \frac{\alpha^2 X_t^2}{\sigma^2} dt \right)
$$

이때 $\exp$ 내부를 미분하여 최대값이 나오는 $\alpha$를 구하면 
$$
\hat{\alpha} = \frac{\int_0^T X_t dX_t}{\int_0^T X_t^2 dt}
$$

## Chracteristics of KL Divergence
KL Divergence는 결국 다음과 같이 정의할 수 있다.
- 확률 분포 $p$의 Entropy - 정보량에 두 확률 분포의 결합 Entropy 를 뺀 값 (**정보량의 차이**)
- 확률 분포를 특징짓는 **Moment Parameter간의 차이에 대한 단순 합** 
###Symmetrised divergence
Let a symmetric and nonnegative divergence such that
$$
\mathbb{KL}(P||Q) + \mathbb{KL}(Q||P)
$$
An Alternative divergence which has convex property with $\lambda$
$$
\mathbb{KL} (P||Q) = \lambda \mathbb{KL}(P|| \lambda P + (1-\lambda)Q) + (1-\lambda) \mathbb{KL}(Q||\lambda P + (1-\lambda)Q)
\tag{2}
$$
#### the Jensen–Shannon divergence, 
The value $λ = 0.5 $ and $M = \frac{1}{2}(P+Q)$then,
$$
\mathbb{KL}_{JS} = \frac{1}{2}\mathbb{KL}(P||M) + \frac{1}{2}\mathbb{KL}(Q||M)
$$



