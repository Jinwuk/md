Kullback-Leibler Divergence
================
KL Divergence를 최소화 하는 문제는 결국 Maximum Likelihood를 최대화 하는 문제로 귀결됨을 보인다. (뭐 당연하지만..)
Assume that two probability distribution $p$ and $q$.

##Definition KL Divergence
### Discrete Version
$$
\begin{align}
\mathbb{KL}(p||q) &\triangleq \sum_{k=1}^K p_k \log \frac{p_k}{q_k} \\
&= \sum_{k=1}^K p_k \log p_k - \sum_{k=1}^K p_k \log q_k \\
&= -\mathbb{H(p)} + \mathbb{H(p,q)}
\end{align}
$$
where $\mathbb{H(p)}$ is the entropy of $p$ such that
$$
\mathbb{H}(X) \triangleq -\sum_{k=1}^Kp(X=k) \log_2 p(X=k)
$$
and $\mathbb{H(p,q)}$ is called the **cross entropy** such that
$$
\mathbb{H}(p,q) \triangleq -\sum_k p_k \log q_k
$$

### Continuous Version
$$
\mathbb{KL}(p||q) \triangleq \int_{\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx
$$
If $p(x), q(X)$ are probablity measure over a set $X$ and $p(x)$ is absolutely continuous with repect to $q(x)$ then
$$
\mathbb{KL}(p||q) = \int_X \log \frac{dP}{dQ}dP  \tag{1}
$$

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
\mathbb{KL} = \int_X \left( -\int_0^t H_s dB_s - \frac{1}{2} \int_0^t H_s^2 ds \right) \log \Lambda dQ
$$
If we want to **minimize the KL Divergence**, it **maximize the Radon-Nykodym derivative**.
Therefore, it is a same problem of **maximum likelihood** problem.

#### Example : Estimation in Ornstein-Uhlenbeck Model
$$
dX_t = -\alpha X_t dt + \sigma dB_t \;\;\; t \in \mathbb{R}[0, T]
$$
$X_t$ 에 대한 Probality는 P, $σB_t$의 Probablity는 $Q$라고 할 때
이렇게 되면 방정식 (1)에 따라 Likelihood 함수는 다음과 같다.
$$
\Lambda(X)_T = \frac{dP}{dQ} = \exp \left( \int_0^T \frac{-\alpha X_t}{\sigma^2} dX_t - \frac{1}{2} \int_0^T \frac{\alpha^2 X_t^2}{\sigma^2} dt \right)
$$

이때 $\exp$ 내부를 미분하여 최대값이 나오는 $\alpha$를 구하면 
$$
\hat{\alpha} = \frac{\int_0^T X_t dX_t}{\int_0^T X_t^2 dt}
$$

## Chracteristics of KL Divergence
###Symmetrised divergence
Let a symmetric and nonnegative divergence such that
$$
\mathbb{KL}(P||Q) + \mathbb{KL}(Q||P)
$$
An Alternative divergence which has convex property with $\lambda$
$$
\mathbb{KL} (P||Q) = \lambda \mathbb{KL(P|| \lambdaP + (1-\lambda)Q) + (1-\lambda) \mathbb{KL}(Q||\lambda P + (1-\lambda)Q)
\tag{2}
$$
#### the Jensen–Shannon divergence, 
The value $λ = 0.5 $ then,
$$
\mathbb{KL}_{JS} = \frac{1}{2}\mathbb{KL}(P||M) + \frac{1}{2}\mathbb{KL}(Q||M)
$$
where $M = \frac{1}{2}(P+Q)
$$


