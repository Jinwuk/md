Diffusions: Basic Properties
==========================
[TOC]

일반적인 형태인 다음의 SDE 
$$
dX_t = b(t, X_t)dt + \sigma(t, X_t) dB_t, \;\; X_t \in \mathbb{R}^n, \; b(t,x) \in \mathbb{R}^n, \; \sigma(t,x) \in \mathbb{R}^{n \times m}
$$
여기서 $ b(t, X_t)$ 를 Drift coefficient, $\sigma(t,x)$ 혹은 $\frac{1}{2}\sigma \sigma^T$ 를 Diffusion coefficient 라고 한다. 

## Time-Homogeneous Ito Diffusion
$X_t (\omega) = X(t, \omega):[0, \infty] \times \Omega \rightarrow \mathbb{R}^n$ 가 다음의 SDE를 만족하면
$$
dX_t = b(X_t)dt + \sigma (X_t)dB_t, \;\; t \geq s ; \; X_s = x
$$
**Time-Homogeneous Ito Diffusion** 이라고 한다.
이때, SDE의 각 계수는 다음의 관계를 만족한다.
$$
|b(x) - b(y)| + |\sigma(x) - \sigma(y)| \leq D|x-y|; \;\; x,y \in \mathbb{R}^n
$$
where $|\sigma|^2 = \sum |\sigma_{ij}|^2$ .

## Markov property for Ito diffusions
Let $f$ be a bounded Borel function from $\mathbb{R}^n$ to $\mathbb{R}$. Then, for $t, h \geq 0$.
$$
E^x [f(X_{t+h}) | \mathcal{F}^m_t]_{\omega} = E^{X_t(\omega)}[f(X_h)]
$$
여기서 $E^x$ 는 Probablity Measure $Q^x$ 에 대한 평균값을 의미한다. 그러므로 $E^y[f(X_h)]$ 는 $E[f(X^y_h)]$ 를 의미한다. 

증명은 생략한다.

## Stopping Time
Let $\tau$ be a **stopping time** w.r.t. $\mathcal{N}_t$ ans let $\mathcal{N}_{\infty}$ be the smallest $\sigma$-algebra containing $N \in \mathcal{N}_{\infty}$ such that
$$
N \cap  \{ \tau \leq t\} \in \mathcal{N}_t, \;\; \forall t \geq 0.
$$
즉, Stopping time에서 측정되는 모든 사건은 시간이 무한대 일때 나타나는 사건 집합을 모두 기술 가능하다. 즉,Stopping Time 이후의 사건들을 무의미하다는 의미이다. 거기까지 Diffusion이 유의미 하다고 생각해도 된다. 그 이상 시간에서 발생하는 사건은 Stopping Time 까지의 사건들로 모두 기술된다. 즉, Deterministic하게 결정된다.

## The Generator of an Ito Diffusion
### Definition 
Let $\{X_t\}$ be a (time homogeneous) Ito diffusion in $\mathbb{R}^n$ . The infinitesimal generator $A$ of $X_t$ is defined by
$$
Af(x) = \lim_{t \downarrow 0} \frac{E^x[f(X_t)] - f(x)}{t} \;\; x \in \mathbb{R}^n
$$
여기서 $f : \mathbb{R}^n \rightarrow \mathbb{R}$ 은 $\mathcal{D}_A(x)$ 로 표시되는 $x$ 에서의 극한값이 존재한다.
여기서 $\mathcal{D}_A$ 는 $\forall x \in \mathbb{R}^n$ 의 극한값에 대한 함수의 집합이다.

## Lemma
Let $Y_t = Y^x_t$ be an Ito process in $\mathbb{R}^n$ of the form
$$
Y^x_t(\omega) = x + \int^t_0 u(s, \omega) ds + \int^t_0 v(s,\omega) dB_s(\omega)
$$
where $B \in \mathbb{R}^m$. 
Let $f \in C^2_0 (\mathbb{R}^n), \;i.e.\; f \in C^2(\mathbb{R}^n)$ and $f$ has compact support, and let $\tau$ be a stopping time w.r.t. $\{\mathcal{F}^{(m)}_t \}$, and assume that $E^x[\tau ] < \infty$. 
Assume that $u(t,\omega)$ and $ v(t, \omega)$ are bounded on the set of $(t, \omega)$ such that $Y(t, \omega)$ belongs to the support of $f$. Then 

$$
E^x [f(Y_{\tau})] = f(x)+E^x \left[ \int^{\tau}_0 \left( \sum_i u_i(s,w) \frac{\partial f}{\partial x_i} (Y_s) + \frac{1}{2} \sum_{i,j} (vv^T)_{i,j} (s,w) \frac{\partial^2 f}{\partial x_i \partial x_j}(Y_s) \right) ds \right]
$$

where $E^x$ is ths expectation w.r.t. the natural probablity law $R^x$ for $Y_t$ starting at $x$:
$$
R^x[Y_{t_1} \in F_1, \cdots, y_{t_k} \in F_k] = P^0 [Y^x_{t_1} \in F_1, \cdots , Y^x_{t_1} \in F_k], \;\; F_i \text{is Borel Sets}
$$

### proof
$Z=f(Y)$ 라 하고 Ito Formula를 적용하면
$$
\begin{align*}
dZ &= \sum_i \frac{\partial f}{\partial x_i}(Y)dY_i + \frac{1}{2}\sum_{i,j}\frac{\partial^2 f}{\partial x_i \partial x_j}(Y)dY_i dY_j \\
&= \sum_i u_i \frac{\partial f}{\partial x_i}dt + \frac{1}{2}\sum_{i,j}\frac{\partial^2 f}{\partial x_i \partial x_j}(vdB)_i(vdB)_j + \sum_i \frac{\partial f}{\partial x_i}(vdB)_i
\end{align*}
$$
Since
$$
\begin{align*}
(vdB)_i(vdB)_j = (\sum_k v_{ik} dB_k)(\sum_n v_{jn} dB_n) = (\sum_k v_{ik}v_{jk}) dt = (vv^T)_{ij}dt
\end{align*}
$$
, this gives
$$
f(Y_t)=f(Y_0)+ \int^t_0 \left(\sum_i u_i \frac{\partial f}{\partial x_i} + \frac{1}{2} \sum_{i,j} (vv^T)_{ij} \frac{\partial^2 f}{\partial x_i \partial x_j}\right) ds + \sum_{j,k} \int^t_0 v_{ik} \frac{\partial f}{\partial x_i}dB_k . 
$$
Hence
$$
E^x[f(Y_t)]=f(Y_0)+ E^x \left[\int^t_0 \left(\sum_i u_i \frac{\partial f}{\partial x_i}(Y) + \frac{1}{2} \sum_{i,j} (vv^T)_{ij} \frac{\partial^2 f}{\partial x_i \partial x_j}\right)(Y) ds \right] + \sum_{j,k} E^x \left[\int^t_0 v_{ik} \frac{\partial f}{\partial x_i} (Y) dB_k \right]. 
$$

따라서, 
$$
E^x \left[\int^t_0 v_{ik} \frac{\partial f}{\partial x_i} (Y) dB_k \right] = 0
$$ 
임을 보이면 증명 끝이다. 일단, 이 증명 자체는 자명해 보인다. 구체적인 중명은 다음을 참조한다.[^1] 증명의 방향은 다음과 같다. Stopping time $\tau$ 에 대하여 
1. $\forall k \in \mathbb{Z}$ 에 대하여 Bounded Borel function $|g| \leq M$ 이면 Stopping Time에서 가측이므로 
2. 1차 Moment 는 0, 2차 Moment Bounded 가 된다. 고로 위 평균 값은 0 이다.
이를 Uniform Integrability and Martingale Convergence 라고 한다.
즉, $f \in C^2_0 (\mathbb{R}^n), \;i.e.\; f \in C^2(\mathbb{R}^n)$ 에서 Uniform Integrability 가 성립하고, Ito Integral이 Martingale 이므로 0 이 된다. (Martingale이면 경계면에서 적분이 0 이면 다른 곳에서도 모두 0 이 될 수 밖에 없다.)

## Dynkin Formula
Let $f \in C^{2}_0 (\mathbb{R}^2)$. Suppose $\tau$ is a stopping time, $E^x [\tau] < \infty $, then
$$
E^x [f(X_{\tau})] = f(x) + E^x  [\int^{\tau}_0 A f(X_s) ds]
$$

다시 말해서 위 Lemma 가 Dynkin Formula 이다. Noise 항이 0이 되기 떄문이다.

## Chrateristic Operator
지금까지는 분모가 절대 시간으로 결정된 Infinitesimal Operator에 대하여 논하였다. 그런데, 만일 Stopping Time으로 결정되는 Operator를 생각하면 어떻게 될까? 이것을 Charsteristoc Operator라고 한다.

### Definition 1
Let $\{X_t \}$ be a Ito diffusion. The charateristic operator $\mathcal{A} = \mathcal{A}_X$ of $\{X_t \}$ is defined by
$$
\mathcal{A}f = \lim_{U \downarrow x} \frac{E^x [f(X_{\tau_{U}})] - f(x)}{E^x [\tau_U]}
$$
where the $U$'s are open sets $U_k$ decreasing to the point $x$, in the sense that $ U_{k+1} \subset U_k$ and $ \bigcap_k U_k = \{x\}$, and $\tau_U = \inf \{ t> 0 ; X_t \notin  U\}$ is the first exit time $U$ for $X_t$. 

### Definition 2
The set of functions $f$ such that the limit derived by the infinitesimal operator $\mathcal{A}$ exists for all $x \in \mathbb{R}^n$ (and all $\{ U_k\})$ is denoted by $\mathcal{D}_{\mathcal{A}}$.  If $E^x[\tau_U] = \infty$ for all open $ U \ni x$, we define $\mathcal{A} f (x) = 0$.

It turns out that $\mathcal{D}_{A} \subseteq \mathcal{D}_{\mathcal{A}}$ always and that 
$$
Af = \mathcal{A} f \;\;\;\; \forall f \in \mathcal{D}_{A}
$$

## Theorem 
Let $f \in C^2$. Then $f \in \mathcal{D}_{\mathcal{A}}$ abd
$$
\mathcal{A} f = \sum_i b_i \frac{\partial f}{\partial x_i} + \frac{1}{2} \sum_{i,j} (\sigma \sigma^T)_{ij} \frac{\partial^2 f}{\partial x_i \partial x_j}.
$$

### Remark
1. Ito Diffusion 은 Continuous, Strong Markov Process 로서 그것의 Chracteristic operator 정의의 Domain은 $C^2$를 포함한다.
2. 그러므로 Ito Diffusion은 in the sense of Dynkin Diffusion이다.




[^1]: B. Oksendal, 'Stochastic Differential Equations : An Introduction with Applications', Springer, pp.117.