Stochastic Differential Equation (SDE) on Manifold
========================
먼저, Manifold가 아닌 경우에 대하여는 다음과 같다.

## SDE and Fokker Plank Equation
다음 SDE에 대하여
$$
dx_t = h(x_t) dt + H(x_t) dW_t, \;\;\; x_t, h(\cdot) \in \mathbb{R}^n, \; dW_t \in \mathbb{R}^m \; H(\cdot) \in \mathbb{R^{n \times m}}
$$

State $x_t$ 에 대하여 출력 $y_t = f(x_t) \in \mathbb{R} $ 라고 하면 이에 대한 SDE는 다음과 같다.
$$
dy_t = \left(\sum_j \frac{\partial f_i}{\partial x_j}h_j(x_t)+ \frac{1}{2} \sum_{k,l} \frac{\partial f_i^2}{\partial x_k \partial x_l} [H(x_t)H^T(x_t)]_{kl}\right)dt + \sum_{k,l} \frac{\partial f_i}{\partial x_j} H_{kl}(x_t)dW_{tl}
$$

만일, $x = x_t$ 이고 $y = x_{t -dt}$ 이고 state에 대한 SDE를 따른다고 하고 Transition Probability $p(x|y,dt)$를 생각하면
$$
\begin{align}
\int_{\mathbb{R}^n} (x - y) p(x|y,dt) dy &= \mathbb{E}(x - y) = h(x_t)dt \\
\int_{\mathbb{R}^n} (x - y)(x - y)^T p(x|y,dt) dy &= \mathbb{E}(x - y)(x - y)^T = \sum_k^m H_{ik}(x_t)H_{kj}^T(x_t)dt \\
\end{align}
$$
이에 대하여 Fokker-Plank 방정식은 다음과 같다.

$$
\frac{\partial p(x|y,t)}{\partial t} = - \sum_{i=1}^n \frac{\partial}{\partial x_i}(h_i(x_t)p(x|y,t)) + \frac{1}{2} \sum_{i,j=1}^n \frac{\partial^2}{\partial x_i \partial x_j} \left( \sum_{k=1}^m H_{ik}(x_t) H_{kj}^T(x_t) p(x|y,t)\right)
$$

- 즉, 순수 SDE 해석은 Stochastic Gradient Rule을 Expectation 만을 Update 한다고 가정할때 사용할 수 있으며
- Fokker-Plank Equation 해석은 $p(x|y,t)$를 사용하여 Monte Carlo 방식으로 Weight를 Update할 때 사용할 수 있다.

## SDE and Fokker Plank Equation on Manifold
만일 Manifold $M$의 Metric tensor가 $G = [g_{ij}]$ 로 주어질 경우 (Inverse는 $G^{-1} = [g^{ij}]$ )

$$
\frac{\partial p(x|y,t)}{\partial t} = - |G|^{-\frac{1}{2}}\sum_{i=1}^n \frac{\partial}{\partial x_i}(|G|^{\frac{1}{2}}h_i(x_t)p(x|y,t)) + \frac{1}{2} |G|^{-\frac{1}{2}}\sum_{i,j=1}^n \frac{\partial^2}{\partial x_i \partial x_j} |G|^{\frac{1}{2}}\left( \sum_{k=1}^m H_{ik}(x_t) H_{kj}^T(x_t) p(x|y,t)\right)
$$

그러나 일반적인 SDE의 경우 Manifold 위에서의 일반적인 Martingale항을 정의하는 것은 대단히 어렵다. 그 이유는 $dW_t$ 가 일반적인  Manifold 위에 어떻게 정의 되는가 자체가 어렵기 때문이다. 일단, Manifold 위의 Tangent Space상의 Wiener Process중, Orthogonal Component는 없다고 가정하고, Horizontal 성분만 있다고 가정해도 Wiener Process에 대한 정의를 내리기 위해서 Geodesic위에서 먼저 Horizontal Vector Field를 정의하여야 Local coordinates 자체를 정의할 수 있다. 즉, 

만일 Parameterized curve $t \rightarrow u_t e_m = e_m^k (t) X_k$ 로 주어졌다고 가정하면 (여기서 $X_k = \frac{\partial }{\partial x^k}$, $X_{km} = \frac{\partial }{\partial e_m^k}$ )
Parallel Transportaion 조건 ($e_m$ 은 $\mathbb{R}^d$ 상의 Coordinate Index)
$$
\dot{e}_m^k(t) + \Gamma_{jl}^k(x_t)\dot{x}_t^j e_m^l = 0
$$
에서 
$$
\dot{e}_m^k(t) = -e_i^j e_m^l \Gamma_{jl}^k(x_t), \;\;\;\dot{x}_0^j = e_i^j
$$
일 경우 다음과 같은 Brownian Motition 이 주어질떄 ($\circ$ 는 Starnotovici SDE 기호)
$$
\begin{align}
dX_t^i &= e_j^i(t) \circ dW_t^j \\
de_j^i(t) &= - \Gamma_{kl}^i(X_t) e_j^l(t) e_k^m(t) \circ dW_t^m
\end{align}
$$

Euclidean Brownian Motion $dM_t = \sigma(X_t)dB_t$ 로 주어졌을 때, Manifold위의 Horizontal Brownian Motion $X_t$ 는 다음과 같이 유도된다.
$$
dX_t^i = \sigma_j^i(X_t) dB_t^i - \frac{1}{2} g^{lk}(X_t) \Gamma_{kl}^i(X_t)dt, \;\;\; g^{ij} = \sum_{k=1}^d e_k^i e_k^j
$$
이를 사용하여 Operator를 만들던가 혹은 Girsanov Theorem등을 사용하여 Wiener Process와 Martingale을 분리하여 SDE를 성립시켜야 한다.

Local Coordinater에서 Generator는 다음과 같다. 
$$
\Delta_M f = \frac{1}{\sqrt{G}} \frac{\partial}{\partial x^j} \left(\sqrt{G} g^{ij} \frac{\partial f}{\partial x^i} \right) = g^{ij} \frac{\partial}{\partial x^i} \frac{\partial f}{\partial x^j} + b^i \frac{\partial f}{\partial x^i}
$$
where
$$
b^i = \frac{1}{\sqrt{G}} \frac{\partial (\sqrt{G} g^{ij})}{\partial x^j} \;\; \Rightarrow b^i = g^{jk} \Gamma_{jk}^i
$$

#### Note
즉,
$$
b^i = \frac{1}{\sqrt{G}} \frac{\partial (\sqrt{G} g^{ij})}{\partial x^j} = \frac{\partial g^{ij}}{\partial x^j} = \frac{\partial }{\partial x^j} \sum_{k=1}^d e_k^i e_k^j = \nabla_{X_j} \sum_{k=1}^d e_k^i e_k^j
$$

그러므로 Let $X_k = e_k^i$ or $X_k = e_k^j$ 이면
$$
\nabla_{X_j} \sum_{k=1}^d e_k^i X_k \cdot e_k^j = \sum_{k=1}^d e_k^i e_k^j \nabla_{X_j} X_k = g^{ij} \Gamma_{jk}^i
$$
