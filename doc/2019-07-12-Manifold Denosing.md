Manifold Denosing
====

본 문서는 [1] 논문을 간략히 요약한 것이다.

## Introduction

## The noise model and problem statement

데이터는 $m$ dimensional abstract manifold $M$ 위에 존재한다고 가정한다,
이 데이터는  smooth regular embedding $i : M \rightarrow \mathbb{R}^d$ 으로 Feature space $\mathbb{R}^d$ 에 embedding 된다. 즉,

$$
i(M) \subset M
$$

이때 data Generating process $X \in \mathbb{R}^d$ 는 다음과 같이 정의된다.

$$
X = i (\Theta) + \epsilon
$$

where $\theta \sim P_M$ and $\epsilon \sim N(0, \sigma)$. 
Probability measure $P_M$은 $M$ 의에서 해당되는 Volume $dV$의 비율로 정의된다,  그러므로, $P_X (x)$는 $P_M$ 으로 부터 다음과 같이 정의된다.


$$
P_X (x) = (2 \pi \sigma^2)^{-\frac{d}{2}} \int_M e^{-\frac{\| x - i(\theta) \|^2}{2 \sigma^2}} p(\theta) dV(\theta)
\tag{1}
$$

- 이때 Volume $dV$는, Local coordinates 가 $\theta_1, \cdots, \theta_m$ 이면 다음과 같다. 

$$
dV = \sqrt{ \det g} d\theta_1, \cdots d\theta_m
$$

이때, $\det g$는 **metric tensor** $g$의 determinent 이다. 


## Denosing Algorithm

위 정의에 따라 $X$ 는 i.i.d sample of $P_X$ 이다,

### Structure on the sample-based graph

- Sample $X$로 부터 Diffusion process를 정의한다.
- 여기에서 diffusion process의 Generator, 즉, the Graph Laplacian을 유도한다.
- Graph vertices 를 $X_i$로 정의하고, k-nn distance $\{ h(X_i) \}_{i=1}^n$ 일때, **the weight of the k-NN graph** 는 다음과 같이 정의한다.

if $\| X_i - X_j \| < \max \{h(X_i), h(X_j) \}$ 
$$
w(X_i, X_j) = \exp \left( - \frac{\| X_i - X_j \|^2}{(\max \{h(X_i), h(X_j) \})^2 }\right),
$$

Otherwise $w(X_i, X_j) = 0$ 
또한 graph에 Loop가 없어도 0이다. 

- **The degree function $d$**

$$
d(X_i) = \sum_{j=1}^n w(X_i, X_j)
$$

-**The inner production (or Riemmanian metric)** between two hilbert space $\mathcal{H}_V, \mathcal{H}_E$ ($V$ denotes vertices, $E$ denotes Edges)
$$
\langle f, g \rangle_{\mathcal{H}_V} = \sum_{i=1}^n f(X_i) g(X_i) d(X_i), \;\; \langle \phi, \psi \rangle_{\mathcal{H}_E} = \sum_{i,j=1}^n w(X_i, X_j) \phi(X_i, X_j) \psi(X_i, X_j)
$$

- **The discerete differential** 

$$
\nabla : \mathcal{H}_V \rightarrow \mathcal{H}_E, \;\; (\nabla f)(X_i, X_j) = f(X_i) - f(X_j)
$$

- **the Graph Laplacian**

$$
\Delta : \mathcal{H}_V \rightarrow \mathcal{H}_V, \Delta = \nabla * \nabla, \;\; (\Delta f)(X_i) = f(X_i) - \frac{1}{d(X_i)} \sum_{j=1} w(X_i, X_j)f(X_j)
$$

- Defining the matrix $D$ with the degree function on the **diagonal** the graph Laplacian in matrix form (see [2])

$$
D = I - D^{-1}W
$$

위 방정식은 Graph Laplacian 정의를 Matrix 형태로 바꾸었을 뿐이다. 즉, 
$$
\Delta f = D f = (I - D^{-1}W) f = f - D^{-1} W f
$$


### The denoising algorithm
Graph Laplacian이 Graph 상에서의 Diffusion Process의 Generator 이므로다음과 같이, 그래프상에서의 미분 방정식을 정의한다.

$$
\partial_t X = - \gamma \Delta X 
$$

where $\gamma > 0$ is the diffusion constant

- By the Implicit Euler-scheme, the above equation is

$$
X(t+1) - X(t) = - \delta t \gamma \Delta X(t+1)
$$

- The solution of  the implicit Euler scheme for one time step  can be computed as : 

$$
X(t+1) = (\mathbb{1} + \delta t \Delta)^{-1} X(t)
$$

- proof

$$
\begin{aligned}
X(t+1) + \delta t \gamma \Delta X(t+1) = X(t) \\
(\mathbb{1} + \delta t \gamma \Delta) X(t+1) = X(t) \\
X(t+1) = (\mathbb{1} + \delta t \gamma \Delta)^{-1} X(t)
\end{aligned}
$$

- **Manifold Denoising Algorithm**
	- Choose $\delta t, k$
	
	- **while** Stopping Criterion not satisied **do**
		
		- Compute the k-NN distances $h(X_i), \; i=1, \cdots , n$,
		- Compute the weights $w(X_i, X_j)$ of the graph with $w(X_i, X_j) = 0$
		$$
		w(X_i, X_j) = \exp \left( - \frac{\| X_i - X_j \|^2}{(\max \{h(X_i), h(X_j) \})^2 }\right), \;\; \text{if } \| X_i - X_j \| < \max \{h(X_i), h(X_j) \},
		$$
		- Compute the graph Laplacian $\Delta, \; \Delta = \mathbb{1} - D^{-1} W,$
		- Solve $X(t+1) - X(t) = - \delta t \gamma \Delta X(t+1) \Rightarrow X(t+1) = (\mathbb{1} + \delta t \Delta)^{-1} X(t).$
		
	- **end while**

#### Diffusion and Tikonov regularization 

위에서 나온 The solution of  the implicit Euler scheme 은 다음의 Regularization problem on the graph의 solution과 동등하다. 

$$
\underset{Z^{\alpha} \in \mathcal{H}_V}{\arg \min} S(Z^{\alpha}) := \underset{Z^{\alpha} \in \mathcal{H}_V}{\arg \min} \sum_{\alpha=1}^d \| Z^{\alpha} - X^{\alpha}(t) \|_{\mathcal{H}_V}^2 + \delta t \sum_{\alpha=1}^d \| \nabla Z^{\alpha} \|_{\mathcal{H}_V}^2
$$

where $Z^{\alpha}$ denotes the $\alpha$-component of the vector $Z \in \mathbb{R}^d$, and $\| \nabla Z^{\alpha} \|_{\mathcal{H}_V}^2 = \langle Z^{\alpha}, \Delta Z^{\alpha} \rangle$. 

위 방정식을 미분하면 
$$
\frac{\partial S(Z^{\alpha})}{\partial Z^{\alpha}} = 2 (Z^{\alpha} - X^{\alpha}(t) ) + 2 \delta t \Delta Z^{\alpha} = 0 , \;\;\;\alpha = 1, \cdots, d.
$$
따라서
$$
Z = (\mathbb{1} + \delta t \Delta)^{-1} X_t
$$
그러므로 Diffusion Process의 매 스텝은 Regression Problem으로 보이게 되고  새로운 Step  $Z$는 $X(t)$에 대한 Filtering된 결과임.

### K-Nearest neighbor graph versus $h$-neighborhood graph

- K-NN을 사용하는 이유는 K-NN을 통해 얻어진 Graph가 h-Neighborhood 방법으로 얻어진 Graph 보다 좋은 3가지 특징이 있기 때문이다. 
  - Graph가 더 좋은 Connectivity를 가지고 있다.
    - Data의 Density 차이로 인해 h-Neighborhood 에서 끊어지거나 가깝게 붙을 수 있는 graph가 k-NN 그래프에서는 더 좋으 ㄴ결과를 보여준다.
    - 매우 큰 Noise 환경에서 강인하다.
    - 변수 $k$의 조절에 따라 Weight matrix $W$와  Laplacian $\Delta$의 sparsity를 조절하기 쉽다. 

#### Stopping Criterion 

- Diffusion이 너무 오랫동안 이루어져 데이터가 Disconnected 되거나 하나의 클러스터로 몰리는 경우
- 또 하나의 경우는 Intrinsic Dimension에 대한 정보를 선험적으로 가지고 있어 sample의 Estimated dimension 이 Intrinsic Dimension과 같을 때.

알고리즘을 Stop 시킨다.

## Large sample limit and theoritical analysis

### Theorem 1 

Let $\{X_i\}_{i=1}^n$ be an i.i.d. samples of a probablity measure $P_M$ on a m-dimensional compact submanifold $M$ of $\mathbb{R}^d$ has a density $p_M \in C^3(M)$. Let $f \in C^3(M)$ and $x \in M \setminus \partial M$, then if $g \rightarrow 0$ and $nh^{m+2}/\log n  \rightarrow \infty$,

$$
\lim \frac{1}{h^2} (\Delta f)(x) \sim -(\Delta_M f)(x) - \frac{2}{p} \langle \nabla f, \nabla p \rangle_{T_x M}
$$

where $\Delta_M$  is the Laplace-Beltrami operator of $M$ and $\sim$ means upto  a constant which depends on the kernel function $k (\| x - y\| )$ used to define the weights $W(x, y) = k (\| x - y \|)$ of the graph.









## Appendix 
### Lapalce-Beltrami Operator
surface parameter $x(q)$ 를 $x \in \mathbb{R}^n$, $q \in \mathbb{R}^d$  for $ n > d$ 라 정의하고 다음과 같은 metric tensor를 정의하자.

$$
g_{ij} = \frac{\partial x}{\partial q_i} \cdot \frac{\partial x}{\partial q_j}
$$

$G = [g_{ij}]$ 는 symmetruc matrix 로서 curve의 길이, surface내 patch의 면적등을 계산하기 위한 모든 Information을 가지게 된다. 

이때 **The first fundamental form**을 다음과 같이 정의하자.

$$
\mathcal{F}^{(1)} (dq, dq) \doteq dq^T G(q) dq
$$

- 이를 이렇게 생각해보자. $t:\mathbb{R} \rightarrow q(t) \in \mathbb{R}^d \rightarrow x(q) \in \mathbb{R}^n$ 인 상태에서

$$
\frac{dx}{dt} = \left[ \frac{\partial x}{\partial q} \right] \frac{\partial q}{\partial t} \in \mathbb{R}^n
$$

이어야 한다. 그렇다면 $\frac{\partial q}{\partial t} \in  \mathbb{R}^d$ 일때, $\left[ \frac{\partial x}{\partial q} \right] \in \mathbb{R}^{n \times d}$ 이다. 

따라서, $\left[ \frac{\partial x}{\partial q} \right]$는 일반적인  Matrix 이므로 다루기가 쉽지 않다. 여기에 Symmetry 성질을 부여한 $d \times d$ matrix를 생각하면, 다음과 같이 정의할 수 있다.

$$
G(q) = \left[ \frac{\partial x}{\partial q} \right]^T \left[ \frac{\partial x}{\partial q} \right] \in \mathbb{R}^{d \times d}
$$


한편 $q = q(s)$ 라고 할 때 parameterize surface의 Coordinate Change는 Chain-rule에 의해 다음과 같다.

$$
dq = J(s) ds, \;\;\;\; \text{where }J(v) = \left[ \frac{\partial q}{\partial s_i},  \frac{\partial q}{\partial s_j} \right]
$$

**좌표게 변환에 대하여 Fundamental form은 Invariance 이므로** $\mathcal{F}_q^{(1)} = \mathcal{F}_s^{(1)}$ i.e.
$$
ds^T G_s(s) ds = ds^T J^T(s) G_q(q(s)) J(s) ds
$$

다시말해, The metric Tensor Transform under coordinate change as 

$$
G_s(s) = J^T(s) G_q(q(s)) J(s)
$$

그러므로 만일 $G(s)$를 알 수 있다면 Differential form의 주요 정보를 알 수 있다. 예를 들어,  $\tilde{x}(t) = x(q(t)) \text{for } t \in [t_1, t_2]$ 로 정의되는 Curve의 Arc Length를 구하는 경우
$$
L(t_1, t_2) = \int_{t_1}^{t_2} \left( \frac{d \tilde{x}}{dt} \cdot \frac{d \tilde{x}}{dt}\right)^{\frac{1}{2}} dt = \int_{t_1}^{t_2} \left( \left[ \frac{d q}{dt}\right]^T G(q) \frac{d q}{dt} \right)^{\frac{1}{2}} dt
$$

**Notice**
$$
\left( \frac{d \tilde{x}}{dt} \right)^T \left( \frac{d \tilde{x}}{dt} \right) = \left( \left[ \frac{\partial x}{\partial q} \right] \frac{\partial q}{\partial t} \right)^T \left( \left[ \frac{\partial x}{\partial q} \right] \frac{\partial q}{\partial t} \right) = \left( \frac{\partial q}{\partial t} \right)^T \left[ \frac{\partial x}{\partial q} \right]^T \left[ \frac{\partial x}{\partial q} \right] \left( \frac{\partial q}{\partial t} \right) = \left( \frac{\partial q}{\partial t} \right)^T G(q) \left( \frac{\partial q}{\partial t} \right)
$$


또한 element of surface area는 다음과 같다.
$$
dS = |G(q)|^{\frac{1}{2}} dq_1 \wedge \cdots \wedge dq_n
$$
where $|G(q) |^{\frac{1}{2}} \doteq \sqrt{ \det G(q)}$ 

$G = [g_{ij}]$ 로 정의되었는데, Metric Tensor의 Inverse를 다음과 같이 표시한다.
$$
G^{-1} = [G^{ij}]
$$
- For example, 

**Gradient vector field** of a real-valued function on the surface cna be defined as 
$$
(\nabla f)_i \doteq \sum_j g^{ij} \frac{\partial f}{\partial q_j}
$$

**proof**
$f(x(q))$ 에 대하여 생각하면 간단하다. 
먼저 $x$에 대한 Orthogonal Frame을 $e_k^x \triangleq \frac{\partial }{\partial x_k}$ 라 정의하면 $q$에 대한 Frame vector는 
$$
e_i^q \triangleq \frac{\partial }{\partial q_i} = \frac{\partial }{\partial x_k}\frac{\partial x_k}{\partial q_i} = e_k^x \frac{\partial x_k}{\partial q_i} \;\;\; \text{or }
$$

Gradient는 각 Frame에 대하여 다음과 같다.
$$
\nabla_x f(x) = \sum_k \frac{\partial f}{\partial x_k} e_k^x, \;\; \nabla_q f(x(q)) = \sum_i \frac{\partial f(x(q))}{\partial q_i} e_i^q
$$

위 Gradient 식은 좌항의 $e_k^x$ Frame상의 Gradient를 $e_i^q$ Frame에 대하여 표현하는 것이므로 

$$
\begin{aligned}
\nabla_q f(x(q)) 
&= \sum_i \frac{\partial f}{\partial q_i} e_i^q \\
&= \sum_i \sum_k \frac{\partial f}{\partial x_k} \frac{\partial x_k}{\partial q_i} e_i^q = \sum_j \sum_i \sum_k \frac{\partial f}{\partial x_k} \frac{\partial x_k}{\partial q_i} e_j^q \\
&= \sum_k \sum_{i, j} \frac{\partial f}{\partial x_k} \frac{\partial x_k}{\partial q_i} \cdot \frac{\partial x_k}{\partial q_j} e_k^x = \sum_k \sum_{i, j} \frac{\partial x_k}{\partial q_i} \frac{\partial x_k}{\partial q_j} \frac{\partial f}{\partial x_k} e_k^x\\
&= \sum_k \sum_{ij} g_{ij} \frac{\partial f}{\partial x_k} e_k^x
\end{aligned}
$$

따라서 $g_{ij}$ 의 Inverse에 의해 
$$
(\nabla_x f)_i = \frac{\partial f}{\partial x_k} = \sum_j g^{ij} \frac{\partial f}{\partial q_j}
$$
$G$가 $d \times d$ matrix이므로 단순 Matrix 표현식으로는 $n \times 1$ 벡터인 $\nabla_x f$ 를 표현할 수 없으며, 내부 Component의 관계를 통해 얻어 낼 수 밖에 없다.  **Q.E.D**

**Divergence of a vector field** on the surface 따라서 다음과 같이 정의될 수 있다.
$$
\nabla \cdot f \doteq | G |^{-\frac{1}{2}} \sum_i \frac{\partial }{\partial q_i} (|G|^{\frac{1}{2}}f_i)
$$

- **The Laplace (or Laplace Beltrami Operator)** of the smooth real-valued function os defined as **the divergence of the gradient**

$$
\nabla \cdot \nabla f \doteq |G|^{\frac{1}{2}} \sum_i \frac{\partial }{\partial q_i} \left( |G|^{\frac{1}{2}} \sum_j g^{ij} \frac{\partial f}{\partial q_j} \right)
$$







## Reference
[1] Matthias Hein, Markus Maier, "Manifold Denosing", 
[2] Hein, Matthias, Audibert, Jean-Yves von Luxburg, Ulrike, "From Graphs to Manifolds – Weak and Strong Pointwise Consistency of Graph Laplacians"