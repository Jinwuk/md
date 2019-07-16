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













## Reference
[1] Matthias Hein, Markus Maier, "Manifold Denosing", 
[2] Hein, Matthias, Audibert, Jean-Yves von Luxburg, Ulrike, "From Graphs to Manifolds – Weak and Strong Pointwise Consistency of Graph Laplacians"