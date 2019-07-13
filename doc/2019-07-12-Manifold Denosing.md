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

- Defining the matrix $D$ with the degree function on the diagonal the graph Laplacian in matrix form (see [2])

$$
D = I - D^{-1}W 
$$

### The denoising algorithm


## Reference
[1] Matthias Hein, Markus Maier, "Manifold Denosing", 
[2] 