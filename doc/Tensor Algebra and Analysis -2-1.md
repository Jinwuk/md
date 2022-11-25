Tensor Algebra and Analysis -2-1
===
[toc]

### Christoffel Symbol $\Gamma_{ijk}, \; \Gamma_{ij}^k$ (Contiued)
#### Christoffel Symbol 을 사용한 covariant derivation 정의
For the vector valued function $\mathbf{x} = \mathbf{x}(\theta^1, \theta^2, \cdots, \theta^n)$, 이를 $\theta^j$ 에 대하여 covariant derivate 하면 ($\mathbf{x} = x^i \mathbf{g}_i$)
$$
\begin{equation}
\mathbf{x}_{, j}
=(x^i \mathbf{g}_i)_{, j}
= x_{, j}^i \mathbf{g}_i + x^i \mathbf{g}_{i, j}
\label{eq-2.58-1}
\end{equation}
$$

$\Gamma_{ij}^k = \mathbf{g}_{i, j} \cdot \mathbf{g}^k$ 에 의해,
$$
\begin{equation}
\mathbf{g}_k \Gamma_{ij}^k
= \mathbf{g}_{i, j} \cdot \mathbf{g}^k \cdot \mathbf{g}^k
= \mathbf{g}_{i, j}
\label{eq-2.58-2}
\end{equation}
$$

$\eqref{eq-2.58-2}$ 를 $\eqref{eq-2.58-1}$ 에 대입하면,
$$
\begin{equation}
\begin{aligned}
x_{, j}^i \mathbf{g}_i + x^i \mathbf{g}_{i, j}
&= x_{, j}^i \mathbf{g}_i + x^i \mathbf{g}_k \Gamma_{ij}^k \\
&= x_{, j}^i \mathbf{g}_i + x^k \Gamma_{kj}^i \mathbf{g}_i \\
&= \left( x_{, j}^i + x^k \Gamma_{kj}^i \right)\mathbf{g}_i,
\end{aligned}
\quad \therefore \mathbf{x}_{, j} = \left( x_{, j}^i + x^k \Gamma_{kj}^i \right)\mathbf{g}_i
\label{eq-2.83}
\end{equation}
$$

Covariant tangent vector $\mathbf{g}^i$ 에 대하여 생각해 보면, $\Gamma_{kj}^i = -\mathbf{g}_{, j}^i \cdot \mathbf{g}_k$
$$
\begin{equation}
\begin{aligned}
\mathbf{x}_{, j}
&=(x_i \mathbf{g}^i)_{, j}
= x_{i, j} \mathbf{g}^i + x_i \mathbf{g}_{, j}^i \\
&= x_{i, j} \mathbf{g}^i - x_i \Gamma_{kj}^i \mathbf{g}^k \\
&= x_{i, j} \mathbf{g}^i - x_k \Gamma_{ij}^k \mathbf{g}^i \\
&= \left( x_{i, j} - x_k \Gamma_{ij}^k \right) \mathbf{g}^i
\label{eq-2.84}
\end{aligned}
\end{equation}
$$

이것을 $\eqref{eq-2.63}$ 과 비교해 보면
$$
\begin{equation}
x^i \vert_j = \left( x_{, j}^i + x^k \Gamma_{kj}^i \right), \quad
x_i \vert_j = \left( x_{i, j} - x_k \Gamma_{ij}^k \right), \quad i, j=1, 2, \cdots n
\label{eq-2.85}
\end{equation}
$$

- **Notice!!** 여기서 중요한 사실은 Christoffel 기호와 연동되는 Tangent vector가 covariant냐, contra variant냐에 따라, 부호가 바뀌어 진다.
$$
\begin{equation}
\Gamma_{kj}^i \mathbf{g}_i = -\Gamma_{ij}^k \mathbf{g}^i
\label{note-2.84}
\end{equation}
$$

- 번외로 $\Gamma_{ij}^k = \Gamma_{kj}^i$ 임을 증명해 보면, $\mathbf{g}_k = \frac{\partial \mathbf{r}}{\partial \theta^k}, \; \mathbf{g}^k = g_i^k \mathbf{g}^i$ 에서
$$
\begin{equation}
\Gamma_{ij}^k
= \mathbf{g}_{i, j} \cdot \mathbf{g}^k
= \mathbf{g}_{i, j} \cdot g_i^{k}\mathbf{g}^i
= g_i^{k} \mathbf{g}_{i, j} \cdot \mathbf{g}^i
= \mathbf{g}_{k, j} \cdot \mathbf{g}^i
= \Gamma_{kj}^i
\end{equation}
$$

#### Christoffel Symbols for Tensor Valued Function
Let a tensor valued function $\mathbf{A} = \mathbf{A}(\theta^1, \theta^2, \cdots \theta^n)$:

$$
\begin{equation}
\begin{aligned}
\mathbf{A}_{, k}
&= \left( A^{ij} \mathbf{g}_i \otimes \mathbf{g}_j \right)_{, k} \quad	\because \text{기본은 covariant tangent vector} \\
&=  A^{ij}_{, k} \mathbf{g}_i \otimes \mathbf{g}_j + A^{ij} \mathbf{g}_{i, k} \otimes \mathbf{g}_j + A^{ij} \mathbf{g}_i \otimes \mathbf{g}_{j, k} \\
&=  A^{ij}_{, k} \mathbf{g}_i \otimes \mathbf{g}_j + A^{ij} \left( \Gamma_{ik}^l \mathbf{g}_l \right) \otimes \mathbf{g}_j + A^{ij} \mathbf{g}_i \otimes \Gamma_{jk}^l \mathbf{g}_l\\

&=  A^{ij}_{, k} \mathbf{g}_i \otimes \mathbf{g}_j + A^{lj} \Gamma_{lk}^i \mathbf{g}_i \otimes \mathbf{g}_j + A^{il} \mathbf{g}_i \otimes \Gamma_{lk}^j \mathbf{g}_j\\

&= \left(A^{ij}_{, k} + A^{lj} \Gamma_{lk}^i + A^{il} \Gamma_{lk}^j \right)  \mathbf{g}_i \otimes \mathbf{g}_j
\end{aligned}
\end{equation}
$$

Thus,
$$
\begin{equation}
A^{ij} \vert_k = A^{ij}_{, k} + A^{lj} \Gamma_{lk}^i + A^{il} \Gamma_{lk}^j , \quad i,j,k= 1, 2, \cdots, n
\label{eq-2.87}
\end{equation}
$$

Notice $\eqref{note-2.84}$ 에서 볼 수 있듯이 $\Gamma_{kj}^i \mathbf{g}_i = -\Gamma_{ij}^k \mathbf{g}^i$ 이므로
$$
\begin{equation}
\begin{aligned}
A_{ij} \vert_k   &= A_{ij, k}    - A_{lj} \Gamma_{ik}^l - A_{il} \Gamma_{jk}^l \\
A_{.j}^i \vert_k &= A_{j, k}^{i} + A_j^l \Gamma_{lk}^i  - A_l^i  \Gamma_{jk}^{l}
\end{aligned}
\label{eq-2.88}
\end{equation}
$$


#### Note 1. What is the Christoffel symbol?

- $\Gamma_{ijk}$ : **Scalar**, Inner product of tangent vector $\mathbf{g}^k$ and the partial differential of one tangent vector $\mathbf{g}_i$ to one coordinate component $\theta^j$ such that
$$
\begin{equation}
\Gamma_{ijk}
= \mathbf{g}_{i, j} \cdot \mathbf{g}_k
= \mathbf{g}_{j, i} \cdot \mathbf{g}_k = \Gamma_{jik}

,\quad \therefore
\Gamma_{ijk} =
\frac{\partial \mathbf{g}_i}{\partial \theta^j} \cdot \mathbf{g}_k
\label{note-001}
\end{equation}
$$

- 만일, contra derivative Tangent vector $\mathbf{g}^k$ 를 따라 (inner product 하여) 정의한다면, 다음과 같이 covariant derivative용  Christoffel symbol $\Gamma_{ij}^k$은 쉽게  정의할 수 있다.
$$
\Gamma_{ij}^k = \mathbf{g}_{i, j} \cdot \mathbf{g}^k
$$

- For covariant derivative, we require a Christoffel symbol $\Gamma_{ij}^k$ along to covarint tangent vector $\mathbf{g}_k$ so that
$$
\begin{equation}
0
= (\delta_j^i)_{,k}
= (\mathbf{g}^i \cdot \mathbf{g}_j)_{, k}
= \mathbf{g}_{, k}^i \cdot \mathbf{g}_j + \mathbf{g}^i \cdot \mathbf{g}_{j, k}
= \mathbf{g}_{, k}^i \cdot \mathbf{g}_j + \Gamma_{jk}^i, \\
\quad \therefore
\Gamma_{jk}^i
= - \mathbf{g}_{, k}^i \cdot \mathbf{g}_j
= - \mathbf{g}_{, j}^i \cdot \mathbf{g}_k
= \Gamma_{kj}^i,
\quad
\Gamma_{jk}^i = -\frac{\partial \mathbf{g}^i}{\partial \theta^k} \cdot \mathbf{g}_j
\label{note-002}
\end{equation}
$$

- $\Gamma_{ijk}$ : Contravariant $\mathbf{g}^k$ compatible, 순서대로 해석이 된다. $i$  번째 벡터를 $j$ 번째 Component로 미분해서 $k$ 번째 Tangent vector와 Inner-product.

- $\Gamma_{jk}^i$ : Covariant $\mathbf{g}_j$ compatible. Covariant는 행렬의 수학적 표시대로 왼쪽 아래 $j$ 번째 Tangent vector와 **minus Inner-product**.  미분은 $i$  번째 벡터를 $k$ 번째 Component로 미분.  **minus 가 있어서 $k, j$를 바꾸어진다** 즉, **inner product로 붙는 index가 아래앞으로 온다**. 라고 기억해도 된다. 그러면 위의 것과 같은 해석

- 즉, 아래 식 처럼, $j$로 미분한다는 것만 fix가 된다는 의미가 된다.
$$
\begin{equation}
\Gamma_{ij}^k = \mathbf{g}_{i, j} \cdot \mathbf{g}^k = - \mathbf{g}_{,j}^k \cdot \mathbf{g}_i
\end{equation}
$$
