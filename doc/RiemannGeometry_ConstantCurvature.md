RiemannGeometry : Space of constant curvature
=========

[toc]

- Euclidean space $\mathbb{R}^n$ with $K \equiv 0$
- The Unit Sphere $S^n \subset \mathbb{R}^{n+1}$ with $K \equiv 1$ 
- The Hyperbolic space $H^n$ which has sectional curvature $K \equiv 01$
- $\mathbb{R}^n, S^n, H^n$ 은 모두 Complete and Simply Connected Manifold.
- 본 장에서는 Complete and Simply Connected Manifold with **constant sectional curvature**를 가진 manifold를 다룬다.

## Theorem of Cartan on the determination of the metric by means of the curvature
Let $M$ and $\bar{M}$ be two Riemannian manifolds of dimension n, and let $p \in M$, and $\bar{p} \in \bar{M}$.
Choose a linear isometry $i : T_p M \rightarrow T_{\bar{p}} \bar{M}$.
Let $V \in M$ be a normal neighborhood of $p$ such that $\exp_{\bar{p}}$ is defined at $i \circ \exp_p^{-1}(V)$. 
Define a mapping $f : V \rightarrow \bar{M}$ by
$$
f(q) = \exp_{\bar{p}} \circ i \circ \exp_p^{-1}(q), \;\;\; q \in V
$$

$\forall q \in V$, $\exists \gamma [0,t] \rightarrow M$ with $\gamma(0) = p$, $\gamma(1) = q$.
Denote by $P_t$ the parallel transport along $\gamma$ from $\gamma(0)$ to $\gamma(t)$
Define $\phi_t : T_q M \rightarrow T_{f(q)} M$ by

$$
\phi_t(v) = \tilde{P}_t \circ i \circ P_t^{-1}(v), \;\;\; v \in T_q(M)
$$

위 관계는 다음 그림과 같이 도시된다. 
![Fig01](http://jnwhome.iptime.org/img/DG/LARG_14.png)

그림에서 $v$와 $\phi_t(v)$는 각각 $q$와 $\bar{q}$를 지나가는 것 처럼 표현 되었지만, 실제로는 속도 벡터의 끝점이다. 

### Theorem 2.1 : Cartan Theorem
With the notation above, if $\forall q \in V$ and $\forall x, y, u, v \in T_qM$ we have
$$
\langle R(x, y)u, v \rangle = \langle \tilde{R}(\phi_t(x), \phi_t(y)) \phi_t(u), \phi_t(v) \rangle 
$$

then $f:V \rightarrow f(V) \subset \bar{M}$ is a local isometry and $df_p = i$

#### proof
Let $q \in V$
Let $\gamma:[0, l] \rightarrow M, \; \gamma(0)=p, \gamma(l)=q$ be a normalized geodesic.
Let $v \in T_q M$ 
Let $J,\; J(0)=0, J(l)=v$ be a Jacobi field along $\gamma$.
Let $e_1, \cdots, e_n = \gamma'(0)$ be an orthonormal basis of $T_p M$ 
Let $e_i (t), i=1, \cdots , n$ be the parallel transport of $e_i$ along $\gamma$.

Set
$$
J(t) = \sum_i y_i (t) e_i (t)
$$

By definition of Jacobi equation
$$
y_j'' + \sum_i \langle R(e_n, e_i) e_n, e_j \rangle y_i(t) = 0 \;\;\; j=1, \cdots, n
$$
: J(t) 에 $e_j$ 를 Inner product 하여 얻게 된다.

Now Let $\tilde{\gamma}[0, l] \rightarrow \tilde{M}, \tilde{\gamma}(0) = \tilde{p}, \tilde{\gamma}'(0)= i(\gamma'(0))$ be a normalized geodesic.
Let $\tilde{J}(t)$ be the field along $\tilde{\gamma}$ given by
$$
\tilde{J}(t) = \phi_t(J(t)) \;\;\; t \in [0, l]
$$

Let $\tilde{e}_j (t) = \phi_t(e_j(t))$. Then, from the linearity of $\phi_t$
$$
\tilde{J}(t) = \sum_i y_i (t) \tilde{e}_i (t)
$$

Since, by hypothesis
$$
\langle R(e_n, e_i)e_n, e_j \rangle = \langle \tilde{R}(\tilde{e}_n, \tilde{e}_i)\tilde{e}_n, \tilde{e}_j \rangle 
$$
Thus,
$$
y_j'' + \sum_i \langle \tilde{R}(\tilde{e}_n, \tilde{e}_i)\tilde{e}_n, \tilde{e}_j \rangle y_i = 0,  j=1, \cdot, n
$$

즉, Jacobi 방정식의 특성이 $\phi$ 변환에도 만족된다. 즉, $\tilde{J}(0) = 0$ 이고 Parallel Transport는 isometry이므로 $|\tilde{J}(l) | = |J(l)|$ 이다. (그러므로 $\tilde{J}(l) = df_q(v) = df_q(J(l))$ 이 만족되면 증명 끝이다.)

Since $\tilde{J}(l) = \phi(J(t))$, $\tilde{J}'(0) = i(J'(0))$ 이다. (초기 값에서는 그렇다. 0으로 사라지므로 - Jacobi field의 특징)
Jacobi field의 Corollary 2.5 에서 
$$
\begin{align}
J(t) &= (d \exp_p)_{t \gamma'(0)} (tJ'(0)) \\
\tilde{J}(t)&= (d \exp_p)_{t \tilde{\gamma}'(0)} (t\tilde{J}'(0))
\end{align}
$$

그러므로
$$
\begin{align}
\tilde{J}(t) &= (d \exp_{\tilde{p}})_{l\tilde{\gamma}'(0)} l i(J'(0)) \\
&= (d \exp_{\tilde{p}})_{l\tilde{\gamma}'(0)} \circ i \circ ((d \exp_p)_{l \gamma'(0)} )^{-1}(J(l)) = df_q(J(l))
\end{align}
$$

### Corollary 2.2
Let $M$ and $\tilde{M}$ be spaces with the same constant curvature and the same dimension $n$ .
Let $p \in M$ and $\tilde{p} \in \tilde{M}$.
Choose arbitrary orthonormal bases $\{e_j \} \in T_p M$ and $\{\tilde{e}_j \} \in T_p \tilde{M}, \; j =1, \cdots, n$.
Then there exist a neighborhood $V \subset M$ of $p$, a neighborhood $\tilde{V} \subset \tilde{M}$ of $\tilde{p}$, and isometry $f : V \rightarrow \tilde{V}$ such that $df_p (e_j) = \tilde{e}_j$.

- proof는 생략, 하지만 Cartan Theorem에 의해 이는 명백

### Corollary 2.3
Let $M$ be a constant curvature and 
let $p$ and $q$ be any two points of $M$.
Let $\{e_j \}$ and $\{f_j\}$ be arbitrary orthonormal bases of $T_p M$ and $T_q M$, respectivewly.
Then there exist neighborhoods $U$ of $p$ and $V$ of $q$ and an isometry $g:U \rightarrow V$ such that $dg_p(e_j) = f_j$

- 얼핏보면 쉬워 보이지만, 매우 어려운 증명이다. 4차원 이상에서는 증명되었으며 3차원에서는 Yau 등에 의해 다루어졌다.

이 따름정리는 결국 다음과 같은 diffeomorphism이 존재하느내 하는 문제이다. 
- $\forall p \in M and \forall X, Y< Z, T \in T_p M$ there exists a diffeomorphism $f:M \rightarrow \tilde{M}$ which preserves the curvature in the sense that

$$
\langle R(X, Y)Z, T \rangle_p = \langle \tilde{R}(df_p (X), df_p(Y)) df_p (Z), df_p (T)\rangle_{f(p)}
$$

## Hyperbolic Curvature
**Constant curvature가 -1 인 경우**를 생각해보자.

Consider the half-space of $\mathbb{R}^n$ given by
$$
H^n = \{(x_1, \cdots, x_n) \in \mathbb{R}^n; x_n > 0 \}  \tag{1}
$$
Introduce on $H^n$ the metric
$$
g_{ij}(x_1, \cdots, x_n) = \frac{\delta_{ij}}{x_n^2}
$$

- $H^n$은 Simply Connected
- Hyperbolic Space with dimension $n$ 이라 한다.
- Introduction of two metric $\langle, \rangle$ ,and  $\langle\langle ,\rangle\rangle$
if there exists a positive differentiable function $f: M \rightarrow \mathbb{R}$ such that $\forall p \in M$ and $\forall u, v \in T_p M$
$$
\langle u,v \rangle_p = f(p) \langle\langle u,v \rangle\rangle_p
$$
The metric (1) of $H^n$ is conformal (등각) to the usual metric of Euclidean space $\mathbb{R}^n$
Consider on $H^n$, the metric
$$
g_{ij}= \frac{\delta_{ij}}{F^2}
$$
- $f$ is a positive differnetiable function on $H^n$; 으로서 유클리드 공간 $\mathbb{R}^n$과 등각이되도록 한다. 그러면 역행렬의 Component는 $g^{ij} = F^2 \delta_{ij}$. 
- Put $\log F = f$.  그리고 $\frac{\partial f}{x_j}= f_j$ 이면

$$
\frac{\partial g_{ik}}{\partial x_j} = -\delta_{ik}\frac{2}{F^3}F_j = -2\frac{\delta_{ik}}{F^3}f_j
$$

그러므로 Christoffel Symbol의 정의에서

$$
\begin{align}
\Gamma_{ij}^k &=  \frac{1}{2} \sum_m \left\{ \frac{\partial }{\partial x_i} g_{jm} + \frac{\partial }{\partial x_j} g_{mi} - \frac{\partial }{\partial x_m} g_{ij} \right\} g^{mk}\\
&= \frac{1}{2} \sum_m \left\{ \frac{\partial }{\partial x_i} g_{jm} + \frac{\partial }{\partial x_j} g_{mi} - \frac{\partial }{\partial x_m} g_{ij} \right\} \delta_{mk} F^2 \\
&= \frac{1}{2}\left\{ \frac{\partial }{\partial x_i} g_{jk} + \frac{\partial }{\partial x_j} g_{ki} - \frac{\partial }{\partial x_k} g_{ij}  \right\}F^2 \\
&= -\delta_{jk}f_i -\delta_{ki}f_j + \delta_{ij}f_k
\end{align}
$$
따라서 $i, j, k$가 모두 다르면 $\Gamma_{ij}^k = 0$ 따라서 두개 씩 같아야 하므로 그럴 경우
$$
\Gamma_{ij}^i = -f_j, \;\; \Gamma_{ii}^j = f_j, \;\; \Gamma_{ii}^j = -f_i, \;\; \Gamma_{ii}^i = -f_i
$$
이를 사용하여 Curvature를 계산하면
$$
\begin{align}
R_{ijij} &= \sum_{l}R_{iji}^l g_{lj} = R_{iji}^j g_{jj} = R_{iji}^j \frac{1}{F^2} \\
&= \frac{1}{F^2} \left\{ \sum_l \Gamma_{ii}^l \Gamma_{jl}^j - \sum_l \Gamma_{ji}^l \Gamma_{il}^j + \frac{\partial}{\partial x_j} \Gamma_{ii}^j - \frac{\partial}{\partial x_i} \Gamma_{ji}^j \right\}
\end{align}
$$
Since
$$
\frac{\partial}{\partial x_j}\Gamma_{ii}^{j}= \frac{\partial f_j}{\partial x_j} = f_{jj}, \;\;\frac{\partial}{\partial x_i}\Gamma_{ji}^{j}= -\frac{\partial f_i}{\partial x_i} = f_{ii} 
$$
, so that
$$
\begin{align}
F^2 R_{ijij} &= - \sum_{l, l\neq i, \ \neq j} f_l f_l + f_i ^2 - f_j^2 - f_i ^2 + f_j^2 + f_{jj} + f_{ii} \\
&= -\sum_l f_l^2 + f_i^2 + f_j^2 + f_{ii} + f_{jj}
\end{align}
$$

여기서
- 4개의 index가 모두 다르면 $R_{ijkl} = 0$ 
- 3개가 다르면 

$$
R_{ijk}^i = -f_k f_j - f_{kj}, \;\; R_{ijk}^j = -f_i f_k - f_{ki}, \;\; R_{ijk}^k = 0 
$$

마지막으로 $\frac{\partial}{\partial x_i}, \frac{\partial}{\partial x_j}$로 생성되는 평면의 Sectional Curvature는
$$
\begin{align}
K_{ij} &= \frac{R_{ijij}}{g_{ii}g_{jj}} = R_{ijij} F^4 \\
&= (-\sum_l f_l^2 + f_i^2 + f_j^2 + f_{ii} + f_{jj})F^2
\end{align}
$$

여기서 $F^2 = x_n^2$ 이면 $f = \log x_n$ 인 경우에 $i \neq n, j \neq n$ 에서 $\sum_l f_l^2 =f_n^2 = \frac{1}{x_n^2}$ 따라서
$$
K_{ij} = (- \frac{1}{x_n^2})x_n^2 = -1
$$

if $ i = n, j \neq n$, 이면
$$
K_{nj} = (-f_n^2 + f_n^2 + f_{nn})F^2 = -\frac{1}{x_n^2} x_n^2 = -1
$$

if $ i \neq n, j = n$, 이어도 $K_{nj} = -1$ 이다 그러므로 **Sectional Curvature of $H^n$은 -1** 이다.

### Proposition 3.1 
The stright lines perpendicular to the hyperplane $x_n = 0$, and the circle of $H^n$ whose planes are perpendicular to the hyperplane $x_n = 0$ and **whose centers are in this hyperplane are the geodesics of $H^n$**.

마찬가지로 **Geodesic의 Uniqueness Existence도 쉽게 증명**된다. 

## Space forms
### Theorem
Let $M^n$ be a complete Riemanian manifold with constant curvature $K$. Then the universal covering $\bar{M}$ of $M$, with the covering metric, is isometroc to:
- $H^n$, if $K = -1$
- $\mathbb{R}^n$, if $K = 0$
- $S^n$, if $K = 1$

위 Theorem을 증명하기 위해서는 다음의 Lemma가 필요하다.

### Lemma
Let $f_i : M \rightarrow N, i=1,2$ be two local isometries of the (connected) Riemannian manifold $M$ to the Riemannian manifold $N$.Suppose that there exists a p[oint $p \in M$ such that $df_1(p) = f_2(p)$ and $(df_1)_p = (df_2)_p.$ Then $f_1 = f_2$.

- 본 Lemma의 증명은 P 163을 참조한다.
- 위 Theorem의 증명은 P164를 참조한다.

## Isometry of the hyperbolic space; Theorem of Liouville
이는 Conformal Transformation of $\mathbb{R}^n$과 연관이 있다.

### Conformal Transform 
Open set $U \subset \mathbb{R}^n$에 대하여 a map $f:U \subset \mathbb{R}^n \rightarrow \mathbb{R}^n$ 이 **conformal transform** 이라는 것은
$v_1$, $v_2$ at $p \in U$ 가 이루는 각과 $df_p(v_1), df_p(v_2)$ 가 이루는 각이  동일한 것을 의미한다.
metric $\frac{\delta_{ij}}{x_n^2}$을 가진 Hyperbolic space에서 $\mathbb{R}^n$의 Conformal transformation은 $H^n \subset \mathbb{R}^n$을 $H^n$에 Mapping 한다. 


