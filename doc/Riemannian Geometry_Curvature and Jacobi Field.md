Riemannian Geometry (Curvature)
=============

**The sectional curvature** of $M$ at $p$ with respect to $\sigma$
$$
K(p,\sigma)
$$
If $M = \mathbb{R}^n$, $K(p, \sigma) = 0$ for all $p$ and all $\sigma$.

## Curvature
### Definition
The curvature $R$ of a Riemannian manifold $M$ is a correspondence that associates to every pair $X, Y \in \mathcal{X}(M)$ a mapping $R(X, Y):\mathcal{X}(M) \rightarrow \mathcal{X}(M)$ given by
$$
R(X,Y)Z = \nabla_Y \nabla_X Z - \nabla_X \nabla_Y Z + \nabla_{[X,Y]} Z, \;\;\; Z \in \mathcal{X}(M)
$$

- Tangent space $T_p M$의 Basis에 대하여 생각해 보면 Since $\left[ \frac{\partial}{\partial x_i}, \frac{\partial}{\partial x_j} \right] = 0$,
$$
R\left( \frac{\partial }{\partial x_i}, \frac{\partial }{\partial x_j} \right) \frac{\partial }{\partial x_k} = \left( \nabla_{\frac{\partial }{\partial x_j}} \nabla_{\frac{\partial }{\partial x_i}} - \nabla_{\frac{\partial }{\partial x_i}} \nabla_{\frac{\partial }{\partial x_j}} \right) \frac{\partial}{\partial x_k}
$$

### Proposition
The curvature $R$ of a Riemannian manifold has the following properties:
- $R$ is bilinear in $\mathcal{X}(M) \times \mathcal{X}(M)$, that is,
$$
\begin{align}
R(fX_1 + gX_2, Y_1) &= fR(X_1, Y_1) + gR(X_2, Y_1) \\
R(X_1, fY_1 + gY_2) &= fR(X_1, Y_1) + gR(X_1, Y_2) \\
\end{align}
$$
- For any $X,Y \in \mathcal{X}(M)$, the curvature operator $R(X,Y):\mathcal{X}(M) \rightarrow \mathcal{X}(M)$ is linear, that is
$$
\begin{align}
R(X,Y)(Z+W) &= R(X,Y)Z + R(X,Y)W \\
R(X,Y)fZ &= fR(X,Y)Z
\end{align}
$$

증명은 90P.
#### Remark 
사실 $\nabla_{[X,Y]} Z$ 항의 존재는 Curvature가 위와 같은 Bilinear 특성을 가지도록 추가한 것이다.

### Bianchi Identity 
$$
R(X,Y)Z + R(Y,Z)X + R(Z,X)Y = 0
$$
증명은 91p.

### Proposition
$$
\begin{align}
(X, Y, Z, T) + (Y, Z, X, T) + (Z, X, Y, T) &= 0 \\
(X, Y, Z, T) &= - (Y, X, Z, T) \\
(X, Y, Z, T) &= - (Y, X, T, Z) \\
(X, Y, Z, T) = (Z, T, Y, X) 
\end{align}
$$

#### Note
- From the Levi-Civita Theorem (p55) for Affine Connection
$$
X \langle Y, Z \rangle = \langle \nabla_X Y, Z \rangle + \langle Y, \nabla_X Z \rangle
$$
so that 
$$
\langle \nabla_Y \nabla_X Z, Z \rangle = Y \langle \nabla_X Z, Z \rangle - \langle \nabla_X Z, \nabla_Y Z \rangle
$$
Proof is (Let $K = \nabla_X Z$) 
$$
\begin{align}
Y \langle \nabla_X Z, Z \rangle = Y \langle K, Z \rangle = \langle \nabla_Y K, Z \rangle + \langle K, \nabla_Y Z \rangle \\
Y \langle K, Z \rangle - \langle K, \nabla_Y Z \rangle = \langle \nabla_Y K, Z \rangle \\
Y \langle \nabla_X Z, Z \rangle - \langle \nabla_X Z, \nabla_Y Z \rangle = \langle \nabla_Y \nabla_X Z, Z \rangle
\end{align}
$$

- Let $[X,Y] = K$. The proof of 
$$
\langle \nabla_{[X,Y]} Z, Z \rangle = \frac{1}{2}[X, Y]\langle Z, Z \rangle
$$
From the above equation
$$
\begin{align}
K \langle Z, Z \rangle &= \langle \nabla_K Z, Z \rangle + \langle Z, \nabla_K Z \rangle \\
K \langle Z, Z \rangle &= 2 \langle \nabla_K Z, Z \rangle
\end{align}
$$

### Curvature under Christoffel symbol to Riemann Connection
Set $ \frac{\partial}{\partial x_i} = X_i$ 
$$
R(X_i, X_j)X_k = \sum_l R_{ijk}^l X_l
$$
$R_{ijk}^l$ are the components of the curvature $R$ in $(U,x)$.
If 
$$
X= \sum_i u^i X_i, \; Y=\sum_j v^j X_j,\; Z = \sum_k w^k X_k
$$
then
$$
R(X,Y)Z = \sum_{i,j,k,l} R_{i.j.k.l}^l X_l
$$
- To express $R_{i.j.k.l}^l$ in terms of the coefficients $\Gamma_{ij}^k$
$$
\begin{align}
R(X_i, X_j)X_k &= \nabla_{X_j} \nabla_{X_i} X_k - \nabla_{X_i} \nabla_{X_j} X_k \\
&= \nabla_{X_j} (\sum_l \Gamma_{ik}^l X_l) - \nabla_{X_i} (\sum_l \Gamma_{jk}^l X_l) \\
&=  \sum_l \frac{\partial}{\partial x_j}\Gamma_{ik}^l X_l + \sum_l \Gamma_{ik}^l \Gamma_{jl}^s X_s - \sum_l \frac{\partial}{\partial x_i} \Gamma_{jk}^l X_l - \sum_l \Gamma_{ik}^l \Gamma_{il}^s X_s
\end{align}
$$

따라서

$$
R_{ijk}^s = \sum_l \Gamma_{ik}^l \Gamma_{jl}^s - \sum_l \Gamma_{jk}^l \Gamma_{il}^s + \frac{\partial}{\partial x_j}\Gamma_{ik}^s - \frac{\partial}{\partial x_i}\Gamma_{jk}^s
$$

이에 따라 Curvature Proposition은 다음과 같이 간략화된다.
$$
\begin{align}
R_{ijks} + R_{jkis} + R_{kijs} = 0\\
R_{ijks} = -R_{jiks} \\
R_{ijks} = -R_{ijsk} \\
R_{ijks} = R_{ksij}
\end{align}
$$

### Sectional Curvature
- Definition $x \wedge y$
$$
|x \wedge y| = \sqrt{|x|^2 |y|^2 - \langle x, y \rangle^2}
$$
이것은 그냥 **Determinent** 이다.

#### Proposition 3.1
Let $\sigma \subset T_p M$, and let $x, y \in \sigma$ be two linearly independent vectors. Then
$$
K(x,y) = \frac{(x,y,x,y)}{|x \wedge y|^2}
$$
**does not depend on the choice of the vectors $x,y \in \sigma$.**
- $K(x,y)$는 다음에 불변이다.
1. $.K(x,y) = K(y,x)$ 
2. $.K(x,y) = K(\lambda x, y)$
3. $.K(x,y) = K(x + \lambda y, y)$

#### Definition (Sectional Curvature)
Given a point $p \in M$ and a two-dimensional subspace $\sigma \subset T_p M$, then real number $K(x,y) = K (\sigma)$ where $\{ x, y\}$ is any basis of $\sigma$, is called the **sectional curvature** of $\sigma$ at $p$.
- Scalar 값인 $K (\sigma)$를 통해 $\forall \sigma$, the curvature $R$을 Completely 하게 결정할 수 있다.

#### Lemma  
Let $V$ be a vector space of dimension $\geq 2$ , provided with an inner product $\langle , \rangle$. Let $R:V \times V \times V \rightarrow V$ and $R':V \times V \times V \rightarrow V$ be tri-linear mappings such that conditions in proposition 2.5 are satisfied by
$$
(x,y,z,t) = \langle R(x,y)z, t \rangle, \;\;\; (x,y,z,t)' = \langle R'(x,y)z, t \rangle
$$
If $x,y$ are two linearly inde;endent vectors, we mat write
$$
K(\sigma) = \frac{(x,y,x,y)}{|x \wedge y |^2}, \;\;\; K'(\sigma) = \frac{(x,y,x,y)'}{|x \wedge y |^2}
$$
where $\sigma$ is the bi-dimensional subspace generated by $x$ and $y$. If for all $\sigma \subset V$, $K(\sigma)=K'(\sigma)$, then $R = R'$.

#### Lemma (p96): Very Important
Let $M$ be a Riemmanian manifold and $p$ a point of $M$. 
**Define a tri-linear mapping $R':T_p M \times T_p M \times T_p M \rightarrow T_p M$ by**
$$
\langle R'(X,Y,W),Z \rangle = \langle X,W \rangle \langle Y,Z \rangle - \langle Y,W \rangle \langle X,Z \rangle
$$
for all $X,Y,W,Z \in T_p M$. Then $M$ has constant sectional curvature equal to $K_0$  if  and only if $R = K_0 R'$, where $R$ is the curvature of $M$

#### proof
Assume that $K(p, \sigma) = K_0$ , $\forall \sigma \subset T_p M$, and set $\langle R'(X,Y,W,Z), Z \rangle = (X,Y,W,Z)'$ 
$$
\langle R'(X,Y,W), Z \rangle = (X, Y, W, Z)'
$$
Lemma 에서 정의된 대로 
$$
(X, Y, X, Y)' = \langle X, X \rangle \langle Y, Y \rangle - \langle X, Y \rangle^2
$$
Sectinal curvature의 정의에 의해 
$$
K_0 = \frac{(X, Y ,X, Y)}{| X \wedge Y|^2} \Rightarrow (X, Y ,X, Y) = K_0 (\langle X, X \rangle \langle Y, Y \rangle - \langle X, Y \rangle^2) = K_0 R'(X,Y,X,Y) 
$$
Lemma 3.3 implies that 
$$
R(X,Y,W,Z) = K_0 R'(X, Y, W, Z)
$$

### Corollary 3.5
Let $M$ be a Riemmanian manifold, $p$ a point of $M$ and $\{ e_1, \cdots e_n\}$ $n = \dim M$, an orthonormal basis of $T_p M$.
Define $R_{ijkl} = \langle R(e_i, e_j)e_k, e_l \rangle, i,j,k,l = 1, \cdots n$ Then $K(p, \sigma)=K_0$ for all $\sigma \subset T_p M$, if only if
$$
R_{ijkl} = K_o (\delta_{ik} \delta_{jl} - \delta_{il} \delta_{jk})
$$
where
$$
\delta_{ij} = 
\begin{cases}
1, &i=j \\ 
0, &i \neq j
\end{cases}
$$

다시말해 $K(p,\sigma)=K_0$, $\forall \sigma \subset T_p M$ if and only if $R_{ijij} = -R_{ijji} = K_0$, $\forall i \neq j$ and $R_{ijkl} = 0$ in other cases.

## Ricci Curvature and scalar curvature
- Sectional curvature의 Combination의 한 형태

- Let $x=z_n$ be a unit vector in $T_p M$.
- Take an orthnormal basis $\{z_1, z_2, \cdots z_{n-1} \}$ of the hyperplane in $T_p M$ orthogonal to $x$
- Consider the following **averages**:
	- **Ricci Curvatuere** in the direction $x$
$$
\text{Ric}_p (x) = \frac{1}{n-1} \sum_{i=1}^{n-1} \langle R(x,z_i)x,z_i \rangle \\
$$

	- **Scalar curvature** at $p$
$$
K(p) = \frac{1}{n} \sum_{j=1}^n \text{Ric}_p(z_j) = \frac{1}{n(n-1)} \sum_{ij} \langle R(z_i, z_j) z_i, z_j \rangle
$$

### Ricci Tensor
- Let $x, y \in T_p M$ and put

$$
Q(x,y) = \text{trace of the mapping} \;\; z \mapsto R(x,z)y 
$$

- $Q$ is **bilinear** For orthonormal basis $\{z_1, \cdots z_{n-1},z_n = x \}$ of $T_p M$

$$
\begin{align}
Q(x,y) &= \sum_i  \langle R(x,z_i)y, z_i \rangle \\
&= \sum_i  \langle R(y,z_i)x, z_i \rangle = Q(y,x)
\end{align}
$$

- $Q$ is symmetric and 

$$ 
Q(x,x) = \sum_{i} \langle R(x, z_i)x, z_i \rangle = (n-1)\frac{1}{n-1} \sum_{i} \langle R(x, z_i)x, z_i \rangle = (n-1) \text{Ric}_p(x)
$$

- Let a self-adjoint mapping $K$ corresponding the bilinear form $Q$ on $T_p M$ 

$$
\langle K(x),y \rangle = Q(x.y)
$$

- For an orthonormal basis $\{z_1, \cdots z_n \}$

$$
\begin{align}
\text{Trace of K} &= \sum_j \langle K(z_j), z_j \rangle = \sum_j Q(z_j, z_j) \\
&= (n-1)\sum_j \text{Ric}_p(z_j) = n(n-1)K(p)

\end{align}
$$

이때, **bilinear form** $\frac{1}{n-1} Q$를 **Ricci Tensor**라 한다.

- Let $X_i = \frac{\partial}{\partial x_i}$, $g_{ij} = \langle X_i, X_j \rangle$, and $g^{ij}$ is the inverse matrix of $g_{ij}$ i.e. $\sum_k g_{ik}g^{kj} = \delta_i^j$
- Then the coefficient of the bilinear form $\frac{1}{n-1}Q$ in the basis $\{X_i\}$ are given by

$$
\frac{1}{n-1}R_{ik} = \frac{1}{n-1} \sum_j R_{ijk}^j = \frac{1}{n-1} \sum_j R_{ijks} g^{sj}
$$ 

- The scalar curvature in the coordinate system $(x_i)$ is given by

$$
K = \frac{1}{n(n-1)} \sum_{ik} R_{ik} g^{ik}
$$

생각해 보면 Ric Curvature 혹은 Scalar Curvature는 Riemannian Metric 의 Inverse가 필요하다는 것이다.

### Lemma 4.1
Let $f:A \subset \mathbb{R} \rightarrow M$ be a parameterized surface. let $(s,t)$ be a usual coordinates of $\mathbb{R}^2$. Let $V = V(s,t)$ be a Vector field alomnng. Then
$$
\frac{D}{dt}\frac{D}{ds}V - \frac{D}{ds}\frac{D}{dt}V = R(\frac{\partial f}{\partial s}\frac{\partial f}{\partial t})V
$$

#### proof
증명 과정은 매우 Simple 하다. 
1. Let $V = \sum_i v^i X_i$ where $v^i = v^i(s,t)$, and $X_i = \frac{\partial}{\partial x_i}$ 
2. Let $f(s,t)=(x_1(s,t), \cdots x_n(s,t))$ so that $\frac{\partial f}{\partial s} = \sum_j \frac{\partial x_j}{\partial s} X_j$ and $\frac{\partial f}{\partial t} = \sum_k \frac{\partial x_k}{\partial t} X_k$ 

$$
\begin{align}
\frac{D}{\partial s}V &= \frac{D}{\partial s}\sum_i v^i X_i = \sum_i \frac{\partial v^i}{\partial s} X_i + \sum_i v^i \frac{D}{\partial s}X_i \\
\frac{D}{\partial t}\frac{D}{\partial s}V &= \sum_i \frac{\partial^2 v^i}{\partial t \partial s} X_i + \sum_i \frac{\partial v^i}{\partial s} \frac{D}{\partial t} X_i + \sum_i \frac{\partial v^i}{\partial t} \frac{D}{\partial s}X_i + \sum_i v^i \frac{D}{\partial t} \frac{D}{\partial s}X_i 
\end{align}
$$
그러므로
$$
\frac{D}{dt}\frac{D}{ds}V - \frac{D}{ds}\frac{D}{dt}V = \sum_i v^i \left( \frac{D}{\partial t} \frac{D}{\partial s} X_i - \frac{D}{\partial s} \frac{D}{\partial t} X_i \right)
$$
Since
$$
\frac{D}{\partial s}X_i = \nabla_{\frac{\partial f}{\partial s}} X_i = \nabla_{\sum_j \frac{\partial x_j}{\partial s} X_j} X_i = \sum_j \frac{\partial x_j}{\partial s} \nabla_{X_j}X_i
$$
그러므로
$$
\begin{align}
\frac{D}{dt}\frac{D}{ds}X_i &= \frac{D }{dt} \left( \sum_j \frac{\partial x_j}{\partial s} \nabla_{X_j}X_i \right) \\
&= \sum_j \frac{\partial^2 x_j}{\partial t \partial s} \nabla_{X_j}X_i + \sum_j \frac{\partial x_j}{\partial s} \nabla_{\sum_k \frac{\partial x_k}{\partial t} X_k}\nabla_{X_j}X_i \\ 
&= \sum_j \frac{\partial^2 x_j}{\partial t \partial s} \nabla_{X_j}X_i + \sum_{jk} \frac{\partial x_j}{\partial s} \frac{\partial x_k}{\partial t} \nabla_{ X_k}\nabla_{X_j}X_i
\end{align}
$$
따라서
$$
(\frac{D}{dt}\frac{D}{ds} - \frac{D}{ds}\frac{D}{dt})X_i = \sum_{jk} \frac{\partial x_j}{\partial s} \frac{\partial x_k}{\partial t} \left( \nabla_{ X_k}\nabla_{X_j}X_i -  \nabla_{X_j} \nabla_{ X_k}X_i \right)
$$
그러므로, $\sum$ 내부의 각 항이 대응되는 Vector field로 합쳐지므로
$$
(\frac{D}{dt}\frac{D}{ds} - \frac{D}{ds}\frac{D}{dt})V = \sum_{ijk} v^i \frac{\partial x_j}{\partial s} \frac{\partial x_k}{\partial t} R(X_j,X_k)X_i = R(\frac{\partial f}{\partial s}, \frac{\partial f}{\partial t})V
$$

## Tensor on Riemannian Manifold

### Definition : Tensor
A Tensor $T$ of order $r$ on a Riemannian manifold is a **multilinear mapping**
$$
T : \underset{r \text{ factors}} {\underbrace{\mathcal{X}(M) \times \cdots \times \mathcal{X}(M)}}  \rightarrow \mathcal{D}(M)
$$

### Example : The Curvature Tensor
$$
T : \mathcal{X}(M) \times \mathcal{X}(M) \times \mathcal{X}(M) \times \mathcal{X}(M) \rightarrow \mathcal{D}(M)
$$
is defined by
$$
R(X, Y, Z, W) = \langle R(X, Y)Z, W \rangle, \;\;\; X, Y, Z, W \in \mathcal{X}(M)
$$
It is a tensor of order 4. 
The components in the frame is $\left\{ X_i = \frac{\partial}{\partial x_i}\right\}$ associated with 
$$
R(X_i, X_j, X_k, X_l) = R_{ijkl}
$$

### Example : Metric Tensor
$G : \mathcal{X}(M) \times \mathcal{X}(M) \rightarrow \mathcal{D}(M)$  is defined by $G(X,Y) = \langle X, y \rangle, \;\; X, Y \in \mathcal{X}(M)$.  $G$ is a tensor of order 2. The components in the frame $\{ X_i \}$ are the coeddicients ** $g_{ij}$ of the Riemannian metric** 

### Example
The Riemannian Connection $\nabla$ defined by:
$$
\nabla : \mathcal{X}(M) \times \mathcal{X}(M) \times \mathcal{X}(M) \rightarrow \mathcal{D}(M) \\
\nabla(X,Y,Z)= \langle \nabla_X Y,Z \rangle, \;\;\; X, Y, Z \in \mathcal{X}(M)
$$
is not a Tensor, because $\nabla$ is not linear with respect to the argument $Y$.

### Definition : Covariant Differential $\nabla T$ of T
Let $T$ be a tensor of order $r$. The Covariant Differential $\nabla T$ of T is a tensor of order $(r+1)$ given by
$$
\nabla T (Y_1, \cdots , Y_r, Z) = Z(T(Y_1, \cdots, Y_r)) - T(\nabla_Z Y_1, \cdots, Y_r) - \cdots - T(Y_1, \cdots, Y_{r-1}, \nabla_Z Y_r)
$$
For each $Z \in \mathcal{X}(M)$, the covariant derivative $\nabla_Z T$ of $T$ relative to $Z$ is a tensor of order $r$ given by
$$
\nabla_Z T(Y_1, \cdots, Y_r) = \nabla T (Y_1, \cdots, Y_r, Z)
$$

### Excercise
Let $G$ be a Lie group with a bi-invariant metric $\langle, \rangle$. Let $X, Y, Z \in \mathcal{X}(G)$ be unit left invariant vector field  on $G$.
1. $\nabla_X Y = \frac{1}{2}[X,Y]$

