RiemannGeometry : Isometric Immersion
============

Riemannian Geometry 상에서의 알고리즘 개발에서는 Isometric Immersion과 Complete Manifold가 더 중요할 것으로 판단된다.
- Isometric Immersion 은 Riemannian Metric이 보존되는 조건을 가지는 Manifold (의 Construction)
- Complete Manifold 는 알고리즘이 Convcerge 할 수 있는 Manifold의 조건이라는 측면에서 중요하다.
- Variations of Energy 는 Geodesic이 최소 Enegy를 따라간다는 것에 대한 증명일 뿐이다. 
	- First Formulation과 Second Formulation으로 증명되는 것이다.
	- Hamiltonian 증명과 유사하다. 

Let $f:M \rightarrow \bar{M}$ be a **differentiable immersion** of a manifold $M$ of dimension $n$ into a Riemmanian manifold $\bar{M}$ of dimension equal to $k = n+m$.

- Riemmanian metric for Isometric Immersion
If $v_1, v_2 \in T_p M$, $\langle v_1, v_2 \rangle = \langle df_p(v_1), df_p(v_2) \rangle $ ... $f$가 $\mathbb{R}$ 로의 함수가 아닌 Manifold로의 함수이므로 $df$는 Scalar가 아닌 Vector field 이다.

이떄, $f$는 $M$ 에서 $\bar{M}$으로의 **Isometric Immersion** 이라 한다.
- $\mathbb{R}^3$ 에서 $S$에 대한 고찰에서 비롯됨. 즉,

$$
\{(x,y,z) \in \mathbb{R}^3; z = f(x,y) \}, \;\;\; f(0,0)=0, \;f_x(0,0)=f_y(0,0)=0
$$

에 대한 고찰, 결국 $S$의 Tangent Space는 $(x,y)$의 Tangent Plane.
- The behavior of $S$ in a neighborhood of the origin $0 \in \mathbb{R}^3$ is strongly influenced by the **Quadratic Form **
- The Second Fundamental form of $S$ : 

$$
\Pi (x,y) =f_{xx}(0)x^2 + 2f_{xy}(0)xy + f_{yy}(0)y^2
$$
이에 대한 Gaussian Curvature는 
$$
K = f_{xx}f_{yy} - f^2{xy}
$$
여기에서 출발한다.

## The Second Fundamental Form
Let $f:M^n \rightarrow \bar{M}^{n+m=k}$ be an immersion. Then
for each $p \in M$, $\exists U \subset M$ of $p$ ($U$ : Neighborhood) such that $f(U) \subset \bar{M}$ is a submanifold of $\bar{M}$.
i.e. there exist a neighborhood $\bar{U} \subset \bar{M}$ of $f(p)$ and a diffeomorphism $\varphi: \bar{U} \rightarrow V \subset \mathbb{R}^k$. such that $\varphi$ maps $f(U) \cap \bar{U}$ diffeomorphically onto an openset of subspace of $\mathbb{R}^n \subset \mathbb{R}^k$. 

- $v \in T_q M, \; q \in U$, with $df_q(v) \in T_{f(q)} \bar{M}$.
- 즉, Local vector field on $M$을 Local vector field $\bar{M}$ 로 보낸다고 생각한다.

![Fig01](http://jnwhome.iptime.org/img/DG/Isometry_Immersion01_01.png)

- **Main Idea**

$$
T_p \bar{M} = T_p M \oplus (T_p M)^{\perp}
$$

- 즉, $T_p \bar{M}$ 은 $T_p M$ 보다 더 크고 $T_p M$에 Orthogonal Space가 덧붙여짅 것이다. 고로 $f:M^n \rightarrow \bar{M}^{n+m=k}$은 $T_p M$을 더 확장 시키는 ... 그런 것으로 보면 된다.

- 고로 $v \in T_p \bar{M}, \; p \in M$에 대하여

$$
v = v^T + v^N, \;\; v^T \in T_p M, \;\; v^N \in (T_p M)^{\perp}
$$

- **Riemmanian Connection on $\bar{M}$**
For local vector fields $X, Y \in M$, and $\bar{X}, \bar{Y} \in \bar{M}$. define
$$
\nabla_X Y = (\bar{\nabla}_{\bar{X}} \bar{Y} )^T
$$

- Let $B(X,Y)$ is local vector field on $\bar{M}$ normal to $M$ such that

$$
B(X,Y) = \bar{\nabla}_{\bar{X}} \bar{Y} - \nabla_X Y
$$

다시말해, $\bar{\nabla}_{\bar{X}} \bar{Y} \in \bar{M}$ 에서 $\nabla_X Y \in M$을 뺐다. 그러므로 $B(X,Y) \in M^{\perp}$ 이다. 
Tangent Space 상에서 생각하면 더욱 간단히 이해할 수 있다. (Proposition 2.1 참조)

- Let $\mathcal{X}(U)^{\perp}$ to be the differentiable vector fields on $U$ that are normal to $f(U) \approx U$.

### Propostion 2.1
If $X,Y \in \mathcal{X}(U)$, the mapping $B:\mathcal{X}(U) \times \mathcal{X}(U) \rightarrow \mathcal{X}(U)^{\perp}$ given by
$$
B(X,Y) = \bar{\nabla}_{\bar{X}} \bar{Y} - \nabla_X Y
$$
is **bilinear and symmetric**.

#### Sketch of proof
- **bilinear**

$$
B(fX, Y) = B(X, fY) = fB(X, Y), \;\; f \in \mathcal{D}(U)
$$

- ** Symmetric **

$$
B(X, Y) = B(Y, X)
$$

- How to proof of Symmetry
Since $[X, Y] = \nabla_X Y - \nabla_Y X$, 
$$
B(X,Y) = \bar{\nabla}_{\bar{X}} \bar{Y} - \nabla_X Y = \bar{\nabla}_{\bar{Y}} \bar{X} + [\bar{X}, \bar{Y}] - \nabla_Y X - [X,Y]
$$
Since $[\bar{X}, \bar{Y}] = [X, Y]$, we conclude that $B(X,Y) = B(Y,X)$.

- Let $p \in <$ and $\eta \in (T_p M)^{\perp}$, The mapping $H_{\eta} : T_p M \times T_p M \rightarrow \mathbb{R}$ such that

$$
H_{\eta} = \langle B(x,y), \eta \rangle, \;\;\; x,y \in T_p M
$$
$H_{\eta}$는 당연하지만 Proposition 2,1에 의해 Symmetric이고 Bilinear이다.

### Definition 2.2
The Quadratic form $\Pi_{\eta}(x)$ defined on $T_p M$ by
$$
\Pi_{\eta}(x) = H_{\eta}(x,x) = \langle B(x,x), \eta \rangle
$$
is called ** the second fundamental form ** of $f$ at $p$ along **the normal vector $\eta$**.

- Self adjoint operator $S_{\eta}: T_p M \rightarrow T_p M$ 과 $H_{\eta}$와의 관계.

$$
\langle S_{\eta}(x), y \rangle = H_{\eta}(x, y) = \langle B(x,y), \eta \rangle
$$

### Propostion 2.3
Let $p \in M, x \in T_p M$ and $\eta \in (T_p M)^{\perp}$. 
Let $N$ be a local extension of $\eta$ normal to $M$. Then
$$
S_{\eta}(x) = - (\bar{\nabla}_x N)^T
$$
- Proof는 간단하다 (see p128).
- Covariant Derivative의 관점에서 $S_{\eta}(x)$를 정의한 것
- 즉, $x \in T_P M$을 따라 Normal Vector의 변화를 보는 것, 그리고 그 변화는 $T_p M$을 따라 발생한다. 
- 내가 원하던, Tangent Space에서의 변화량에 따른 Normal Vector의 변화량을 그리고 이를 통해 Curvature의 변화량을 계측할 수 있는 도구가 바로 $S_{\eta}(x)$ 이다. 

#### Sketch of Proof
Let $y \in T_p M$ and let X, Y be a local extension of $x, y \in T_p M$. Then $\langle N, Y \rangle = 0$, and
$$
\begin{align}
\langle S_{\eta}(x), y \rangle &= \langle B(X,Y), N \rangle  &\because B(X,Y), N \in (T_p M)^{\perp} \\
&= \langle \bar{\nabla_X} Y - \nabla_X Y, N \rangle(p)  &\bar{\nabla}_X Y \in \bar{M},\; \nabla_X Y \in M \\
&= \langle \bar{\nabla_X} Y, N \rangle (p) = - \langle Y, \bar{\nabla_X} N \rangle(p) &\because \bar{\nabla}_X \langle Y, N \rangle = \langle \bar{\nabla}_X Y, N \rangle + \langle Y, \bar{\nabla}_X N \rangle \\
&= \langle -\bar{\nabla}_x N, y \rangle \;\;\;\;\;\forall y \in T_p M
\end{align}
$$

### Example 2.4 : Hypersurface
Consider the particular case in which the codimension of the immersion is 1, i.e. $f:M^{n} \rightarrow \bar{M}^{n+1}; f(M) \subset \bar{M}$ is the called a hypersurface.

### Note : Important thing of Eigenvalue and vectors
Let $p \in M$ and $\eta \in (T_p M)^{\perp}$, $\|\eta\| = 1$.
$S_{\eta}: T_p M \rightarrow T_p M$ 가 Symmetric이기 때문에, Eignevectors로 Orthognormal Basis $\{ e_1, \cdots e_n \}$ of $T_p M$ 을 대응하는 Eigen values $\lambda_1, \cdots \lambda_n$ 에서 만들 수 있다. 
- i.e. $S_{\eta}(e_i) = \lambda_i e_i, \; 1 \leq i \leq n$.
- 그러므로 $M, \bar{M}$ 둘 다 Orientable and Oriented 되어 있다고 하면 
  - $\{ e_1, \cdots, e_n \} $은 $M$의 **Basis**
  - $\{ e_1, \cdots, e_n, \eta \} $는 $\bar{M}$의 **Basis**

- 이때, $e_i$들을 **Principal Directions**이라 한다. 그리고 $\lambda_i = k_i$들을 **Principal Curvature** of $f$ 라고 한다.
- $\lambda_1, \cdots \lambda_k$의 symmetric functions들은  Immersion에 Invarients 들이다. 예를 들면...
  - $\det(S_n) = \lambda_1, \cdots \lambda_n$은 **Gauss-Kronecker curvature** of $f$ 이다.
  - $\frac{1}{n} (\lambda_1 + \cdots + \lambda_n)$은 $f$은 **Mean Curvature**이다.

### Gauss spherical mapping 
When $\bar{M} = \mathbb{R}^{n+1}$, $N \perp M$ be a local extension of $\eta$, such that $|\eta| = 1, \eta \perp M$
Let $S_1^n = \{ x \in \mathbb{R}^{n+1}; |x| = 1  \} \subset \mathbb{R}^{n+1}$
Define **the Gauss spherical mapping**, $g:M^n \rightarrow S_1^n$ (이는 $N$의 원점을 $\mathbb{R}^{n+1}$의 원점으로 보내는 Mapping이다.)
$$
g(q) = \text{endpoint of the translation of} N(q)
$$

고로 $T_q M$과 $T_g(q)S_1^n$ 은 서로 **Parallel** 이다. (원점이 같으므로 당연). 고로 **이 둘을 같은 것으로 볼 수 있다.**
Let $dq_q : T_p M \rightarrow T_q M$ is given by
$$
dg_q(x) = \frac{d}{dt}(N \circ c(t))_{t=0} = \bar{\nabla}_x N = (\bar{\nabla}_x N )^T = -S_{\eta}(x)
$$
where $c:(-\varepsilon, \varepsilon) \rightarrow M$ is a curve with $c(0)=q, c'(0)=x$. 또한 $\langle N, N \rangle = 1$에서 $\bar{\nabla}_x N = (\bar{\nabla}_x N )^T$ 이다.
- 즉, $-S_{\eta}$ 는 **Gauss spherical mapping 의 미분이다.**

### Curvature of $M, \bar{M}$ 그리고 Second Funcdamental Form과의 관계
Let $x, y \in T_p M \subset T_p \bar{M}$ to be linearly independent.
Let $K(x,y), \bar{K}(x,y)$ be a sectional curvature of $M, \bar{M}$.

### Theorem 2.5 : Gauss
Let $p \in M$ and let $x, y \in T_p M$ be orthnormal vectors. Then
$$
K(x,y) - \bar{K}(x,y) = \langle B(x,x), B(y,y) \rangle - |B(x,y)|^2
$$
#### Sketch of Proof
Let $X, Y$ be a local extension of $x, y$ such that $X \perp Y \in T_p M$, and
let $\bar{X}, \bar{Y}$ be a local extension of $X, Y$ to $\bar{M}$. Then
$$
\begin{align}
K(x,y) - \bar{K}(x,y) &= \langle \nabla_Y \nabla_X X - \nabla_X \nabla_Y X - (\bar{\nabla}_{\bar{Y}} \bar{\nabla}_{\bar{X}} \bar{X} - \bar{\nabla}_{\bar{X}} \bar{\nabla}_{\bar{Y}} \bar{X}), Y \rangle(p) \\
&+ \langle \nabla_{[X,Y]} X - \bar{\nabla}_{[\bar{X}, \bar{Y}]} \bar{X}, Y \rangle (p)
\end{align}
$$
마지막 항은 다음과 같이 유도되어 0이다.
$$
\langle \nabla_{[X,Y]} X - \bar{\nabla}_{[\bar{X}, \bar{Y}]}\bar{X}, Y \rangle(p) = - \langle (\bar{\nabla}_{[\bar{X}, \bar{Y}] } \bar{X})^{\perp}, Y \rangle (p) = 0
$$
Let an orthonormal fields $E_1, \cdots E_m$, $m = \dim \bar{M} - \dim M$ to $M$, then we write $B(X,Y)$ such that
$$
B(X,Y)=\sum_i H_i (X,Y)E_i, \;\;\; H_i = H_{E_i}, \;\; i=1, \cdots , m
$$
따라서,
$$
\begin{align}
\bar{\nabla}_{\bar{Y}} \bar{\nabla}_{\bar{X}} \bar{X} &= \bar{\nabla}_{\bar{Y}} \left( \sum_i H_i (X, X) E_i + \nabla_X X  \right) \\
&= \sum_i  \{ H_i(X,X) \bar{\nabla}_{\bar{Y}} E_i + \bar{Y} H_i (X,X) E_i \} + \bar{\nabla}_{\bar{Y}} \nabla_X X
\end{align}
$$
여기에서
$$
\begin{align}
\bar{\nabla}_{\bar{Y}} \langle E_i, Y\rangle &= \langle \bar{\nabla}_{\bar{Y}} E_i, Y \rangle + \langle E_i, \bar{\nabla}_{\bar{Y}} {\bar{Y}} \rangle & \\
0 &= \langle \bar{\nabla}_{\bar{Y}} E_i, Y \rangle + \langle E_i, \bar{\nabla}_{\bar{Y}} {\bar{Y}} \rangle &\because E_i \perp Y \\
\langle \bar{\nabla}_{\bar{Y}} E_i, Y \rangle &= -\langle E_i, \bar{\nabla}_{\bar{Y}} {\bar{Y}} \rangle + \langle E_i, \nabla_{Y} Y \rangle  &\because E_i \perp \nabla_{Y} Y \in M, \;\; \langle E_i, \nabla_{Y} Y \rangle = 0 \\
&= -\langle E_i, \bar{\nabla}_{\bar{Y}} {\bar{Y}} - \nabla_{Y} Y \rangle = -H_i (Y,Y)
\end{align}  
$$
그리고 $\bar{Y} = Y + Y^{\perp}$ 이므로 $\langle \bar{\nabla}_{\bar{Y}} \nabla_X X, Y \rangle = \langle \nabla_Y \nabla_X X, Y \rangle$. 그러므로, at $p$,
$$
\langle \bar{\nabla}_{\bar{Y}} \bar{\nabla}_{\bar{X}} {\bar{X}}, Y \rangle = - \sum_i H_i (X,X) H_i (Y,Y) + \langle \nabla_Y \nabla_X X, Y \rangle
$$
마찬가지로
$$
\langle \bar{\nabla}_{\bar{X}} \bar{\nabla}_{\bar{Y}} {\bar{X}}, Y \rangle = - \sum_i H_i (X,Y) H_i (X,Y) + \langle \nabla_X \nabla_Y X, Y \rangle
$$
이를 첫 식에 대입하여 정리하면 증명 끝.

### Remark 2.6 : Very Important to develope a Algorithm
Hypersurface $f:M^n \rightarrow \bar{M}^{n+1}$ 의 경우 Gauss 공식은 매우 쉽게 표현된다.
Let $p \in M$, $\eta \in (T_p M)^{\perp}$.
Let $\{ e_1, \cdots, e_n \}$ be orthonormal basis of $T_p M$, where $S_{\eta} = S$ is diagonal, that is $S(e_i) = \lambda_i e_i, i=1, \cdots n$ where $\lambda_1, \cdots \lambda_n$ are eigenvalues of $S$.
당연히 $e_i$ 는 Orthonormal Basis 이고 Eigne Vector이므로 $H(e_i, e_i) = \lambda_i$ and $H(e_i, e_j) = 0$, if $i \neq j$. 그러므로 Theorem 2.5는 다음과 같이 Simple 하게 나온다. 
Since
$$
H(e_i, e_i) = \langle B(e_i, e_i), \eta \rangle = \lambda_i, \;\;\;  S_{\eta}(e_i) = S(e_i) = \langle \lambda_i e_i, e_i \rangle = H(e_i, e_i) = \langle B(e_i, e_i), \eta \rangle
$$
즉, Normal 측 변화가 $\lambda$ 로 간단히 나타나므로. ($H$에 대한 Normal의 차원은 1이므로)
$$
K(e_i, e_j) - \bar{K}(e_i, e_j) = \lambda_i \lambda_j \;\;\; 
$$

### Remark 2.7
$M = M^2 \subset \bar{M} = \mathbb{R}^3$ 의 경우 $\lambda_1 \lambda_2$ 가 면에 대한 The principle curvature coincides with Gaussian curvature 이다. 이때, ㅎ면냐무 curvature는 coincides with sectional curvature가 되고 이것이 그 유명한 **Theorem Egregium of Gauss** 가 된다. 

### Example 2.8 (Curvature of $S^n$)
$S^n \subset \mathbb{R}^{n+1}$의 경우 sectional curvature는 항상 1이다. P132 살펴본다. 직관적으로도 $S^n$에 Mapping이므로 Curvature는 1이다.

### Proposition 2.9 (Immersion and Geodesic)
An Immersion $f:M \rightarrow \bar{M}$ is geodesic at $p \in M$ if and only if every geodesic $\gamma$ of $M$ starting from $p$ is a geodesic of $\bar{M}$ at $p$.
#### Sketch of proof
Let $\gamma(0)=p, \gamma'(0)=x$. Let $N$ be a local extention normal to $M$ of a vector $\eta$ norma to $M$ at $p$.
Let $X$ be a local extension of $\gamma'(t)$ to a tangent field on $M$.  Since $\langle X, N \rangle = 0$. at $p$

![Fig02](http://jnwhome.iptime.org/img/DG/Isometry_Immersion02.png)

$$
\begin{align}
H_{\eta}(x,x) &= \langle S_{\eta}(x), x \rangle   &\because \text{by definition} \\
&= - \langle \bar{\nabla}_X N, X \rangle   &\because \text{by definition} \\
&= - X\langle N, X \rangle + \langle N, \bar{\nabla}_X X \rangle   &\because X\langle N, X \rangle = \langle \bar{\nabla}_X N, X \rangle + \langle N, \bar{\nabla}_X X \rangle \\
&= \langle N, \bar{\nabla}_X X \rangle
\end{align}
$$

여기서 $f$가 $p$에서 $\forall x \in T_p M$ 에 대하여Geodesic이면 $\gamma \in T_p M$ 이므로 $\bar{\nabla}_X X$ 에는 Normal Component가 없으므로 $\langle N, \bar{\nabla}_X X \rangle = 0$ 이 될 것이다. 그러면, $\bar{M}$에서의 Geodesic이나 $M$에서의 Geodesic은 같다. 

#### Note
다음 그림에서 Gauss Formula 에서, 점 p에서의 Gaussian Curvature는 $K_S(p) = K(p, \sigma)$

![Fig03](http://jnwhome.iptime.org/img/DG/Isometry_Immersion03.png)

다시말해, the sectional curvature $K(p, \sigma)$ 는 Gaussian Curvature at $p$, of a small surface formed by geodesics of $M$ that start from $p$ and are tangent to $\sigma$.

또한 다음 그림과 같이 좀 더 극단적으로 $\bar{M}=S^n \subset \mathbb{R}^{n+1}$의 경우를 생각할 수 있다 (P133)

![Fig04](http://jnwhome.iptime.org/img/DG/Isometry_Immersion04.png)

이 경우에도 Totally Geodesic of $S^n$과 Locally Geodesic of $\Sigma$는 서로 동일하다.
위 그림과 같은 경우는 Sectional curvature까지 동일하게 되는 경우이다.

### Definition 2.10 : A minimal Immersion
An Immersion $f:M \rightarrow \bar{M}$ is called **minimal** if for every $p \in M$ and every $\eta \in (T_p M)^{\perp}$, the trace of $S_{\eta} = 0$.

- Minimal의 의미는 Immersion이 가능하게 하는 $\bar{M}$ 의 Volume을 최소화 시킨다는 의미가 있기 때문이다. 

- $E_1, \cdots E_m$ 을 $\mathcal{X}(U)^{\perp}$의 Orthonormal Frame이라 놓자. where $U$ is a neighorhood of $p$ in which $f$ is an embedding. 그러면

$$
B(x,y) = \sum_i H_i (x,y) E_i, \;\;\; x,y \in T_p M, i=1, \cdots, m
$$
where $H_i = H_{E_i}$. 그러면 The mean curvature vector of $f$를 다음과 같이 놓을 수 있으며 이는 Frame의 선택과는 관계 없다.

$$
H = \frac{1}{n}\sum_i (\text{trace } S_i) E_i
$$
이 경우 $f$가 Minimal이기 위한 필요충분 조건은 $H(p) =0, \;\forall p \in M$.

#### Note 꼭 기억하자.
$$
S_{\eta}(X) = -(\bar{\nabla}_X \eta)^T \;\; 
$$
- Transpose가 아니라 Tangent Space상에 존재하는 Component들로 이루어졌다는 의미이다. 
- -Tangent가 $S_{\eta}(X)$ 의 의미이다. 
- $S_{\eta}: T_p M \rightarrow T_p M$ 임을 기억하자.
$$
\langle S_{\eta}(x), y \rangle = H_{\eta}(x, y) = \langle B(x,y), \eta \rangle
$$

## Fundamental Equations
### Normal Connection $\nabla^{\perp}$
$$
\nabla_X^{\perp} \eta = (\bar{\nabla}_X \eta)^N = \bar{\nabla}_X \eta - (\bar{\nabla}_X \eta)^T = \bar{\nabla}_X \eta + S_{\eta}(X)
$$
- Linear 특징 

$$
\nabla^{\perp}_X (f\eta) = f \nabla^{\perp}_X \eta + X(f) \eta,\;\; f \in \mathcal{D}(M)
$$

- Normal Curvature $R^{\perp}$

$$
E^{\perp}(X,Y)\eta = \nabla_Y^{\perp} \nabla_X^{\perp} \eta - \nabla_X^{\perp} \nabla_Y^{\perp} \eta + \nabla_{[X,Y]}^{\perp} \eta
$$

### Proposition 3.1
#### Gauss equation
$$
\langle \bar{R}(X,Y)Z, T \rangle = \langle R(X,Y)Z, T \rangle -\langle B(Y,T), B(X,Z)) \rangle + \langle B(X,T), B(Y,Z) \rangle
$$

#### Ricci equation
$$
\langle \bar{R}(X,Y)\eta, \zeta \rangle - \langle R^{\perp}(X,Y) \eta, \zeta \rangle = \langle [S_{\eta}, S_{\zeta}]X, Y \rangle
$$
where $[S_{\eta}, S_{\zeta}]$ denotes the operator $S_{eta} \circ S_{\zeta} - S_{\zeta} \circ S_{\eta}$.

The Second fundamental form of the immersion 은 Tensor로 볼 수 있으므로 다음과 같이 정의된다고 하자.
$$
B: \mathcal{X}(M) \times \mathcal{X}(M) \times \mathcal{X}(M)^{\perp} \rightarrow \mathbb{R} \\
B(X, Y, \eta) = \langle B(X, Y), \eta \rangle
$$

### Proposition 3.4 : Codazzi Equation
$$
\langle \bar{R}(X,Y)Z, \eta \rangle = (\bar{\nabla}_Y B)(X, Z, \eta) - (\bar{\nabla}_X B)(Y, Z, \eta)
$$
#### Note
$(\bar{\nabla}_Y B)(X, Z, \eta)$ 에서 $B(X, Y, \eta) = \langle B(X, Y), \eta \rangle$ 인 Tensor 표기이므로 이를 먼저 정의에 의해 일반 연산으로 바꾸어서 해석한다. Tensor지만 결과는 스칼라이므로 비교적 간단한 편이다.

#### Note : Remind
### Definition 5.7 : Tensor의 Covariant Derivation
Let $T$ be a tensor of order $r$. The covariant differential $\nabla T$ of $T$ is a tensor of order $(r+1)$ given by
$$
\nabla T (Y_1, \cdots, Y_r, Z) = Z(T(Y_1, \cdots, Y_r)) - T(\nabla_Z Y_1, \cdots, T_r) - \cdots - T(Y_1, \cdots, Y_{r-1}, \nabla_Z Y_r).
$$
For each $Z \in \mathcal{X}(M)$, **the covarinat derivative $\nabla_Z T$ of $T$ relative to $Z$** is a tensor of order $r$ given by
$$
\nabla_Z T(Y_1, \cdots, Y_r) = \nabla T(Y_1, \cdots, Y_r, Z)
$$
- 다시말해 Tensor의 Covariant Derivative는 Tensor의 맨 마지막 Vector field에 의한 Covariant Derivative 라는 것이다. 따라서 다음과 같다.

$$
\nabla_Z T(Y_1, \cdots, Y_r) = Z(T(Y_1, \cdots, Y_r)) - T(\nabla_Z Y_1, \cdots, T_r) - \cdots - T(Y_1, \cdots, Y_{r-1}, \nabla_Z Y_r).
$$



