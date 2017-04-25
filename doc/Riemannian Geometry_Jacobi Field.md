Riemannian Geometry : Jacobi Field
=========
The Curvature $K(p,\sigma), \sigma \subset T_p M$ 은 Geodesic이 얼마나 빨리 잔행하는 가를 결정한다. 즉, 점 $p$에서 출발하여 Tangent Space $\sigma$ 에서 얼마나 빨리 Spread 되는가를 결정한다.

Geodesic의 변화 속도를 정확하게 계측하기 위한 수단이 Jacobi Field.
Jacobi Field 는 Geodesic을 따라 정의되는 Vector Field 이다. 

## Jacobi Equation
**Gauss Lemma**를 사용하여 알아본다.
Let $M$ be a Riemannian manifold and let $p \in M$. If $exp_p$ is defined at $v \in T_p M$ and if $w \in T_v(T_p M)$, then
$$
(d \exp_p)_v w = \frac{\partial f}{\partial s}(1, 0)
$$
where $f$ is a parameterized surface given by
$$
f(t,s) = \exp_p tv(s), \;\;\; 0 \leq t \leq 1, \;\; -\varepsilon \leq s \leq \varepsilon \\
v(s) \in T_p M \;\;\text{with}\;\; v(0) = v,\; v'(0) = w
$$

이를 기반으로 다음과 같은 좀 더 일반적인 경우에 대하여 생각해 보자.

$$
(d \exp_p)_{tv} (tw) = \frac{\partial f}{\partial s}(t, 0)
$$
along the geodesic $\gamma(t) = \exp_p(tv), \;\; 0 \leq t \leq 1$
$\gamma(t)$가 Geodesic 이므로 for all $(t,s)$, $\frac{D}{\partial t}\frac{\partial f}{\partial t} = \frac{D}{\partial t} \gamma' =0$

그러므로 Lemma 4.1
$$
\frac{D}{dt}\frac{D}{ds}V - \frac{D}{ds}\frac{D}{dt}V = R(\frac{\partial f}{\partial s}\frac{\partial f}{\partial t})V
$$
에서 $V = \frac{\partial f}{\partial t} = \gamma'$ 로 놓고 즉, **Geodesic velovity**에 대하여 적용하면 
$$
\begin{align}
0 = \frac{D}{\partial s} (\frac{D}{\partial t} \frac{\partial f}{\partial t}) &= \frac{D}{dt}\frac{D}{ds} \frac{\partial f}{\partial t} - R(\frac{\partial f}{\partial s}, \frac{\partial f}{\partial t}) \frac{\partial f}{\partial t} \\
&= \frac{D}{dt}\frac{D}{dt} \frac{\partial f}{\partial s} + R(\frac{\partial f}{\partial t}, \frac{\partial f}{\partial s}) \frac{\partial f}{\partial t}
\end{align}
$$

여기서 $\frac{\partial f}{\partial s}(t,0) = J(t)$ 로 놓으면 위 방정식은 다음과 같다.
$$
\frac{D^2 J}{dt^2} + R(\gamma', J(t)) \gamma' = 0
$$
이를 **Jacobi Equation** 이라 한다.

### Jacobi Equation의 해석
Jacobi 방정식을 해석하기 위하여 다음과 같이 놓아보자. 먼저 $e_1(t), \cdots, e_n(t)$ be parallel, orthonormal fields along $\gamma$. 이때 다음과 같이 놓는다.
$$
J(t) = \sum_i f_i(t) e_i(t), \;\;\; a_{ij} = \langle R(\gamma'(t), e_i(t))\gamma'(t), e_j(t) \rangle
$$
여기서 $i,j= 1, \cdots n = \dim M$이다. 그러면
$$
\frac{D^2 J}{\partial^2 t} = \sum_i f_i''(t) e_i(t) \\
R(\gamma', J)\gamma' = \sum_j \langle R(\gamma', J)\gamma', e_j \rangle e_j = \sum_{ij} \langle f_i R(\gamma', e_i)\gamma', e_j \rangle e_j 
$$
이 방정식을 합치면 다음과 같이 $J$의 한 Component에 대하여 다음의 방정식을 얻는다.
$$
f_j''(t) + \sum_{i} a_{ij} f_i = 0
$$
이는 선형 2차 미분 방정식. 

### Remark
$\gamma'$, $t\gamma'$는 Jacobi Field along $\gamma$. 어쨌든 이것의 derivative는 0이 된다.
그러므로 **Jacobi field along $\gamma$는 $\gamma'$에 Normal 이다.**

### Example 
Let $M$ : Riemannian Manifold of **constant curvature $K$**
Let $\gamma:[0,l] \rightarrow M $ be a **normalized geodesic on $M$.**
$J$ ba a Jacobi field along $\gamma$, normal to $\gamma'$ .
이때, $|\gamma'|=1$ 이면 Lemma 3.4 (of Ch.4 : $\langle R'(X,Y,W),Z) \rangle = \langle X, W \rangle \langle Y, Z \rangle - \langle Y, W \rangle \langle X, Z \rangle$, $R=K_0 R'$에서
$$
\langle R(\gamma', J)\gamma', T \rangle = K \langle R', T \rangle = K \left(\langle \gamma', \gamma' \rangle \langle J, T \rangle - \langle J, \gamma' \rangle \langle \gamma', T \rangle \right) = K\langle J, T \rangle \\
\because \langle \gamma', \gamma' \rangle = |\gamma'|^2 = 1, \;\;\langle \gamma', T \rangle = 0
$$
따라서 $R(\gamma', J)\gamma' = KJ$
이를 Jacobi 방정식에 대입하면
$$
\frac{D^2 J}{dt^2} + KJ = 0
$$
여기서 $w(t)$ parallel field along $\gamma$ with $\langle \gamma'(t), w(t) \rangle = 0$ and $|w(t)| = 1$ 의 경우
$$
J(t) = 
\begin{cases}
\frac{\sin(t\sqrt{K})}{\sqrt{K}}w(t) &\text{if}\;\; K>0\\
tw(t) &\text{if}\;\; K=0\\
\frac{\sin(t\sqrt{K})}{\sqrt{K}}w(t) &\text{if}\;\; K < 0
\end{cases}
$$

이것이 위 방정식의 Solution with initial conditions $J(0)=0, J'(0) = w(0)$

### Construction of Jacobi field
- Jacobi field의 전제 조건
$p \in M, v \in T_p M$ and $ w \in T_v(T_p M)$ Geodesic $\gamma:[0,1] \rightarrow M$ given by $\gamma(t) = \exp_p tv$. 
Parameterized surface $f(t,s)= \exp_p tv(s)$, $v(s) T_p M$ os a curve with $v(0)=v, v'(0)=w$ 
Take $J(t) = \frac{\partial f}{\partial s}(t, 0)$ and $J(0) = 0$ 

### Proposition 2.4
Let $\gamma:[0,a] \rightarrow M$ be a geodesic and let $J$ be a Jacobi field along $\gamma$ with $J(0) = 0$. 
Put $\frac{DJ}{\partial t}(0) = w$ and $\gamma'(0) = v$.
Consider $w$ as an element of $T_{av}(T_{\gamma(0)} M)$ and construct a curve $v(s)$ with $v(0)=av, v'(0)=w$.
Put $f(t,s)= \exp_p(\frac{t}{a}v(s)), \; p=\gamma(0)$, and define a Jacobi field $\bar{J}(t)= \frac{\partial f}{\partial s }(t,0)$. Then $\bar{J}=J$ on $[0,a]$.

#### Remind 
1. $d (\exp_p)_o$ is **identity** of $T_p M$
$$
d (\exp_p)_o(v) = \frac{d}{dt}(\exp_p (tv))|_{t=0} = v = \gamma'(0)
$$

2. From Gauss Lemma
$$
(d \exp_p)_{tv} (tw) = \frac{\partial f}{\partial s}(t, 0), \;\; tw = (d \exp_p)_{tv} (tw)= t (d \exp_p)_{tv} (w) = t \cdot w
$$

####Sketch of Proof
For $s = 0$
$$
\begin{align}
\frac{D}{dt} \frac{\partial f}{\partial s} &= \frac{D}{dt} (d \exp_p)_{tv} (tw) = \frac{D}{dt} t (d \exp_p)_{tv} (w) \\
&= (d \exp_p)_{tv} (w) + t \frac{D}{dt} (d \exp_p)_{tv} (w)   \;\;\;\;\;\;\;\;\;\;\text{(1)}
\end{align}
$$
For $t = 0$
$$
\frac{D \bar{J}}{dt}(0) = \frac{D }{dt}\frac{\partial f}{\partial s}(0,0) = (d \exp_p)_o (w) = w
$$
위 두 식에서 $\frac{DJ}{dt} = \frac{D \bar{J}}{dt} = w$ 그리고 $J(0) = \bar{J}(0) = 0$ From the uniquness theorem that $J = \bar{J}$.

### Corollary 
Let $\gamma : [0,a] \rightarrow M$ be a geodesic, Then a Jacobi field $J$ along $\gamma$ with $J(0) = 0$ is given by
$$
J(t) = (d \exp_p)_{t \gamma'(0)} (tJ'(0)), \;\;\; t \in [0, a]
$$

### Proposition : Rate of Spreading of Geodesic 
Put $\frac{DJ}{dt} = J', \; \frac{D^2 J}{dt^2} = J''$
Let $p \in M$ and $\gamma:[0,a] \rightarrow M$ be a geodesic with $\gamma(0)=p, \gamma'(0)=v$.
Let $w \in T_v(T_p M)$ with $|w|=1$ and let $J$ be a Jacobian field along $\gamma$ given by
$$
J(t) = (d \exp_p)_{tv}(tw), \;\;\; 0 \leq t \leq a
$$
Then the Taylor expansion of |J(t)|^2 about $t=0$ is given by
$$
|J(t)|^2 = t^2 - \frac{1}{3} \langle R(v,w)v, w \rangle t^4 + R(t)
$$
where $\lim_{t \rightarrow 0} \frac{R(t)}{t^4} = 0$

#### Note
1. **Bianchi Identity** 및 관련 Curvature Tensor 연산
$$
\begin{align}
(X,Y,Z,T)+(Y,Z,X,T)+(Z, X,Y,T) &= 0\\
(X, Y, Z, T) &= -(Y, X, Z, T) \\
(X, Y, Z, T) &= -(X, Y, T, Z) \\
(X, Y, Z, T) &= (T, Z, X, Y) 
\end{align}
$$
이 중에서 다음이 가장 많이 사용된다.
$$
(X, Y, Z, T) = (T, Z, X, Y) 
$$

Curvature Tensor에 대한 Covariant Derivation
기본적으로 다음과 같다. ($X' = \frac{D}{dt} X$)
$$
0 = \frac{d}{dt}\langle X, Y \rangle = \langle X', Y \rangle + \langle X, Y' \rangle 
$$
따라서
$$
\frac{d}{dt} \langle R(\gamma', W)\gamma', J \rangle = \langle \frac{D}{dt} (R(\gamma', W)\gamma'), J \rangle + \langle R(\gamma', W')\gamma'), J \rangle + \langle R(\gamma', W)\gamma', J' \rangle
$$

그러므로
$$
\frac{d}{dt} \langle R(\gamma', W)\gamma', J \rangle = \frac{d}{dt} \langle R(\gamma', J)\gamma', W \rangle = \langle \frac{D}{dt} R(\gamma', J)\gamma', W \rangle + \langle R(\gamma', J)\gamma', W' \rangle \\
\begin{align}
\langle \frac{D}{dt} (R(\gamma', J)\gamma'), W \rangle &= \frac{d}{dt} \langle R(\gamma', W)\gamma', J \rangle - \langle R(\gamma', J)\gamma', W' \rangle \\
&= \langle \frac{D}{dt} (R(\gamma', W)\gamma'), J \rangle + \langle R(\gamma', W')\gamma'), J \rangle + \langle R(\gamma', W)\gamma', J' \rangle - \langle R(\gamma', J)\gamma', W' \rangle \\
&= \langle \frac{D}{dt} (R(\gamma', W)\gamma'), J \rangle + \langle R(\gamma', J)\gamma'), W' \rangle + \langle R(\gamma', W)\gamma', J' \rangle - \langle R(\gamma', J)\gamma', W' \rangle \\
&= \langle \frac{D}{dt} (R(\gamma', W)\gamma'), J \rangle + \langle R(\gamma', W)\gamma', J' \rangle \\
&= \langle R(\gamma', W)\gamma', J' \rangle \;\;\;\because \frac{D}{dt} (R(\gamma', W)\gamma') \perp J \implies \langle \frac{D}{dt} (R(\gamma', W)\gamma'), J \rangle = 0 \\
&= \langle R(\gamma', J')\gamma', W \rangle
\end{align}
$$

따라서,
$$
\nabla_{\gamma'} (R(\gamma', J) \gamma')(0) = R(\gamma', J')\gamma'(0)
$$

### Corollary 2.9
If $\gamma:[0, l] \rightarrow M$ is parametrized by arc length, (i.e. $|v|=1$) and $\langle w, v \rangle = 0$, the expression $\langle R(v,w)v,w \rangle $ is the sectional curvature at $p$ with respect to the plane $\sigma$ generated by $v$ and $w$. Therefore, in this situation,
$$
|J(t)|^2 = t^2 - \frac{1}{3} K(p, \sigma)t^4 + R(t)
$$

### Corollary 2.10
With the same conditions as in the previous colorollary
$$
|J(t)| = t - \frac{1}{6} K(p, \sigma)t^3 + \tilde{R}(t), \;\;\; \lim_{t \rightarrow 0} \frac{R(t)}{t^3} = 0
$$

#### Note
Corollary 2.10 이 **Geodesic**과 **Curvature** 의 본질적인 **Relation** 이다.

1. Parametrized surface 
$$
f(t,s) = \exp_p tv(s), \;\;\; t \in [0, \delta], \;\;\; s \in (-\varepsilon, \varepsilon)
$$
이 정의되어 있고 $v(s)$ 는 $T_p M$ 위의 Curve이며 $|v(s)|=1, v(0)= v, v'(0) = w$ 이다. 이때, 
$t \rightarrow tv(s), t \in [0, \delta]$ 에 대하여 $T_p M$의 원점에서의 $t \rightarrow tv(s)$의 속도는
$$
\left| \left( \frac{\partial}{\partial s}tv(s)) \right)(0) \right| = |tw| = t
$$

2. Jacobi Field를 고려하면 **Geodesics** $t \rightarrow \exp_p(tv(s))$ 위에서의 속도는 
$$
|J(t)| = \left| \frac{\partial f}{\partial s} (t,0)\right| = t - \frac{1}{6} K(p, \sigma)t^3 + \tilde{R}(t)
$$
$K > 0$ 에서 $\frac{1}{6} K(p, \sigma)t^3$ 만큼 $T_p M$ 에서 보다 느리다.
$K < 0$ 에서는 $\frac{1}{6} K(p, \sigma)t^3$ 만큼 $T_p M$ 보다 빠르다.

**Geodesic위에서의 속도는 $K(p, \sigma)t^3$ 에 비례하여 느리거나 빨라진다.**
Geodesic위에서의 거리 역시 $K(p, \sigma)t^3$ 에 따라 변화한다고 보아야 한다. 

## Conjugate Points
### Definition 3.1
Let $\gamma:[0,a] \rightarrow M$ be a geodesic. The point $\gamma(t_0)$ is said to be **conjugate** to $\gamma(0)$ along $\gamma$, $t_0 \in (0, a]$, if there exists a Jacobi Field $J$ along $\gamma$, not **identically zero**, with
$$
J(0) = 0 = J(t_0)
$$

또한, The maximum number of such linearly independent fields is called the **multiplicity** of the conjugate point $\gamma(t_0)$.

#### Remark
간단히 생각해도 $J(t) = t \gamma'(t)$ 형태이면 Conjugate Point가 생성되지 않는다. 그러나, $J(t) = \sin t \gamma'(t)$ 이면 $t = k \pi$ 에서 Conjugate point가 생성된다.

### Definition 3.2
The set of (first) conjugate points to the point $p \in M$, for all the geodesics that start at $p$, is called the **conjugate locus** of $p$ and is denoted by $C(p)$.

### Proposition 3.5
Let $\gamma:[0,a] \rightarrow M$ be a geodesic and put $\gamma(0)=p$.
The point $q = \gamma(t_0), t_0 \in (0,a]$ is conjugate to p along $\gamma$ if and only if $v_0 = t_0 \gamma'(0)$ is critical point of $\exp_p$.
In addition, the multiplicity of $q$ as a conjugate point of $p$ is equal to the dimension of the kernel of the linear map $(d\exp_p)_{v_0}$.

### Proposition 3.6
Let $J$ be a Jacobi field along the geodesic $\gamma:[0,a] \rightarrow M$. Then
$$
\langle J(t), \gamma'(t) \rangle = \langle J'(0), \gamma'(0) \rangle t + \langle J(0), \gamma'(0) \rangle, \;\;\; t \in [0,a]
$$

#### Note
Jacobi Equation
$$
\frac{D^2 J}{dt^2} + R(\gamma', J(t)) \gamma' = 0
$$
이것을 Inner Product형으로 바꾸어 써보면
$$
\langle J'', \gamma' \rangle + \langle R(\gamma', J) \gamma', \gamma') = 0
$$
Inner Product의 우측항 $\gamma'$를 $T \in T_p M$ 으로 바꿀 수도 있다.

#### Sketch of proof
변수 $t$를 생략하고 Note를 참조하면
$$
\langle J', \gamma' \rangle' = \langle J'', \gamma' \rangle = - \langle R(\gamma', J) \gamma', \gamma' \rangle = 0 \;\;\;\because R(\gamma', J) \gamma' \perp \gamma'
$$
Therefore $\langle J', \gamma' \rangle' = $\langle J'(0), \gamma'(0) \rangle'$. In addition,
$$
\langle J, \gamma' \rangle' = \langle J', \gamma' \rangle = \langle J'(0), \gamma'(0) \rangle 
$$
위 식을 t에 대하여 적분하면
$$
\langle J, \gamma' \rangle = \langle J'(0), \gamma'(0) \rangle t + \langle J(0), \gamma'(0) \rangle 
$$

### Corollary 3.7