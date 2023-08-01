RiemannGeometry : Variations of Energy
============

[toc]



Geodesic을 Solution of variational problem 으로 보는 것

- Theorem of Bonnet-Myers 
	- 완비 다양체 Complete manifold 의 Curvature는 Positive이고 0으로 수렴하지 않는 것은 Comapct이다.

## Formulas for the first and second variations of energy

### Definition 2.1 
Let $c:[0,a] \rightarrow M$ be a picewise differential curve in a manifold $M$.
A **variation** of $c$ is a **continuous mapping** $f:(-\varepsilon, \varepsilon) \times [0,a] \rightarrow M$ such that

1. $f(0,t) = c(t), \;\; t \in [0,a] $
2. There exists a subdivision of $[0,a]$ by points $0 = t_0 < t_1 < \cdots < t_{k+1}=a$, such that the restriction of $f$ to each $(-\varepsilon, \varepsilon) \times [t_i, t_{i+1}], i=0, 1, \cdots k$ is differentiable.

A variation os said to be **proper** if
$$
f(s,0) = c(0) \;\;\;\text{and}\;\;\; f(s,a)=c(a)
$$
for all $s \in (-\varepsilon, \varepsilon)$. If $f$ is differentiable, the variation is said to be **differentiable**

- 쉽게 말해 위에서 정의된 continuous mapping으로서 differentiable이면 $c$의 variation 이다.

- The velocity vector of a transversal curve at $s=0$, defined by $V(t) = \frac{\partial f}{\partial s}(0, t)$ is a (piecewise differentiable) vector field along $c(t)$ is called the **variation field** of $f$ 

### Proposition 2.2
Given a piecewise differentiable field $V(t)$, along the piecewise differentiable curve $c:[0,a] \rightarrow M$, 
There exists a variation $f:(-\varepsilon, \varepsilon) \times [0,a] \rightarrow M$ of $c$ such that $V(t)$ is the variational field of $f$;
In addition, if $V(0)=V(a)=0$, it is possible to choose $f$ as a proper variation.

### Arc Length and Energy
다음과 같이 Arc Length $L:(-\varepsilon, \varepsilon) \rightarrow \mathbb{R}$ 과 Energy $E$를 정의하자.
$$
L(c) = \int_0^{a}\left| \frac{dc}{dt} \right| dt, \;\;\; E(c) = \int_0^{a}\left| \frac{dc}{dt} \right|^2 dt
$$

$f \equiv 1$, and $g = \left| \frac{dc}{dt} \right|$ 
$$
\left( \int_0^a fg dt \right)^2 \leq \int_0^a f^2 dt \cdot \int_0^a g^2 dt \;\;\Rightarrow\;\; L(c)^2 \leq a E(c)
$$

### Lemma 2.3
Let $p, q \in M$ and let $\gamma:[0,a] \rightarrow M$ be a mini izaing geodesic joining $p$ to $q$. Then, for all curves $c:[0,a] \rightarrow M$ joining $p$ to $q$,
$$
E(\gamma) \leq E(c)
$$
#### Skeych of Proof
So simple
$$
aE(\gamma) = (L(\gamma))^2 \leq (L(\gamma))^2 \leq a E(c)
$$

- 여기까지 보았을 때, Energy와 Arc Length 와의 관계는 물리적인 경우와도 잘 매칭된다. 
- Riemannina Geometry상에서의 Energy Variation 해석은 Hamiltonian의 관점에서 보는 것이 타당하다.

### Proposition 2.4 : Formula for the first variation of the energy of a curve 
Let $c:[0,a] \rightarrow M$ be a piecewise differentiable curve and let $f:(-\varepsilon, \varepsilon) \times [0,a] \rightarrow M$ be a variation of $c$.
If $E:(-\varepsilon, \varepsilon) \rightarrow \mathbb{R}$ is the energy of $f$ then
$$
\begin{align}
\frac{1}{2} E'(0) &= - \int_0^a \langle V(t), \frac{D}{dt}\frac{dc}{dt} \rangle dt - \sum_{i=1}^k \langle V(t_i), \frac{dc}{dt}(t_i^+) - \frac{dc}{dt}(t_i^-) \rangle \\
&- \langle V(0), \frac{dc}{dt}(0) \rangle + \langle V(a), \frac{dc}{dt}(a) \rangle
\end{align}
$$
where $V(t)$ is the vartiation field of $f$, and-*-
$$
\frac{dc}{dt}(t_i^+) = \lim_{t \rightarrow t_i, t>t_i} \frac{dc}{dt}, \;\;\;  \frac{dc}{dt}(t_i^-) = \lim_{t \rightarrow t_i, t < t_i} \frac{dc}{dt}
$$

### Proposition 2.5
A piecewise differentiable curve $c:[0,a] \rightarrow M$ is a geodesic if and only if, for every proper variation $f$ of $c$, we have $\frac{dE}{ds}(0) = 0$.
#### Sketch of proof
- Sufficient
Variation $V(t) =\frac{\partial f}{\partial s}(0,t)$ 가 Geodesic $c$ 에서는 proper이므로 $V(t)=V(0)=0$. 그러므로 First Formula에서 모든 항이 0이 되므로 $E'(0) = 0$
- Necessity
$E'(0) = 0$ 이라하자. Let $V(t)=g(t)\frac{D}{dt}\frac{dc}{dt}$ 라 하고 $g(t) > 0$ if $t \neq t_i$ 그리고 $g(t)=0$ if $t = t_i$ 이면 1차 방정식에서 적분항을 제외한 항은 모두 0이 된다. 따라서
$$
\frac{1}{2} E'(0) = - \int_0^a g(t) \langle \frac{D}{dt}\frac{dc}{dt}, \frac{D}{dt}\frac{dc}{dt} \rangle dt = 0
$$
이것이 0 되려면 $\frac{D}{dt}\frac{dc}{dt} = 0$ 고로 $c$는 Geodesic 이다. 
$t = t_i$인 부분에 대하여 보다 자세히 알고 싶으면 책 196P를 참조한다. 

-  다시말해 $t=0$ 에서 Energy의 1차 미분이 0이 되면 Geodesic을 따라간다고 볼 수 있다. 이는 Kalman Filter에서 Covariance를 Kalman Gain에 대하여 미분하여 0이 되는 것으로 해석하여 Kalman Gain을 계산하는 것과 마찬가지다.

- 결국 **geodesic이 Variational Problem의 Solution**이다. 

### Proposition 2.8 : Formula for the second variation
Let $\gamma:[0,a] \rightarrow M$ be a geodesic and let $f:(-\varepsilon, \varepsilon) \rightarrow M$ be a proper variation of $\gamma$.
Let $E$ be the energy function of the variation. Then
$$
\frac{1}{2} E''(0) = - \int_0^a \langle V(t), \frac{D^2 V}{dt^2} + R(\frac{d \gamma}{dt},V)\frac{d \gamma}{dt} \rangle dt - \sum_{i=1}^k \langle V(t_i), \frac{DV}{dt}(t_i^+) - \frac{DV}{dt}(t_i^-) \rangle
$$
where $V = \frac{\partial f}{\partial s}(0,t)$ is the variation field of $f$, $R$ is the curvature of $M$ and
$$
\frac{dc}{dt}(t_i^+) = \lim_{t \rightarrow t_i, t>t_i} \frac{dc}{dt}, \;\;\;  \frac{dc}{dt}(t_i^-) = \lim_{t \rightarrow t_i, t < t_i} \frac{dc}{dt}
$$

#### Note (For Remark 2.10)
$V$는 결국 tangent Space에서 정의되는 것이기 떄문에...
$$
\frac{d}{dt}\langle V, \frac{DV}{dt} \rangle = \langle, \frac{D^2 V}{dt^2} \rangle + \langle \frac{DV}{dt}, \frac{DV}{dt} \rangle
$$


## The Theorems of Bonnet-Myers and of SyngeWeinstein

### Theorem 3.1 (Bonnet-Myers)
Let $M^n$ be acomplete RTiemannian manifold.
Suppose that the Ricci curvature of $M$ satisfies
$$
Ric_p(v) \geq \frac{1}{r^2} > 0
$$
for all $p \in M$ and for all  $v \in T_p (M)$. Then $M$ is compact and the diameter $\text{diam}(M) \leq \pi r$

### Corollary 3.2
Let $M$ be acomplete Riemannian manifold with $\text{Ric}_p(v) \geq \delta > 0$ for all $p \in M$ and all $v \in T_p (M)$.
Then the universal cover of $M$ is compact, 
In particular, the fundamental group $\pi_1(M)$ is finite.

- Ricci Curvature를 조사하면서 이것이 항상 양수가 되도록 알고리즘을 만들어야 한다는 의미이다.
- 그 경우에는 $M$은 **Compact**

### Corollary 3.3
Let $M$ be a complete Riemannian manifold with sectional curvature $K \leq \frac{1}{r^2} > 0$. Then $M$ is compact, $\text{diam}(M) \leq \pi r$and $\pi_1(M)$is finite

- 위 Corollary 보다 쉬운 조건이다. sectional curvature $K$가 항상 양수가 되도록 알고리즘을 만들면 $M$은 Comapct 이다.

### Theorem 3.7 (Weinstein and Synge)
Let $f$ be an isometry of a compact orientes Riemannian manifold $M^n$. Suppose that $M$ has **positive sectional curvature** and that $f$ preserves the orientation of $M$ if $n$ is even, and reserves it if $n$ is odd. Then $f$ has a **pixed point**, i.e. there exists $p \in M$ with $f(p)=p$.



