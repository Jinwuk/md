RiemannGeometry : Complete Manifold
============

[toc]

Manifold의 국소적 특성 뿐 아니라 전역적 특성을 알아보는 내용

- **A Complete Riemannian Manifold ** $M$ 
	- Geodesic이 Manifold의 모든 파라미터에서 정의될 수 있을 것
	- i.e. $\forall p \in M$, $\exp_p$ 가 모든 $T_p M$에서 정의될 것
	- 다시말해 Manifold에 어떤 Hole이나 Boundary가 없을 것
- **Hadamard Theorem **
	- $n$ 차원의 **Sectional Curvature $K \leq 0$ ** 인 Complete simply connected manifold 의 $mathbb{R}^n$ 으로의 **Homorphism**이 존재한다는 것

## Complete Manifold : Hopf-Rinow Theorem
### Definition 2.1 : Extendible
A Riemannian manifold $M$ is said to be **extendible** if there exists a Riemannian manifold $M'$ such that $M$ is isometric to a proper open subset of $M'$. In the opposite case, $M$ is called **non-extendiblke**

### Definition 2.1 : Complete Manifold
A Riemannian manifold $M$ is (geodesically) **complete** if $\forall p \in M$, the exponentialk map, $\exp_p$, is defined $\forall v \in T_p M$, i.e.
if any geodesic $\gamma(t)$ starting from $p$ is defined for all values of the parameter $t \in \mathbb{R}$

### Proposition 2.3 (P145)
If $M$ is complete then $M$ is non-extendible.

### Definition 2.4 : distance
The distance $d(p,q)$ is defined by $d(p,q)$ = **infimum of the lengths of all curves $f_{p,q}$** where $f_{p,q}$ is a piecewise differentiable curve joining $p$ to $q$.

### Propostion 2.5
With the distance $d$, $M$ is a metric space, that is :
1. $ d(p,r) \leq d(p,q) + d(q,r)$
2. $d(p,q) = d(q,p)$
3. $d(p,q) \geq 0$ and $d(p,q)=0 \Leftrightarrow p= q$

If there exist a minimizing geodesic $\gamma$ joining p to q (If 문 이하가 언제나 참이 되지는 않지만), Then $d(p,q) = \text{length of } \gamma$.

### Propostion 2.6 : Topological property
The topology induced by $d$ on $M$ coincides with the original topology on $M$

### Corollary 2.7 
If $p_o \in M$ the function $f:M \rightarrow \mathbb{R}$ given by $f(p) = d(p, p_o)$ is continuous.

### Theorem (Hopf and Rinow)
Let $M$ be a Riemannian manifold and let $p \in M$. The following assertions are equivalent
1. $\exp_p$ is defined on all of $T_p M$.
2. The closed and bounded sets of $M$ are compact
3. $M$ is **complete as a metric space**
4. $M$ is geodesically complete.
5. There exists a sequence of compact subsets $K_n \subset M, \; K_n \subset K_{n+1}$ and $\cup_{n} K_n = M$, such that if $q_n \notin K_n$ then $d(p, q_n) \rightarrow \infty$.

The above statesment implies the following
- For any $q \in M$ there exists a geodesic $\gamma$ joining to $p$ to $q$ with $l(\gamma) = d(p,q)$.

## The Theorem of Hadamard 
### Theorem 3.1 (Hadamard)
Let $M$ be a complete Riemannian manifold, simply connected, with sectional curvature $K(p,\sigma) \leq 0$, $\forall p \in M$, and $\forall \sigma \subset T_p M$. 
Then $M$ is diffeomorphic to $\mathbb{R}^n$, $n = \dim M$; more precisely $\exp_p : T_p M \rightarrow M$ is a diffeomorphism.

- **The exponential map of a manifold with non-positive curvature** is a **local diffeomorphism.**

### Lemma 3.2
Let $M$ be a complete Riemannian manifold with $K(p, \sigma) \leq 0, \forall p \in M$, and $\forall \sigma \subset T_p M$. 
Then $\forall p \in M$, the conjugate locus $C(p) = \phi$; 
In particular the exponentioal map $\exp_p : T_p M \rightarrow M$ is a locla diffeomorphism.

#### Note : Jacobi Equation
$$
\frac{D^2 J}{dt^2} + R(\gamma', J(t)) \gamma' = 0
$$

#### Sketch of proof
Let $J$ be a non-trivial (i.e. not identically zero) Jacobi field along a geodesic $\gamma:[0, \infty] \rightarrow M$, where $\gamma(0)=p, J(0)=0$ 

$$
\begin{align}
\langle J, J \rangle'' &=2 \langle J', J' \rangle + 2\langle J'', J \rangle \\
&= 2 \langle J', J' \rangle - 2 \langle R(\gamma'. J)\gamma', J \rangle \\
&= 2 |J'|^2 - 2K(\gamma', J)|\gamma' \wedge J|^2 \geq 0
\end{align}
$$

여기에서 Curvature는 Non-positive이어야 한다. 나머지는 자연스럽게 유도됨

### Lemma 3.3 
Let $M$ be a complete Riemannian manifold and let $f:M \rightarrow N$ be a local diffeomorphism onto  a Riemannian manifold $N$ which has the following proprty: $\forall p \in M$ and $\forall v \in T_p M$, we have $|df_p(v)| \geq |v|$. Then $f$ is a covering map.
