Riemannian Geometry (Geodesic)
=====================

[toc]


## Geodesic Flow

$M$ will be a Riemannian manifold, together with its Riemannian connection.



### Definition : Geodesic

A parameterized curve $\gamma:I \rightarrow M$ is a geodesic at $t_0 \in I$, if

$$
\frac{D}{dt}\left( \frac{d \gamma}{dt}\right) = 0 \;\;\;\text{at the point } t_0
$$

즉, 시간에 따른 Parametrized curve의 변화량이 일정할때 such that $\gamma(t, \cdot) = t \cdot h(\cdot)$ 이것의 시간의 Covariant Derivation 이 0 이면 Geodesic



- If $\gamma:I \rightarrow M$ is a **geodesic**, then

$$
\frac{D}{dt} \langle \frac{d \gamma}{dt}, \frac{d \gamma}{dt} \rangle = 2 \langle \frac{D}{dt} \frac{d \gamma}{dt}, \frac{d \gamma}{dt}\rangle = 0
$$



간단히 생각하면 $\frac{d \gamma}{dt} = c \neq 0 $ 이어서 거리가

$$
s(t) = \int_{t_0}^t \left| \frac{d \gamma}{ds} \right| ds = c(t - t_0)
$$

그런데, 이것은 Differential 1-form 으로도 계산 가능하다.



A parameterized curve $\gamma$ 가 Coordinate system $(U, x)$ about $\gamma(t_0)$ 근방에 다음과 같이 정의되었다고 가정하자.

$$
\gamma(t) = (x_1(t), \cdots x_n(t))
$$

Geodesic이 되려면 $V = \sum_i v^i X_i$ 에서 

$$
\frac{DV}{dt} = \sum_k \left( \frac{dv^k}{dt} + \sum_{i,j} v^j \frac{dx_i}{dt} \Gamma_{ij}^k \right)X_k
$$

이므로 $\frac{d\gamma}{dt} = \sum_i \frac{dx_i}{dt} X_i$, where $X_i = \frac{\partial}{\partial x_i}$ 에서

$$
0 = \frac{D}{dt} \frac{d\gamma}{dt} = \sum_k \left( \frac{d^2 x_k}{dt} + \sum_{i,j} \frac{dx_i}{dt} \frac{dx_j}{dt} \Gamma_{ij}^k \right) X_k
$$

따라서 Geodesic 조건은 

$$
\frac{d^2 x_k}{dt} + \sum_{i,j} \Gamma_{ij}^k \frac{dx_i}{dt} \frac{dx_j}{dt} = 0
$$



### Tangent Bundle $TM$

Tangent Bundle $TM$ 은 $(q, v), q \in M, v \in T_q M$ 을 의미. 즉, 각 점 $q$에 해당하는 Tangent vector $v$로 정의되는 **Tangent Vector들의 모임**이다.

이런 것을 도입하는 이유는 Geodesic위의 일반적인 법칙을 세우기에 Tangent Bundle의 개념이 편리하기 때문이다.

Tangent Bundle이 도입되면 Geodesic위의 임의의 Tangent space에서 세워진 법칙은 다은 Tangent Space에서도 동일하게 적용될 것이다. (좌표변환에도 불구하고..)



- Tangent Space $T_q M$ 은 $t \rightarrow \gamma(t)$ 로 결정

- Tangent Bundle $TM$ 은 $t \rightarrow (\gamma(t), \frac{d\gamma}{dt}(t))$ 로 결정. 따라서 Geodesic에서 

$$
\begin{cases}

\frac{dx_k}{dt} &= y_k \\

\frac{dy_k}{dt} &= -\sum_{i,j} \Gamma_{i,j}^k y_i y_j

\end{cases}
$$



### Definition : flow

The mapping $\varphi_t:V_0 \rightarrow V$ given by $\varphi_t(q) = \varphi(t,q)$ 를 ** the flow of $X$ on $V$ ** 라고 한다.

- There exist a **unique vector field $G$ on $TM$** whose trajectories are of the form **$t \rightarrow (\gamma(t), \gamma(t)')$**, where $\gamma$ is a geodesic on $M$ 

- 위와 같은 Vector Field를 **Geodesic Vector field** 그리고 그것의 flow를 **Geodesic flow on $TM$** 이라고 한다.

- **Homogeneity of Geodesic** 

$$
\gamma(t, q, av) = \gamma(at, q, v)
$$



### Exponential map (on $\mathcal{U}$)

Let $p \in M$ and let $\mathcal{U} \subset TM$  be an open set given by Proposition 2.7 (p 64). Then the map $\exp: \mathcal{U} \rightarrow M$ given by 

$$
\exp (q,v) = \gamma(1, q, v) = \gamma(|v|, q, \frac{v}{|v|}) , \;\; (q,v) \in \mathcal{U}
$$

- 특징

$$
\exp_q : B_{\varepsilon}(0) \subset T_qM \rightarrow M
$$

by $\exp_q(v) = \exp(q,v)$



### Proposition 

Given $q \in M$, there exists an $\varepsilon > 0$ such that $\exp_q : B_{\varepsilon} (0) \subset T_q M \rightarrow M$ is a diffeomorphism of $B_{\varepsilon}(0)$ onto an open subset of $M$ 

$$
\begin{align}

d(\exp_q)_o(v) &= \frac{d}{dt}(\exp_q (tv))|_{t=0} = \frac{d}{dt}(\gamma(1, q, tv))|_{t=0} &\;\;\; \text{by definition of exponential map} \\

&= \frac{d}{dt}(\gamma(t, q, v))|_{t=0} = v &\;\;\;\text{by definition of exponential map}  

\end{align}
$$

Exponential Map의 시간에 대한 미분이 $v$ 이므로 이것은 $T_q M$을 만들기 위한 Parametrized Curve와 같다. 그러므로 Local Diffeomorphism.

#### Note

사실, 정의에 의해 어떤 점에서건 속도 벡터가 $\alpha$ 로 주어진 $\exp_q(\alpha)$의 Diffeomorphism $d\exp_q(\alpha) = \alpha$ 이다. 

꼭 $t=0$ 가 아니어도 위, Proposition에 의해 모든 점에서 $B_{\varepsilon}(x) \subset T_q M$ 에서 성립한다. 



## Minimizing Properties of Geodesics

### Definition : piecewise differentiable curve

piecewise differentiable curve is continuous mapping $c:[a,b] \rightarrow M$ of a closed interval $[a,b] \subset \mathbb{R}$ into $M$ satisfying thefollowing conditrion: there exists a partition $a = t_0< t_1 < \cdots < t_{k-1} < t_k = b$ of $[a, b]$ such that the restrictions $\left. c \right|_{[t_i, t_{i+1}]}, \; i=0, \cdots , k-1$ are differentiable.

 - $\lim_{t \rightarrow t_i^+} c'(t)$ 와 $\lim_{t \rightarrow t_i^+} c'(t)$ 사이의 각도를 **vetex angle** 이라고 함



다음 그림 처럼 정의된다.



![Fig02](http://jnwhome.iptime.org/img/DG/RMGM_002.png)



- Piecewise differentiable curve를 도입하는 이유는 Geodesic에 변화를 주었을 경우 Geodesic이 다양체 상의 두점을 잇는 최소 거리임을 증명하기 위함임

- 마치 Hamiltonian 증명돠 비슷하게 하기 위함



### Definition

A segment of geodesic $\gamma:[a,b] \rightarrow M$ is called **minimizing** if $l(\gamma) \leq l(c)$



### Definition

Let $A$ be a connected set in$\mathbb{R}^2$, $U \subset A \subset \bar{U}$, $U$ is open, such that the boundary $\partial A$ of $A$ is a piecewise differentiable curve with vertex angles different from $\pi$. A **parametrized surface in $M$** is a differentiable mapping $s: A \subset \mathbb{R}^2 \rightarrow M$



|  |   |

|---|---|

| ![Fig02](http://jnwhome.iptime.org/img/DG/RMGM_003.png) | ![Fig03](http://jnwhome.iptime.org/img/DG/RMGM_004.png) |



- Vector field $V$ along $s$ is a mapping which associates to each $q \in A$, a vector $V(q) \in T_s(q)M$

- $V(q) = (ds \frac{\partial}{\partial u}, ds \frac{\partial}{\partial v}) = (\frac{\partial s}{\partial u}, \frac{\partial s}{\partial v})$ is a velocity vector along $s$

- Covariant Derivative $\frac{D}{\partial u}(u,v)$ is also possible to be defined.



### Lemma : Symmetry

If $M$ is a differentiable manifold with a symmetric connection and $s : A \rightarrow M$ is a parameterized surface then :

$$
\frac{D}{\partial v} \frac{\partial s}{\partial u} = \frac{D}{\partial u} \frac{\partial s}{\partial v}
$$



#### Note : Tangent space to $T_p M$

For $v \in T_p M $ with $T_p M $ **itself.** wmr, Tangent Space 자체에 대한 Tangent Space (실은 같은 공간이다.)에 대한 표현은 다음과 같다.

$$
T_pM \approx T_v(T_p M)
$$

### Lemma : Gauss

Let $p \in M$ and let $v \in T_p M$ such that $\exp_p v$ is defined. Let $w \in T_p M \approx T_v(T_p M).$ Then

$$
\langle (d \exp_p)_v (v), (d\exp_p)_v(w) \rangle = \langle v, w \rangle
$$



#### Note

$$
d(exp_q)_o(v) = v
$$

에서, 

$$
\langle (d \exp_p)_o (v), (d\exp_p)_o(w) \rangle = \langle v, w \rangle
$$

Set $\exp_p u$ is defined for 

$$
u = t v(s), \;\; 0 \leq t \leq 1, \;\; -\varepsilon < s < \varepsilon
$$

where $v(s)$ is a curve in $T_pM$ with $v(0) = v, v'(0)=w_N$ and Let $f(t,s)= \exp_p tv(s)$ then

$$
\begin{align}

\frac{\partial f}{\partial s} &= \frac{\partial }{\partial s} \exp_p tv(s) = (d\exp_p)_v (tv'(s))\\

\frac{\partial f}{\partial t} &= \frac{\partial }{\partial t} \exp_p tv(s) = (d\exp_p)_v (v(s)) 

\end{align}
$$



![Fig05](http://jnwhome.iptime.org/img/DG/RMGM_005.png)



#### Sketch of Proof

먼저 $w = w_T + w_N$ 으로 놓으면 $w_T$는 Tangent Space 위에 있고 $v$에 평행이다. 그러므로 

$$
\langle (d \exp_p)_v (v), (d\exp_p)_v(w_T) \rangle = \langle v, w_T \rangle
$$



For $w = w_N$, 에서 Riemannian 계량이 0이면 증명 성립

$$
\langle \frac{\partial f}{\partial s}, \frac{\partial f}{\partial t} \rangle (1,0) = \langle (d\exp_p)_v(w_N), (d\exp_p)_v(v) \rangle 
$$

For all $(t,s)$,

$$
\frac{\partial}{\partial t} \langle \frac{\partial f}{\partial s}, \frac{\partial f}{\partial t} \rangle = \langle \frac{D}{\partial t}  \frac{\partial f}{\partial s}, \frac{\partial f}{\partial t} \rangle + \langle \frac{\partial f}{\partial s}, \frac{D}{\partial t} \frac{\partial f}{\partial t} \rangle
$$

여기서 $\frac{\partial f}{\partial t}$이 **Geodesesic의 Tangent vector이기 때문**에 이것의 Covariant Differential 은 0. 

따라서,

$$
\langle \frac{\partial f}{\partial s}, \frac{D}{\partial t} \frac{\partial f}{\partial t} \rangle = 0
$$



Symmetry에 의해 첫번째 항은 

$$
\langle \frac{D}{\partial t}  \frac{\partial f}{\partial s}, \frac{\partial f}{\partial t} \rangle = \langle \frac{D}{\partial s}  \frac{\partial f}{\partial t}, \frac{\partial f}{\partial t} \rangle = \frac{1}{2} \frac{\partial }{\partial s} \langle \frac{\partial f}{\partial t}, \frac{\partial f}{\partial t} \rangle = 0 \;\;\; \because \langle \frac{\partial f}{\partial t}, \frac{\partial f}{\partial t} \rangle = \text{constant}
$$

그러므로



$$
\frac{\partial}{\partial t} \langle \frac{\partial f}{\partial s}, \frac{\partial f}{\partial t} \rangle = 0
$$

따라서 모든 $(t,s)$ 에 대하여 $\langle \frac{\partial f}{\partial s}, \frac{\partial f}{\partial t} \rangle$는 $s$의 함수이다. $s=0$의 값을 알기 위해 극한값을 취하면



$$
\lim_{t \rightarrow 0} \frac{\partial f}{\partial s}(t,0) = \lim_{t \rightarrow 0}(d\exp_p)_{tv} tw_N = 0
$$

그러므로 $\langle \frac{\partial f}{\partial s}, \frac{\partial f}{\partial t} \rangle (1,0) = 0$  따라서,

$$
\langle \frac{\partial f}{\partial s}, \frac{\partial f}{\partial t} \rangle (1,0) = \langle (d\exp_p)_v(w_N), (d\exp_p)_v(v) \rangle = \langle w_N,v \rangle  = 0 
$$



#### Note

$\exp_p$ 가 the origin in $T_p M $ 근방 $V$의 Diffeormorphism 이면

- **Normal Neighborhood** : $\exp_p V = U$



만일 $B_{\varepsilon}(0)$ 가 $\bar{B_{\varepsilon}(0)} \subset V$  

- **Normal ball** with center $p$ and radius $\varepsilon$ : $\exp_p B_{\varepsilon}(0) = B_{\varepsilon}(p)$

- **Normal Sphere** $S_{\varepsilon}(p)$ : Normal ball의 Bondary를 따라 $p$에서 시작된 Geodesic에 Orthogonal인 Hyper Plane



![Fig06](http://jnwhome.iptime.org/img/DG/RMGM_006.png)





### Proposition : Geodesic의 Length Minimize 특성

Let $p \in M$, $U$ a normal neighborhood of $p$, and $B \subset U$ a normal ball of center $p$.

Let $\gamma:[0,1] \rightarrow B$ be a geodesic segment with $\gamma(0) = p$. 

If $c:[0,1] \rightarrow M$ is any piecewise differentiable curve joining $\gamma(0)$ to $\gamma(1)$,

then $l(\gamma) \geq l(c)$ and if equality holds then $\gamma([0,1]) = c([0,1])$

#### Note

맨 마지막 줄 $l(\gamma) \geq l(c)$ 이 이 명제의 의미. $\gamma$는 geodesic.



#### Sketch of proof

Set $t \rightarrow v(t)$ be a **curve** in $T_p M$ with $|v(t)| = 1$, and

$r:(0,1] \rightarrow \mathbb{R}$ is a positive piecewise differentiable **function**, so that the curve $c(t)$ is that

$$
c(t) = \exp_p(r(t) \cdot v(t)) = f(r(t), t)
$$

then

$$
\frac{dc}{dt} = \frac{\partial f}{\partial r}r'(t) + \frac{\partial f}{\partial t}
$$

Gauss Lemma 에서 $\langle \frac{\partial f}{\partial r}, \frac{\partial f}{\partial t} \rangle = 0$, Since $\left| \frac{\partial f}{\partial t} \right| =0 $,

$$
\left| \frac{dc}{dt} \right|^2 = \left| r'(t) \right|^2 + \left| \frac{\partial f}{\partial t} \right|^2 \geq \left| r'(t) \right|^2
$$

and so

$$
\int_{\varepsilon}^1 \left| \frac{dc}{dt} \right| dt \geq \int_{\varepsilon}^1 \left| r'(t) \right| dt \geq \int_{\varepsilon}^1 r'(t) dt = r(1) - r(\varepsilon)
$$

위에서 $r(1) = l(\gamma)$ 이므로 $\varepsilon \rightarrow 0$ 에서 $l(c) \geq l(\gamma)$  

- 일단 위에서 $l(c) > l(\gamma)$.

- $l(c) = l(\gamma)$ 의 경우는 $\left| \frac{\partial f}{\partial t} \right| = 0$ , 이 경우는 $v= const$ 그리고 $|r'(t)| = r(t) > 0$ 이어서 $c(t)$가 monotone reparametrizatioin of $\gamma$ 즉, $c([0,1]) = \gamma([0,1])$을 의미 



#### Remark

임의의 $q_1, q_2 \in W$ 를 지나는 **a unique minimizaing geodesic** 이 존재한다는 것 such that $\gamma'(0) = v$



### Theorem 3.7 : Normal Neighborhood

For any $p \in M$, there exist a nrighborhood $W$ of $p$ and a number $\delta > 0$, such that 

$\forall q \in W$, $\exp_q$ is a diffeomorphism on $B_{\delta}(0) \subset T_q M$ and $\exp_q(B_\delta(0)) \supset W$,

that is, $W$ is a **normal neighborhood** of each of its points



![Fig01](http://jnwhome.iptime.org/img/DG/RMGM_001.png)



- 즉 위의 그림 처럼 $p$의 근방에 속하는 $q$의 $T_q M$에 Open Ball $B_{\delta}(0) \subset T_p M$ 을 놓고 이것의 Exponential Map $\exp_q(B_\delta(0))$ 이 $p$의 Neighborhood를 포함할 수 있는 $p$의 Neighbor hood을 **Normal Neighborhood** 라고 한다.

- Normal Neighborhood가 되면 $p$를 중심으로 하는 Geodesic 혹은 $v$에 대한 미분에 대하여 $q$와의 관계를 알 수있게 되므로 미분을 보다 확장적으로 사용할 수 있게 된다.

- 예를 들어 $q_1, q_2$ 를 지나는 Geodesic의 Length가 $\delta$ 보다 작다면 There exists a **unique** $\; v \in T_{q_1} M$  that depends differetiably on $(q_1, q_2)$ such that $\gamma'(0)= v$.

	- 이것이 $p$의 근방 $W$에서 전반적으로 만족될 수 있다면 **totally normal neighborhood of $p$**.



## Convex Neighborhoods

위에서 정의한 totally normal neighborhood of $p$ 가 $W$에서 존재하지 않을 수 있다, 왜냐하면 $M$의 모양에 따라 Convex가 아닐 수 있기 때문이다.

- 만일, $p$의 Neighborhood $W$가 **Convex** 이면 **$W$ 내의 어떤 점을 잇는 $Geodesic$은 모두 $W$ 내에 존재하게 될 것이다.** 

	- 이것은 Euclidean Space에서의 Convex의 특성과 유사하다.



### Lemma 4.1 

$\forall p \in M, \exists c > 0$ such that any geodesic in $M$ that is tangent at $q \in M$ to the geodesic sphere $S_r(p)$ of radius $r < c$  stays out of the geodesic ball $B_r(p)$ for some neighborhood of $q$



#### Sketch of proof

Let $W$ be a totally normal neighborhood of $p$ 그러면 $q \in W$ 이다. 이떄 Tangent Bundle $T_1 W$를 다음과 같이 정의하자

$$
T_1 W = \{(q,v); q\in W, v \in T_q M, |v| =1 \}
$$

Let $\gamma:I \times T_1W \rightarrow M, \; I= (-\varepsilon, \varepsilon)$ be the differentiable mapping such that $t \rightarrow \gamma(t,q,v)$ is **the geodesic**. $t=0$ 에서 $q$를 지나며 속도는 $v, |v|=1$ 이다.

Define $u(t,q,v) = \exp_p^{-1}(\gamma(t,q,v)))$ and 

$$
F:I \times T_1 W \rightarrow \mathbb{R}, \;\;\; F(t,q,v) = | u(t,q,v) |^2
$$

이때 , $F$는 the square of the distance from $p$ to a point that is moving along the geodesic $\gamma$ 이다.

즉, Geodesic은 $\gamma$ 하나 뿐이며 앞에서 살펴 보았듯이 $q$를 지나면서 어떤 Boundary에 접하고 있다. 



그리고 $u(t,q,v)$ 자체는 하나의 위치를 가리키고 있으며 $u(t, q, v) = tv$ 이다.

여기서 $F$를 미분하면 

$$
\begin{align}

\frac{\partial F}{\partial t} &= 2 \langle \frac{\partial u}{\partial t}, u \rangle \\

\frac{\partial^2 F}{\partial t^2} &= 2 \langle \frac{\partial^2 u}{\partial t^2}, u \rangle + 2 \left| \frac{\partial u}{\partial t} \right|^2

\end{align}
$$



마지막으로 let $r > 0$ be chsoen so that

$$
\exp_p B_r(0)= B_r (p) \subset W
$$



만일 **Geodesic $\gamma$가 $S_r(p)$에 Tangent at $q=\gamma(0,q,v)$** 이면 Gauss Lemma에서 

$$
\langle \frac{\partial u}{\partial t}, u \rangle (0, q, v) = 0
$$

Since $\frac{\partial u}{\partial t} = \gamma'$ ($\gamma$가 $S_r(p)$에 Tangent 이므로) 가 $u (0, q, v) = v$ 이므로 ($q$ 점에서 속도 $v$로 나가기 떄문) 서로 Orthogonal 하다.



그러므로 $\frac{\partial F}{\partial t} (0, q, v) = 0$ 최소 $F$는 Constant 값이다. 또한 $r$이 충분히 작으면 $F$는 Minimum point이다. (이로서 증명은 완성)

다시 살펴보면, $u(t,p,v) = tv$ (즉, $p$에서 $q$점을 보게 되면 최소 $ tv$  만큼의 거리가 나타난다.) 그러므로

$$
\frac{\partial^2 F}{\partial t^2} (0, p, v) = 2|v|^2 = 2
$$

이는 there exists a neighborhood $V \subset W$ of $p$ such that $\frac{\partial^2 F}{\partial t^2} (0, p, v) > 0, \forall q \in V, \forall v \in T_p M, |v|=1$.

그러므로 $c>0$ 를 잡아서 다음과 같이 잡으면 

$$
\exp_p B_c(0) \subset V
$$

앞에서 증명한 바와 같이 any geodesic in $B_c(p)$ that is tangent to the geodesic sphere of radius $r < c$ at the point $\gamma(0, q, v)$ 는 a strict local minimum for $F$ at $(0, q, v)$이다. 이것은 $q$의 Neighborhood에 대하여 the points of $\gamma$가 $B_r(p)$에 있음을 의미한다.



![Fig08](http://jnwhome.iptime.org/img/DG/RMGM_007.png)







### Proposition 4.2 : Convex Neighborhood 

For any $p \in M$ there exists a number $\beta > 0$ such that the geodesic ball $B_{\beta}(p)$ is **strongly convex**



#### Sketch of proof

다음과 같이 놓아 보자. choose $\delta > 0$ and $\beta <\delta < \frac{c}{2}$ . 그리고 $B_{\delta} (p)$ 안에 $q_1, q_2$ 가 있고 $B_{\delta} (p)$ 내부에 Geodesic이 있다고 하면. Strongly Convex 이다. 

$q_1$을 중심으로 하는 Open Ball을 생각해보면 $2\delta < c$ 이므로 $B_{2\delta}(q_1) \subset W$ 이다 여기에는 $q_2, p$가 모두 포함된다. 따라서 Normal Neighborhood이다. 또한 Lemma 4.1도 만족된다. 또한 W에 모두 포함되므로 Strongly Convex 이다.



만일, $B_{\beta}(p)$ 외부에 그림의 붉은선 처럼 Geodesic이 존재한다고 하면 위에서 보듯이 $W = B_c(p)$ 내부에 여전히 Geodesic이 존재하기 때문에 Lemma 4.1을 위배한다. 고로 증명 끝.



![Fig10](http://jnwhome.iptime.org/img/DG/RMGM_009.png)



















## Exponential Maps and Log Maps

- Let $v \in T_p M$ be a vector on the tangent plane to $M$ at $p \in M$ and $v \neq 0$. 

- $\gamma_p^v$ be the geodesic that pass through point $p$ (a.k.a. the base point) in the direction of $v$. 

- The Riemannian Exponential map of $v$ at base point $p$, denoted by $\exp_p(v)$, maps $v$ to the point, say $x$, on $M$ along the geodesic at distance $v$ from $p$, i.e.

$$
x = Exp_p(v).
$$

- Note that the Exponential map preserves the geodesic distance from the base point to the mapped point, i.e. 

$$
d(p, x) = d(p, Exp_p(v)) = \| v \| = \| Log_p (x) \|
$$

The Riemannian Log map is the inverse of Riemannian Exponential map, i.e.

$$
v = Log_p (x).
$$



### Properties of Log Maps

#### Gradient of Log maps

$$
\nabla_x d(p,x)^2 = -2 Log_p(x) \;\;\text{for}\; x \in V(p)
$$

** proof **

$p, x \in T_p M$ 이고 Geodesic의 $T_p M$ 은 Euclidean Space 이기 때문에 

$$
\nabla_x d(p,x)^2 = - 2 d(p,x) = -2 Log_p(x)
$$



#### Least Square Estimation 

For $(p,v) \in TM$, Define the sum of suared error of the data from the geodesic given by $(p,v)$ as

$$
E(p,v) = \frac{1}{2} \sum_{i=1}^N d(\exp(p, x_i v), y_i)^2
$$

the gradient of the sum-of-squares energy is 

$$
\begin{align}

\nabla_p E(p,v) &= - \sum_{i=1}^N d_p \exp(p, x_i v)^T Log(\exp(p, x_i v), y_i) \\

\nabla_v E(p,v) &= - \sum_{i=1}^N x_i d_v \exp(p, x_i v)^T Log(\exp(p, x_i v), y_i)

\end{align}
$$



### Intrinsic Average and Weighted Intrinsic Average

#### Intrinsic average

- The intrinsic average of the N points ${x_1, x_2, · · · , x_N}$ lying on a Riemannian manifold $M$ is defined as

$$
\bar{x} = \text{IntrinsicAvg} (x_1; x_2; · · · ; x_N) = \arg \min_{x \in M} \sum_{i=1}^N d(x, x_i)^2
$$

- Iterative gradient-descent method to solve the aforementioned minimization problem (by Pennec[1])

  first proposed an 

$$
x_{j+1} = Exp_{\bar{x}_j} \left( \frac{\tau}{N} \sum_{i=1}^N Log_{\bar{x}_j} (x_i) \right)
$$

where $\tau$ is the step size. 

- The uniqueness of the solution can be guaranteed when the data are well localized.



#### Weighted Intrinsic Average

- Let $w_i$ be the weight value of $x_i$, $w_i \geq 0,\; 1 \leq i \leq N$ . 

- For $x = \text{WIntrinsicAvg}(w_1, x_1;w_2, x_2; · · · ;w_N, x_N)$, The Weighted Intrinsic Average can be computed using the following iteration equation:



$$
\bar{x}_{j+1} = Exp_{\bar{x}_j} \left( \frac{\tau}{\sum_{i=1}^N w_i} \sum_{i=1}^N w_i \cdot Log_{\bar{x}_j} (x_i) \right)
$$



#### Example : SOM on Riemannian manifold

- Euclidean Space에서의 SOM Learning Equation

$$
w_{t+1} = w_{t} + \alpha(t) \cdot h_{C(x(t)),i}(t) \cdot (x(t) - w_i (t))
$$



- Riemannian Manifold에서의 SOM Learning Equation

$$
w_{t+1} = \exp_{w_i(t)} \left( \alpha(t) \cdot h_{C(x(t)),i}(t) \cdot Log_{w_i(t)} (x(t))\right)
$$

   - Step 1: Riemannian Log Map을 사용하여 Tangent Space에서의 $x(t), w_i(t) \in M$의 거리를 계산한다.

   - Step 2: Tangent Space에서 $\Delta = \alpha(t) \cdot h_{C(x(t)),i}(t) \cdot Log_{w_i(t)} (x(t))$를 계산한다.

   - Step 3: Exponential Map을 사용하여 Tangent Space위에서 계산된 $\Delta$ 를 Exponential Map을 통해 Riemanisn Manifold $M$위로 가져온다.



여기서 보면 $\exp_{w_i(t)}$ 에서 이전 시간의 weight를 중심으로 Exponential Mapping을 수행하므로 Eiclidena Space에서 차분 방정식의 역할이 Embedded 되어 있음을 알 수 있다.



