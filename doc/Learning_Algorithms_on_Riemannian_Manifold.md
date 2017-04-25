Learning Algorithms on Riemannian Manifold
==================
[TOC]

Riemannian Geometry 상에서의 해석 전에 먼저, 일반적인 Euclidean Space $\mathbb{R}^n$에서의 Linear Regression 을 먼저 알아보자.

## Linear Regression 
- $X \in \mathbb{R}$ : A non-random independent variable 
- $Y \in \mathbb{R}^n$ : A random dependent variable 
- $\alpha \in \mathbb{R}^n$ : An unobservable intercept parameter 
- $\beta \in \mathbb{R}^n$ : An unobservable slope parameter 
- $\epsilon \in \mathbb{R}^n$ : unobservable random variable representing the error

$$
Y = \alpha + X \beta + \epsilon
$$

![Fig01](http://jnwhome.iptime.org/img/DG/LARG_01.png)

- Manifold에서의 해석을 위해 다음과 같이 생각한다.
	- $\alpha$ is the **starting point** of the line
	- $\beta$ is a **velocity** vector

The least square estimates $\hat{\alpha}, \hat{\beta}$ corresponding to the data $(x_i, y_i) \in \mathbb{R} \times \mathbb{R}^n$ for $i=1, \cdots N$ by solving the minimization problem
$$
(\hat{\alpha}, \hat{\beta}) = \arg \min_{\alpha, \beta} \sum_{i=1}^N \| y_i - \alpha - x_i \beta \|^2
$$

### Propostion L1
The solution of the above equation is as follows:
$$
\begin{align}
\hat{\beta } &= \frac{\frac{1}{N} \sum x_i y_i - \bar{x} \bar{y}}{\frac{1}{N} \sum x_i^2 - \bar{x}} \\
\hat{\alpha} &= \bar{y} - \bar{x}\hat{\beta}
\end{align}
$$

#### Proof
Let  $f(\alpha, \beta) = \frac{1}{2} \sum_{i=1}^N \| y_i - \alpha - x_i \beta \|^2$, and $\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i, \;\; \bar{y} = \frac{1}{N} \sum_{i=1}^N y_i$
- For $\hat{\alpha}$

$$
0 = \frac{\partial f}{\partial \alpha} = - \sum_{i=1}^N (y_i - \alpha - x_i \beta) = N \alpha - \sum_{i=1}^N y_i - \beta \sum_{i=1}^N x_i \\
\alpha = \frac{1}{N} \sum_{i=1}^N y_i - \beta \frac{1}{N} \sum_{i=1}^N x_i = \bar{y} - \beta \bar{x}
$$

If $\beta$ would be an optimal solution $\hat{\beta}$, we can obtain the optimal solution of $\alpha$ such that
$$
\hat{\alpha} = \bar{y} - \hat{\beta} \bar{x}
$$

- For $\hat{\beta}$

$$
0 = \frac{\partial f}{\partial \beta} = - \sum_{i=1}^N (y_i - \alpha - x_i \beta) x_i = -\sum_{i=1}^N (x_i y_i - \alpha x_i - x_i^2 \beta)
$$
Since $\alpha = \bar{y} - \beta \bar{x}$, then
$$
0 = \sum_{i=1}^N (x_i y_i - \bar{y} x_i + {\beta} \bar{x} x_i - x_i^2 \beta) \\
\begin{align}
\bar{y} \sum_{i=1}^N x_i - \sum_{i=1}^N x_i y_i &= {\beta} \bar{x} \sum_{i=1}^N x_i - \beta \sum_{i=1}^N x_i^2 \\
\bar{y} \frac{1}{N} \sum_{i=1}^N x_i - \frac{1}{N} \sum_{i=1}^N x_i y_i &= {\beta} \bar{x} \frac{1}{N} \sum_{i=1}^N x_i - \beta \frac{1}{N} \sum_{i=1}^N x_i^2 \\
\bar{y} \bar{x} - \frac{1}{N} \sum_{i=1}^N x_i y_i &= {\beta} \bar{x}\cdot \bar{x} - \beta \frac{1}{N} \sum_{i=1}^N x_i^2 \\
{\beta} \left(\frac{1}{N} \sum_{i=1}^N x_i^2 - \bar{x}^2 \right) &= \frac{1}{N} \sum_{i=1}^N x_i y_i - \bar{x} \bar{y} 
\end{align}
$$
Conclude that
$$
\hat{\beta} = \frac{\frac{1}{N} \sum_{i=1}^N x_i y_i - \bar{x} \bar{y} }{\frac{1}{N} \sum_{i=1}^N x_i^2 - \bar{x}^2 }
$$

- Propostion L1은 다음 그림으로 도시 된다.

![Fig01](http://jnwhome.iptime.org/img/DG/LARG_01_01.png)

#### Note : Linear Regression and Neural Network (or Single Layer Perceptron)
$\beta = w \in \mathbb{R}^n$, $x \in \mathbb{R}^2$ 로 놓고 


### Geodesic Regression 
다음 그림과 같이 데이터 $y_1, \cdots y_N$ (청색 점)이 Riemannian Geometry $M$ 상에 존재한다면 $x_1 \cdots x_N \in \mathbb{R}$ 에 대하여 그림과 같이 나타나게 될 것이다.

![Fig02](http://jnwhome.iptime.org/img/DG/LARG_02.png)

- 좌측 그림에서 $\hat{\alpha}$는 y의 절편 혹은 **Intersection** 이라고 하며, $\hat{\beta}$는 **slope** 라고 한다. 혹은
	- $\alpha$ is the **starting point** of the line
	- $\beta$ is a **velocity** vector
- 같은 개념으로 manifold 위에서는 $(p, v) \in TM$에서 $p$ 가 Intersection, $v$를 slope 로 볼 수 있다.
	- Riemannian Geometry에서는 $p$는  **starting point**, $v$는 **velocity** vector 이다.

- 그러므로 $\mathbb{R}^n$ 상에서의 Regression Equation $Y = \alpha + X \beta + \epsilon$은 다음과 같이 Riemannian Geometry상의 관계식으로 변경할 수 있다.

$$
Y = \text{Exp}(\text{Exp}(p, Xv), \epsilon), \;\;\;\; \epsilon \in T_{\text{Exp}_p(Xv)} M
$$

- 만약, 일반적인 $M$ 이 Euclidean Space $\mathbb{R}^n$ 이라면 간단히 Exponential Map은 다음과 같이 표현된다.

$$
\text{Exp}_p (v) = \text{Exp}(p, v) = p + v
$$

### Least Square Estimation
- For $(x_i, y_i) \in \mathbb{R} \times M, \;\; \text{for } i \in \mathbb{N}[1, N]$, $(p,v)$ 로 주어지는 Geodesic에 대한 the sum of squarred Error는 다음과 같이 정의된다.[1] 

$$
E(p,v)= \frac{1}{2}\sum_{i=1}^N d(\text{Exp}(p, x_i v), y_i)^2
$$

- Formulation of a least mean square estimator of the geodesic model.

$$
(\hat{p}, \hat{v}) = \arg \min_{p,v} E(p,v)
$$

#### Proposition : Derivatives of Riemann Distance 
The gradient of the squared distance function is 
$$
\nabla_x d(p,x)^2 = -2 \text{Log}_x(p) \;\;\; \text{for }x \in V(p)
$$

** proof **

Let $d(p, x) \equiv \Phi_p (x) : T_p M \rightarrow \mathbb{R}$ then
$$
\begin{align}
\nabla_x d(p, x)^2 &= \nabla_x \Phi_p^2 (x) = 2 \cdot \Phi_p (x)  \\
&= 2 \cdot d(p, x) = 2 \cdot \text{Log}_x (p), \;\;\;\; \text{for } x \in V(p) \;\;\;\; \Box \\
\end{align}
$$

- Log Map에 의해 Distance는 $T_p M$ 위에서 정의되므로 $d(p,q) = \| Log_p (q) \|$ 에 의해 위외 같이 유도된다.  

#### Proposition : Derivatives of Exponential Map

** Variation of Geodesic **
- 다음과 같이 주어진 Geodesic의 Variation을 생각해보자.

$$
c_1(s,t) = \text{Exp}(\text{Exp}(p, su_1), tv(s)) 
$$

where $u_1 \in T_p M$ defines a **variation of the initial point** along the geodesic $\eta(s) = \text{Exp}(p, su_1)$.
여기에 대하여 $v \in T_p M$ 을 확장하여 $\eta$ 를 $v$를 따라 평행 이동(Parallel Translation) 시킨다. 

![Fig07](http://jnwhome.iptime.org/img/DG/LARG_07.png " " "float:center")

- 이와는 다르게 다음과 같이 주어진 Geodesic의 Variation을 생각해보자.

$$
c_2(s,t) = \text{Exp}(p, su_2 + tv ) 
$$
where $u_2 \in T_p M$.
위 정의와 같은 경우, $v$에 의해 확장된 Tangent Space에 $u_2$가 존재하므로 $u_2 \in T_v (T_p M)$ 이 된다. 하지만, 이 경우, $T_v(T_p M)$은 기본적으로 $v$에 의해 이동된 $T_p M$이므로 이것의 위상적 특성은 동일하여 Natural Isomorphism을 형성한다. i.e. $T_v (T_p M) \cong T_p M$  (즉, 같다고 본다.)

![Fig08](http://jnwhome.iptime.org/img/DG/LARG_08.png)

** Derivatives repect to the initial point $p$ **
$$
d_p \text{Exp}(p,v) \cdot u_1 = \left. \frac{d}{ds} c_1 (s,t) \right|_{s=0, t=1} = J_1(1)
$$

**proof**
Assume that $\text{Exp}(p, su_1) = q(s) \in T_p M$ then
$$
\begin{align}
\left. \frac{d}{ds} c_1 (s,t) \right|_{s=0, t=1} &= \left. d \text{Exp} (\text{Exp}(p, su_1), tv(s)) \right|_{s=0, t=1}\\
&= \left. d \text{Exp} (q, tv(s)) \right|_{s=0, t=1} = \left. t v(s) \cdot dq(s) \right|_{s=0, t=1} \\
&= \left. t v(s) \cdot d\text{Exp}(p, su_1) \right|_{s=0, t=1} = \left. t v(s) \cdot u_1 \right|_{s=0, t=1} \\
&= v(0) \cdot u_1 = d \text{Exp} (p, v) \cdot u_1 
\end{align}
$$

** Derivatives repect to the initial velocity $v$ **
$$
d_v \text{Exp}(p,v) \cdot u_1 = \left. \frac{d}{ds} c_2 (s,t) \right|_{s=0, t=1} = J_2(1)
$$

**proof**
Assume that $z (s,t) = su_2 + tv \in T_v(T_p M) $ then
$$
\begin{align}
\left. \frac{d}{ds} c_2 (s,t) \right|_{s=0, t=1} &=  \left. d \text{Exp}(p, z(s,t) ) \right|_{s=0, t=1} \\
&= \left. \frac{d}{dz} \text{Exp}(p, z(s,t) ) \cdot \frac{dz}{ds}\right|_{s=0, t=1} \\
&= \left. \frac{d}{dz} \text{Exp}(p, su_2 + tv ) \cdot su_2\right|_{s=0, t=1} \\
& = d \text{Exp}(p, v )\cdot u_2
\end{align}
$$

$J_i$ 는 Jacobi Fields along the geodesic $\gamma(t) = \text{Exp}(p, tv)$ 를 의미하며 초기 값은 
$$
\begin{align}
J_1 (0) = u_1, &J_1'(0) = 0 \\
J_2 (0) = 0, &J_2'(0) = u_2
\end{align}
$$
이다. $J_1$ 의 경우 $t=0$ 에서 $u_1$ 이 존재해야 하므로 당연하며, $J_1$의 미분은 $u_1$이 상수이므로 0이다.
$J_2$의 경우 $t=0$에서는 그 어떤 속도벡터도 존재하지 않으므로 0이며, $J_2(0)$의 미분에서는 $u_2$ 가 $t=0$ 에서 존재한다.

이에따라 the gradient of the sum of squares energy 는 다음과 같다.

** The gradient of the sum of squares energy **

$$
\begin{align}
\nabla_p E(p,v) &= - \sum_{i=1}^N d_p \text{Exp}(p, x_i v)^T \text{Log}(\text{Exp}(p, x_i v), y_i) \\
\nabla_v E(p,v) &= - \sum_{i=1}^N x_i d_v \text{Exp}(p, x_i v)^T \text{Log}(\text{Exp}(p, x_i v), y_i) 
\end{align}
$$

**proof**
$$
E(p,v)= \frac{1}{2}\sum_{i=1}^N d(\text{Exp}(p, x_i v), y_i)^2
$$
에서 

$$
\begin{align}
\nabla_p E(p,v)&= \nabla_p \frac{1}{2}\sum_{i=1}^N d(\text{Exp}(p, x_i v), y_i)^2 & \\
&= \frac{1}{2}\sum_{i=1}^N \nabla_p d(\text{Exp}(p, x_i v), y_i)^2 &\because \text{by the Linearity of Derivation}\\
&= -\sum_{i=1}^N d_p\text{Exp}(p, x_i v)^T  \text{Log}(\text{Exp}(p, x_i v), y_i)  &\because \nabla_x d(p,x)^2 = -2 \text{Log}_x(p)
\end{align}
$$

Suppose that $x_i v = z(v)$
$$
\begin{align}
\nabla_v E(p,v)&= \nabla_v \frac{1}{2}\sum_{i=1}^N d(\text{Exp}(p, z(v)), y_i)^2 & \\
&= \frac{1}{2}\sum_{i=1}^N \nabla_v d(\text{Exp}(p, z(v)), y_i)^2 & \\
&= \frac{1}{2} \sum_{i=1}^N \frac{dz}{dv} \nabla_z d(\text{Exp}(p, z(v)), y_i)^2 & \\
&= -\sum_{i=1}^N x_i d_v\text{Exp}(p, x_i v)^T  \text{Log}(\text{Exp}(p, x_i v), y_i)  & 
\end{align}
$$

- 그러므로 Linear Regression에서 $\hat{\alpha}$를 찾기위해 $0 = \frac{\partial f}{\partial \alpha}$구했듯이, Riemannian Geometry상의 Regression은 $\nabla_p E(p,v)$ 로 구할 수 있다.
- 또한, $\hat{\beta}$를 찾기 위해 $0 = \frac{\partial f}{\partial \beta}$구했듯이, $\nabla_v E(p,v)$ 로 구할 수 있다.
- 하지만, 이 방법론은 Iteration을 기반으로 하지 않고 한번에 $(p, v)$를 찾는 과정이므로 통계적인 방법론을 통해 구하여진 $(\hat{p}, \hat{v})$가 적절한지를 계속 테스트해야 한다. 예를 들면 Minimum Variance를 만족하는지를 알아보아야 한다.
- Iteration을 통해 기계적으로 Parameter를 찾는 방법론은 다음과 같은 접근 방법을 취한다.


#### Intrinsic Mean on Manifold[2]
Let $x_1 \cdots x_n \in M$ be in the neighborhood of $x \in M$.
To minimize the sum of squared distance function (with the same way on the above equation)

$$
f(x) = \frac{1}{2N} \sum_{i=1}^N d(x, x_i)^2 
$$

For minimizinf the $f(x)$, it is necessary to define the gradient of $f$ such that
$$
\nabla f(x) = - \frac{1}{N} \sum_{i=1}^N \text{Log}_x (x_i).
$$

It is a negative gradient to minimize the object function $f$.
Then, under a given current estimator $\mu_j$ , the updateing mena is evaluated as follows
$$
\mu_{j+1} = \text{Exp}_{\mu_j} \left( \frac{\tau}{N} \sum_{i=1}^N \text{Log}_{\mu_j}(x_i) \right)
$$
where $\tau$ is the step size.

Step size의 경우 $\tau = 1$의 경우 단순 평균이고 Euclidean Space 혹은 Sphere의 경우에는 이것으로 충분하다는 연구도 있다.[3]


## Application : Self Organizing Feature Map 
- Teuvo Kohonen에 의해 개발된 Competitive Learning 알고리즘
- Back Propagation 알고리즘과 같은 Error-Correction 알고리즘과는 다르게 입력 데이터의 위상적 성질을 추상화 하는 방식이다.

![Fig09](http://jnwhome.iptime.org/img/DG/LARG_09.svg)

- 학습 알고리즘은 다음과 같다. 
	- $i \in \mathbb{N}[1, N]$ 는 Weight vector의 Index
	- $W_(t) \in \mathbb{R}^n$ 는 시간 t에서 i 번째 Weight Vector
	- $x(t) \in \mathbb{R}^n$ 는 시간 t에서 입력 데이터
	- $\alpha(t) \in \mathbb{R} $ 는 Learning rate, $ t \uparrow \infty$ as $\alpha(t) \downarrow 0$

$$
w_i (t+1) = w_i (t) + \alpha (t) h_{c(x(t)), i} (t) \cdot (x(t) - w_i (t))
$$

- 학습 알고리즘에 있는 $h_{c(x(t)), i} (t)$ 는 Neighborhood function 으로서
	- $r_i \in \mathbb{R}^2$ 는 i번째 Weight Vector 가 가리키는 Neuron의 위치
	- $r_{c(x(t))} \in \mathbb{R}^2$ 는 입력 벡터 $x(t)$가 인가 되었을 때 Winner 혹은 가장 가까운 곳에 위치하는 Weight Vector 를 가리키는 Neuron의 위치
	- $c(x(t)) = \arg \min_{\forall i \in \mathbb{N}[1,N]} \| x(t) - w_i (t) \|^2$
$$
h_{c(x(t)), i} (t) = \exp \left( - \frac{\| r_i - r_{c(x(t))}\|^2}{2 \sigma^2 (t)} \right) \in \mathbb{R}
$$

- $h_{c(x(t)), i} (t)$의 동작을 도시하면 다음 그림과 같다. 

![Fig10](http://jnwhome.iptime.org/img/DG/LARG_10.png)

- SOFM은 입력 데이터 집합의 Topological Property를 찾는 알고리즘이기 때문에 자체 만으로는 어떤 의미를 주지 못한다.
- 그러므로 SOFM의 경우는 각 Weight Vector에 의미를 줄 수 있는 또 다른 종류의 신경망을 결합시키거나 통계적 특성을 분석하는데 사용된다.
- 일반적으로 입력 데이터에 대한 전처리를 위해 사용된다.
- 대표적인 Unsupervised Learning이기 때문에 Unsupervised Learning의 입문용으로 많이 연구된다.
- 기본적으로 Kalman Filter 알고리즘을 기반으로 한다.

### Riemannian SOFM [4]
Euclidean 공간에서의 SOFM을 일반적인 다양체상의 알고리즘으로 확장하여 SOFM 학습 방정식을 재 정의한다. 


#### Exponential and Log Maps on $S^2(n)$
Unit 3-sphere manifold $S^2 \subset \mathbb{R}^3$ 을 생각한다.
- Let a point $p_0 = (0, 0,1)^T \in S^2$ 를 Base point로 놓고 Tangent space $T_{p_0} S^2$ 를 생각한다. 
- $T_{p_0} S^2$ 상에서 Geodesic을 형성하는 속도벡터 $v = (v_x, v_y)^T \in T_{p_0} S^2$를 생각하자.
- 이 경우, Exponential Map은 다음과 같다.

$$
\text{Exp}_{p_0}(v) = \left( v_x \cdot \frac{\sin \| v \|}{\| v \|}, v_y \cdot \frac{\sin \|v\|}{\| v \|}, \cos \|v\| \right)^T
$$ 

- 여기에 대한 Log Map for $q = (q_x, q_y, q_z)^T \in S^2$ 은 다음과 같다. $\beta = \arccos(q_z)$ 이다,

$$
\text{Log}_{p_0}(q) = \left( q_x \cdot \frac{\beta}{\sin \beta}, q_y \cdot \frac{\beta}{\sin \beta}  \right)^T
$$

이를 이해하기 위하여 다음과 같은 그림을 놓는다. ($q_y = 0$ 이라 보자.)

![Fig11](http://jnwhome.iptime.org/img/DG/LARG_11_01.png)

- Exponential Map의 경우 Let $v_y = 0$ 으로 놓으면 $\| v \| = v_x$ 고로 자연적으로 $S^2$상의 점은 $\sin \|v\|$ 이다.
	- 나머지 부분들도 간단히 유도된다.

- Log Map의 경우 $\text{Log}_{p_0}(q) = \left. (\beta_x, 0) \right|_{q_y = 0}$ 라 놓자 
	- Log Map에 의해 $T_{p_0} S^2$ 상의 점을 $\bar{v}_x$ 라 하면, 직각 삼각형의 비례에 따라

$$
\begin{align}
q_x : \sin \beta &= \bar{v}_x : \beta \\
\sin \beta \cdot \bar{v}_x &= q_x \cdot \beta \\
\bar{v}_x &= q_x \frac{\beta}{\sin \beta} 
\end{align}
$$

- $q_y$ 에 대하여도 마찬가지로 풀면된다.

#### Adaptation Error in Conventional SOM on $S^2$ 

다음 그림을 살펴보면 Riemannian Geometry 상에서의 알고리즘이 필요한 이유가 명확하다.

- Weight Vector, Input Vector가 모두 Riemannian Manifold $S^2$ 위에 있다고 가정하자. 
- 기존의 Euclidean 방식의 SOM 알고리즘을 적용하면 Weight Vector는 $S^2$ 내부로 들어가게 되어 정확한 입력 데이터의 Toplogical Property를 보존한다는 SOFM의 특성이 깨지게 된다. 
	- SOFM의 Toplogical Property 보존을 통해 Classfication, NP-hard Optimization문제 (TSP 문제등)등을 풀 수 있는데 이것이 어렵게 된다.

- 그러나, Riemannian Geometry를 적용한 경우 Weight Vector는 $S^2$ 상에 항상 존재하게 되어 SOFM의 특성을 보존할 수 있게 된다.

![Fig12](http://jnwhome.iptime.org/img/DG/LARG_12.png)

### Experimental Results
Swiss Roll Data에 대한 기존 SOFM과 Riemannian SOFM의 Topological Structure를 훨씬 잘 보존하는 것을 볼 수 있다.

![Fig13](http://jnwhome.iptime.org/img/DG/LARG_13.png)

- Facial Recognition등에 사용된 경우 보통 기존 알고리즘 보다 3~6% 가량 분류 성능이 향상됨을 보인다. 











#### Sequential Learning Algorithm
다음의 순서를 통해 학습 방정식을 구한다.
- 입력 데이터 $x(t)$를 Riemannian Log Map을 통해 Tangent Space로 옮긴다.
- Tangent Space에서 다음의 Update 항을 계산한다.

$$
\Delta = \alpha(t) h_{c(x(t)), i} (t) \text{Log}_{w_i(t)}(x(t))
$$

- $\Delta$를 다시 Exponential Map을 통해 다양체 위로 옮긴다. 

이를 하나의 방정식으로 나타내면 다음과 같다. 

$$
w_i (t+1) = \text{Exp}_{w_i (t)} \left(\alpha (t) h_{c(x(t)), i} (t) \cdot \text{Log}_{w_i(t)}(x(t)) \right) 
$$

#### Batch Learning Algorithm
Batch Learning은 위와 같이 Sequential하게 하는 것이 아닌 여러개의 데이터에 대하여 Class별 대표 입력 데이터에 대해 Weight Vector를 구하는 방법이다. 

- K개의 Class가 존재하고 각 Class의 Intrinsic Mean을 $\bar{x}_j \in \mathbb{R}^n$, $j \in 1, \cdots K$ 로 표시한다.
- 각 Class 별로 Intrinsic Mean 을 Iterative하게 구한다. 여기서 $n_j$는 j번째 Class에 할당된 입력 데이터의 갯수를 의미한다.

$$
\bar{x}_{j}(t+1) = \text{Exp}_{\bar{x}_j (t)} \left( \frac{\tau}{n_j} \sum_{i=1}^{n_j} \text{Log}_{\bar{x}_j (t)}( x_i )\right)
$$

- 구하여진 Class별 Intrinsic mean $\{\bar{x}_1, \cdots, \bar{x}_K \}$ 에 대하여 i 번쨰 Weight Vector 와 다음과 같은 관계를 가지고 있다고가정하자.  

$$
h_{ji}(t) = \exp \left( -\frac{\| \bar{x}_j - w_i (t) \|^2}{2 s^2(t)} \right) \;\;\;\text{ where } s(t) \downarrow 0 \text{ as } t \uparrow \infty
$$

- 다음과 같이 Weighted sum을 위한 weight $\bar{w}_{ji}(t)$ 를 구한다.

$$
\bar{w}_{ji}(t) = \frac{h_{ji}(t) \cdot n_j}{\sum_{j=1}^K h_ji(t) \cdot n_j} 
$$

- Learning 방정식은 다음과 같다. 

$$
w_i (t+1) = \text{Exp}_{w_i (t)} \left( \frac{\tau}{\sum_{j=1}^{K} w_{ji}} \sum_{j=1}^K w_{ji} \text{Log}_{w_i(t)}(\bar{x}_j) \right) 
$$ 

- **Initialization of Weight vector**
	- 임의로 2개의 입력 벡터 $x, y$를 선택한다.
	- Weight Vector는 두 입력 벡터의 중간 값으로 정한다.

$$
w_i = \text{Exp}_x(\frac{1}{2} \cdot \text{Log}_x (y))
$$

## Principal Geodesic Analysis
Euclidean Space에서 Principal Component Analysis (PCA) 가 되는 것을 다양체 위에서는 Principal Geodesic Analysis가 된다. 그 이유는 Geodesic이 다양체 위에서 가장 짧은 거릭가 되기 때문에 모든 거리를 Geodesic의 거리로 정의하기 때문이다. 그렇다고 해서 기존의 다양체위에서의 해석과 다를 것은 크게 없다.

### PCA 요약
입력 데이터를 $x_1, \cdots x_N \in V$ ($V$는 Vector space or Hilbert space라고 하자) 일때, 입력 데이터의 단순 평균은 다음과 같다.
$$
\mu = \frac{1}{N} \sum_{i=1}^N x_i
$$

이때, Covariance Matrix $S$는 다음과 같으므로
$$
S = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)(x_i - \mu)^T
$$

여기서  PCA는 $S$에 대한 Eigen Value $\lambda_k$에 대한 Eigen vector $v_k$ 를 통해 다음과 같이 원래 신호를 복원하는 것이다. (for $k \in \mathbb{N}[1, N]$) 
$$
x = \mu + \sum_{k=1}^d \alpha_k v_k
$$
여기서 $\alpha_k \mathbb{R}$은 the mode of variation의 가중값 혹은 PCA 계수이다. 이를 조금 더 살펴보면 $\mu = 0$인 상황에서 $v_k$가 모두 Orthogonormal 하다고 하면
$$
\langle x, v_i \rangle = \sum_{k=1}^d \alpha_k \langle v_k, v_i \rangle = \sum_{k=1}^d \alpha_k \delta_k^i = \alpha_i
$$
여기에서 $\alpha_i$는 index $i$에 따라 크기가 다르게 나타날 것이다. 이를 사용하여 $v_i$를 중요도에 따라 정렬시킬 수 있다.  
따라서 어떤 Orthonormal basis를 Eigen vector로 부터 찾는 과정은 다음과 같이 Recursive 하게 이루어 질 수 있다.
$\{v_1, \cdots, v_d \} \in \mathbb{R}^d$ 인 Orthonormal Vector들이 존재한다고 할 때 이 중 하나를 $v$라고 하면
$$
\begin{align}
v_1 &= \arg \max_{\|v\|=1} \sum_{i=1}^N \langle v, x_i \rangle^2 \\
v_k &= \arg \max_{\|v\|=1} \sum_{i=1}^N \sum_{j=1}^{k-1} \langle v_j, x_i \rangle^2 + \langle v, x_i \rangle^2
\end{align}
$$
그런데, Symmetric Matrix에 대한, Eigen Vector는 서로 Orthogonal 하므로 Orthonormal한 Basis는 위와 같은 Gram-Schmidt Process를 통해 얻거나(컴퓨터 해석) Eigen vector의 Rescaling을 통해 얻을 수 있다.

### PGA 유도 
일반적인 다양체에서는 Euclidean Space에서 처럼 Inner product에 의한 Induced Norm을 전역적으로 사용할 수 없다.
따라서 Riemannian 계량으로 정의되는 Tangent Space상에서만 이와 같은 방법을 사용할 수 있으며, 다양체위에서는 거리 함수 $d$를 정의해야 한다.
Euclidean Space상에서의 Covariance를 다음과 같이 간략하게 놓으면 
$$
\sigma^2 = \mathbb{E}(x - \mu)^2
$$ 
다양체 위에서는 다음과 같이 표현된다.
$$
\sigma^2 = \mathbb{E} d(\mu, x)^2
$$
그러므로 이를 확대하여 주어진 데이터 $x_1 \cdots x_N$에대한 Sample Variance는 다음과 같다.
$$
\sigma^2 = \frac{1}{N} \sum_{i=1}^N d(\mu, x)^2 = \frac{1}{N} \sum_{i=1}^N \| Log_{\mu}(x_i) \|^2
$$
- 즉, $\mu$와 $x_i$의 거리는 $\mu$를 중심으로 하는 Tangent Space $T_{\mu}$상에서의 거리로 정의할 수 있다. 
   - 엄밀히 얘기하면 $T_{\mu}$ 상에 존재하는 Normal Ball $B^o(\mu, \delta)$상에 $x_1, \cdots x_N$이 모두 포함된다고 가정해야 하며 그래야 Tangent space상의 Euclidean 거리가 다양체 위의 Geodesic으로 Mapping 된다.

다음과 같이 임의의 다양체 $M$상의 $x \in M$에 대하여 Geodesic Submanifold $H \subset M$으로의 Projection $\pi_H$이 존재하여 다음과 같이 정의된다고 하자.
$$
\pi_H(x) = \arg \min_{y \in H} d(x, y)^2
$$
이때, $p \in H$에 대한 Tangent Space $T_p H$가 존재하면 이것은 $T_x M$과 다르므로 다음과 같이 근사화 된다.
$$
\begin{align}
pi_H(x) &= \arg \min_{y \in H} \| \text{Log}_{x}(y) \|^2 \\
&\approx \arg \min_{y \in H} \| \text{Log}_{p}(x) - \text{Log}_{p}(y) \|^2
\end{align}
$$
이때, $\text{Log}_p (y)$ 를 $T_p H$에서의 (단위) 속도 벡터 $v$로 보면 다음과 같이 $T_p H$상에서의 Tangent Vector로 $x, y$사이의 Projection을 놓을 수 있다.
$$
\text{Log}_p (\pi_H (x)) \approx \arg \min_{v \in T_p H} \| \text{Log}_{p}(x) - v \|^2
$$
이 방정식은 $\text{Log}_p(x)$에 대한 Linear Subspace $T_p H$로의 Linear Projection이다. (Minimization 이므로) 즉, $x \in M$과 가장 유사한 $v \in T_p H$를 찾는 문제이다.
여기서 $T_p H$의 Orthonormal Basis를 $v_1, \cdots v_k$라고 하자. 그러면, 위 방정식은 $T_p H$상의 Orthonormal base로 $\text{Log}_p(x)$를 만드는 것이므로 
$$
\text{Log}_p(\pi_H (x)) \approx \sum_{i=1}^k \langle v_i , \text{Log}_p (x) \rangle
\tag{1}
$$

따라서 이를 사용하여 임의의 다양체 위에서 PCA를 정의할 수 있다. 
- $T_{\mu} M$ 은 $x_i \in M$의 Intrinsic Mean $\mu$의 $M$에 대한  Tangent Space이다.
- $U \subset T_{\mu} M$ 은 $\mu$의 Neighborhood 로서 $\text{Exp}_{\mu} (U)$의 모든 Geodesic submanifold 에 대하여 Well-defined 되어 있다.
   - 다시말해 $\text{Exp}_{\mu} (U)$ 상의 모든 점에 대하여 Geodesic을 줄 수 있다.
   - 당연히 이러한 neighborhood는 매우 협소한 공간이다. 대역적으로 $M$이 $S^n$ 과 같지 않는 한.
   - 그러므로 알고리즘을 만든다고 하면 국소적으로 $S^n$으로 가정하고 알고리즘을 만드는 것은 타당하다.
- Geodesic Submanifold에서의 Tangent Space는 Orthonormal basis $v_1, \cdots , v_d \in T_{\mu} M$으로 Span 된다.
- Vector Space $V_k = \text{span}(\{ v_1, \cdots , v_d \}) \cap U$ 이다.
- 이때, Geodesic Submanifold $H_k$는 $H_k = \text{Exp}_{\mu}(V_k)$ 로 정의된다.
   
위 조건하에서 첫번쨰 Principoal Direction은 다음으로 정의한다.
$$
v_1 = \arg \max_{\| v \| = 1} \sum_{i=1}^N \| \text{Log}_{\mu}(\pi_H(x_i)) \|^2
$$   
where
$$
H = \text{Exp}_{\mu} (\text{span}(\{ v \}) \cap U).
$$

나머지 성분은 
$$
v_1 = \arg \max_{\| v \| = 1} \sum_{i=1}^N \| \text{Log}_{\mu}(\pi_H(x_i)) \|^2
$$
where
$$
H = \text{Exp}_{\mu} (\text{span}(\{ v_1, \cdots v_{k-1}, v \}) \cap U).
$$

식 (1)의 근사식을 여기에 대입하면 다음과 같이 PGA 방정식을 구할 수 있다.
$$
\begin{align}
v_1 &\approx \arg \max_{\| v \| =1 } \sum_{i=1}^N \langle v, \text{Log}_{\mu}(x_i) \rangle^2 \\
v_k &\approx \arg \max_{\| v \| =1 } \sum_{i=1}^N \sum_{j=1}^{k-1} \langle v_j, \text{Log}_{\mu}(x_i) \rangle^2 + \langle v, \text{Log}_{\mu}(x_i) \rangle^2
\end{align}
$$

이를 사용하여 다음과 같이 PGA 알고리즘을 만들 수 있다.

| Principal Geodesic Analysis |  |
|---| ---|
| Input  | $x_1, \cdots, x_M \in M$ |
| Output | Principal Directions, $v_k \in T_{\mu} M $  and Variance $\lambda_k \in \mathbb{R}$|
|        |          |
| step 1 | Get the intrinsic maean $\mu$ of $\{x_i \}$ |
| step 2 | $u_i = \text{Log}_{\mu} (x_i)$              |
| step 3 | $\mathbf{S} = \frac{1}{N} \sum_{i=1}^N u_i u_i^T$ |
| step 4 | $\{ v_k, \lambda_k \}$ is **eigenvectors** and **eigenvalues** of $\mathbf{S}$ |


## 한계점과 시사점 
Riemannian Geometry 기반 알고리즘은 현재 Manifold에 대한 정보가 주어진 경우에 대하여 적용되는 것이 대부분이다. 
- M-rep 등 의료 영상의 경우 Manifold 에 대한 정보가 주어져 Exponential Map, Log Map에 대한 정보가 주어진 상태에서 알고리즘 개발이 이루어진다.

Riemannian SOFM의 경우와 같이 다양체를 $S^n$ 으로 대역적으로 가정하고 적용할 수도 있다.
- 다양체 정보가 주어진 경우에는 해당 정보를 기반으로 알고리즘을 변경 시킨다.

따라서, 매 Iteration에 따라 Riemannian Gemetry 변화에 대한 정보를 업데이트 할 수 있는 알고리즘의 개발이 필요하다.

## My Idea
위에서 언급한 한계점 및 다양체에서의 신경망 적용을 위한여 다음과 같은 생각을 해본다.

### Classification 문제
다음 그림과 같이 두개의 분포를 가진 데이터가 서로 중첩 되어 있는 경우를 생각해 보자.
![Fig_Ai01](http://jnwhome.iptime.org/img/AI/Idea01.png)

이러한 경우, 다음의 가정을 놓는다.
- 두 개의 분포를 가진 데이터는 원래 동일한 분포를 가지고 있다. 
	- 다양체위의 데이터를 단순 Euclidean Space에 Mapping 할 경우 그림과 같은 현상이 일어난다. 
	- 따라서, 다양체 위에서 데이터를 분석 할 경우, 보다 효율적으로 데이터를 분류 할 수 있다.  
- 다양체는 구면 ($S^n$)을 가정한다. 

이러한 경우 그림처럼 분포의 Principle 벡터중 한 요소가 다른 분포의 요소와 다르다고 하면, 해당 분포는 구면의 Geodesic을 따라 구면상의 다른 부분에 위치하게 된다.
![Fig_Ai02](http://jnwhome.iptime.org/img/AI/Idea02.png)

따라서, 다양체 위에 Tangent Space를 사용하여 문제를 접근하게 되면 Classification 문제를 보다 쉽게 해결 할 수 있을 것으로 생각된다.

### 다양체 위의 Classification 문제 분석
다음 그림과 같이 다양체 위에서 두 분포가 접하고 있는 경우 이를 단순 Euclidean Projection을 하게 되면 잘못된 분류가 발생할 수 있다.

![Fig_Ai03](http://jnwhome.iptime.org/img/AI/Idea03.png)

- Weight Vector가 학습 과정에서 넓은 분포를 가진 데이터에 교란되어 중심점을 잘못 찾을 가능성이 높다.
- 다양체에 Mapping될 경우에는 분포가 모두 같다고 볼 수 있으므로 중심점을 잘못 찾을 가능성이 낮아진다. 
- 학습의 경우 Weight Vector를 중심으로 하는 Tangent Space위에서 수행한다.

또한 분류를 위한 Hyper Plane은 
- 구면상의 Tangent Space상의 주 분포 상의 $\text{Log}$ Map 만으로 해석하는 경우 
- $\text{Log}$ Map 후 $\text{Exp}$ Map을 통해 해석하는 경우 

#### Regression 응용 
기본적으로 Classification 문제와 다를 바 없으며 Regression 을 통해 Hyper Plane이 교란되는 경우를 줄일 수 있다.
Regression을 통한 Classification은 결과가 위 문제에 대하여 Orthogonal하게 나타난다고 생각할 수 있다. 

### Curvature 이용 문제 
위의 문제를 확장하여 다음과 같이 생각해 본다. 

만일 평균 값, 분산이 서로 다른 분포가 그림과 같이 있다고 가정하면 주황색의 분포와 청색의 분포는 확연하게 구별할 수 있으나 청색의 두 분포의 경우는 인접해 있기 떄문에 Normal 분포를 특성할 수 없어 Monte Carlo 기반의 알고리즘을 사용해야 Feasible한 결과를 얻을 수 있다.

![Fig_Ai04](http://jnwhome.iptime.org/img/AI/Idea04.png)

그러나 만일, 평균간 다르고 분산이 동일한 분포가 그림과 같은 다양체 위에 존재한다고 하면, 분산의 경우 Curvature가 급격히 변하는 부분에 존재한다고 볼 수 있다.

![Fig_Ai05](http://jnwhome.iptime.org/img/AI/Idea05.png)

이러한 경우, 다양체의 대역적인 Curvature를 구하는 것이 아닌 다양체 위의 점 $p \in M$ 상의 $T_p M$ 에서의 Independent Vector를ㄹ 사용하여 Sectional Curvature를 구하는 것이므로, 만일, 데이터와 Weight 간의 분산을 구하면서 Tangent Space상의  Orthogonal한 Variance의 2개의 주 성분, 혹은 2개의 Gradient로 유도 가능한 Orthogonal 성분 (Conjugated 성분) 을 사용하여 Sectional Curvature를 매 Iteration 마다 계산하여 다른 Weight에서의 Tangent Space와 그 만큼 다른 곳에서 존재한다고 하면 보다 효율적인 알고리즘을 만들 수 있을 것으로 생각된다. 

- 이 부분에 대해서는 추가 연구 필요
- Riemmanian Geometry 상에서의 Curvature는 다음 참고
http://jnwhome.iptime.org/?p=324


$$
J(x,w) = \frac{1}{2} \sum_{C=0}^{n-1} \frac{1}{C(k)}\sum_{C(k=0)}^{C(k-1)} (x_{C(k)} - w^{C})^2
$$

$$
w_{t+1}^C = w_t^C - \varepsilon_t \nabla_{w_t^C} J(x,w) = w_t^C + \varepsilon_t \frac{1}{C(k)} \sum_{C(k=0)}^{C(k-1)}(x^{C(k)} - w_t^C)
$$


$$
w_{t+1}^C = Exp_{w_t}^C  \varepsilon_t \frac{1}{C(k)} \sum_{C=0}^{n-1} Log_{w_t^C}x^{C(k)} 
$$

$$
w_{t+1}^C = \text{Exp}_{w_t^C}  \frac{\varepsilon_t}{C(k)} \sum_{C=0}^{n-1} \text{Log}_{w_t^C}\frac{d}{dt} \gamma (x^{C(k)}, w_t^C) 
$$


$$
dw_{t+1}^C = -\varepsilon_t \nabla_{w_{t}^C} J(x,w_t^c) dt + \sqrt{2T(t)} dW_t
$$

$$
\begin{align}
\int_{t}^{t+1} dw_{\tau}^C &=  \text{Exp}_{w_{\tau}^C}  \left[\int_{t}^{t+1} \frac{\bar{\varepsilon}_{\tau}}{C(k)} \sum_{C=0}^{n-1} \text{Log}_{w_{\tau}^C}\frac{d}{d{\tau}} \gamma (x^{C(k)}, w_{\tau}^C) d\tau  + \int_{t}^{t+1} \sqrt{T(\tau)} dW_{\tau} \right] \\
w_{t+1}^C - w_{t}^C&= \text{Exp}_{w_{\tau}^C}  \left[ \frac{\varepsilon_{\tau}}{C(k)} \sum_{C=0}^{n-1} \text{Log}_{w_{\tau}^C}\frac{d}{d{\tau}} \gamma (x^{C(k)}, w_{\tau}^C) - w_{t}^C  + \sqrt{\bar{T}(\tau)} W_{t} \right] \\
&= - w_{t}^C + \text{Exp}_{w_{\tau}^C}  \left[ \frac{\varepsilon_{\tau}}{C(k)} \sum_{C=0}^{n-1} \text{Log}_{w_{\tau}^C}\frac{d}{d{\tau}} \gamma (x^{C(k)}, w_{\tau}^C)   + \sqrt{\bar{T}(\tau)} W_{t} \right] \\
\therefore w_{t+1}^C &= \text{Exp}_{w_{\tau}^C}  \left[ \frac{\varepsilon_{\tau}}{C(k)} \sum_{C=0}^{n-1} \text{Log}_{w_{\tau}^C}\frac{d}{d{\tau}} \gamma (x^{C(k)}, w_{\tau}^C)   + \sqrt{\bar{T}(\tau)} W_{t} \right]
\end{align}
$$

$$
\begin{align}
\sqrt{\bar{T}(\tau)} &= \sigma \cdot \frac{c}{\log(2+t)} \\
\sqrt{\bar{T}(\tau)} &= \sqrt{E(x_{C(k)} - w_t^{C} )^2} \cdot \frac{c}{\log(2+t)}
\end{align}
$$



## 진행상황

- 현재 최적화 문제 등에 많이 나타나는 Hyperbolic Manifold에 대한 특성 해석인 Lioville 이론 연구 
   - Riemannian Geometry 기초 연구로는 마지막 부분 
   - Hyperbolic Manifold는 $f:U \subset \mathbb{R}^n \rightarrow \mathbb{R}^n$ 에 대하여 다음과 같은 특성을 가지고 있다. $p$점에서의 vector $v_1, v_2$에 대하여 

$$
\langle df_p(v_1), df_p (v_2) \rangle = \lambda^2 (p) \langle v_1, v_2 \rangle, \lambda^2 \neq 0
$$

- Stochastic Gradient 등에 대한 이론 연구 
   - 거의 완료 추후, 추가 연구 계획 (Riemannian Geometry 상의 Wiener Process 유도와 이에 따른 Probability  Measure 변화 등에 대한 연구)

- Python과 Tensorflow를 사용한 Regression 실험
   - Tensorflow를 사용한 간단한 Linear Regression 완료
   - 이를 바탕으로 Riemannian Geometry 상에서의 Regression 실험 예정 

- 다양체 위에서의 Classification 문제 실험 예정 
   - Python과 Tensorflow를 응용하여 실험 데이터 생성 중 
   - 현재 두 개의 서로 다른 $(x, y)$ 분산을 가지는 실험 데이터 생성, Plotting까지 진행
   
![Fig_Ai06](http://jnwhome.iptime.org/img/AI/Idea06.png)
  
   
   - 간단한 Classification 을 위한 Competitive Learning 알고리즘 구현 중
      - Simple Competitive Learning
      - Learning Vector Quantization 

   - Euclidean Space상에서의 테스트 완료 후, 해당 데이터 Set을 $S^n$ 상에 Mapping 시킨 후, Riemannian Geometry 상에서 상기 두 알고리즘을 실험할 예정 


### Mathematical Base
- **Isometric Immersion** 
Let $f:M^n \rightarrow \bar{M}^{n+m=k}$ be an immersion. Then
for each $p \in M$, $\exists U \subset M$ of $p$ ($U$ : Neighborhood) such that $f(U) \subset \bar{M}$ is a submanifold of $\bar{M}$.
i.e. there exist a neighborhood $\bar{U} \subset \bar{M}$ of $f(p)$ and a diffeomorphism $\varphi: \bar{U} \rightarrow V \subset \mathbb{R}^k$. such that $\varphi$ maps $f(U) \cap \bar{U}$ diffeomorphically onto an openset of subspace of $\mathbb{R}^n \subset \mathbb{R}^k$. 

- 원래 의미는 다양체 $M^n$을 보다 고차원의 다양체 $\bar{M}^k$로 옮겨 물리적 현상을 쉽게 해석하려는 것에 있다.
- 공학적으로 $\mathbb{R}^n$ 에 정의되어 있는 데이터를 부가된 차원을 포함하는 다양체 위에서 해석하는 하는 것도 같은 의미이다.
	- 예를 들어, $\mathbb{R}^2$에 정의된 데이터를 $S^n$상에 옮겨 보다 쉽게 데이터를 분석할 수 있다.

- 이를 위해서는 다음의 수학적 분석을 통해 기존의 Riemannian Geometry 기반 알고리즘을 업그레이드 할 필요성이 있다.
	- Cartan Theory 기반 Learning Algorithm
	- 입력 데이터의 Covariance에 대한 Eigen Vector 분석에 의한 Curvature 기반 Geodesic Distance 유도
	- Wiener Process 영향에 의한 Stochastic Gradient 변화 유도


## Simple Riemannian Geometry
### Definition : Differentiable Manifold
A Differnetiable manifold of dimension $n$ is a **set** $M$ and a family of **injective mappings** $x_{\alpha} : U_{\alpha} \subset \mathbb{R}^n \rightarrow M$ of openset $U_{\alpha}$ of $\mathbb{R}^n$ into $M$ such that
- $\cup_{\alpha} x_{\alpha}(U_{\alpha}) = M$ 
- **Differentiable Structure** : for any pair $\alpha, \beta$ with $x_{\alpha}(U_{\alpha}) \cap x_{\beta}(U_{\beta}) = W \neq \varnothing$ the sets $x_{\alpha}^{-1}(W)$ and $x_{\beta}^{-1}$ are open sets in $\mathbb{R}^n$ and **the mappings $x_{\beta}^{-1} \circ  x_{\alpha}$ are differentiable**.
- The family $\{U_{\alpha}, x_{alpha} \}$ is maximal relative to the above two conditions.

![Fig03_02](http://jnwhome.iptime.org/img/DG/LARG_03_02.png)

- Differentiable Manifold 는 줄여서 Manifold 라는 이름으로도 호칭된다.
- 미분을 줄 수 있는 다양체 (미분다양체) 라는 의미이며 다양체는 $많을 다, 접힐 양$ 이라는 의미 즉, 국소적으로 Euclidean 인 공간을 서로 미분 가능하게 접어서 붙인다는 의미
	- 이때 어떤 구조로 미분이 적용되는가가 위 정의에서 Differential Structure 임 
- 위 정의에서 $x : \mathbb{R}^n \rightarrow M$ 을 Parameterization이라고 함 

### Definition : Tangent Vector, Tangent Space, Tangent Bundle
Let $M$ be a differentiable manifold. A differentiablke function $\alpha:(-\varepsilon, \varepsilon) \rightarrow M$ is called a **(differentiable) curve** in $M$. 
Suppose that $\alpha(0) = p \in M$, and let $\mathcal{D}$ be the set of functions on $M$ that are differentiable at $p$.
**The tangent vector** to the curve $\alpha$ at $t=0$ is a function $\alpha'(0): \mathcal{D} \rightarrow \mathbb{R}$ given by
$$
\alpha'(0) f = \left. \frac{d(f \circ \alpha)}{dt} \right|_{t=0}, \;\; f \in \mathcal{D}
$$

- A **Tangent vector at $p$** 는 $t=0$ 에서 어떤 curve $\alpha:(-\varepsilon, \varepsilon) \rightarrow M$ with $\alpha(0)=p$. 
- 모든 Tangent vector to $M$ at $p$의 집합을 **Tangent Space $T_p M$** 이라 한다.
- Parameterization $x:U \rightarrow M^n$ at $p= x(0)$가 있을 때 

$$
f \circ x(q) = f(x_1, \cdots x_n), \;\;\; q=(x_1, \cdots, x_n) \in U \\
x^{-1} \circ \alpha(t) = (x_1(t), \cdots, x_n(t) )
$$
이라 할 떄
$$
\begin{align}
\alpha'(0) f &= \left. \frac{d}{dt}(f \circ \alpha) \right|_{t=0} = \left. \frac{df}{dt}(x_1(t), \cdots x_n(t)) \right|_{t=0} \\
&= \sum_{k=1}^n \frac{\partial x_k}{\partial t}(0) \cdot \frac{\partial f}{\partial x_k} = \left( \sum_{k=1}^n x_k'(0) \frac{\partial }{\partial x_k} \right)f
\end{align}
$$
- 위 식을 정의와 비교하면 $\alpha'(0) = \sum_{k=1}^n x_k'(0) \frac{\partial }{\partial x_k}$ 따라서 **Tangent Space는 $\frac{\partial }{\partial x_k}$ 를 Basis로 하는 Vector Space** 이다. 

![Fig04](http://jnwhome.iptime.org/img/DG/LARG_04.png)

- **Tangent Bundle** : 점 $p \in M$, Tangent vector $v \in T_p M$ 모든 순서쌍 $(p, v)$ 집합을 Tangent Bundle 이라 하고 다음과 같이 정의한다.

$$
TM = \{ (p,v); p \in M, v \in T_p M \}
$$

### Geodesic and Exponential Map 
#### Definition : Geodesic
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

- Geodesic 은 다양체위의 임의의 점 사이에서 **다양체를 따라 가장 가까운 거리**를 의미한다[1]. 
	- 다양체위에서는 Euclidean 좌표계와 같은 절대 좌표계를 줄 수 없다. (절대 좌표계를 생각하지 않는다.)
	- 다양체는 Euclidean 좌표계가 아닌 다양체 자체의 좌표계가 존재한다고 생각한다. (다양체의 정의에 따라.)
	- Euclidean 좌표계는 다양체위의 Tangent Space에서 국소적으로 성립한다.
	- 그러므로 다양체위에서는 **거리를 새로 정의**해야 하며 이때 Geodesic은 다양체위의 최소 거리이기 때문에 Gedodesic을 따라 거리를 정의한다. 
	- 이렇게 정의된 거리는 Norm Property를 만족한다. 
	- 그러나 Geodesic 은 언제나 다양체위에서 Global 하게 정의되지 않는다. 국소적으로만 정의된다.
		- Global 하게 정의되는 다양체는 Complete Manifold 라고 하며 $\mathbb{R}^n, S^{n-1}, H^n$의 세가지가 대표적이다.

#### Definition : Exponential Map 
Let $p \in M$ and let $\mathcal{U} \subset TM$  be an open set. Then the map $\exp: \mathcal{U} \rightarrow M$ given by 
$$
\exp (q,v) = \gamma(1, q, v) = \gamma(|v|, q, \frac{v}{|v|}) , \;\; (q,v) \in \mathcal{U}
$$
- 특징
$$
\exp_q : B_{\varepsilon}(0) \subset T_qM \rightarrow M
$$
by $\exp_q(v) = \exp(q,v)$

- **Exponential Map**은 **Tangent Bundle에서 다양체로의 Mapping**이다. 

### Proposition 
Given $q \in M$, there exists an $\varepsilon > 0$ such that $\exp_q : B_{\varepsilon} (0) \subset T_q M \rightarrow M$ is a diffeomorphism of $B_{\varepsilon}(0)$ onto an open subset of $M$ 
$$
\begin{align}
d(\exp_q)_o(v) &= \frac{d}{dt}(\exp_q (tv))|_{t=0} = \frac{d}{dt}(\gamma(1, q, tv))|_{t=0} &\;\;\; \text{by definition of exponential map} \\
&= \frac{d}{dt}(\gamma(t, q, v))|_{t=0} = v &\;\;\;\text{by definition of exponential map}  
\end{align}
$$
Exponential Map의 시간에 대한 미분이 $v$ 이므로 이것은 $T_q M$을 만들기 위한 Parametrized Curve와 같다. 그러므로 Local Diffeomorphism.

![Fig05](http://jnwhome.iptime.org/img/DG/LARG_05.png)


#### Definition : Riemannian Log Map 
- Exponential Map은 Locally Diffeomorphism 이므로 $p \in M$ 주변에 **Neighborhood $V(p)$를 형성**한다.
- $V(p)$ 에서 **Exponential Map에 대한 역함수**가 존재하고 이를 **Riemannian Log Map**이라고 하며 다음과 같이 정의된다.

$$
\text{Log}_p : V(p) \rightarrow T_p M 
$$

#### Definition : Riemannian Distance by Riemannian Log Map  
- $V(p)$ 내의 임의의 점 $q$에 대하여 $p, q \in T_p M, p, q \in V(p)$ 사이에는 다음과 같은 Riemannian Distance를 정의한다.

$$
d(p,q) = \| \text{Log}_p (q) \|
$$

- 다음은 같은 표현이다.

$$
\text{Exp} (p, v) = \text{Exp}_p (v) \;\;\; \text{Log} (p,q) = \text{Log}_p (q)
$$

![Fig06](http://jnwhome.iptime.org/img/DG/LARG_06.png)


## Jacobi Field and Jacobi Equation

The Curvature $K(p,\sigma), \sigma \subset T_p M$ 은 Geodesic이 얼마나 빨리 잔행하는 가를 결정한다. 즉, 점 $p$에서 출발하여 Tangent Space $\sigma$ 에서 얼마나 빨리 Spread 되는가를 결정한다.

Geodesic의 변화 속도를 정확하게 계측하기 위한 수단이 Jacobi Field.
Jacobi Field 는 Geodesic을 따라 정의되는 Vector Field 이다. 


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



## References
[1] P.T.Fletcher, "Geodesic Regression and the Theory of Least Squares on Riemannian Manifolds", Proceedings of the Third International Workshop on Mathematical Foundations of Computational Anatomy - Geometrical and Statistical Methods for Modelling Biological Shape Variability. pp75-86, 2011 
International Journal of Computer Vision, Volume 105, Issue 2,  pp 171–185, November 2013.

[2] P.T, Fetchert, C. Lu, S, M, Pizer, S. Joshi, "Principal geodesic analysis for the study of nonlinear statistics of shape", IEEE tran. Medical Imaging, Vol. 23, No. 8, pp.995-1005, August 2004

[3] S. R. Buss, J. P. Fillmore, "Spherical average and applications to spherical spline analysis", Inform. Processing Med. Imag., pp.502-516, 2001

[4] P. Foggia, C. Sansone, and M. Vento, "A Riemannian Self-Organizing Map", ICIAP 2009, LNCS 5716, pp. 229–238, 2009.