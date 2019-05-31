General Eigenvalue Problems
====

Fourier Transform의 기본을 알기 위하여  PDE 분석을 하고자 한다. 그 전에 먼저 PDE 에서Eigenvalue의 특성을 살펴 본다. 

개인적으로 나는, CNN 의 다층 Convolution - Pooling 과정이 미분 다양체 상에서의 합성 Differential mapping 과정이라 예상하고 있으며 Fourier Transform과 PDE의 과정이 이와 같은 것이라는 증명을 통해, CNN의 과정이 Linear Transform 혹은 비선형  Transform 과정임을 증명하고자 한다.

그 전에 Eigenvalue Problem 분석에서 다음과 같이 진행이 된다.
**Eigenvalue는 어떤 제한조건하에서 언제나 에너지 함수를 최소화** 시킨다는 것을 증명할 것이다. 이를 역 이용하여, 해당 상태에서 EIgenvalue를 구하는 방법을 제시할 것이다.
그 다음에는 일반 EIgenvalue 문제, Sturm-Liouville 문제와 같은 것을 체크할 것이다.

## The Eigenvalues are Minima of the Potential Energy
다음의 Dirichlet Bounadry 를 가지는 Eigenvalue 문제를 생각해 보자.

$$
- \Delta u = \lambda u \text{ in }D, \;\;\; u = 0 \text{ on the bdy } D
\tag{1}
\label{eig01}
$$

여기서 $D$는 임의의 Domain으로서 Piecewise  smooth 이다. 여기서 Eigenvalue는 다음과 같다고 하자.

$$
0 < \lambda_1 \leq \lambda_2 \leq \leq \lambda_3 \cdots \lambda_n \leq \cdots
$$

Section 7.1 Dirichlet Priniple 에서 Energy 를 최소화 시키고 inhomogeneous boundary condition을 만족시키는 함수는 **Harmonic** 함수.

Function $E : u \rightarrow \mathbf{R}$ for $u \in \mathbf{R}^n$ 을 생각하자.  그러면 $\nabla E(u) = 0$ 에서 Minimum 이다. 그리고 제한조건 $F(u) = 0$  이 주어지면 Kuhn-Tucker Condition에 의해 $\nabla E(u) = \lambda \nabla F(u)$ 이다. 이때, $\lambda$ 는 Lagrange Multiplier 이다.

다음의 Minimum 문제를 생각해 보자

$$
m = \min \left\{ \frac{\| \nabla w \|^2}{\| w \|^2 } : w=0 \text{ on bdy } D, w \not\equiv 0 \right\}
\tag{MP}
\label{Eigen02}
$$

where $w(x) \in C^2$. 

이는 다음 **Rayleigh quotinet** $Q$를 최소화 시키는 것과 같음

$$
Q = \frac{\| \nabla w \|^2}{\| w \|^2}
$$

이때 Solution of $\eqref{Eigen02}$ 은  $\forall w=0 \text{ on bdy } D, w \not\equiv 0$ (Boundary에서 0 이고 그 외에서는 0이 아니다.) 에 대하여 다음을 만족한다.

$$
\frac{\| \nabla u \|^2}{\| u \|^2} \leq \frac{\| \nabla w \|^2}{\|w\|^2}
\tag{2}
\label{defmin}
$$

### Theorem 1. Minimum Principle for the First Eigenvalue
Assume that $u(x)$ is a solution of $\eqref{Eigen02}$. Then the value of the minimum equals the first (smallest) eigenvalue $\lambda_1$ of $\eqref{eig01}$, and $u(x)$ is its eigenfunction.
That is
$$
\lambda_1 = m = \min \left\{  \frac{\| \nabla w \|^2}{\| w \|^2} \right\} \text{ and } -\Delta u = \lambda_1 u \text{ in } D.
$$

즉, **The first eigenvalue is the minimum of the energy**.  그리고 first eigen function $u(x)$을 Ground State라고 한다. 

#### proof
Let $v(x)$ be any other trial function, and let $w(x) = u(x) + \epsilon v(x)$ where $\epsilon$ is any contant. Then

$$
f(\epsilon) = \frac{\int \| \nabla (u + \epsilon v) \|^2}{\| u + \epsilon v \|^2}
\tag{3}
\label{1_proof}
$$

정의 $\eqref{defmin}$에 의해 $\epsilon = 0$ 일때, $\eqref{1_proof}$는 최소값을 가진다.  이를 전개하면

$$
f(\epsilon) = \frac{\int \left( \| \nabla u \|^2 + 2 \epsilon \nabla u \cdot \nabla v + \epsilon^2 \| \nabla v \|^2 \right)}{\int (u^2 + 2\epsilon u \cdot v + \epsilon^2 v^2 )}
$$

여기에서 $F(\epsilon) = \| \nabla u \|^2 + 2 \epsilon \nabla u \cdot \nabla v + \epsilon^2 \| \nabla v \|^2$,  $G(\epsilon) = u^2 + 2\epsilon u \cdot v + \epsilon^2 v^2$ 로 놓으면 

$$
\begin{aligned}
\frac{\partial }{\partial \epsilon} f(\epsilon) 
&= \frac{\partial }{\partial \epsilon} F(\epsilon) G^{-1}(\epsilon) \\
&= \frac{\partial F(\epsilon)}{\partial \epsilon} G^{-1}(\epsilon) + F(\epsilon) \frac{\partial G^{-1}(\epsilon)}{\partial \epsilon}\\
&= \frac{\partial F(\epsilon)}{\partial \epsilon} G^{-1}(\epsilon) + G^{-2}(\epsilon) F(\epsilon) \frac{\partial G(\epsilon)}{\partial \epsilon} \\
&= G^{-2}(\epsilon) \cdot \left( \frac{\partial F(\epsilon)}{\partial \epsilon} G(\epsilon) +  F(\epsilon) \frac{\partial G(\epsilon)}{\partial \epsilon} \right)
\end{aligned}
$$

여기에서
$$
\begin{aligned}
\frac{\partial F(\epsilon)}{\partial \epsilon} \bigg\vert_{\epsilon = 0}
&= 2 \nabla u \cdot \nabla v + 2 \epsilon \| \nabla v \|^2  \bigg\vert_{\epsilon = 0}
= 2 \nabla u \cdot \nabla v \\
\frac{\partial G(\epsilon)}{\partial \epsilon} \bigg\vert_{\epsilon = 0}
&= 2 u \cdot v + 2 \epsilon v^2 \bigg\vert_{\epsilon = 0}
= 2 u \cdot v
\end{aligned}
$$

이를 대입하여 정리하면

$$
0 = f'(0) = \frac{(\int u^2)(2 \int \nabla u \cdot \nabla v) - (\int \| \nabla u \|^2)(2 \int uv)}{(\int u^2 )^2}
$$

그러므로 
$$
\begin{aligned}
(\int u^2)(2 \int \nabla u \cdot \nabla v) &= (\int \| \nabla u \|^2)(2 \int uv) \\
\int \nabla u \cdot \nabla v &= \frac{\int \| \nabla u \|^2}{\int u^2} \int uv = m \int u v\\
\end{aligned}
$$

Green's First Identity $\eqref{Green01}$ 에서, 좌항  $\int_{\partial D} v \frac{\partial u}{\partial n} dS$ 은 위  가정에서  $v = 0$ 이므로 $\int_{\partial D} v \frac{\partial u}{\partial n} dS = 0$,  이를 정리하면


$$
\begin{aligned}
0 
&= \int_{\partial D} \nabla v \cdot \nabla u dx + \int v \Delta u dx = m \int u v dx + \int v \Delta u dx \\
&= \int (\Delta u + m u)(v) dx
\end{aligned}
\tag{4}
\label{2_proof}
$$
식 $\eqref{2_proof}$ 가  성립하기 위해서는 $\Delta u + mu = 0$ 이어야 한다. 이떄, $m$은 **Eigenvalue** .  즉,  the minimum value $m$ of $Q$ 는 Eigenvalue of $-\nabla$ 이고 $u(x)$ 는 이것의 **Eigen Function** 이 된다. 

$m$ 이 the smallest eigenvalue of $-\Delta$ 임을 보이자.

Let $-\Delta v_j = \lambda_j v_j$ 라 놓으면 Green's First Identity에 의해 다음과 같다. 
$$
m \leq \frac{\int \| \nabla v_j \|^2}{\int v_j^2} = \frac{\int -\Delta v_j (v_j)}{\int v_j^2} = \frac{\int \lambda_j v_j \cdot v_j}{\int v_j^2} = \lambda_j
$$
**Q.E.D**

## Minimum Principle for the n-th Eigenvalue

Suppose that  $\lambda_1 , \cdots \lambda_{n-1}$ are already known, with the **eigenfunctions** $v_1(x), \cdots v_{n-1}(x)$, respectively. Then

$$
\begin{aligned}
\lambda_n = \min \left\{  \frac{\| \nabla w \|^2}{\| w \|^2} \right\} : w \not\equiv, w=0 \text{ on cdy } D, w \in C^2, \\
0 = (w, v_1) = (w, v_2) = \cdots = (w, v_{n-1})
\end{aligned}
\tag{MPn}
\label{MPn}
$$



### proof

Let $u(x)$ denote the minimizing function for $\eqref{MPn}$ , 이는 가정에 의해 존재한다.

Let $m^*$ denote this new minimum value, so that  $m^*$ is the value of the Rayleigh quotient at $u(x)$.

따라서, $u=0$ on $\partial D$, and $u$ is orthogonal to $v_1, \cdots v_{n-1}$. 그리고 quotient $Q$는 $w$에 대하여 $u$ 보다 작다고 하자.



Theorem 1 에서 $w = u + \epsilon v$로 놓자. 그럼, 앞에서와 마찬가지로












## Green's First Identity 

### Definitions

$$
\forall u : \mathbf{R}^3 \rightarrow \mathbf{R} \Delta u = \nabla \cdot \nabla u = u_{xx} + u_{yy} + u_{zz}
$$

$$
\| \nabla u \|^2 = u_x^2 + u_y^2 + u_z^2
$$

#### Divergence Theorem

$$
\int \int \int_{D} \nabla \cdot F dx = \int \int_{\partial D} \mathbf{F} \cdot \mathbf{n} dS
$$
where $\partial D$ is $\text{bdy } D$, $D$ is a bounded solid region, and $\mathbf{n}$ is the unit outer normal on $\text{bdy } D$.

### Green's First Identity 

Let $\mathbf{n} = \frac{\partial u}{\partial n}$ and $\mathbf{F} = v$. Then, by the law of partial integration, we can obtain

$$
\int \int_{\partial D} v \frac{\partial u}{\partial n} dS = \int \int \int_{\partial D} \nabla v \cdot \nabla u dx + \int \int \int v \Delta u dx
\tag{G1}
\label{Green01}
$$

where  $\partial u /\partial n = \mathbf{n} \cdot \nabla u$ is the **directional derivative** in outward normal direction. 

### Example

만일  $v \equiv 1$ 이면 다음과 같다.

$$
\int \int_{\partial D} \frac{\partial u}{\partial n} dS = \int \int \int_{D} \Delta u dx 
\tag{G2}
\label{GreenEx01}
$$

여기서 다음과 같은 *Neumann Problem*을 생각해 보자

$$
\begin{cases}
\Delta u = f(x) & x \in D \\
\frac{\partial u}{\partial n} = h(x) & x \in \partial D
\tag{G3}
\label{Neumann01}
\end{cases}
$$

By $\eqref{Green01}$, we have

$$
\int \int_{\partial D} h dS = \int \int \int_D f dx
\tag{G4}
\label{GreenEx02}
$$

즉, Neumann Problem $\eqref{Neumann01}$ 이 해를 가지려면 $\eqref{GreenEx02}$가 성립해야 한다.
