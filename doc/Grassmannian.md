Grassmannian
====

## Introduction
Linear Optimal Control로 살펴본 Grassmannian 개요

- For $x \in \mathbf{R}^n$ 
    $$
    \dot{x} = Ax + bu
    $$

-  Object function
    $$
    J = \frac{1}{2} \int \left( x^T Q x + u^T R u \right) dt = \int \mathcal{L}(x, u) dt
    $$
    - where $Q, R$ is a symmetric matrices and $R$ is invertible.

- Hamiltonian to the object function $J$
    $$
    \mathcal{H} = \min_{u} \left( \mathcal{L} + p^T (Ax +bu) \right)
    $$
    - Hamiltonian 해석 방정식들은 다음과 같다.
        $$
        \begin{aligned}
        \frac{\partial \mathcal{H}}{\partial u} &= 0 &\text{Optimal control condition}\\
        \dot{x}^T &= \frac{\partial \mathcal{H}}{\partial p} &\text{System condition}\\
        -\dot{p}^T &= \frac{\partial \mathcal{H}}{\partial x} &\text{Co-state equation}
        \end{aligned}
        $$

    - 이를 Hamiltonian에 적용하면
$$
\begin{aligned}
0 = \frac{\partial \mathcal{H}}{\partial u} = \frac{\partial}{\partial u}(u^T R u) + \frac{\partial}{\partial u} (p^T b u) = R u + b^T p \\
u = - R b^T p \\
\dot{x} = Ax + bu = Ax -b R b^T p \\
-\dot{p} = \frac{\partial \mathcal{H}}{\partial x} = \frac{\partial }{\partial x} (x^T Q x + p^T A x) = Qx + A^T p  
\end{aligned}
$$

- 여기서, Largrangian Multiplier $p$의 시간에 대한 변화율은 $x$와 $p$의 결합으로 되어 있어 이에 대한 해석이 어렵다.
- Largrangian Multiplier $p$를 해석하기 위한 Matrix 방정식을 놓으면
    $$
    \frac{d}{dt} 
    \begin{bmatrix} 
    x \\ p
    \end{bmatrix}
    = 
    \begin{bmatrix} 
    A & b R b^T \\ -Q & -A^T
    \end{bmatrix}
    \begin{bmatrix} 
    x \\ p
    \end{bmatrix}
    = H 
    \begin{bmatrix} 
    x \\ p
    \end{bmatrix}
    $$
    - 간략히 보기위해 $\begin{bmatrix} x \\ p  \end{bmatrix} = v$라 놓으면 위 방정식은 
        $$
        \dot{v} = H v
        $$

- **두개의 변수가 결합된 미분방정식을 해석 하기 위해 다음과 같은 변환을 생각해보자**
    $$
    \begin{bmatrix}
    \xi \\ \eta
    \end{bmatrix}
    =
    \begin{bmatrix}
    I & 0 \\ -P & I
    \end{bmatrix}
    \begin{bmatrix}
    x \\ p
    \end{bmatrix}
    ; \;\;\;
    \begin{bmatrix}
    x \\ p
    \end{bmatrix}
    = 
    \begin{bmatrix}
    I & 0 \\ P & I
    \end{bmatrix}
    \begin{bmatrix}
    \xi \\ \eta
    \end{bmatrix}
    $$
    - 위 방정식을 다음과 같이 간략하게 표현한다.
        $$
        \gamma = \tilde{\mathbf{P}} v, \;\;\; v = \mathbf{P} \gamma
        $$

    - 여기에서 $\gamma = \begin{bmatrix} \xi \\ \eta \end{bmatrix}$ 이고, $\tilde{\mathbf{P}} = \begin{bmatrix} I & 0 \\ -P & I \end{bmatrix}$, $\mathbf{P} = \begin{bmatrix} I & 0 \\ P & I \end{bmatrix}$. 
    - 따라서, $\xi, \eta, x, p$ 간의 관계는 다음과 같다. 
        $$
        \begin{aligned}
        \xi  &= x \\
        \eta &= \tilde{\mathbf{P}}x + p
        \end{aligned}
        $$

- Hamiltonian 해석을 위해 $\gamma$의 시간에 대한 미분을 계산해 보면 

$$
\dot{\gamma} = \frac{\partial }{\partial t}(\tilde{\mathbf{P}} v) = \frac{\partial \tilde{\mathbf{P}}}{\partial t} v + \tilde{\mathbf{P}} \dot{v} = \frac{\partial \tilde{\mathbf{P}}}{\partial t} \mathbf{P} \gamma + \tilde{\mathbf{P}} H v = \left( \frac{\partial \tilde{\mathbf{P}}}{\partial t} + \tilde{\mathbf{P}} H \right) \mathbf{P} \gamma 
$$

- 따라서 다음과 같다.
    $$
    \begin{aligned}
    \frac{d}{dt}
    \begin{bmatrix}
    \xi \\ \eta
    \end{bmatrix}
    &= \left(   
    \begin{bmatrix}
    I & 0 \\ -\dot{P} & I
    \end{bmatrix}
    + 
    \begin{bmatrix}
    I & 0 \\ -P & I
    \end{bmatrix}
    \begin{bmatrix}
    A & bRb^T \\ -Q & -A^T
    \end{bmatrix}
    \right)
    \begin{bmatrix}
    I & 0 \\ P & I
    \end{bmatrix}
    \begin{bmatrix}
    \xi \\ \eta
    \end{bmatrix} \\
    &= 
    \begin{bmatrix}
    A -bR^{-1}b^TP & -bR^{-1}b^T \\
    Z(P) & -(A - bR^{-1}b^TP)^T
    \end{bmatrix} 
    \begin{bmatrix}
    \xi \\ \eta
    \end{bmatrix} 
    \end{aligned}
    $$
    - where $Z(P) = - \dot{P} - PA - A^T P + P bR^{-1}b^T P - Q$

- 이 방정식을 살펴보면
	- $\xi = x$ 이고 $p$와 관련이 없으므로 $p$와 관련이 있는 $\eta$가 $\dot{\xi}$에 들어오면 안됨. 고로, $\dot{\xi} = (A - bR^{-1}b^T P)\xi$ 이어야 함 그런데, $b R^{-1}b^T \neq 0$ 이므로 결국, 초기 조건에서 $\eta = 0$ 이어야 하며 $\dot{\eta} = 0$ 이어야 함.
	- 초기조건에서 $\eta = 0$ 이라면, $\dot{\eta} = Z(P) \xi - (A - bR^{-1}b^TP)^T \eta |_{\eta = 0} = Z(P) \xi = 0$ 에서 $Z(P) = 0$ 이어야 함, 그러므로 (**Riccati Equation**)
    $$
    -\dot{P} = PA + A^T P - P b R^{-1} b^T P + Q 
    $$
  
    - 그리고 $\eta = \tilde{\mathbf{P}}x + p$ 에서 $\eta = 0$ 이므로 $0 = -Px + p$ 그러므로
        $$
        p(t_0) = P(t_0) x(t_0)
        $$

- 위 방정식이 의미하는 바는, Largrange Multiplier $p(t)$ 는 state $x(t)$를 어떤 n-dimensional 다양체위의 curve로 Mapping 한다는 것이다. 
- 최적화 문제를 풀기 위한 방법론에서 Largrangian Multiplier를 해석하기 위하여 나온 방법 (by Grassmann) 
	- parameter set으로 형성된 space상의 curve (state)와 parameter space상에 유도된 curve (Largrangian Multiplier)간의 Realtion의 연구에서 비롯 
	- 여기서 $x(t)$는 어떤 parameter를 가진 curce 이고, Riccati 방정식으로 부터 유도된 Transformation에 의해, Largrangian Multiplier와 연결된다.

## The Grassmanian $G^p(V)$
