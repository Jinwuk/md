Partial Differential Equation
==========================

## First Order Linear Equations

### The Constant Coefficient Equation

책에서보다 더 간단한 해법을 생각해보자.
$$
a u_x + b u_y = 0
$$
의 경우 우리는 이렇게 생각할 수 있다. (Geometric Method 기반으로 생각해 보면..)
$$
0 = \nabla u \cdot (a , b)^T = \nabla u \cdot (dx , dy)^T
$$
고로 
$$
\frac{dy}{dx} = \frac{b}{a}
$$ 
이므로
$$
dy = \frac{b}{a} dx \;\Rightarrow\; y=\frac{b}{a}x \; \Rightarrow\; bx - ay = c 
$$
그러므로 위 편미분 방정식의 해 $u(x,y) = f(by - ax)$ 로 치환시킬 수 있다. ($u(x,y)$는 $by -ax$ 에만 종속이므로)
즉, 
$$
u(x,y) = f(c) = f(by -ax)
$$
### The Variable Coefficient Equation
$$
u_x + y u_y=0
$$
의 경우 위의 해법을 생각해 보면 
$$
\frac{dy}{dx} = \frac{y}{1}
$$
따라서
$$
\frac{1}{y} dy = dx \;\Rightarrow y = C \cdot e^x 
$$
이를 $u(x,y)$ 에서 $y$를 치환하여 대입해 보면 (Chain Rule에 의해 이는 확실)
$$
\frac{d}{dx} u(x, Ce^x) = \frac{\partial u}{\partial x} + Ce^x \frac{\partial u}{\partial y} = u_x + y u_y = 0
$$
그리고 이에 따라, $x=0$ 일때 $y = C e^0 = C$ 이므로 
$$
C = e^{-x}y
$$
로 놓으면 $u(0, C) = u(0, e^{-0}y)$ 에서 $x$ 에 대하여 Independent 하므로 
$$
u(x,y)=f(e^{-x}y)
$$
가 된다.

## Well Posed Problems
#### Existence
There exists at least one solution $u(x,t)$ satisfying all these conditions.
#### Uniquness
There is at most one solution
#### Stability
The unique solution $u(x,t)$ depends in a stable manner on the data of the problem. This means that if the data are changed a little, the corresponding solution changes only a little.

## Types of PDE
### Simple Transport
$$
u_t + c u_x = 0
$$
### Vibrating 
#### Wave Equation (String: 1-Dimension)
$T(x,t)$ is magnitude of tension vector. $\rho$ is density.  
$$
u_{tt} = c^2 u_xx \;\;\textit{where } c= \sqrt{\frac{T}{\rho}}
$$

#### Laplacian
Vibrating Drumhead (Two-Dimensional)
$$
u_{tt} = c^2 \nabla \cdot (\nabla u) \equiv c^2 (u_{xx} + u_{yy})
$$

Three-Dimensional (light, Radar or electromagnetic wave, Linear lized supersonic wave ...)
$$
u_{tt} = c^2 (u_{xx} + u_{yy} + u_{zz})
$$

### Diffusion
Diffusion 방정식의 유도 
$$
M(t) = \int_{x_0}^{x_1} u(x,t) dx \Rightarrow \frac{dM}{dt} \int_{x_0}^{x_1} u_t(x,t) dx 
$$
그리고 Fick's Law 에 따라 
$$
 \frac{dM}{dt} = \text{Flow In} - \text{FLow Out} = k (u_x(x_1,t) - u_x(x_0,t))
$$
따라서
$$
\int_{x_0}^{x_1} u_t(x,t) dx =  k (u_x(x_1,t) - u_x(x_0,t))
$$
이를 $x_1$ 에 대하여 (즉, 확산 종단점에 대하여) 미분하면 다음의 Diffusion 방정식을 얻는다.


1-Dimensional
$$
u_t = k u_{xx}
$$
3-Dimensional
$$
u_t = k (u_{xx} + u_{yy} + u_{zz}) = k \Delta u
$$
When there exist externel source $f$ then it is a more general inhomogeneous equation
$$
u_t = \nabla \cdot (k \nabla) + f(x,t)
$$

### Heat Flow
$$
c \rho \frac{\partial u}{\partial t} = \nabla \cdot (\kappa \cdot \nabla u)
$$

### Schrodinger Equation
$$
-ihu_t = \frac{h^2}{2m}\Delta u + \frac{e^2}{r}u
$$

## Diffusion with a source
다음 방정식을 생각해보자
$$
u_t - ku_{xx} = f(x,t) \;\; \forall x \in (-\infty, \infty), t \in (0, \infty) \\
u(x,0) = \phi(x)
$$

위 방정식의  Solution은 다음과 같다.

$$
\begin{align}
u(x,t) &= \int_{-\infty}^{\infty} S(x-y,t)\phi(y)dy \\ 
       &+ \int_{0}^t \int_{-\infty}^{\infty} S(x-y, t-s)f(y,s)dy ds
\end{align}
$$

### 이해를 위한 ODE
$$
\frac{du}{dt} + Au(t)= f(t), \;\; u(0)= \phi \tag{1}
$$
방정식 (1)의 Solution은 다음과 같다.
$$
u(t)=e^{-tA}\phi + \int_0^t e^{(s-t)A}f(s)ds
$$
위 방정식을 유도하기 위해 다음과 같이 풀이한다. 
먼저 $f(t) = 0$ 의 경우, 즉, Homogeneous 경우를 생각해보자.
그 경우 
$$
u^H (t) = e^{-tA} \phi
$$
이다. 
이때 $S(t) = e^{-tA}$ 라 놓고, t=0인 경우에 대하여 생각해보자. 이 경우, 
$$
u(t)|_{t=0} = u^H (0) = e^{tA} \cdot e^{-tA} \phi = S(-t) u^H(t)
$$
로 볼 수 있다, 이를 $t$에 대하여 미분하면 ($u^H (t) = S(t) \phi$)
$$
\frac{d}{dt}(S(-t)u^H(t))= S(-t) \frac{du^H(t)}{dt} + A S(-t) u^H(t) 
$$
에서
$$
S(-t) \frac{du(t)}{dt} + S(-t) A u(t) = S(-t)f(t)
$$
가 됨을 알 수 있다. 그러므로
$$
\frac{d}{dt}(S(-t)u(t)) = S(-t)f(t) \Rightarrow S(-t)u(t) - \phi = \int_0^t S(-s)f(s)ds
$$
이고 이를 정리하면
$$
u(t)= S(t)\phi + \int_0^t S(t-s)f(s)ds
$$
이다. 

### Homogeneous PDE Solution에서 유도
$f(x,t) = 0$ 인 경우 Homogeneous Solution은 다음과 같다.
$$
u(x,t) = \int_{-\infty}^{\infty}S(x-y,t)\phi(y)dy = (\mathcal{L}(t) \phi)(x)
$$
이때 $x- y =0$ 즉, $x$ 항의 값이 0이 되는 경우 다음과 같다.
$$
u(t) = \mathcal{L}(t)\phi + \int_{0}^{t} \mathcal{L}(t-s)f(s)ds
$$
여기서 $x$항을 되살려 내면
$$
u(x,t) = \int_{-\infty}^{\infty} S(x-y,t)\phi(y)dy 
       + \int_{0}^t \int_{-\infty}^{\infty} S(x-y, t-s)f(y,s)dy ds
$$