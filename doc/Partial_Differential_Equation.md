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

