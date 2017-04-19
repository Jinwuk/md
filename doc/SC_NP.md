Short Course of Nonlinear Programming
=====================
## Preliminaries

### Continuity and Uniform Continuity
We say a function $f : \mathbb{R}^n \rightarrow \mathbb{R}^{m}$ is **continuous** at $x \in \mathbb{R}^n$
$$
\text{If}\;\; \forall \delta > 0, \;\; \exists \varepsilon > 0 \;\;\text{such that}\;\; \left\| f(y) - f(x) \right\| < \delta \;\; \forall y \in B^o(x, \varepsilon)
$$
(Locally Continuouis)

We say a function is **uniform continuous**
$$
\text{If}\;\; \forall \delta > 0, \;\; \exists \varepsilon > 0 \;\;\text{such that}\;\; \forall x \in \mathbb{R}^n,\; \left\| f(y) - f(x) \right\| < \delta \;\; \forall y \in B^o(x, \varepsilon)
$$
($\forall x \in \mathbb{R}^n$ 조건에 의해 매우 강력한 Continuous 가 된다, 전체 영역에 대한 것이므로..)

### Taylor's Formula (or Differentiable)
Suppose that $f:\mathbb{R}^n \rightarrow \mathbb{R}^m$ is continously differentiable (계속 미분 할 수 있다는 의미), then 
$$
\exists y, x \in \mathbb{R}^n, \;\; f(y) - f(x) = \int_0^1 \frac{\partial f(x+ s(y-x))}{\partial x} \cdot (y-x) ds
$$

### Twice Differentiable
Let $f:\mathbb{R}^n \rightarrow \mathbb{R}$ be a **twice differentiable**, then, for any $x, y \in \mathbb{R}^n$ 
$$
f(y) - f(x) = \langle \frac{\partial f(x)}{\partial x}, y-x \rangle + \int_0^1 (1-s) \langle y-x, \frac{\partial^2 f(x+s(y-x))}{\partial x^2}\cdot (y-x) \rangle ds 
$$


## First and Second Order Necessary Condition
### First Order Necessary Condition
Let $\Omega \supset \mathbb{R}^n, \;\; \textit{and} \;\; f \in \mathbb{C}^1, \Omega \Rightarrow \mathbb{R} $.
If $x^*$ is a local minimum point of $f$ over $\Omega$, then $\forall d \in \mathbb{R}^n$ that is a feasible direction at $x^*$, we have
$$
\langle \nabla f(x^*), d \rangle \geq 0
$$
- 간단히 말하면 $\nabla f(x^*) = 0$, 이것이 **First Order Necessary Condition**

### Second Order Necessary Condition
Suppose that $f$ is twice continuosly differentiable. Let $\hat{x}$ is a local minimizer of $\min_{x \in B^o(\hat{x}, \rho)} f(x)$.
Let $H = \frac{\partial^2 f}{\partial x^2}$ then $\langle h, H(\hat{x})h \rangle \geq 0, \;\; \forall h \in \mathbb{R}^n$.

즉, $x^*$ is Local Minimum 이면

|First order Necessary conditions | Second order Necessary conditions |
|---|---|
|$\langle \nabla f(x^*), h \rangle \geq 0, \;\; h \textit{: Feasible }$ | $\langle h, H(x^*) h \rangle \geq 0, \;\; h \textit{: Feasible }$ |


### Theorem F-3 (Sufficient Condition)
Suppose that  $f$ is twice differential continously differentiable, and that 
$$
\hat{x} \in \mathbb{R}^n \;\;\textit{such that}\;\; \nabla f(\hat{x}) = 0 \; \textit{and}\; H(\hat{x}) > 0
$$
Then $\hat{x}$ is a **local minimizer** of $f$.


## Convex Function
### Definition : Convex function
$f:\mathbb{R}^n \rightarrow \mathbb{R}$ is said to be ** convex **.  If $\forall x, y \in \mathbb{R}^n, \;\; \lambda \in [0,1]$, then 
$$
f(x + \lambda(y-x)) \leq (1-\lambda) f(x) + \lambda f(y)
$$

### Theorem : Differentiable and Convex
$f$ is **differentiable** then $f$ is **convex iff ** $\forall x, y \in \mathbb{R}^n$ 
$$
f(y) - f(x) \geq \langle \nabla f(x), (y-x) \rangle
$$

### Theorem : Twice Continuosly Differentiable and Convex
$f$ is **twice continuosly differentiable**, then $f$ is **convex iff** 
$$
\frac{\partial^2 f(x)}{\partial x^2} \geq 0, \forall x \in \mathbb{R}^n
$$

| Convex function | Differentiable and Convex  |
|---|---|
|![Fig13](http://jnwhome.iptime.org/img/Nonlinear_Optimization/01_13.png)| ![Fig16](http://jnwhome.iptime.org/img/Nonlinear_Optimization/01_16.png)|

## Gradient Method
### Corollary : descent direction
If $x \in \mathbb{R}^n$ is such that $\nabla f(x) \neq 0$, then any vector $h \in \mathbb{R}^n$ such that $\langle \nabla f(x), h \rangle < 0$ is a **descent direction** for $f(\cdot)$ at $x$. i.e. 
$$
\exists \hat{\lambda} > 0 \;\;\textit{such that}\;\; f(x+\hat{\lambda} h) - f(x) < 0
$$

## Directional Derivative
Directional Derivative at $x$ along the vector $h$  is defined by
$$
df(x,h) = \lim_{\lambda \downarrow 0} \frac{f(x+\lambda h) - f(x)}{\lambda} = \langle \nabla f(x ), h \rangle 
$$
where $\lambda \downarrow 0$ means that $\lambda$ is **Monotone decreasing** on $\mathbb{R}^+$, and $\lambda \uparrow 0$ is **Monotone increasing** on $\mathbb{R}^-$.
- If $f$ is continuosly differentiable, then

$$
f(x+\lambda h) - f(x) = \int_0^1 \langle \nabla f(x + s \lambda h), \lambda h \rangle \implies \frac{f(x+\lambda h) - f(x)}{\lambda} = \int_0^1 \langle \nabla f(x + s \lambda h), h \rangle ds
$$
Let $\lambda \downarrow 0$, then
$$
\begin{align}
df(x, h) &= \lim_{\lambda \downarrow 0} \frac{f(x+\lambda h) - f(x)}{\lambda} = \lim_{\lambda \downarrow 0} \int_0^1 \langle \nabla f(x + s \lambda h), h \rangle ds \\
&= \int_0^1 \langle \lim_{\lambda \downarrow 0} \nabla f(x + s \lambda h), h \rangle ds = \langle \nabla f(x ), h \rangle \int_0^1 ds = \langle \nabla f(x ), h \rangle  
\end{align}
$$
### Note
많은 경우 $x$를 생략하고 $df(h)$ 를 많이 사용한다. 여기서는 최적화 알고리즘 도출을 위해서 $x$를 굳이 표시하였다. 그 외에 $h[f] = h(f)$ 역시 같은 의미로 사용한다. (미분 기하학, 리만 기하학, 미분 다양체론 등에서..)

### Steepest Descent Algorithm.
$$
\textit{Problem: } \;\; \min_{x \in \mathbb{R}^n} f(x)
$$

| Procesdure | Processing|
|---|---|
| Data | $x_o \in \mathbb{R}^n$ |
| Step 0 | Set $i=0$ |
| Step 1 | Compute the Steepest Desvemt Direction |
|        | $ h_i = h(x_i) = -\nabla f(x_i)$ |
|        | Stop If $\nabla f(x_i) = 0$ |
| Step 2 | Compute the **Step Size** |
|      * | $\lambda_i = \lambda(x_i) = \arg \min_{\lambda \geq 0} f(x_i + \lambda h_i)$ |
| **Step 3** | Update $x_{i+1} = x_i + \lambda_i h_i$ replace $i $ by $i+1$ and goto Step 1 |

여기에서 **Step 2 * **는 실제로 Implementation 할 수 없다. (할 수 있는 것 처럼 보이지만, 해당 문제를 푸는 것이 Problem을 푸는 것과 같은 것)

### Armijo Gradient Algorithm
| Procesdure | Processing|
|---|---|
| Data | $x_o \in \mathbb{R}^n$, $\alpha, \beta \in (0,1)$ |
| Step 0 | Set $i=0$ |
| Step 1 | Compute the Descent Direction |
|        | $ h_i = h(x_i) = -\nabla f(x_i)$ |
|        | Stop If $\nabla f(x_i) = 0$ |
| Step 2 | Compute the Step Size rule |
|      * | ![Fig_Armio](http://jnwhome.iptime.org/img/Nonlinear_Optimization/Armijo_01.svg)|
| **Step 3** | Update $x_{i+1} = x_i + \lambda_i h_i$ replace $i $ by $i+1$ and goto Step 1 |

### Note : Core Idea 
The following rule of adaptation gain is the key idea of Armijo's algorithm 
$$
\lambda_i = \lambda(x_i) \triangleq \arg \max_{k \in \mathbb{N}} \{ \beta^k | f(x_i + \beta^k h_i) - f(x_i) \leq - \beta^k \cdot \alpha \left\| \nabla f(x_i) \right\|^2 \}
$$
Let $\phi(\lambda) = f(x + \lambda h(x)) - f(x)$
$\phi(0) = 0$
$\phi'(0) = - \left\| \nabla f(x) \right\|^2 |_{x=0}$

![Fig25](http://jnwhome.iptime.org/img/Nonlinear_Optimization/01_25.png)

### Newton's Method (Local case)
#### Assumption
- $f$ is twice differentiable and locally Lipschitz, i.e. for given any bounded set $S \subset \mathbb{R}^n$
$$
\exists L < \infty \;\; \textit{such that} \;\; \forall x', x \in S, \;\; \| f(x') - f(x) \| \leq L \| x' - x \|
$$
- $\exists 0 < m \leq M < \infty$ such that $\forall y \in \mathbb{R}^n$
$$
m \|y\|^2 \leq \langle y, H(\hat{x}) y \rangle \leq M \| y \|^2
$$
#### Basic Idea 
For given $f(\cdot)$, we can approximate $f$ near $x_i$ using a Taylor expansion such that
$$
f(x) \approx f(x_i) + \langle \nabla f(x_i), x - x_i \rangle + \frac{1}{2} \langle x - x_i, H(x_i)(x - x_i) \rangle 
$$
Local minimizer $x$에서 $\nabla f(x)=0$ 이므로 위 식을 $x$에 대하여 미분하면
$$
\nabla f(x) = \nabla f(x_i) + H(x_i)(x - x_i) = 0 \Rightarrow x - x_i = -H^{-1}(x_i) \nabla f(x_i)
$$
| Procesdure | Processing|
|---|---|
| Data | $x_o \in \mathbb{R}^n$ |
| Step 0 | Set $i=0$ |
| Step 1 | Compute the Search Direction |
|        | $ h_i =  -\nabla f(x_i)$ |
|        | Stop If $\nabla f(x_i) = 0$ |
| Step 2 | Compute the **Step Size** |
|      * | $\lambda_i = \lambda(x_i) = \arg \min_{\lambda \geq 0} f(x_i + \lambda h_i)$ |
| **Step 3** | Update $x_{i+1} = x_i + \lambda_i h_i$ replace $i $ by $i+1$ and goto Step 1 |

- $\lambda_i$를 Armijo 방식에 의해 선정하면 Armijo Newton 

## Quadratic Problem and Conjugate Descent
Consider the Following problem 
$$
\min f(x) : f(x) = \frac{1}{2} \langle x, Qx \rangle -b^T x, \;\;\; \textit{Minimizer}\;\; x^* = Q^{-1}b
$$
Let $\nabla f(x) = Qx - b = g(x)$ 
Steepest Descent Method를 생각해 보면
$$
x_{i+1} = x_i - \alpha_i g_i, \;\;\;\textit{where}\;\; \alpha_i = \arg \min_{\lambda \geq 0} f(x_i - \lambda g_i) \implies \frac{\partial f}{\partial \lambda} = 0
$$
그러므로
$$
0 = \frac{\partial f}{\partial \lambda} = \frac{\partial f}{\partial x} \frac{\partial x}{\partial \lambda} = \left( Q(x - \lambda g_i) -b \right)^T (-g_i) \implies \lambda g_i^T Q g_i = (Qx - b)^T g_i \implies \lambda = \frac{g_i^T g_i}{g_i Q g_i}
$$
위 $\lambda$를 만족하면 **현재 Gradient와 다음 Gradient가 Orthogonal** 하다는 의미가 된다. 이런 특성을 사용하여 만든 최적화 알고리즘이 Conjugate Gradient

| Procesdure | Processing|
|---|---|
| Data | $x_o \in \mathbb{R}^n$ |
| Step 0 | Set $i=0$, $h_0 = g_0 = Qx_0 - b$ |
| Step 1 | Compute the Step Length |
|        | $ \lambda_i = \arg \min f(x_i - \lambda h_i)$ |
| Step 2 | Update : Set |
|        | $ x_{i+1} = x_i - \lambda_i h_i$  |
|        | $ g_{i+1} = Qx_{i+1} - b$ |
|        | $ h_{i+1} = g_{i+1} + h_i r_i$ with $r_i = -\frac{\langle Qh_i, g_{i+1} \rangle}{\langle h_i, Q h_i \rangle}$|
|        | so  that $\langle h_{i+1}, Qh_i \rangle = 0$ ... 이렇게 되어야 $\{g_k \}$ 가 Orthogonal | 
| Step 3 | replace $i $ by $i+1$ and goto Step 1 |

### General Conjugate Gradient 
Quadratic 문제가 아니어도 Conjugate Algorithm을 적용하고 싶다. 그런데 Hessian이 Constant가 아닌 경우, Orthogonal한 Gradient를 어떻게 보장할 것인가?
- 일반적인 Quadratic 문제에서는 다음과 같이 $r_i$를 선정한다 
$$r_i = -\frac{\langle Qh_i, g_{i+1} \rangle}{\langle h_i, Q h_i \rangle}$$
이렇게 되면  $\langle h_{i+1}, Qh_i \rangle = 0$ 이 되고 이렇게 되어야 $\{g_k \}$ 가 Orthogonal해진다. (증명 생략)

- $r_i$를 잘 선정하여 최대한 Orthogonal 하게 만들어 본다.
- Polak-Riebel : Gradient 끼리 조금 Orthogonal이 아니라면 해당 부분 만큼 보정 한다.
$$ r_i = \frac{ \langle \nabla f(x_{i+1}) - \nabla f(x_i), \nabla f(x_{i+1}) \rangle }{ \| \nabla f(x_i) \|^2 }$$
- Fletcher-Reeves : Gradient는 다 Orthogonal 하다고 본다.
$$ r_i = \frac{ \| \nabla f(x_{i+1}) \|^2 }{ \| \nabla f(x_i) \|^2 }$$

## Quasi Newton
원래 Newton Method는 $x_{i+1} = x_i - \lambda_i H^{-1} \nabla f(x_i)$ 의 형태이다. 그런데, 최적화 문제가 언제나 Quadratic 하게 주어지는 것도 아니므로 Hessian이 고정적으로 나타난다는 보장도 없다. 따라서 Hessian을 최대한 Estimation하여 Newton Method와 비슷하게 만들고자 하는 것이 Quasi Newton이다.

### Idea
Problem $\min_{x \in \mathbb{R}^n} f(x)$ 에서 $f(x)$를 $f(x) = \langle d, x \rangle + \frac{1}{2} \langle x, Hx \rangle $ 로 근사화 하고 Search Direction을 다음에서 구한다.
$$
h(x) = \arg \min \{ \frac{1}{2} \| h \|_{H}^2 + df(x, h) \}
$$
- $H = I$ 인 경우 Steepest Descent와 똑같다. 
  $$
    \frac{\partial h(x)}{\partial h} = \frac{\partial}{\partial h} (\frac{1}{2} h^T h + \nabla f(x)^T h) = h + \nabla f(x) = 0 \implies h = -\nabla f(x)
  $$
- $H$가 Hessian이면 Newton Method와 같다.
  $$
    \frac{\partial h(x)}{\partial h} = \frac{\partial}{\partial h} (\frac{1}{2} h^T H h + \nabla f(x)^T h) = Hh + \nabla f(x) = 0 \implies h = -H^{-1}\nabla f(x)
  $$
- Hessian을 근사화 하기 위해 $B$ 라는 것을 놓자. 위의 Newton Method를 보면 $\Delta x_i = x_{i+1} - x_i$, $g_i = \nabla f(x_i) = H x_i + d$ 에서 
$$
\Delta g_i = g_{i+1} - g_i = H x_{i+1} + d - H x_i - d = H \Delta x_i \;\;\; \therefore H^{-1} \Delta g_i = \Delta x  
$$
여기서 $B \Delta g_i = \Delta x_i, \;\; \forall i$ 라고 생각하고 $\Delta_i$ 가 Nonsingular 이면 $B = H^{-1} = \Delta X \Delta G^{-1}$ 로 생각한다. ($\Delta X$는 $\Delta x_i$를 Column vector로 놓은 Matrix, $\Delta G$는 $\Delta g_i$를 Column Vector로 놓은 Matrix)

#### DFP and BFGS 알고리즘
$B$ 를 업데이트 하는 방법에 따라 두 가지 알고리즘이 나온다.
##### DFP : Davison, Fletcher, Powell 알고리즘
$$
B_{i+1}^{DFP} = B_i + \frac{1}{\langle \Delta x_i, g_i \rangle}\Delta x_i \Delta x_i^T - \frac{(B_i \Delta g_i)(B_i \Delta g_i)^T}{\langle \Delta g_i, B_i \Delta g_i \rangle}
$$
##### BFGS : Broyden, Fletcher, Goldfarb, Shanno 알고리즘
$$
B_{i+1}^{BFGS} = B_i + \left( \frac{1 + \Delta x_i^T B_i \Delta x_i}{\langle \Delta x_i, \Delta g_i \rangle} \right) \frac{\Delta g_i \Delta g_i^T}{\langle \Delta g_i, \Delta x_i \rangle} - \frac{\Delta g_i \Delta x_i^T B_i + B_i \Delta x_i \Delta g_i^T}{ \langle x_i, \Delta g_i \rangle}
$$

| Procesdure | Processing|
|---|---|
| Data | $x_o \in \mathbb{R}^n$, $B_0$ : a symmetric positive definite $n \times n$ Matrix , ex) $I \in \mathbb{R}^{n \times n}$|
| Step 0 | Set $i=0$,  |
| Step 1 | If $g_i = 0$ Stop else continue |
|        | Step size rule : |
|        | $ \lambda_i = \arg \min f(x_i - \lambda B_i g_i)$ where $g_i = \nabla f(x_i)$|
| Step 2 | Update  |
|        | $ x_{i+1} = x_i - \lambda_i B_i g_i$  |
|        | $ \Delta x_i = x_{i+1} - x_i $ |
|        | $ \Delta g_i = g_{i+1} - g_i $|
|        | $B_{i+1} = B_{i+1}^{DFP}$ or  $B_{i+1} = B_{i+1}^{BFGS}$| 
| Step 3 | replace $i $ by $i+1$ and goto Step 1 |

- Broyden Family 
$$
B_{i+1}^{\phi} = \phi B_{i+1}^{DFP} + (1 - \phi) B_{i+1}^{BFGS} \;\; \phi \in [0, 1]
$$

## Constrained Optimization
제약 조건이 있는 최적화는 보통 Khun-Tucker Condition에 의해 해석하게 된다.

