Mathematical Note
======

## $\frac{\varepsilon}{1 + \varepsilon}$을 사용한 확률 공간에서의 Convexity

Convex 조건은, Domain이 만일 Convex 일 경우 함수 $f$가 contimuity 이면 (최소 Lipschitz Continuous 이면), Compact 조건을 만족 시킬 수 있기 떄문에 알고리즘의 수렴성 증명에 있어 매우 중요한 조건이다.

그러나 일반적인 Convex 조건에 의한 해석은 Euclidean Space $\mathbb{R}^n$ 이나, Manifold $M^n$ 에서 유효하다.
따라서 확률 변수의 경우에는 적분이 무한대 적분으로 나타내야 하는 경우가 많으므로 Convexity를 체크하기 위한 변수가 무한대에서 잡을 수 있도록 해야 한다.
따라서 이러한 경우, Convexity 변수를 $s \in \mathbb[0,1]$ 대신 $\varepsilon \in \mathbb{R}$로 놓아야 한다. 이때 Convexity Check를 위해 다음과 같이 놓아보자.

$$
\lim_{\varepsilon \rightarrow 0} \frac{\varepsilon}{1 + \varepsilon} = 0, \;\;\; \lim_{\varepsilon \rightarrow \infty} \frac{\varepsilon}{1 + \varepsilon} = 1 
$$

그러므로 어떤 확률 변수 $z$가 1과 z에서 변한다고 가정해 보자. 즉 Deterministic 한 경우에
$$
h = x + s(y - x), \;\;\; s \in \mathbb{R}[0,1]
$$

이것을 1과 z과 바꾸어 보면
$$
h = z + s(1 - z) = (1 -s)z + s = \left( 1 - \frac{\varepsilon}{1 + \varepsilon} \right) z + \frac{\varepsilon}{1 + \varepsilon} = \frac{1}{1 + \varepsilon}z + \frac{\varepsilon}{1 + \varepsilon}
$$

z와 1의 위치를 바꾸어 보면
$$
h = 1 + s(z - 1) = (1 -s) + sz = \left( 1 - \frac{\varepsilon}{1 + \varepsilon} \right) + \frac{\varepsilon}{1 + \varepsilon}z = \frac{1}{1 + \varepsilon} + \frac{\varepsilon}{1 + \varepsilon}z
$$


## Quadratic Problem
$$
\min f(x) : f(x) = \frac{1}{2} \langle x, Qx \rangle - b^T x \;\;\;\;\;\text{ minimizer } x^* = Q^{-1}b \\
\nabla f(x) = Qx -b = g(x)
$$ 

### Steepest Descent
$$
x_{i+1} = x_i - \alpha_i g_i \;\;\;\;\; \alpha_i = \underset{\lambda \geq 0}{\arg}  \min f(x_i - \lambda g_i) 
$$

식(1)에서 최적 $\lambda$ 는 $\frac{\partial f}{\partial \lambda} = 0$을 만족한다. 즉, ($x = x_{i+1}$ 이라 생각하면, 그리고 $\alpha$ 대신 $\lambda$로 놓고 생각하면)
$$
\frac{\partial f}{\partial \lambda} = {\frac{\partial f}{\partial x}}^T \frac{\partial \lambda}{\partial x} = [Q(x_i - \lambda g_i) -b]^T(-g_i) = 0 
$$
그러므로
$$
\begin{align}
0 &= [Q(x_i - \lambda g_i) -b]^T g_i = x_i^T Q g_i - \lambda g_i^T Q g_i -b^T g_i = (Q x_i - b)^T g_i - \lambda g_i^T Q g_i \\
&\implies (Q x_i - b)^T g_i = g_i^T g_i = \lambda g_i^T Q g_i \\
&\implies \lambda = \frac{g_i^T g_i}{g_i^T Q g_i}
\end{align}
$$
따라서,
$$
x_{i+1} = x_i - \frac{g_i^T g_i}{g_i^T Q g_i} g_i
$$

### Usage 
for any $f(x)$ and given $x_i$ $\rightarrow$ Second order approximation near $x_i$
$$
f(x) \approx f(x_i) + \langle \nabla f(x_i), x- x_i \rangle + \frac{1}{2} \langle x - x_i, Q(x- x_i) \rangle
$$

이러면 문제 
$$
\left. \min f(x) \right|_{x^*}, \;\;\; f(x) = \frac{1}{2} \langle x, Qx \rangle - b^T x
$$
에 대하여  Let
$$
F(x) = \frac{1}{2} \langle x - x^*, Q(x - x^*)
$$

#### Claim  
$\min f(x)$ and $min F(x)$ construct a same $\{x_i\}$ sequence.
#### proof of claim
$$
\begin{align}
F(x) &= \frac{1}{2} \langle x, Qx \rangle - {x^*}^T Q x + \frac{1}{2} \langle x^*, Qx^* \rangle &\\
&= \frac{1}{2} \langle x, Qx \rangle - b^T x + \frac{1}{2} \langle x^*, Qx^* \rangle  &\because \nabla f(x^*) = Qx^* - b = 0 \\
&= f(x) + \frac{1}{2} \langle x^*, Qx^* \rangle
\end{align}
$$
따라서 $\frac{\partial f}{\partial x} = \frac{\partial F}{\partial x}$ 그러므로 같은 최적화 문제가 된다. 

### Lemma
$$
F(x_{i+1}) = \left[ I - \frac{(g_i^T g_i)^2}{(g_i Q g_i)(g_i Q^{-1} g_i)}  \right] F(x_i)
$$
#### proof of Lemma
$$
g_i \triangleq Qx_i -b = Qx_i - Qx^{*} = Q(x_i - x^{*}) = Qy_i
$$
따라서, $F(x_i) = \frac{1}{2}y_i^T Q y_i$. 고로
$$
F(x_i) - F(x_{i+1}) = \frac{1}{2} [y_i Q y_i - y_{i+1} Q y_{i+1}]
$$
로 놓을 수 있으므로
$$
\begin{align}
y_{i+1}^T Q y_{i+1} &= (x_{i+1} - x^*)^T Q (x_{i+1} - x^*) = (x_i - \alpha_i g_i - x^*)^T Q (x_i - \alpha_i g_i - x^*) \\
&= (y_i - \alpha_i g_i)^T Q (y_i - \alpha_i g_i) \\
&= y_i^T Qy_i - 2 \alpha_i g_i^T Q y_i + \alpha_i^2 g_i^T Q g_i 
\end{align}
$$

따라서, Since $ \alpha = \lambda = \frac{g_i^T g_i}{g_i^T Q g_i} $
$$
\begin{align}
\frac{F(x_i) - F(x_{i+1})}{F(x_i)} &= - \frac{- 2 \alpha_i g_i^T Q y_i + \alpha_i^2 g_i^T Q g_i }{y_i^T Q y_i} = \frac{ 2 \alpha_i g_i^T Q Q^{-1} g_i - \alpha_i^2 g_i^T Q g_i}{g_i^T Q^{-1} Q Q^{-1} g_i} \\
&= \frac{ 2 \alpha_i g_i^T g_i - \alpha_i^2 g_i^T Q g_i}{g_i^T Q^{-1} g_i} = \frac{2 \frac{g_i^T g_i}{g_i^T Q g_i} g_i^T g_i - \frac{(g_i^T g_i)^2}{(g_i^T Q g_i)^2} g_i^T Q g_i}{g_i^T Q^{-1} g_i} \\
&= \frac{\frac{1}{g_i^T Q g_i}\left( 2(g_i^T g_i)^2 - g_i^T g_i \right)}{g_i^T Q^{-1} g_i} \\
&= \frac{g_i^T g_i}{(g_i^T Q g_i)(g_i^T Q^{-1} g_i)}
\end{align}
$$

### Note
- 이 Lemma는 마치, Gaussian Curvature의 Normalized Version 처럼 보인다. 실제로 그러한지 증명 필요
   - 만일 그렇다면, Curvature를 사용하는, 매우 유력한 Iterative Algorithm for manifold가 나올 수 있다.
- 또한 모든 알고리즘을 2파 함수로 근사화 하고 차이점을 White Noise로 모델링하면 Conjugate Method의 아이디어를 Stochastic Differential에 근거한 알고리즘으로 치환할 수 있다.
   - 살펴보면 2차 편미분 (Hessian) $Q$ 만 존재하면 나머지는 Gradient $g_i$로 만들 수 있다. 




