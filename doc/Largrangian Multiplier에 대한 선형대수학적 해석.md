Largrangian Multiplier에 대한 선형대수학적 해석
====
다음과 같은 일반적인 등식 제한 조건을 가진 최적화 문제를 생각해 보면

$$
\begin{align*}
\textit{Minimize}& \;\; f(x) \in \mathbb{R} \;\;\forall x \in \mathbb{R}^n\\ 
\textit{Subject to}& \;\; g(x)=0
\end{align*}
$$

이에 대한 Largrangian을 다음과 같이 보통 정의한다.
$$
\mathscr{L} = f(x) + \lambda \cdot g(x)
$$

위 Largrangian의 최소값을 찾기 위하여 다음의 Khun-Tucker Condition을 취한다.
$$
\nabla f(x) + \lambda \cdot \nabla g(x) = 0
$$

하지만 이 경우에 단순하게 Khun-Tucker Necessary Condition 을 적용하면 잘못된 해 즉, Local minima, Maxima등을 찾게 될 가능성이 매우 높다. 따라서, 일반적으로는 다음과 같은 Tangent Space $T_{x^{0}}$ 와 이에 대한 Normal Space $N_{x^{0}} = T_{x^{0}}^{\perp}$ 를 놓고 각 Component가 어떠한 공간에 위치하는가를 생각하여 해석한다.

##Tangent Space

다음 그림과 같은 경우를 생각해보자. 여기에서 Surface $S$는 다음 방정식을 만족하는 Level Space of $g$라고 가정하자.
$$
g(x_1, x_2, x_3) = 0
$$
여기에서 $g = 0$ 이지만 보통의 경우라면 $g=c, \; c \in \mathbb{R}$ 이다.

이때, Tangent plane of $S$ at $x^{(0)}$ i.e. $T_{x^{(0)}}(S)$ 은 모든 벡터 $x^{0} + y$에 의해 만들어진다고 하면 다음이 성립한다.
$$
y \cdot \nabla g(x^{(0)}) = 0
$$
**proof**
$x$ 가 $t \in I$ 인 Parameterized curve 이고 $x^{(0)} = I(t)|_{t=0}$ 이러고 하자 이떄 $g(x)$의 t에 대한 변화율은 $g(x) =0 $ 혹은 $g(x) = c$ 이므로

$$
0 = \frac{dg}{dt}|_{x^{(0)}} = \frac{dg}{dx}\frac{dx}{dt} |_{x^{(0)}} = \nabla g \frac{dx}{dt} |_{x^{(0)}}
$$
여기서, $x^{(0)}$ 에서의 Tangent Space의 정의에 따라 $\frac{dx}{dt} |_{x^{(0)}} \in T_{x^{0}} (S)$ , 그러므로
$$
y \cdot \nabla g(x^{(0)}) = 0
$$
**(Q.E.D)**

그러므로 다음과 잩이 정리할 수 있다.
$$
y \in T_{x^{(0)}}, \;\; \nabla g(x) \in N_{x^{(0)}}
$$

결국 Tangent Space를 생각하게 된다면, 위의 Lagrangian에서 Largrangian Multiplier $\lambda$ 가 Level Surface $S$의 Tangent Space에 존재하는 것으로 생각할 수 있다.

이것을 좀 더 확대시켜 생각해 본다면 다음과 같은 1-form이 존재한다고 생각할 수 있다.
$$
dg(y) = 0
$$
만일, y 대신 Lagrangian Multiplier $\lambda$로 대치한다면
$$
dg(\lambda) = 0 
$$

![Largrangian_01](http://jnwhome.iptime.org/Local_Docs/MDdocs/Testhtml/WorkDirectory\2016\Largrangian_01.jpg)

## Regular Point
A feasible point $x^*$ is a **regular point** if the set of vectors
$$
\{ \nabla g_j (x^*): j \in J(x\}
$$
is linearly independent where
$$
J(x^*) = \{ j: 1 \leq j \leq p, g_j(x^*) = 0 \}
$$

다시말해, $\nabla g_j(x^*)=0$ 들이 Linearly Independent 이고 이것이 $p$개의 등식 제한 조건에 대하여 Linearly Independent 이므로 P-Dimensional Normal Space가 존재한다는 것이 된다. 따라서, 이에 대한 Lagrangian Multipliier가 존재하면 $N-p$ Dimension의 Lagrangian Multipliier $\lambda$ 들로 Span 되는 Tangent Space 혹은 Kernel Space가 존재하는 것이 된다.
만일 1 Dimension 이라면 $\lambda$로 Span 되는 공간은 없으므로 $\lambda$는 그냥 하나의 값이 된다.

그리고 만일 $N$개의 조건 중, $P$ 개가 위의 조건을 만족하는 등식 제한 조건이고 나머지가 부등식 조건이라고 하면, 부등식 조건은 $x^*$ 에서 Active 이다.

## 일반이론

**Definition of Program**
$$
(P)
\left\{\begin{matrix}
\textit{Minimize}\;\; f(x) \;\; \textit{subject to the constraints}\\ 
g_1(x)=0, \cdots, g_{m-1}(x)=0; \;\;g_m(x) \leq 0, \cdots, g_p(x) \leq 0 \\ 
\textit{where}\; f(x), g_1(x), \cdots g_p(x)\;\; \textit{have continous first partial}\\
\textit{derivatives on some open subset} \;\; C \subset \mathbb{R}^n
\end{matrix}\right. 
$$

**Theorem** Suppose that $x^*$ is a *regular point* for $(P)$. If $x^*$ is a local minimizer for $(P)$ , then there exist a $\lambda^* \in \mathbb{R}^p$ such that:
1. $\lambda_j^* \geq 0 \;\;\; \textit{for}\;\; j = m, \cdots, p$
2. $\lambda_j^* g_j(x^*)=0 \;\;\; \textit{for}\;\; j = m, \cdots, p$
3. $\nabla f(x^*) + \sum_{j=1}^p \lambda_j^* \nabla g_j(x^*)=0$

**proof**
다음과 같이 문제를 단순화 시킨다.

$$
(EP)
\left\{\begin{matrix}
\textit{Minimize}\;\; f(x) \;\; \textit{subject to the constraints}\\ 
g_j(x)=0 \;\;\; \textit{for} \;\; j \in J(x^*)
\end{matrix}\right. 
$$

Let $S \subset \mathbb{R}^n$ is defined by the constraints for $(EP)$.
Let $\psi(t)$ be a path in $S$ s.t. $x^* = \psi(0)$. 
$x^*$ 은 local minimizer이므로 there exist $r$ such that $f(x^*) \leq f(x), \; \forall x \in B^o (x^*, r) \cap S $
$$
f(x^*) = f(\psi(0)) \leq f(\psi(t))\;\;\; \forall t \in I(-\epsilon, \epsilon), \; \epsilon \in \mathbb{R}
$$
$t^*=0$ 가 Local minimizer 이므로 
$$
0 = \frac{d}{dt}f(\psi(t)) |_{t=0} = \nabla f(\psi(0)) \cdot \psi'(0) = \nabla f(x^*) \cdot \psi'(0) 
$$
$x^*$ 가 Regular Point 이므로 Tangent Space $T_{x^*}S$ 는 $y = \psi'(0)$ 으로 결정되며 따라서 $\nabla f(x^*) \in (T_{x^*}S)^{\perp} = N_{x^*}S $ for some path $\psi(t) \in S$ 이다.

그러므로 p개의 linearly independent vector $\nabla g_j (x^*)$ 으로 $N_{x^*}S$ 가 결정되므로 $\nabla f(x^*) \in N_{x^*}S $ 는 $\nabla g_j (x^*)$ 의 1차 결합에 의해 표현 가능하며 그러므로 당연히 there exist a $\lambda^* \in \mathbb{R}^p$ such that
$$
\nabla f(x^*) + \sum_{j=1}^p \lambda_j^* \nabla g_j(x^*)=0
$$
이다. 이로서 3 이 증명된다.

$\lambda_j^* g_j(x^*)=0 \;\;\forall j \in J(x^*)\;\; \text{by}\; g_j(x^*) = 0$ and
$\lambda_j^* g_j(x^*)=0 \;\;\forall j \notin J(x^*)\;\; \text{by}\; \lambda_j^* = 0$ 
이로서 2 증명

1을 증명하기 위하여, 만일, $\lambda_j^* < 0$ for some $j \in \mathbb{Z}[m,p]$ 그러면 $\lambda_i^* = 0$ for $i \notin J(x^*), \;\; m \leq i \leq p$ 이렇게 되므로 $j \in J(x^*)$ 가 된다. 따라서 다음을 만족하는 $S_j$를 놓을 수 잇다.
$$
g_i(x)=0\;\;\forall i \in J(x^*), i \neq j
$$
그러면 $x^* \in S_j$ 이고, 따라서 Tangent Space $T_j(x^*)S_j$ 가 존재하고 따라서 다음을 만족하는 $y \in T_j(x^*)S_j$ 가 존재한다.
$$
\nabla g_j(x^*) \cdot y < 0
$$

앞에서와 마찬가지로 $\psi(0)=x^*$ 그리고 $y = \psi'(0)$ 이라고 하자.그러면
$$
\begin{align*}
\frac{d}{dt}f(\psi(t))|_{t=0} &= \nabla f(x^*) \cdot y\\ 
 &= -(\sum_{i=1}^{m-1}\lambda_i^*\nabla g_i(x)\cdot y + \sum_{i=m}^p\lambda_i^* \nabla_i(x) \cdot y)\\ 
 &= - \lambda_j^*\nabla g_j(x) \cdot y 
\end{align*}
$$

그런데 $\lambda^*_j < 0$ 이고 $\nabla g_j(x) \cdot y < 0$ 이므로 이는 가정에 위배
간단히 생각하면 1. 의 경우는 사실, $\lambda^*_j = 0$ 이어야 한다. 등식 제한 조건이 아닌 Feasible 영역에 해당되는 부분이기 때문에 Tangent Space에 존재하는 $y \neq 0$ 이 존재하게 된다. 즉, 위 에서 $i \neq j$ 에ㅐ 해당되는 부분의 합은 0이 되는데, $j$ 만 0이 아닌 값을 가제 되므로 가정에 위배되는 것이다. 

즉, 등식 제한 조건에 걸리는 $\lambda_j^*$ 는 Positive definite 일 필요가 없으나 그렇지 않은 경우에는 $g_i(x) \leq 0$ 이므로 $\lambda_j^* \geq 0$ 이라는 것이다.

## Special Case : Quadratic Positive Definite Problem

이 부분은 Largrangian Multiplier 의 성격을 극명하게 드러낸다.

Suppose that $A$ is $n \times n$ **symmetric matrix** 그러면 $A$는 Orthonormal Eigen Vector를 가진다.
문제를 보자.
$$
(P_1)=
\left\{\begin{matrix}
\textit{Maximize}\;\; & f(x)=x^TAx\\ 
\textit{subject to}\;\; & g_1(x) = ||x||^2 - 1 = 0
\end{matrix}\right. 
$$

그러면 간단히 보아도 위 문제는
$$
\nabla f(x^{(1)})+ \lambda_1 \nabla g_1 (x^{(1)}) = 0
$$

고로
$$
2 Ax + 2 \lambda_1 x^{(1)} = 0
$$
즉, $x^{(1)}$ 은 A의  Eigen Vector 이다. 
k개의 mutually Orthogonal unit eigenvector $x^{(1)}, \cdots, x^{(k)}$가 주어졌다고 할 때
$$
(P_k)=
\left\{\begin{matrix}
\textit{Maximize}\;\; & f(x)=x^TAx \;\; \textit{subject to}\\ 
 & g_1(x) = ||x||^2 - 1 = 0 \;\;\;\textit{and} \\
 & g_2(x) = x \cdot x^{(1)}=0, \cdots g_{k+1}(x) = x \cdot x^{(k)}=0
\end{matrix}\right.
$$

이것을 풀어보자. 간단히 $x^{(k+1)}$ (이것은 eigen vector인지 아닌지 모른다) 에 대하여 K-T Condition을 적용하면
$$
\nabla f(x^{(k+1)}) + \lambda_1 \nabla g_1(x^{(k+1)}) + \cdots + \lambda_{k+1}\nabla g_{k+1} (x^{(k+1)}) = 0
$$

그러면 문제에 의하여 간단히
$$
2Ax^{(k+1)}+ 2 \lambda_1 x^{(k+1)} + \lambda_1 x^{(1)} + \cdots + \lambda_{k+1} x^{(k)} = 0
$$
여기에 임의의 Eigen vector $x^{(i)}$를 Inner Product 시키면
$$
2 x^{(i)} A x^{(k+1)} + \lambda_i = 0
$$
$x^{(i)}$는 Eigen Vector 이므로 Eigen Value $\mu_i$ 에 대하여
$$
A x^{(i)} = \mu_i x^{(i)}
$$
그러므로
$$
2 \mu_i x^{i} \cdot x^{(k+1)} = -\lambda_i
$$
$\mu_i \neq 0$ 이므로
$$
x^{(k+1)} = - \frac{\lambda}{2\mu_i}x^{(i)} 
$$
즉, Eigen Vector $x^{i}$의 어떤 Scaling 된 값이므로 역시 Eigen Vector이다. 그러므로 위 K-T 조건을 만족하는 $\lambda^* \in \mathbb{R}^{k+1}$ 이 존재한다. 

즉, Largrangian Multiplier는 Eigen Value와는 별 관계가 없으나 (특수한 경우, 부호가 반대일 수는 있다) **Eigen Vector와는 밀접한 관계가 있다.**
