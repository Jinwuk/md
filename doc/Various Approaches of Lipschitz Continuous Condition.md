Various Approaches of Lipschitz Continuous Condition and Big O Notation
====

## Lipschitz Continuous  Condition

A vector valued function $\bold{X}(x,t)$ is said to satify a Lipschitz condition in a region $\mathcal{R}$ in $(x, t)$ -space if, for some constant $L$ (called the Lipschitz constant), we have 

$$
\| \bold{X}(x,t) - \bold{X}(y,t) \| \leq L \| x - y \|
\label{eq01:Intro}
\tag{I01}
$$

whenever $(x, t) \in \mathcal{R}$ and $(y, t) \in \mathcal{R}$.


### Lemma 1.
If $\bold{X}(x,t)$ has continuous partial derivatives $\frac{\partial \bold{X}_i}{\partial x_i}$ on a bounded closed convex domain $\mathcal{R}$, then it satisfies a Lipschitz condition in $\mathcal{R}$.

#### Proof
Denote

$$
M = \sup_{\overset{\bold{X} \in \mathcal{R}}{1 \leq i, j \leq n}}
\left | \frac{\partial \bold{X}_i}{\partial x_i} \right |.
\label{eq01:LM01_proof}
\tag{L01}
$$

Since $\bold{X}$ has continuous partial derivatives on the bounded closed region $\mathcal{R}$, we know that $M$ is finite. For $x, y \in \mathcal{R}$, there exists $0 \leq s \leq 1$ such that $z(s) = (1 - s)x + sy$. 
Since
$$
\frac{d}{ds}\bold{X}_i ((1-s)x + sy, t) = \frac{d}{dz}\bold{X}_i \cdot \frac{dz}{ds}(s) = \frac{d}{dz}\bold{X}_i (y - x)
$$

Therefore,

$$
\frac{d}{ds}\bold{X}_i ((1-s)x + sy, t) = \sum_{k=1}^n \frac{\partial \bold{X}_i}{\partial x_k}((1-s)x + sy, t) (y_k - x_k)
$$

Using **the mean value theorem**, we obtain

$$
X_i(y, t) - X_i (x, t) = \sum_{k=1}^n \frac{\partial \bold{X}_i}{\partial x_k}((1-\sigma_i)x + \sigma_i y, t) (y_k - x_k)
$$

for some $\sigma_i$ such as $ 0 \leq \sigma_i \leq 1$ (By the mean value theorem, s is replaced with $\sigma_i$ which is proportioin to $s$).

The Schwartz inequality gives 
$$
\begin{aligned}
\left | \sum_{k=1}^n \frac{\partial \bold{X}_i}{\partial x_k}((1-\sigma_i)x + \sigma_i y, t) (y_k - x_k) \right | 
&\leq \left( \sum_{k=1}^n \left | \frac{\partial \bold{X}_i}{\partial x_k}((1-\sigma_i)x + \sigma_i y, t) \right|^2 \right)^{1/2} \left( \sum_{k=1}^n | y_k - x_k |^2 \right)^{1/2} \\

&\leq \left( \sum_{k=1}^n M^2 \right)^{1/2} \left\| y - x \right\|^{1/2} = \sqrt{n}M \cdot \| y - x\|.
\end{aligned}
$$
Thus,
$$
\begin{aligned}
\| \bold{X}(y, t) - \bold{X}(x, t) 
&=    \left(\sum_{i=1}^n |X_i(y, t) - X_i(x, t) |^2 \right)^{1/2} \\
&\leq \left(\sum_{i=1}^n (\sqrt{n}M \cdot \| y - x\|)^2 \right)^{1/2} \\
&= nM \| y - x \|
\end{aligned}
$$
**Q.E.D**

### Corollary 1.
Let $\bold{X}(x, t)$ be a bounded gradient of scalar field $f(x, t) \in \mathbf{R}$, then it is rewritten as follows:

$$
\| \nabla f(y, t) - \nabla f(x, t) \| \leq L \| y - x \|, \;\; \forall x, y \in \mathbf{R}^n, \; y \in B^o(x, \rho)
$$

#### Note
- proof is very trivial.
- A bounded gradient means that the Hessian of the $f(x, t)$ is bounded such that

$$
c_l \| v \| \leq \langle v, \frac{\partial^2 f}{\partial x^2} v \rangle \leq C_h \| v \|, \;\; \forall v \in \mathbf{R}^n
$$
   - It is equivalent to the $f(x, t)$ being a convex function.

## Algebraric Approach
Although the above method is clearly proof, there exist another method based on an algebraric approach.

Suppose that the functions $f_1, f_2$ are locally Lipschitz continulius functions, i. e. $f_1, f_2 \in \mathcal{L}_l$, for $y_1, y_2 \in \mathbf{R}^n$, we have

$$
\begin{aligned}
\| (f_1 + f_2)(y_1) - (f_1 + f_2)(y_2) \| 
&= \| f_1(y_1) + f_2(y_1) - f_1(y_2) - f_2(y_2) \| \\
&\leq \| f_1(y_1) - f_1(y_2) \| + \| f_2(y_1) - f_2(y_2) \| \\
&\leq L_1 \| y_1 - y_2 \| + L_2 \| y_1 - y_2 \| \\
&= (L_1 + L_2)\| y_1 - y_2 \|
\end{aligned}
$$

For another case

$$
\begin{aligned}
\| (f_1 - f_2)(y_1) - (f_1 - f_2)(y_2) \| 
&= \| f_1(y_1) - f_2(y_1) - f_1(y_2) + f_2(y_2) \| \\
&\leq \| f_1(y_1) - f_1(y_2) \| + \| f_2(y_2) - f_2(y_1) \| \\
&\leq L_1 \| y_1 - y_2 \| + L_2 \| y_2 - y_1 \| \\
&= (L_1 + L_2)\| y_1 - y_2 \|
\end{aligned}
$$


## Big O Notation

There are two variations.

### Equality of Essential Feature

First, if $f(x), g(x)​$ are real valued functions on the real line, then
$$
f(x) = O(g(x)) \;\;\text{as}\;\; x \rightarrow \infty
$$

if there exists a **positive constant $M$ **and $x_0$ such that

$$
\| f(x) \| \leq M \| g(x) \|, \;\;\forall x > x_0
$$

For example, the number of arithmetic operations used to multiply two $n \times n$ matrices $A = [a_{ij}]$ and $B = [b_{ij}]$ using the formula $c_ik = \sum_{j=1}^n a_{ij} b_{jk}$ is $n^2(2n-1)$ 
- $n^2$ 개의 $i, k$ 에 대하여 $n$개의 곱셈 ($j$ index를 따라) , 그리고 $n-1$ 개의 덧셈 ($\sum$ 의 Notation을 보면..)  따라서 $n^2 \times (n + n-1)$

Therefore

$$
n^2 \cdot (2n -1) = O(n^3)
$$

즉, 위 방정식을 보면 정확히 $n^3$ 개와 동일하다는 것이 아니라, 가장 중요한 Essential Feature가 $n^3$ 에 비례한다는 의미이다.

### Convergence rate

The second type way that Big-O notation is used is

$$
f(x) = O(g(x)) \;\;\text{as}\;\; x \rightarrow \infty \Leftrightarrow \lim_{x \rightarrow 0} \frac{f(x)}{g(x)} = C
\label{eq01:converge}
\tag{C01}
$$

where $0 < \mid C \mid < \infty$.

For example, for a function that has a convergence Taylor series expansion about $x = 0$, it is common to write

$$
f(x) = f(0) + x f'(0) + \frac{1}{2}x^2 f''(0) + O(x^3)
\label{eq02:converge}
\tag{C02}
$$

Big-O notation은 이와 같이 Taylor 급수 전개식을 보다 정확하게 표현하는 데 사용될 수 있다.

- 다음과 같이  $\eqref{eq02:converge}$ 를 생각해 보면 Taylor 급수 3차 이상의 항은

   $$
   \lim_{x \rightarrow 0} \left | \frac{f(x)}{g(x)} \right |= \lim_{x \rightarrow 0} \left | \frac{\sum_{n=3}^{\infty} \frac{x^n}{n!}f^{(n)}(x)}{x^3} \right | = \frac{1}{6} \left| f^{(3)}(0) \right | = |C|
   $$




