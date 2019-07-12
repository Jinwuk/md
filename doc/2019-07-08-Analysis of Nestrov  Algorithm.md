Analysis of Nestrov  Algorithm 
===

본 분석은 Yu.E.Nestrov의 'A Method of solvinf a convex programming problem with convergence rate $O(1/k^2)$'  을 분석한 것이다.

기본적으로 제목에서 보이듯이 **Robins and Monroe Condition**의 Convex optimization에 있어 Strong version이라고 할 수 있다. 

## Definitions

Consider the problem of unconstrained minimization of a convex function $f(x)$. Assume that there exists a positive value $L > 0$ such that
$$
\| \nabla f(x) - \nabla f(y) \| \leq L \| x - y\|, \,\, \forall x, y \in E.
$$

이를 다음으로 정의한다.
$$
f(x) \in C^{1,1}(E)
$$
이에 따라 다음이 성립한다.
$$
f(y) - f(x) \leq \langle \nabla f(x), y - x \rangle + \frac{1}{2} L \| y - x\|^2
$$
여기에서 다음과 같이 초기 값을 놓는다. for $y_0 \in E$. Put
$$
\begin{aligned}
k=0, \; a_0 =1, \; x_{-1} = y_0 \\
\alpha_{-1} = \frac{\| y_0 - z \|}{\| \nabla f(y_0) - \nabla f(z) \|}
\end{aligned}
$$
where $z \in E, z \neq y_0$, and $\nabla f(y_0) \neq \nabla f(z) $. 

- k-th Iteration
  - the smallest index $i \geq 0$ for which

$$
f(y_k) - f(x_k) = f(y_k) - f(y_k -2^{-i} \alpha_{k-1} \nabla f(y_k) ) \geq 2^{-i-1} \alpha_{k-1} \| \nabla f(y_k) \|^2 
\tag{D1}
\label{D1}
$$

- In the k-th Iteration, the other parameters are defined as

$$
\begin{aligned}
\alpha_k = 2^{-i} \alpha_{k-1}, \;\; x_k = y_k - \alpha_k \nabla f(y_k) \\
a_{k+1} = \frac{1}{2} \left( 1 + \sqrt{4 a_k^2 + 1} \right) \\
y_{k+1} = x_k + \frac{a_k - 1}{a_{k+1}} \left( x_k - x_{k-1} \right)
\end{aligned}
$$

- 방정식 $\eqref{D1}$ 을 생각해 보면 먼저  $\alpha_{-1}$의 정의에서 

$$
\| \nabla f(y_0) - \nabla f(z) \| =  \frac{1}{\alpha_{-1}} \| y_0 - z \|
$$

그러므로 $\frac{1}{\alpha_{-1}} \leq L \Rightarrow  \alpha_{-1} \geq \frac{1}{L}$ 이 된다.   Convex 이므로 
$$
f(y_k) - f(x_k) \geq \langle \nabla f(x_k), y_k - x_k \rangle
$$
이 성립하고,  For a large $k$  and $\| \nabla f(x_k) \|^2 \geq \alpha_{k}^2 \| \nabla f(y_k) \|^2$  ,  and 
$$
\frac{1}{L} \| \nabla f(x_k) - \nabla f(y_k) \| \leq \| x_k - y_k \| = \alpha_k \| \nabla f(y_k) \|
$$
라 하면,  $L = 1$ 인 경우에 대하여 성립하는 $k$ Iteration에 대하여,

$$
\begin{aligned}
&\| \nabla f(x_k) - \nabla f(y_k) \|^2 \leq \alpha_k^2 \| \nabla f(y_k) \|^2 \\
&\Rightarrow \| \nabla f(x_k) \|^2 + \| \nabla f(y_k) \|^2 - 2 \langle \nabla f(x_k), \nabla f(y_k) \rangle \leq \alpha_k^2 \| \nabla f(y_k) \|^2 \\
&\Rightarrow - 2 \langle \nabla f(x_k), \nabla f(y_k) \rangle \leq -\| \nabla f(x_k) \|^2 - (1 - \alpha_k^2 ) \| \nabla f(y_k) \|^2 \\
&\Rightarrow \langle \nabla f(x_k), \nabla f(y_k) \rangle \geq \frac{1}{2} \left( \| \nabla f(x_k) \|^2 + (1 - \alpha_k^2 ) \| \nabla f(y_k) \|^2\right) \\
&\Rightarrow \langle \nabla f(x_k), \nabla f(y_k) \rangle \geq \frac{1}{2} \| \nabla f(y_k) \|^2 \;\;\; \because \| \nabla f(x_k) \|^2 \geq \alpha_{k}^2 \| \nabla f(y_k) \|^2
\end{aligned}
$$

그러므로
$$
f(y_k) - f(x_k) 
\geq \langle \nabla f(x_k), y_k - x_k \rangle 
= \langle \nabla f(x_k), \alpha_k \nabla f(y_k) \rangle 
\geq 2^{-1}\alpha_{k-1} \frac{1}{2} \| \nabla f(y_k) \|^2
$$

