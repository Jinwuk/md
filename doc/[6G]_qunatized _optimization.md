Quantized Optimization
====

## Fundamental Property of Quantized Parameter
### Definition 
Consider the following Quantized value

$$
x = [x] + \epsilon \;\;\; (0 \leq \epsilon < 1)
$$

The above equation means that

$$
[x] \leq x < [x] + 1 \Rightarrow x - 1 < [x] \leq x
$$

It is just equal to the python function of **floor**
However, We want to use the **round** function such that
$$
round(x) = [x + 0.5]
$$

We denote the result of the round function to be a rounded value such that

$$
x^R = round(x) = [x + 0.5].
$$

By the definition of the Gauss symbol

$$
x+0.5 -1 < [x + 0.5] \leq x+0.5 \Rightarrow x - 0.5 < x^R \leq x + 0.5
$$

It means that
$$
\begin{aligned}
x^R &= [x + 0.5] = x + 0.5 - \epsilon = x - \varepsilon \\
x   &= x^R + \varepsilon \;\;\; (-0.5 \leq \varepsilon < 0.5)
\end{aligned}
$$

Thus,  We can consider for sufficient many $\varepsilon_k$ such that $x^R = x + \varepsilon_k$, where $\varepsilon_k$ is distributed uniformly and, by the strong law of large numbers, $\lim_{N \rightarrow \infty} \frac{1}{N} \sum_{k=0}^{N-1} \varepsilon_k = E \varepsilon = 0, $ 

$$
E x^R \triangleq \lim_{N \rightarrow \infty} \frac{1}{N} \sum_{k=0}^{N-1} x^R = \lim_{N \rightarrow \infty} \frac{1}{N} \sum_{k=0}^{N-1} (x - \varepsilon_k) = \lim_{N \rightarrow \infty} \frac{N}{N} x - E \varepsilon = x
$$

$\varepsilon$이 Gaussian이라고 가정 혹은 증명할 수 있는지에 대하여는 나중에 알아보도록 하자.

- 특성상, 이것의 미분을 White Noise로 가정하더라도 큰 문제는 없을 것으로 보인다. 
- Linear로 가정하면 명백하다. 

Consider the other property of $x^R$, the production property.

$$
x y = (x^R + \varepsilon_x)(y^R + \varepsilon_y) = x^R y^R + \varepsilon_y x^R + \varepsilon_x y^R + \varepsilon_x \varepsilon_y
$$

Moreover,

$$
(x y)^R = ( x^R y^R + \varepsilon_y x^R + \varepsilon_x y^R + \varepsilon_x \varepsilon_y)^R = x^R y^R + (\varepsilon_y (x^R + \varepsilon_x) + \varepsilon_x y^R)^R = x^R y^R + (\varepsilon_y x + \varepsilon_x y^R)^R
$$

Since the norm of $\varepsilon$ is less than $0.5$ such that $$ \|\varepsilon\| \leq \frac{1}{2} $$ , and the norm of the rounded value is less than the real value such that $$\|x^R\| \leq \| x \|$$,  the above equation implies that 
$$
-\frac{1}{2} (\| x \| + \| y \|) < (\varepsilon_y x + \varepsilon_x y^R)^R < \frac{1}{2} (\| x \| + \|y \|).
$$

In addition, for $n \in \mathbf{Z}$.
$$
[x + 0.5 + n] = x^R + n
$$

Consequently,  if there exists $y^R \in \mathbf{Z}$, such that $y^R = y + \varepsilon$, hence

$$
\begin{aligned}
x^R + y^R &= (x + y^R)^R \neq (x + y)^R \\
(x + y)^R &= (x^R + \varepsilon_x + y^R + \varepsilon_y)^R = x^R + y^R + (\varepsilon_x + \varepsilon_y)^R 
\end{aligned}
$$

Since $ -1 \leq (\varepsilon_x + \varepsilon_y) < 1$, we can evaluate the $(x + y)^R$ such that

$$
(x + y)^R = 
\begin{cases}
x^R + y^R - 1  & -1   \leq \varepsilon_x + \varepsilon_y < -0.5 \\
x^R + y^R      & -0.5 \leq \varepsilon_x + \varepsilon_y < 0.5  \\
x^R + y^R + 1  & 0.5  \leq \varepsilon_x + \varepsilon_y < 1    \\
\end{cases}
$$

##  Definition of Quantization

We expand the above round function to thew quantization with quantization parameter $Q_p$ such that 

$$
x^Q = \frac{1}{Q_p} \left[ Q_p \cdot  x + 0.5 \right]
$$

It means simply that, let $x = 0.123456789$m and $Qp = 100$ then 

$$
\begin{aligned}
Qp \cdot x &= 100 \cdot 0.123456789 = 12.3456789 \\
Qp \cdot x + 0.5 &= 12.3456789 + 0.5 = 12.8456789 \\
[Qp \cdot x + 0.5] &= [12.8456789] = 12 \\
\frac{1}{Qp}[Qp \cdot x + 0.5] &= 0.12
\end{aligned}
$$

and it has the following relation : 

$$
x - \frac{0.5}{Qp} \leq \frac{1}{Qp}\left(Qp \cdot x \right)^R \leq x + \frac{0.5}{Qp} \Rightarrow  x - \frac{0.5}{Qp} < x^Q < x + \frac{0.5}{Qp}
$$

It is easy to understand, since you let the round value of $Qp \cdot x$ to $y$, and for $\varepsilon$ such that $-0.5 < \varepsilon \leq 0.5$, we obtain

$$
x^Q = \frac{1}{Qp}(Qp \cdot x)^R = \frac{1}{Qp}[Qp \cdot x + 0.5] = \frac{1}{Qp} (Qp \cdot x + \varepsilon) = x + \bar{\varepsilon}
$$

where $-\frac{0.5}{Qp} \leq \bar{\varepsilon}  < \frac{0.5}{Qp}$ and $\bar{\varepsilon} = \frac{\varepsilon}{Qp}$. 

From the multiplication property of $x^R$, we can get

$$
x^Q y^Q = x y - \bar{\varepsilon}_y x^Q - \bar{\varepsilon}_x y^Q - \bar{\varepsilon}_x \bar{\varepsilon}_y
$$

where $\bar{\varepsilon} = x - x^Q $.

##  Parameter Update Rule of Quantized Optimization

다음을 생각해보자
$$
x_{t+1} = x_t - \lambda_t h_t
$$

이것의 Quantized 된 값을 $x_t^Q \equiv [x]_t$  로 정의한다.

$$
x_{t+1}^Q = (x_t - \lambda_t h_t)^Q
$$

여기서 $\lambda_t$는 Step Size 로서 $h_t$가 원하는 양자화 계수에 다음과 같이 맞도록 하면서  

$$
\lambda_t h_t  = (\lambda_t h_t)^Q
$$

다음의 조건을 만족한다.
$$
\lambda_i = \arg \min f(x_i - \lambda h_i)
$$

만일, $x_t$ 도 $x_t^Q$ 라고 하면, 위 방정식은 

$$
x_{t+1}^Q = (x_t^Q - \lambda_t h_t)^Q = (x_t^Q - (\lambda_t h_t)^Q)^Q = x_t^Q - (\lambda_t h_t)^Q
$$

결론적으로 Line Search를 수행할 때, $\lambda_t h_t  = (\lambda_t h_t)^Q$ 조건이 만족되도록 수행하면 된다. 

