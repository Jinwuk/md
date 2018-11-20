Quantized Optimization
====

## Abstract


## Introduction
비디오 코덱의 양자화 계수의 추정을 통한 화질 향상을 위하여는 부호화 및 복호화를 위하여 정수값으로 나오는 양자화 계수 들의 특성을 이용할 수 있어야 하며 동시에, 추정과정에서 가급적 정수 혹은 고정 소수점에 기반한 추정 알고리즘을 통해 추정된 양자화 계수의 값이 정수로 나타날 수 있도록 해야 한다.
For the purpose of the improvement to video quality through the estimation of the quantization parameter in a video codec, it is necessary to use the property of the quantization parameter which is represented with an integer value. To achieve the goal the above mentioned,  the estimated quantization parameter or an equal parameter should be represented with an integer or a fixed point fractional value evaluated by an estimation algorithm.     

또한, Parameter Estimation과 같은 문제의 경우는 일반적인 CNN 기반의 문제들과는 달리, 상당한 수준의 정확도를 가지고 Estimation이 이루어져야 하기 때문에 기존의 Stochastic Gradient 와 같은 단순한 알고리즘 기반의 신경망의 경우, 효과적인 추정이 어려울 수 있다.

Moreover, since the problem of parameter estimation requires the high accuracy level comparing with the general problems using the CNN, it may be difficult to estimate the parameter using the algorithm based on a general stochastic gradient. 

이 때문에, 복잡한 형태의 목적함수과 상대적으로 작은 규모의 입력 데이터를 지닌 Parameter Estimation 문제에서는 많은 경우, 목적함수의 Hessian에 기반한 최적화 알고리즘 (예를 들면 Quasi Newton Method) 을 사용하게 되는경우가 많다. 

Subsequently,  in many cases of the parameter estimation including a complex object function and  relatively small size data set, the optimization algorithm based on the Hessian of  the object function (such as the Quasi-Newton algorithm) is generally employed.

본 논문에서는 정수 혹은 고정 소수점으로 양자화 된 파라미터를 대상으로 최적화 알고리즘에 적용할 경우  발생하는 최적화 오차를 보상하기 위한 방법론을 제시한다. 또한 제안된 방법론을 Losen Brock 함수에 적용하여 양자화 오차를 보상하지 않은 경우에 대하여 최적화 성능이 향상됨을 실험적으로 보인다.  

In this paper, we propose the novel methodology to compensate an error which is raised from the quantized parameter constructed with an integer or a fixed point fractional value  being  applied to a optimization algorithm.  Moreover,  the experimental results shows that the proposed algorithm improve the performance of the non-linear optimization algorithm in comparison to the case of the same optimization algorithm without the proposed methodology.

## Optimization Failure in Constant Step Size
Generally, the non-linear optimization using the gradient descent with a constant step size does not satisfy the convergence of parameter or the consistency of optimization.

In the very special case, such that we have to employ the stochastic gradient descent owing to a relatively simple objective function to a large scaled data, the optimization algorithm with constant step size is able to operate to find a locally optimal point under a limited condition.  (One of the limited condition, maybe it would be a typical condition,  is that the norm of constant step size should be very small.)

- Object function : Rosen-brock $f(x,y) = (1 - x) +100 \cdot  (y - x^2)^2$
- Stepsize: 0.002

- Initial Point : [-1.22, -1.22]
![](https://drive.google.com/uc?id=1sj2hQ5IeYt8qp2dux-Udmk_s1zNZsoQL)

- Initial Point : [-1.232, -1.22]
![](https://drive.google.com/uc?id=1rWJ9qWPH7OqIGnzoZhx5oB-BLD8DMiBU)
The above picture represent the divergence of the gradient descent with constant step size according to where is the initial point. 

For that reason, a decreased step size as time increasing is used widely in the learning equation of the artificial intelligence. However, since there is not a general the time schedule of a decreased step size which has a detailed analysis of an internal dynamics to the optimization algorithm, in most cases, a heuristic or empirical time decayed step size is employed.

Even though such a time decayed step size is worked well in some range where the locally convexity of the object function, it cannot be guaranteed that the updated parameter should be always in the convex region, so that it is possible that the updated parameter can go out of the convex region.  

Therefore, the time decayed step size should be evaluated to satisfy that the updated parameter always will be in the feasible region by that the value of the object function at the updated parameter should be less than the value at the current parameter, i.e.

$$
f(x_{t+1} ) < f(x_t)
$$

### Lipschitz Continuous

다음과 같은 Lipschitz Continuous를 1차 미분항에 대하여 동일하게 표현할 수 있음을 증명한다.
Hessian 의 경우는 비선형 최적화에서 이미 증명되어 있다.

#### 기본 삼각 부등식을 사용한 Difference 부등식

삼각 부등식을 다음과 같이 변경한다. Let $a = c - b$
$$
\begin{aligned}
&\| a + b \| \leq \| a \| + \| b \| \\
&\Rightarrow \| a + b \| - \| b \|  \leq \| a \| \\
&\Rightarrow \| c \| - \| b \| \leq \| c - b \| \;\; \because a = c - b
\end{aligned}
$$

#### Lemma 1.
다음을 만족하는 양수 $k > 0$ 가 존재한다.

$$
\begin{aligned}
& k \cdot \| a + b \| \geq \| a \| + k \cdot \| b \| \\
&\Rightarrow k \cdot (\| a + b \| - \| b \| ) \geq \| a \| \\
&\Rightarrow k \cdot (\| c \| - \| b \| ) \geq \| c - b \| \\
&\Rightarrow \| c \| - \| b \| \geq \frac{1}{k} \| c - b \| 
\end{aligned}
$$

#### proof of Lemma 1
Suppose the lemma is false i.e. $\forall k > 0$
$$
k \cdot \| a + b \| < \| a \| + k \cdot \| b \| 
$$

then 
$$
\begin{aligned}
k^2 \cdot (a^2 + b^2) < (\| a \|  + k \cdot \| b \|)^2 \\
k^2 \cdot (a^2 + b^2) < a^2  + 2 k \cdot \|a \|\|b\| + k^2 b^2 \\
(k^2 - 1) a^2 < 2 k \cdot \|a \|\|b\| \\
k^2 - 1 < 2 K \cdot \frac{\| b \|}{\| a \|} \\
k^2 - 2 K \cdot \frac{\| b \|}{\| a \|} - 1 < 0 \\
(k - \frac{\| b \|}{\| a \|})^2 - \frac{b^2}{a^2} - 1 < 0 \\
(k - \frac{\| b \|}{\| a \|})^2 < \frac{b^2}{a^2} + 1 \\
\frac{\| b \|}{\| a \|} - \sqrt{\frac{b^2}{a^2} + 1} < k < \frac{\| b \|}{\| a \|} - \sqrt{\frac{b^2}{a^2} + 1} \\
\frac{\| b \| - \sqrt{b^2 + a^2}}{\| a \|} < k < \frac{\| b \| + \sqrt{b^2 + a^2}}{\| a \|}
\end{aligned}
$$

Consequently, since $\frac{\| b \| - \sqrt{b^2 + a^2}}{\| a \|} < 0 $, there exists $k \geq \frac{\| b \| + \sqrt{b^2 + a^2}}{\| a \|}$. It contradicts to the assumption.

###  Lipschitz Continuous for the first order differential 

다음과 같이 Lipschitz Continuous 가 만족되고 
$$
\| f(y) - f(x) \| \leq L \cdot \| y - x \|
$$
$f(x)$ 가 Twice differential 이며 $f(y) > f(x)$ 인 경우 

다음이 만족된다.
$$
\| \nabla f(y) - \nabla f(x) \| \leq L \cdot \| y - x \|
$$

#### proof

$$
f(y) - f(x) = f(y) - f(\bar{x}) + f(\bar{x}) - f(x) = -(f(\bar{x}) - f(y)) + (f(\bar{x}) - f(x))
\tag{1}
\label{eq:Lip01}
$$

식 ($\ref{eq:Lip01}$) 에서 각 부분을 뗴어서 정리하면 Twice differential 에서

$$
\begin{aligned}
f(\bar{x}) - f(x) 
&=    \langle \nabla f(x), \bar{x} - x \rangle + \int_0^1 (1-s) \langle \bar{x} - x, H(x+s(\bar{x} - y)(\bar{x} - x) \rangle ds \\
&\leq \| \nabla f(x) \| \cdot \rho + M \cdot \rho \cdot\int_0^1 (1-s) ds \\
&= \rho \cdot \| \nabla f(x) \| + \frac{1}{2} M \cdot \rho 
\end{aligned}
\label{eq:Lip02}
\tag{2}
$$

$$
\begin{aligned}
-f(\bar{x}) + f(y) 
&=  - \langle \nabla f(y), \bar{x} - y \rangle - \int_0^1 (1-s) \langle \bar{x} - y, H(y+s(\bar{x} - y)(\bar{x} - y) \rangle ds \\
&\geq - \| \nabla f(x) \| \cdot \rho - M \cdot \rho \cdot\int_0^1 (1-s) ds \\
&= - \rho \cdot \| \nabla f(x) \| - \frac{1}{2} M \cdot \rho 
\end{aligned}
\label{eq:Lip03}
\tag{3}
$$

식 ($\ref{eq:Lip02}$)는 $f(\bar{x}) - f(x)$ 의 Upper Bound 이고 식 ($\ref{eq:Lip03}$) 은 $-f(\bar{x}) + f(y)$ 의 Lower Bound 이다. 
그러므로 식 ($\ref{eq:Lip02}$) 와 식 ($\ref{eq:Lip03}$) 합은 각각 큰 값 $-f(\bar{x}) + f(y)$ 의 최소값과 작은 값 $f(\bar{x}) - f(x)$의 최대 값이므로 
$$
\begin{aligned}
f(y) - f(\bar{x}) + f(\bar{x}) - f(x) &= f(y) - f(x) \\
& \geq \rho \cdot \| \nabla f(x) \| + \frac{1}{2} M \cdot \rho - \rho \cdot \| \nabla f(x) \| - \frac{1}{2} M \cdot \rho \\
& \geq \rho \cdot (\| \nabla f(x) \| - \| \nabla f(x) \|) 
\end{aligned}
\tag{4}
$$

Lemma 1에서 $\exists k > 0$ 이므로,
$$
\begin{aligned}
f(y) - f(x) 
&\geq \frac{1}{k}\rho \cdot (\| \nabla f(x) - \nabla f(y) \|) \\
& = \frac{1}{k}\rho \cdot (\| \nabla f(y) - \nabla f(x) \|)
\end{aligned}
\tag{5}
$$

따라서 
$$
\begin{aligned}
\frac{1}{k}\rho \cdot (\| \nabla f(y) - \nabla f(x) \|) \leq \| f(y) - f(x) \| \leq L \cdot \| y - x \| \\
\| \nabla f(y) - \nabla f(x) \| \leq \frac{k L}{\rho} \| y - x \|
\end{aligned}
$$

### Properties of Constant Step Size

Suppose that the object function $f(x) \in C^2, \; f:\mathbf{R}^n \rightarrow \mathbf{R}$  is continuosly twice differentiable for all $x, y \in \mathbf{R}^n$, i.e.
$$
\begin{aligned}
f(y) - f(x) 
&= \langle \nabla f(x), y-x \rangle \\
&+ \int_0^1 (1 - s) \langle y-x, H(x + s(y - x))(y-x) \rangle ds
\end{aligned}
\tag{6}
$$
where $H(x) \in \mathbf{R}^{n \times n}$ is the Hessian Matrix of $f(x)$ such that $H(x) \triangleq \frac{\partial^2 f}{\partial x^2}(x)$, and the symbol $\langle \cdot, \cdot \rangle$ is an inner product for two vectors in $\mathbf{R}^n$.

In addition, suppose that there exists a positive value $0 < m < M <\infty$ for the object function $f(x)$ such that
$$
m \| v \|^2 \leq \langle v, \frac{\partial^2 f}{\partial x^2}(x) v \rangle \leq M \|v\|^2
\tag{7}
$$
, and there exists a positive value $L > 0$, so that the object function satisfies the locally Lipschitz continous condition in the sense of the first order differential for $y \in B(x, \rho) \triangleq \{ x | \| y - x \| < \rho, \; \rho > 0 \}$ such that
$$
\| \nabla f(y) - \nabla f(x) \| < L \cdot \| y - x \|
\tag{8}
$$
, where $L$ is a positive value such as $ 0 < L < \infty$.

Let the search equation based on the gradient of the object-function $\nabla f(x)$ including a constant learning rate as follows:
$$
x_{t+1} = x_t - \lambda \cdot \nabla f(x_t)
\tag{9}
$$
, where $\lambda \in \mathbf{R}(0, 1)$ is a constant learning rate, and  we regard $\hat{x} \in \mathbf{R}^n$ is the optimal point such that
$$
\hat{x} = \arg_{x \in \mathbf{R}^n} \min f(x), \;\;\; \nabla f(\hat{x}) = 0.
$$

Suppose that, after $k$ Iterations, the parameter $x_t$ which is generated by the search algorithm defined by (4) converges to the optimal point $\hat{x}$, we can obtain 
$$
\begin{aligned}
x_t - \hat{x} &= x_t - x_{t+k} \\  
&= x_t - x_{t+1} + x_{t+1} - x_{t+2} \cdots + x_{t+k-1} - x_{t+k} \\
&= \lambda \cdot \nabla f(x_t) + \lambda \cdot \nabla f(x_{t+1}) \cdots +  \lambda \cdot \nabla f(x_{t+k-1}) \\
&= \lambda \cdot \sum_{l=0}^{k-1} \nabla f(x_{t+l}) \\
&= \lambda \cdot \sum_{l=0}^{k-1} \left( \nabla f(x_{t+l}) - \nabla f(\hat{x}) \right) 
\;\;\; \because \nabla f(\hat{x}) = 0\\
\end{aligned}
\tag{9}
$$

By the locally Lipschitz continuous (4), setting the norm to both terms in (5), we obtain the following inequality:
$$
\begin{aligned}
\| x_t - \hat{x} \| 
&= \| \lambda \cdot \sum_{l=0}^{k-1} \left( \nabla f(x_{t+l}) - \nabla f(\hat{x}) \right)  \| \\
&\leq \lambda \cdot \sum_{l=0}^{k-1} \| \nabla f(x_{t+l}) - \nabla f(\hat{x}) \| \\
& \leq \lambda \cdot \sum_{l=0}^{k-1} L \cdot \| x - \hat{x} \| \\
& \leq \lambda \cdot k \cdot L \cdot \rho. 
\end{aligned}
\tag{10}
$$

Since the twice differentiable condition is hold, and $\nabla f(\hat{x}) = 0$ by the assumption, the equation (1) is replaced with the following:
$$
f(x_t) - f(\hat{x}) = \int_0^1 (1 - s) \langle x_t - \hat{x}, H(\hat{x} + s(x_t - \hat{x}))(x_t - \hat{x}) \rangle ds. 
$$

Thereby, from the condition of bounbed twice differentiable condition (2) and the equation (6), the norm of each term is evaluated as follows:
$$
\begin{aligned}
\| f(x_t) - f(\hat{x}) \| 
&= \| \int_0^1 (1 - s) \langle x_t - \hat{x}, H(\hat{x} + s(x_t - \hat{x}))(x_t - \hat{x}) \rangle ds.\| \\
&\leq \int_0^1 (1 - s) M \cdot \| x_t - \hat{x} \|^2 ds  \\
&\leq  k \cdot \lambda \cdot M  \cdot L \cdot \rho \int_0^1 (1 - s) ds \\
&=  k \cdot \lambda \cdot \frac{ M L}{2}  
\end{aligned}
\tag{11}
$$

The equation (7) illustrates that  since the learning rate $\lambda$ , the Lipschitz contsant $L$ and $M$ are constant values, the norm of $\| f(x_t) - f(\hat{x}) \|$ is increased as k is increased. It contradicts the assumption that the $x_t$ converges to $\hat{x}$ after k iterations. 



However, since there is not a general the time schedule of a decreased step size which has a detailed analysis of an internal dynamics to the optimization algorithm, in most cases, a heuristic or empirical time decayed step size is employed.



In the equation (1), when we let the step size $\lambda$ be an inverse function of t, the difference of 








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
Consider the following update equation
$$
x_{t+1} = x_t - \lambda_t h_t
$$
이것의 Quantized 된 값을 $x_t^Q \equiv [x]_t$  로 정의한다.
We define the quantized value of  $x_t$ to be $x_t^Q \equiv [x]_t$ such that
$$
x_{t+1}^Q = (x_t - \lambda_t h_t)^Q
$$
여기서 $\lambda_t$는 Step Size 로서 $h_t$가 원하는 양자화 계수에 다음과 같이 맞도록 하면서 where $\lambda_t$ is the step size satisfying that $h_t$ should update $x_t$ or $x_t^Q$ to be $x_{t+1}^Q$ such that
$$
\lambda_t h_t  = (\lambda_t h_t)^Q
$$
다음의 조건을 만족한다.
, and it also satisfies the following condition 
$$
\lambda_i = \arg \min f(x_i - \lambda h_i).
$$
만일, $x_t$ 도 $x_t^Q$ 라고 하면, 위 방정식은 
$$
x_{t+1}^Q = (x_t^Q - \lambda_t h_t)^Q = (x_t^Q - (\lambda_t h_t)^Q)^Q = x_t^Q - (\lambda_t h_t)^Q
$$
결론적으로 Line Search를 수행할 때, $\lambda_t h_t  = (\lambda_t h_t)^Q$ 조건이 만족되도록 수행하면 된다.  
하지만,  $\lambda_t \in \mathbf{R}$ 이고 $h_t \in \mathbf{R}^n$ 이기 때문에 등호 관계는 성립하지 않는다. 

Let $z_t^Q = x_t^Q - x_{t+1}^Q$ .  then  $\lambda_t h_t = z_t^Q$

하지만,  $\lambda_t h_t = z_t^Q$ 혹은,  $\lambda_t h_t  = (\lambda_t h_t)^Q$은 실제로는 $\lambda_t$ 의 Domain과  $h_t$의 Domain의 Dimension이 다르기 때문에, Exact하게 이루어 질 수 없으며  따라서 다음 Object Function을 최소화 시키는 방식으로 $\lambda_t$를 구해야 한다.
$$
f(\lambda_t) = \frac{1}{2}(\lambda_t h_t - z_t^Q)^2 \\
0 = \frac{\partial f}{\partial \lambda_t} = (\lambda_t h_t - z_t^Q)h_t^T  \Rightarrow \lambda_t h_t - z_t^Q = 0 \\
\lambda_t = \frac{h_t^T z_t^Q}{h_t^T h_t} = \frac{(z_t^Q)^T h_t }{h_t^T h_t} = \frac{(x_t^Q - x_{t+1}^Q)^T h_t }{h_t^T h_t}
$$
$x_{t+1}^Q$는 Line Search 과정에서 생성되기 때문에 이를 사용한다.  즉, 이렇게 생각한다.

- $\lambda_t = \arg \min_{\lambda_t \in \mathbf{R}} f(x_t - \lambda h_t)$ 과정에서 $\bar{x}_{t+1} = \arg \min_{x \in \mathbf{R}} f(x_{t+1})$ 으로 찾게 된다. 
- Continuous 이면,  $\bar{x}_{t+1}^Q \neq x_{t+1}^Q$ 이다. 왜냐하면, $\lambda^* = \arg \min_{\lambda_t \in \mathbf{R}} f(x_t - \lambda h_t)$ 일때,  $\bar{x}_{t+1} = \arg \min_{x} f(x - \lambda h_t)$를 만족하지만, $x_{t+1} = x_t - \lambda^* h_t$ 이므로 정확한 minimum 값이 아니기 때문이다. ($\lambda$ 는 $\bar{x}_{t+1}$로  $h_t$를 Projection 시키는 것일 뿐이다.)
- $\bar{x}_{t+1}^Q = x_{t+1}^Q$  이므로 ($\lambda_t$의 정의에 의해 그렇게 된다, 다르다고 하면 $\lambda_t$가 유일하게 결정되지 않는다는 것인데, 이는 위 명제와 위배된다.)  $f(\bar{x}_{t+1}) < f(x_{t+1}^Q)$ 일 수 있다. 따라서, 이를 보완하기 위한 어떤 방법론이 필요하다. 

### Compensated Term 
#### Main Structure

$h_t \in \mathbf{R}^2$ 가 $[a, b]$ 이고 $a < b$ 라고 하면 $a^Q$ 를 1 단위로 하여 $[sgn(a), \| (\frac{b}{a})^Q \| sgn(b)]$ 로 놓고 이것을 단위 Vector로 하여 찾는 방법을 생각할 수 있겠다. 그러므로 만일 $ k \in \mathbf{N}$ 에 대하여
$$
h^Q_c(k) = [k \cdot sgn(a), \| (k \cdot \frac{b}{a})^Q \| sgn(b)]
$$
만일 $k$를 지수함수적으로 증가하는 함수$g(k) \in \mathbf{N}$로 본다면
$$
h^Q_c(k) = [g(k) \cdot sgn(a), \| (g(k) \cdot \frac{b}{a})^Q \| sgn(b)]
$$
이를 일반화 시키면 $h_t = (h_0, h_1, \cdots , h_{n-1}) \in \mathbf{R}^n$ 에 대하여 그 절대값이 가장 작은 Component를  $h_{\text{min}} \triangleq \min \|h_k\|, \; \forall k \in [0, n-1]$ 이라 하면
$$
g^Q_c(k) \triangleq (\| g(k) \cdot \frac{h_0}{h_{\text{min}}} \|^Q,  \| g(k) \cdot \frac{h_1}{h_{\text{min}}} \|^Q, \cdots \| g(k) \cdot \frac{h_{n-1}}{h_{\text{min}}} \|^Q )\in \mathbf{R}^n
$$

$$
h^Q_c(k) = g^Q_c(k) \otimes sgn(h_t)
$$
Python Code는 다음과 같다.

~~~python
    def Q_Compensation(self, x, h, _cost, function):
        rtvalue = 0.0
        if self.bQuantization:
            fbest = _cost
            sh    = np.sign(h)/self.iQuantparameter
            hmin  = np.min(np.abs(h))
            for k in range(-1, self.k_limit):
                g_k   = pow(2, k)
                gQ_c  = np.abs(g_k * h / hmin) + 0.5/self.iQuantparameter
                hQ_ck = self.Qunatization(gQ_c * sh)

                xn    = x - hQ_ck
                fnew  = function(xn)
                if fnew < fbest:
                    rtvalue = hQ_ck
                    break

        return rtvalue
~~~
이떄,  function(k) 는 pow(2, k) 로 하면, 2의 k승으로 증가하는 함수가 된다.

- Line 9의 0.5/self.iQuantparameter 는 Python의 round 함수와 본 논의에서의 round 함수의 특징을 맞추기 위한 반올림 함수이다. 
- 위 예제에서, $g(k) = 2^k$ 으로 하였으며 $2^{-1}$ 부터 적용되도록 하였다.

#### Additional Structure
위 알고리즘을 사용하였을 때 Quantization Algorithm이 정상적으로 동작하였으나, Quantization Level이 높은 경우 다시말해, 고정 소숫점의 해상도를 낮게 하는 경우 전역  최적점을 잘 찾아내지 못하거나, 최적화 알고리즘이 잘 작동하지 않는 문제점이 발생하였다.

- 이러한 문제점에 대응하기 위해 추가적으로 Update Derivation $h_t$에서 특정 Component를 0으로 두는 방법론을 생각해 보자.  즉,  $h^Q_c (k) = [h^Q_c (k)_x, h^Q_c (k)_y]^T$ 이면,  다음 2개의 Vector를 Serarch Point로 추가하는 것이다. 
$$
h^Q_c(k)^{1} = [0, h^Q_c (k)_y]^T \;\;\;\;h^Q_c(k)^{1} = [h^Q_c (k)_x, 0]^T
$$
이를 그림으로 도시하면 다음과 같으며 
![](https://drive.google.com/uc?id=1tJ_d1XuARtQohF_5TFIvWux8e-NYlrrM)
일반적으로는 다음과 같다.
$$
h^Q_c(k)^{k} = [h^Q_c (k)_1, \cdots h^Q_c (k)_{k-1}, 0, h^Q_c (k)_{k+1}, \cdots h^Q_c (k)_n]
$$

- 다음 그림은 최적 Step Size를 찾는 과정을 본 알고리즘이 결합 되었을 때를 도시한 것이다.
![](https://drive.google.com/uc?id=1Uskm1NdSHPCTLN2yRsPcHABqOaJCiyZq)

따라서, 이를 적용한 경우 Q-Compensation 알고리즘은 다음과 같다.

~~~python
    def Q_Compensation(self, x, h, _cost, function, bdebug_info):
        xn = x
        epsilon = prev_cost - self.inference(Xn)
        if self.bQuantization and epsilon <= 0:
            fbest = _cost
            sh    = np.sign(h)/self.iQuantparameter
            hmin  = np.min(np.abs(h))
            normh = np.linalg.norm(h)

            k = -1
            while True : 
                g_k   = pow(2, k)
                gQ_c  = np.abs(g_k * h / hmin) + 0.5/self.iQuantparameter
                hQ_ck = self.Qunatization(gQ_c * sh)

                for i in range(3):
                    hq_i = self.qc_param[i] * hQ_ck
                    xcc    = x - hq_i 
                    fnew  = function(xcc)
                    if fnew < fbest or i==2:
                        xc = xcc
                        break

                # Stop Condition    
                if fnew < fbest or np.linalg.norm(hQ_ck) > normh or k > self.k_limit:
                    if fnew < fbest:
                        xn = xc
                    break
                else:
                    k = k + 1

        return xn
~~~

#### Quasi Newton에서의 실험 결과
Quasi Newton에서 Local 에 빠지거나 혹은 발산하는 경우에 본 알고리즘을 적용할 경우 Local에 빠지거나 발산하지 않고 Global Optimal Point를 정확히 찾아내었다.

- **초기 위치 $( -1.2209 -1.22)$ 의 경우 : 발산**
   - Quasi Newton의 경우 : 발산하여 엉뚱한 Point에 머무르게 됨을 알 수 있다.
![](https://drive.google.com/uc?id=19_sQJThVUeRnXXFR7VLwa4Atu96iKL12)

   - 제안한 알고리즘의 경우 : 발산 Point에서도 정상적으로 Global Optimum으로 다시 수렴해 들어간다. (Qunatrization Level 100에서 까지 정상적으로 수렴) 
![](https://drive.google.com/uc?id=1HrRxHqLM799v9XAB5Jr4Hq0tyJeM59c1)

- 초기위치 $(0.46, -0.47)$ 의 경우 : Local Converge 
   - Quasi Newton의 경우 Local Converge 되어 $(-1.104, 1.2269)​$에 Local Converge됨 
   ![](https://drive.google.com/uc?id=1antz-TELzs6sXBVGX_gNdS5Ae1u0z3rK)

   - 제안한 알고리즘의 경우 : 발산 Point에서도 정상적으로 Global Optimum으로 다시 수렴해 들어간다. 
   ![](https://drive.google.com/uc?id=11vpXjFIvY2ZB2Ia1FbxDnX_OqMcdsFYY)

#### Constant Learning rate에서의 실험결과
Constant Learning rate 에서 Local에서 정지되는 위 결과에 대하여 다음과 같이 Global에 가까운 결과를 얻을 수 있었다.  (시작점 $(-1.232, 1.22)$, 도착점 $(1.17, 1.12)$) 
![](https://drive.google.com/uc?id=1s2YIC9mhoo9arcm-Y3pJCqW8B1tuRTvt)

### Next Step

- Quantization 자체가 Annealing Effect 
- 따라서, Annealing을 적용한다면 Stop Condition Check 부분에서 가능성이 있을 듯
  - Quasi Newton의 경우는 매 Iteration에서 적용되어도 무관함
  - 왜냐하면 $\arg_{\lambda_t \mathbf{R}} \min_{x} f(x_t - \lambda_t h_t) $ 이므로 최적 값을 어쩄든 찾아가기 떄문
  - 그러나, **Armijo Rule 적용 알고리즘은 Stop Condition에서 적용**하여야 한다.










