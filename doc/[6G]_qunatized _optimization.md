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
위 알고리즘을 사용하였을 때 Quantization Algorithm이 정상적으로 동작하였으나, Qunatization Level이 높은 경우 다시말해, 고정 소숫점의 해상도를 낮게 하는 경우 전역  최적점을 잘 찾아내지 못하거나, 최적화 알고리즘이 잘 작동하지 않는 문제점이 발생하였다.



