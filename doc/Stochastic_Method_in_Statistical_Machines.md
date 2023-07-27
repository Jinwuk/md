Stochastic Method in Statistical Machines
=========================================
[TOC]

## Simple Associative Memory

Assume that $x^i \in \mathbb{R}^n$ is Input or **Visible Vector** and $x^j \in \mathbb{R}^n$ is a **Hidden vector**.
In addition, the Visible and Hidden vector, in which each component has a value $+1, -1$, is normal vector such that $\left\| x^k \right\| = 1$, for instance
$$
\begin{equation}
x^k = \frac{1}{\sqrt{N}}(1, -1, 1, -1 \cdots , 1)^T. 
\end{equation}
$$

Let Weight matrix  $W = [w_{kl}]$ such that

$$
\begin{equation}
W = x_j x_i^T   \~  \textit{where}  \~   w_{kl} = x_k^j x_l^i.
\end{equation}
$$

Then
$$
\begin{equation}
W x_i = (x^j {x^i}^T) x^i = x^j ({x^i}^T x^i) = x^j, \~  \because \left\| x^k \right\| = 1
\end{equation}
$$

### Auto Associative Memory
If a visible and a hidden vector are same, then as a result, the weight matrix $W$ is symmetric such that $W = W^T$ i.e. $w_{ji} = w_{ij}$

### For Non-Normalized Case
Introduce the following threshold to decide a output to be 1 or -1 for $W \in \mathbb{R}^{n \times n}$
$$
\theta_k \triangleq -\frac{1}{2} \sum_{l=1}^n w_{kl} \Rightarrow \Theta = [\theta_k] \in \mathbb{R}^n
$$

The formation of associative memory for a non-normalized vector is 
$$
f(x) = Wx + \Theta     \textit{where}     f(x)\in \mathbb{R}^n
$$
The other formulation is
$$
f(x)_k = \sum_{l=1}^n w_{kl} x_l^i + \theta_k
$$

The result is as follows:
$$
f(x)_k = 
\begin{cases}
1  & f(x)_k = \sum_{l=1}^n w_{kl} x_l^i + \theta_k > 0 \\
-1 & f(x)_k = \sum_{l=1}^n w_{kl} x_l^i + \theta_k < 0 \\
f^*(x)_k & f(x)_k = \sum_{l=1}^n w_{kl} x_l^i + \theta_k = 0
\end{cases}
$$

### Binary In/Out
It menas that each component of visible and hiddnen vector is +1 or 0 i.e. $x_k^i \in \{1, 0 \}$
Under the condition, the weight matrix is evaluated as follows:
$$
W = (2x^j - I^n)(2x^i -I^n)^T     \textit{where}    W_{kl} = (2x_k^j - 1)(2x_l^i - 1)
$$
where $I^n \in \mathbb{R}^n$ is defined as $I^n = (1, 1, \cdots 1)^T$.
Since $x^i, x^j, I^n \in \mathbb{R}^n$, the law of associative is eatablished such that $x^i (x^z + I^n)^T = x_i {x^Z}^T + x_i + {I^n}^T$
Subsequently, 
$$
W = 4x^j {x^i}^T - 2x^j {I^n}^T - 2I^n{x^i}^T + I^{n \times n}
$$
where $I^{n \times n}$ is a matrix of which components are all one.
$$
\begin{aligned}
W x^i &= 4 x^j {x^i}^T x^i - 2 x^j {I^n}^T x^i - 2I^n{x^i}^T x^i + I^{n \times n} x^i\\
&= 4 (\sum_{l=1}^n x_l^i) x^j  - 2 (\sum_{l=1}^n x_l^i)x^j - 2 (\sum_{l=1}^n x_l^i) I^n + (\sum_{l=1}^n x_l^i) I^n \\
&= 2 (\sum_{l=1}^n x_l^i) x^j  - (\sum_{l=1}^n x_l^i) I^n \\
&= (\sum_{l=1}^n x_l^i)\cdot(2 x^j - I^n)
\end{aligned}
$$
The above equation can be represented as follows
$$
f(x^i)_k = \sum_{l=1}^n w_{kl} x_l^i = (\sum_{l=1}^n x_l^i)\cdot(2 x_k^j - 1)
$$

#### Threshold
Threshold is not a key component which can decide the result of associative memory to be 1 or 0. It is only a helper of decision.
By Definition of threshold, we can obtain

$$
\begin{aligned}
\theta_k &= -\frac{1}{2} \sum_{l=1}^n w_{kl} = -\frac{1}{2}  \sum_{l=1}^n (2x_k^j - 1)(2x_l^i - 1) \\
&= -\frac{1}{2} (2x_k^j - 1)(\sum_{l=1}^n 2x_l^i - \sum_{l=1}^n 1) \\
&= -\frac{1}{2} (2x_k^j - 1)(\sum_{l=1}^n 2x_l^i - n)\\
&=  -(\sum_{l=1}^n x_l^i)(2x_k^j - 1) + \frac{n}{2}(2x_k^j - 1)\\
\end{aligned}
$$
Therefore, when the threshold is applied to the result of binary associative memory, it is represented as follows:
$$
\begin{aligned}
f(x^i)_k &=  \sum_{l=1}^n w_{kl} x_l^i + \theta_k\\
&= (\sum_{l=1}^n x_l^i)\cdot(2 x_k^j - 1) - (\sum_{l=1}^n x_l^i)(2x_k^j - 1) + \frac{n}{2}(2x_k^j - 1) \\
&= \frac{n}{2}(2x_k^j - 1) \\
\end{aligned}
$$
Consequently, the threshold makes the results to be independent on the summation of the components in a visible vector.

### Mutiple Vectors on Auto Associative Memory
Consider the following auto associative memory system
$$
W = \sum_{m \in \mathcal{T}} (2 x^{m} - 1) (2 x^{m} - 1)^T.
$$

Let $y^k = \frac{1}{\sqrt{n}}(2x^k - I^n)$. Assume that the input vectors are mutually orthonormal such that $\forall y^i \perp y^j \in \mathbb{R}^n,      \forall i, j \in \mathcal{T}$.
The result of the auto associative memory is evaluated as

$$
\begin{aligned}
W x^n &= \sum_{m \in \mathcal{T}} (2 x^{m} - I^n) (2 x^{m} - I^n)^T x^n \\
&= n \sum_{m \in \mathcal{T}} y^m {y^m}^T \frac{1}{2}(\sqrt{n}y^n + I^n) \\
&= \frac{n}{2} \left(\sqrt{n} \sum_{m \in \mathcal{T}} y^m {y^m}^T y^n + \sum_{m \in \mathcal{T}} y^m {y^m}^T I^n \right)  \\
&= \frac{n}{2} \left(\sqrt{n} y^n + \sum_{m \in \mathcal{T}} (\sum_{l=1}^n y_l^m) y^m \right) 
\end{aligned}
$$
where $n \neq m$, $\forall n \in \mathcal{T}$.
The second part of the right term in the equation is **perturbation term** raised by multiple vectors, in spite of the assumption of which all the vectors are mutually othogonal. 

#### Effect of threshold
$$
\theta_k = -\frac{1}{2} \sum_l w_{kl} = -\frac{n}{2} \sum_{l=1}^n \sum_{m \in \mathcal{T}} y_k^m y_l^m = -\frac{n}{2} \sum_{m \in \mathcal{T}} (\sum_{l=1}^n y_l^m) y_k^m
$$
Thus
$$
\begin{aligned}
f_k(x^n) &= \sum_{l=1}^n w_{kl} x_l^n + \theta_k \\
&= \frac{n}{2} \left(\sqrt{n} y_k^n + \sum_{m \in \mathcal{T}} (\sum_{l=1}^n y_l^m) y_k^m \right) - \frac{n}{2} \sum_{m \in \mathcal{T}} (\sum_{l=1}^n y_l^m) y_k^m \\
&=\frac{n \sqrt{n}}{2} y_k^n
\end{aligned}
$$
Threshold를 통해 Perturbation 항을 없애 버릴 수 있다. 그리고 출력은 $y_k^n \in \{-1, 1 \}$ 이므로 이를 $ \mathbb{R}[0,1]$ 에 매핑시키는 함수 $h(x) : \mathbb{R} \rightarrow \mathbb{R}[0,1]$ 로 Compactification 시키면 된다.
한 예가 Sigmoid Function 혹은 Fermi-Dirac 분포함수이다.
$$
h(x) = \frac{1}{1+ \exp (- \lambda x)}
$$
여기서 $\lambda$는 비례상수, Fermi-Dirac 통계함수에서는 온도의 역수이다. 
만일, 입력 Vector가 Normalization 되어 있지 않다고 한다면, $y_k^n$의 비례상수가 변화한다.

### Example
Orthonormal 한 경우가 아니더라도 .. 비슷하게 된다.
Let $a = [1   1   1   1   0   1   1   1   1]$, and $b=[1   0   1   0   1   0   1   0   1]$.
The weight matrix is $W = (2a - I^n)(2a -I^n)^T+(2b -I^n)(2b -I^n)^T$
~~~
W =
   2   0   2   0   0   0   2   0   2
   0   2   0   2  -2   2   0   2   0
   2   0   2   0   0   0   2   0   2
   0   2   0   2  -2   2   0   2   0
   0  -2   0  -2   2  -2   0  -2   0
   0   2   0   2  -2   2   0   2   0
   2   0   2   0   0   0   2   0   2
   0   2   0   2  -2   2   0   2   0
   2   0   2   0   0   0   2   0   2
~~~
Threshold is 
~~~
>> t = -0.5 * sum(W',1)
t =  -4  -3  -4  -3   3  -3  -4  -3  -4
~~~
$f(a) = Wa + t$
~~~
>> f = (W*a + t')'
f =  4   5   4   5  -5   5   4   5   4
>> (f(:)>0)'
ans =  1   1   1   1   0   1   1   1   1
>> f = (W * b + t)'
f =   4  -5   4  -5   5  -5   4  -5   4
>> (f(:)>0)'
ans =   1   0   1   0   1   0   1   0   1
~~~
## Hopfield Network
위 Associative Memory의 가정은 입력 벡터가 Orthonormal 이라는 가정이다. 그러나, 이러한 가정은 실용적으로는 너무 강한 가정이다.(하지만, 입력 벡터를 Mutually Orthonormal 하게 뽑을 수 있다면?)

입력벡터가 Orthogonal 하지 않다고 가정하고 그 대신 어떤 Orthonormal 혹은 최소한 Orthgonal 한 Vector Set으로 Weight를 구성하였다고 하면, 임의의 입력 벡터 $x(0)$에 대하여  Associative Memory의 의미는 다음의 에너지 함수를 최소화 시키는 어떤 $x(t) \rightarrow \hat{x}$로 수렴되는 벡터를 출력 시키는 것이다.

### 기본 알고리즘
For $x(t) \in \mathbb{R}^n (0, 1)$, $\forall t > 0$ 
$$
\begin{aligned}
f_j(x(t)) &= \sum_i w_{ji} x_i(t) + \theta_j \\
x(t+1) &= h(f(x(t)) = \frac{1}{1+ \exp (- \lambda f(x(t)))}
\end{aligned}
$$
where $h(x) : \mathbb{R}^n \rightarrow \mathbb{R}^n(0,1)$

### Orthonormal 가정 조차 없다면?
만일 Weight Matrix에 Orthonormal 한 가정 조차 없다면, 입력 벡터에 대하여 가장 유사한 벡터를 출력시키는 Associative Memory가 되어야 할 것이다. 이는 일의적으로 결정되지 않을 것이고, Enengy Function 혹은 Object Function을 놓고 이것을 최소화 시키는 방식으로 출력값을 입력으로 다시 Feedback 시키면?... 어떻게 될까?


### Derivation of Energy Function
만약 $x(0) \in \mathbb{R}^n \{0,1\}$ 이라는 입력 벡터가 인가 되었다고 하자. 
이 벡터가 Associative Memory에 인가 되었을 때 출력 벡터와 입력 벡터와의 유사도를 Energy $\bar{E}_1$라고 하자.
이러한 Energy중 가장 쉬운 것중 하나는 **Inner Product** 이다. 그리고 **Inner Product**의 최대 값이 최소 값이 되도록 정의하자.
$$
\bar{E}_1(0) = -\frac{1}{2} \langle x(1), x(0) \rangle = -\frac{1}{2} \langle h(0), x(0) \rangle
$$
그런데 위 Associative Memory에서 알 수 있는 것은 $ f(x(0)) $ 를 사용하더라도 위 Energy 값의 특성이 변하지 않는다는 것이다. 
(Fermi-Dirac 통계가 단조 증가 함수이기 때문)
따라서, 다음과 같이 Energy 함수를 정의하자.

$$
\begin{aligned}
E(t) &= -\frac{1}{2} \langle f(x(t)), x(t) \rangle = -\frac{1}{2} \sum_j f_j (x(t)) x_j(t) = -\frac{1}{2} \sum_j (\sum_i w_{ji} x_i(t) + \theta_j) x_j(t) \\
&= -\frac{1}{2} \sum_j \sum_i w_{ji} x_i(t) x_j(t) - \sum_j \theta_j x_j(t)
\end{aligned}
$$

- 이렇게 Energy 함수를 정의하고 Feedback이 들어갈 경우, 시간에 대한 Energy 함수의 변화를 분석하여 과연, 올바른 출력을 낼 수 있는지를 알아봐야 한다.
- Convergence Analysis (Strong Condition), Stability Analysis (Weak Condition)

### Dynamics of Energy Function
시간에 따른 Energy 함수의 변화를 생각해보자.
$$
\frac{\partial E(x, t)}{\partial t} = \frac{\partial E(x, t)}{\partial x} \cdot \frac{\partial x}{\partial t} = \nabla E(x, t) \cdot \dot{x}
$$
시간 지연이 거의 없다고 생각하면 
$$
\dot{x} = f(x(t)) = \sum_i w_{ji} x_i(t) + \theta_j = Wx + \theta
$$
그리고,

$$
\begin{aligned}
\frac{\partial E(t)}{\partial x_j}  = -f_j(x)  = -\sum_i w_{ji} x_i(t) - \theta_j \\
\nabla E(x(t)) = f(x) = - W x - \theta
\end{aligned}
$$
이므로 

$$
\frac{\partial E(x,t)}{\partial t} = \nabla E(x,t) \cdot \dot{x} = - \left\| \nabla E(x(t)) \right\|^2 < 0
$$
에서 **Lyapunov Stability**를 만족한다. 

### Hopfield Network의 특징 
#### Weight  Update가 없다
다시말해 문제가 Fix 되어 있고 문제를 주면 **자동적으로 안정 상태 : 문제를 해결하는 상태 : 국소적으로 낮은 에너지로 수렴**한다는 특징이 있다.

그러므로 만일, **등식 제한조건 및 부등식 제한 조건**이 있는 문제라고 하더라도 자동적으로 국소 해를 찾을 수는 있다.
초기 형태의 Hofield Network은 하나의 등식제한 조건과 부등식 제한 조건을 푸는 문제이다.

예제로 든 **Associative Mempory** 문제는 하나의 다음과 같이 치환해도 동일하다.
$$
\max_{x \in \mathbb{R}^n} h(x)        \textit{subject to}      x=0 \\
\implies \max_{x \in \mathbb{R}^n} h(x) + \lambda x \implies \min_{x \in \mathbb{R}^n} - (x^T W x + \lambda x)
$$
where $h(x) = x^T W x$ and $\lambda = \theta$.

#### Weight Vector는 Orthogonal? 
Associative Memory 를 위해서는 상호 Orthogonal한 입력 Vector로 Weight Matrix를 만든다.
그래야, Associative Memory는 최소 Weight Matrix를 만들기 위한 입력 벡터에 대해서 올바른 출력을 낼 수 있다. (가장 에너지가 작은 결과)
이것을 압축적인 학습 과정이라고 생각해보면 신경망에서 Weight 를 Update하는 과정은 임의의 입력 벡터에 대한 Gram-Schmidt Process라고 생각할 수 있으며 Weight Vector는 이러한 직교 벡터로 생성된 Symmetry Matrix라고 생각할 수 있다. 

$$
\hat{v}_{n} = v_n - \sum_{k=1}^{n-1} \frac{\langle \hat{v}_k, v_n \rangle}{\langle \hat{v}_k, \hat{v}_k \rangle} \hat{v}_k
$$

## From Associative Memory to Neural Networks




### Metropolis Algorithm (MCMC : Markov Chain Monte Carlo)

#### Statistical Property
Suppose that a r.v $X_n$ representing an arbitrary Markov chain, is in state $x_i$ at time $n$.
In addition, a new r.v $Y_n$ is in state $x_j$ such that
$$
P(Y_n = x_j | X_n = x_i) = P(Y_n = x_i |X_n = x_j)
$$
It means that the transition probability is a symmetry.
Let $\Delta E $ denote the energy difference resulting from 한 State $X_n = x_i$ 에서 $Y_n = x_j$ 로의 이동시 에너지의 차이.

$E$를 최소화 시키는 관점에서
1. $\Delta E < 0$ 이면 $X_{n+1} = Y_n$ 이 된다.
2. $\Delta E > 0$ 이면 그냥 안 받아들이는 것이 아니라. 다음과 같이 해서 받아 들일 수도 있다. 즉,
   - 먼저, [0,1] 사이에서 random number $\xi$ 를 하나 만든다. 예를 들어 $\xi = 0.25$ 라고 하자.
   - 다음을 만족하면 $X_{n+1} = Y_n$ 이 된다. $T$ 는 온도 계수이다.
$$
\xi < \exp(\frac{-\Delta E}{T})
$$
   - 만족하지 못하면 $X_{n+1} = X_n$ 변화가 없다.

####  Choice of Transition Probability
위 방법론을 도입하여 Transition Probability를 계산하게 되면 Metropolis Algorithm이 된다.
해당 항목은 **Simon Hykin, 'Neural networks and Learning Machines', pp620-622** 를 참조한다.

#### Simulated Annealing 과 Metropolis Algorithm
Metropolis Algorithm은 Temperature가 고정되어 있다. 그래서 너무 낮으면 Deterministic 한 알고리즘이 되고 너무 높으면 수렴성에 문제가 생긴다. 따라서 시행때 마다 점차 낮아지는 Temperature를 도입해야 한다 such that $T \downarrow 0$. 그러면 어떠한 스케줄로 할 것인가?.

$$
T_k = \alpha T_{k-1},      \forall k \in \mathbb{Z}^+
$$
where $\alpha \in [0.8, 0.99]$ 충분히 큰 $T_0$ 에서 이러한 형식으로 $T_k$를 줄여 나가면 Simulated Annealing 이다. 


### Gibbs Sampling
Let $X \in \mathbb{R}^k$ Random Variable. It means that the components of $X$ are $X_1, X_2, \cdots X_k$
Suppose that we have knowldge of **the conditional distribution** of $X_k$, given the values of all other components of $X$.
이때 초기값 $X = \{ x_1(0), x_2(0), \cdots x_k(0) \}$ 에서 다음과 같은 방법론으로 Marginal Density를 극대화 하는 $X(1)$을 구한다. 이것이 Gibbs Sampling. 즉, MCMC 혹은 Simulated Annealing으로 **Derive** 한다.

- $x_1(1)$ 을 $X_1$에서 Derive 한다. given $x_2(0), x_3(0), \cdots x_k(0)$
- $x_2(1)$ 을 $X_2$에서 Derive 한다. given $x_1(1), x_3(0), \cdots x_k(0)$
- $x_n(1)$ 을 $X_n$에서 Derive 한다. given $x_1(1), x_2(1), \cdots x_{n-1}(1), x_{n+1}(0) \cdots x_K(0)$
- $x_K(1)$ 을 $X_K$에서 Derive 한다. given $x_1(1), x_1(1), \cdots x_{K-1}(1)$

우측 값을 지정한 것은 이떄의 Conditional Density를 계산할 수 있기 때문이다.
그래서, 어떤 Marginal Density 즉, 목적함수를 극대화 시킬 수 있는 또다른 $X$를 구하는 것이다. 
좌우지간 
$$
E(X_t|X_{t-1}) > E(X_{t-1}|X_{t-2})
$$
이 되도록, 그런데 **MCMC** 혹은 **Simulated Annealing**이기 때문에 완벽히 이렇게 되지 않을 수 있다. 어쩄든 이것과 유사하게 되도록 하는 것이다.

#### Note : Diffusions for Global Optimization
**By Steuart Geman and Chii-Ruey Hwang (1986. SIAM J. of Control and Optimization)**

Given a real-valued function $U$ on the unit cube
$$
U:[0,1]^n \rightarrow \mathbb{R}
$$
and an **annealing schedule  ** $T \downarrow 0$, and $X_t \in [0,1]^n$, $W_t \in \mathbb{R}^n$ is standard Brownian motion, then
$$
dX_t = -\nabla U(X_t) dt + \sqrt{2T} dW_t
$$
**converges weakly to a distribution** concentrated on the global minima of $U$.

### Stochastic  Gradient Descent
Consider the following Langevine equation
$$
dX_t = - \nabla E(X_t) dt + \sqrt{2T(t)} dW_t,       \forall t \geq 0    \tag{1}
$$
where $T(t) = \frac{T_0}{ln (2+t)}$
이것의 해는 다음 Probability Density에 비례한다. (Gibbs' Distribution or Boltzmann Distribution)
$$
\frac{1}{Z} \exp\left(\frac{-\nabla E(x)}{T(t)} \right) 
$$
where $Z = \int_{-\infty}^{\infty} \exp (\frac{-\nabla E(x)}{T(t)}) dx $

### Discrete Version of Stochastic  Gradient Descent
From the equation (1)
$$
\int_{X_t}^{X_{t+1}} dX_t = \int_{t}^{t+1} -\nabla E(X_t) dt + \int_{t}^{t+1} \sqrt{2T(t)}dW_t      \forall t \geq 0
$$
Assume that there exist a parametrized curve $\alpha(t) \in \mathbb{R}(0,1)$ such that 
$$
\begin{aligned}
E(x(t+1)) - E (x(t)) &= \langle -\nabla E(x(t)), x(t) \rangle \\ 
& + \int_0^1 (1-s) \langle \alpha(t) \nabla E(x(t)), \nabla^2 E(x(t) - s \alpha(t) \nabla E(x(t))) \cdot \alpha(t) \nabla E(x(t)) \rangle ds \\
&= -\langle \nabla E(x(t)), x(t) \rangle\\
&+ \alpha^2(t) \int_0^1 (1-s) \langle \nabla E(x(t)), \nabla^2 E(x(t) - s \alpha(t) \nabla E(x(t))) \cdot \nabla E(x(t)) \rangle ds < 0
\end{aligned}
$$
이러한 조건 하에서 for large $t > 0$ such that $\sqrt{2T(t+1)} \approx \sqrt{2T(t)}$
$$
X_{t+1} - X_t =  - \alpha(t) \nabla E(X_t) + \sqrt{2T(t)}(W_{t+1} - W_t)      \forall t \geq 0
$$
따라서
$$
X_{t+1} = X_t  - \alpha(t) \nabla E(X_t) + \sqrt{2T(t)}N_{t}      \forall t \geq 0        \tag{2}
$$
여기서 $N_t$는 평균 0, 분산 $t$ 인 Gaussion 분포를 따르는 Random Variable 이다. 

#### Note
에너지 함수 $E(x) \in \mathbb{R}$ 가 만일 Quadratic 한 형태라면 이것의 Lapalacian은 Hessian 이 되고 이 Hessian이 Positive defienite 이고 이것의 Supremum 이 $M \in \mathbb{R}$ 이라 한다면 $\langle x, \nabla^2 E(y)x \rangle < \|x\|^2 \cdot M$ 이므로 
$$
E(x(t+1)) - E (x(t)) < -\langle \nabla E(x(t)), x(t) \rangle + \alpha^2(t) \frac{1}{2} \|\nabla E(x(t))\|^2 \cdot M < 0,       \forall t > 0
$$
이를 만족할 수 있는 $\alpha(t)$를 잘 찾아야 한다.

### Generalized Discrete Stochastic Gradient
##### Including random sequence version
$$
X_{t+1} = X_t - \alpha(t) \left( \nabla E(X(t)) + \xi_t \right) + b(t) N_t        \tag{3}
$$
where $\{\xi_t \in \mathbb{R}^n\}$ is a sequence of random vectors due to noisy or imprecise measurement of the gradient measurement

##### excluding random sequence version
$$
X_{t+1} = X_t - \alpha(t) \nabla E(X(t)) + b(t) N_t        \tag{4}
$$
Typically, for large $t$
$$
\alpha(t) = \frac{A}{t+1},      b_t^2 = \frac{B}{t \ln \ln t} 
$$
여기서, $b_t$ 항은 Iterative Logarithm과 연결된다. 즉, $\lim_{t \rightarrow \infty} \frac{B_t}{\sqrt{t \ln \ln t}} = \pm 1$. 그러므로 적절히 $B$ 값을 선택하여 Simulated Annealing의 효과가 나타날 수 있도록 하여야 한다.

### Stochastic Gradient Descent (SGD) in Machine Learning
The object or energy function is a sum of **Loss function** $E_i (x)$ for a feasible data set $A$ such that
$$
E(x) \triangleq \frac{1}{N} \sum_{i=1}^{N} E_i (x),        E_i(x) \triangleq f(x^i)        x = \{x^i | x^i \in A \}
$$
##### Standard (or Batch) SGD
$$
X_{t+1} = X_t - \eta \nabla E(X_t) = X_t - \eta \frac{1}{N} \sum_{i=1}^{N} \nabla E_i(X_t)
$$
#### Simple Analysis of SGD
Let the expectation value of the loss function on the feasible region such that $\mathbb{E}_A E(x)$, then each loss function is factorized as the expextation term and random term such that
$$
E_i (x) = \mathbb{E}_A E_i(x) + \xi_i
$$
Assume that the random vector $\xi_i \in \mathbb{R}$ is a random process then,
$$
\begin{aligned}
X_{t+1} &= X_t - \eta \frac{1}{N} \sum_{i=1}^{N} \nabla E_i(X_t) = X_t - \eta \frac{1}{N} \sum_{i=1}^{N} \nabla (\mathbb{E}_A E_i(X_t) + \xi_i) \\
&= X_t - \eta \frac{1}{N} \sum_{i=1}^{N} (\mathbb{E}_A \nabla E_i(X_t) + \bar{\xi}_i )= X_t - \eta \frac{1}{N} N \cdot \mathbb{E}_A \nabla E_i(X_t) + \eta \frac{1}{N} \sum_{i=1}^N \bar{\xi}_i \\
&= X_t - \eta \cdot \mathbb{E}_A \nabla E_i(X_t) + \eta \frac{1}{N} \sum_{i=1}^N \bar{\xi}_i 
\end{aligned}
$$
For sufficiently large $N$, suppose that the random variable $B_t = \frac{1}{N} \sum_{i=1}^N \bar{\xi}_i \in \mathbb{R}^n$ has a zero mean and a variance $\sigma \in \mathbb{R}^{n \times n}$ by the **Strong Law of Large Numbers**, it can be rewritted as 
$$
X_{t+1} = X_t - \eta \nabla F(X_t) + \eta B_t
$$
It looks like a generalized discrete stochatic gradient excluding random sequence, so that the above algorithm is told as the **stochastic gradient descent**.

##### Convergence Property
This type of Langevine Equation is not converge to a limit point asymtotically. However, for large $t$, it is converge to a distribution looks like a gaussian distribution. (Weak Converge). Moreover, it is weakly converges to a $\mathbb{E}_A \nabla E_i(X_t)$ with the first moment. (It means that the algorithm converes to an expectation value).

However, the generalized Discrete Stochastic Gradients converges more strongly a distribution in the sense of $l_2$ for $t \uparrow \infty$. 

###### Sketch of Weak Convergence
- Continuous, 및 Lipschitz Continuos 가정이 feasible 한지 Check 한다. 
- $\| X_t - E_A X_t \|$ 의 상한선을 유도하기 위해 Chevyshev 부등식, 삼각 부등식, Parallelo 등식을 사용하여 Cauchy Sequence 화 시킨다. (Cuachy Sequence는 될 수 없다.)
- Cauchy Sequence는 아니므로 Central Limit Theorem을 만족하는지 Check 한다. (증명의 핵심)
- 위 두가지 방법을 통해 $\| X_t - E_A X_t \|$ 상한선이 유도 되었으면 Large $N$에 대하여 Markov 부등식이나 ChevyShev 부등식을 만족하는지 Check 한다.
| Markov inequality | Chevyshev inequality (for Probabilty measure) | Central Limit Theorem |
|---|---|---|
|$P(X \geq a) \leq \frac{EX}{a}$ |![SGD05](http://jnwhome.iptime.org/img/AI/SGD05.png)  | $\frac{X_1 + X_2 + \cdots + X_n - n \mu}{\sigma \sqrt{n}}$ |

## Convergence
수렴성 증명은 개발한 알고리즘의 Stability, Consistency, Well-defined, 특성을 보이기 위함이다. 수렴성 증명이 완벽하지 않으면 보통의 경우 최적화 알고리즘은 다운(!) 된다. 수렴성 증명을 통해 수렴 속도 분석도 가능하다. (Linear/exponential)
### Asymptotically Convergence (For General Sequence)
$$
\lim_{n \rightarrow \infty} x_n = c     \textit{iff}    \forall \epsilon > 0, \exists n_0 \in N, \textit{such that}     n > n_0 \Rightarrow |x_n - c | <\epsilon 
$$
### Strong Convergence (Convergence almost surely)
$$
P(\lim_{n \rightarrow \infty} x_n = X) = 1,\textit{iff}     \forall \epsilon, \eta > 0, \exists n_0 \in N, \textit{such that}     P(\sup_{n \geq n_0} |x_n - X| \geq \epsilon) < \eta,     (\therefore P(\sup_{n \geq n_0} |x_n - X| < \epsilon) > 1 - \eta)
$$
### Weak Convergence (Convergence in Probability)
$$
P(|x_n - X| > \epsilon) \rightarrow 0, \textit{iff}     \forall \epsilon, \eta > 0, \exists n_0 \in N, \textit{such that}     n > n_0 \Rightarrow P(|x_n - X | \geq \epsilon ) \leq  \eta,      (\therefore P(|x_n - X | < \epsilon ) > 1 - \eta)
$$
### Convergence in Distribution (가장 약한 수렴)
분포함수열 $F_n(x) \rightarrow F(x)$. 만일, $X_n$이 대응하는 분포함수 $F_n$을 가지고 이것이 $F_n(x) \rightarrow F(x)$를 만족하면 $X_n$ 은 $X$로 분포수렴한다고 한다.
### Convergence in the r-th mean
$$
\lim_{n \rightarrow \infty} \mathbb{E}(|X_n - X |^r ) \rightarrow 0
$$
### Example 
Proof of Convergence for Simple LMS Algorithm such that
$$
W_{k+1} = W_k + \frac{1}{k}(X_k -W_k),       W_0 = 0
$$
for the i.i.d. process $\{X_k\}$ with the **zero mean** and the variance $\sigma$, such as $\mathbb{E}(X_i, X_j) = 0$ for $i \neq j$. Rewrite the above equation 
$$
W_{k+1} = \frac{k-1}{k}W_k + \frac{1}{k}X_k \implies k W_{k+1} = (k-1)W_k + X_k
$$
We can obtain the following relation
$$
W_k = \frac{1}{k} \sum_{i=1}^{k-1} X_i
$$
For the Chebyshev inequality, it is necessry to ecaluate the mean and variance of $W_k$ such that
$$
\begin{aligned}
\mathbb{E}(W_k) &= \mathbb{E}(\frac{1}{k} \sum_{i=1}^{k-1} X_i) = \frac{1}{k} \sum_{i=1}^{k-1} \mathbb{E}(X_i) = 0 \\
\mathbb{E}(W_k - \mathbb{E}(W_k))^2 &= \frac{1}{(k-1)^2} \mathbb{E}(\sum_{i=1}^{k-1} X_i \sum_{i=j}^{k-1} X_j ) = \frac{(k-1)\sigma^2}{(k-1)^2}
\end{aligned}
$$
For arbitrary $\varepsilon > 0$, by the Chevyshev inequality,
$$
\lim_{k \uparrow \infty} P(|W_k - E(W_k)| \geq \varepsilon) \leq \lim_{k \uparrow \infty} \frac{\sigma^2}{(k-1) \cdot \varepsilon} = 0
$$

