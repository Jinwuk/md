#Recursive Identification 

[TOC]

## Matrix Inverse Lemma

$$
[A + BCD]^{-1} = A^{-1} -A^{-1}B[C^{-1} + DA^{-1}B ]^{-1} DA^{-1}
$$

**Memory Point ** 

Let 
$$ \tilde{A}=
\begin{bmatrix}
A & B\\ 
C & D
\end{bmatrix}
$$ 
로 가정하고 Inverse는 $A$와 $C$만 해당된다. 그리고

**아 - ABC + 답다** 로 외운다. 



**proof **

$$
\begin{align*}
&[A+BCD] \cdot [A^{-1} -A^{-1}B[C^{-1} + DA^{-1}B ]^{-1}DA^{-1}] \\
&= I - B[C^{-1} + DA^{-1}B ]^{-1}DA^{-1} + BCDA^{-1}-BCDA^{-1}B[C^{-1} + DA^{-1}B ]^{-1}DA^{-1}]\\
 &= I + BCDA^{-1} - (B + BCDA^{-1}B)[C^{-1} + DA^{-1}B ]^{-1}DA^{-1}\\
 &= I + BCDA^{-1} - BC(C^{-1} + DA^{-1}B)[C^{-1} + DA^{-1}B ]^{-1}DA^{-1}\\
 &= I
\end{align*}
$$

## Recursive Least Square Method

### Simple Case 
Assume that the model 

$$
y(t) = b + e(t)
$$

where e(t) denotes a disturbance of variance $\lambda^2$
Then the least square estimate of $\theta = b$ is the arithmetic mean,

$$
\hat{\theta}(t) = \frac{1}{t}\sum_{s=1}^t y(s)
$$

Hence,

$$
\begin{align*}
\hat{\theta}(t) &= \frac{1}{t}[\sum_{s=1}^{t-1} y(s) + y(t)] \\
&= \frac{1}{t} \frac{t-1}{t-1} \sum_{s=1}^{t-1} y(s) + \frac{1}{t} y(t)\\
&= \frac{t-1}{t} \hat{\theta}(t-1) + \frac{1}{t} y(t) \\
\end{align*}
$$

so that

$$
\hat{\theta}(t) = \hat{\theta}(t-1) + \frac{1}{t} (y(t) - \theta(t-1)) \tag{1}
$$

Equation (1) is a Simple Linear Regression Equation

### Basic Linear Regression 
** Problem : A Polynomial Trend**

$$
y(t) = a_0 + a_1 \cdot t  + \cdots + a_r t^r
$$

where $a_k \forall k \in Z[0,r]$ is an unknown coefficient
then

$$
\begin{align*}
\psi(t) &= (1, t, \cdots t^r)^T \\
\theta  &= \{a_0, a_1, \cdots a_r \}^T
\end{align*}
$$



** Problem : Weighted Form of Exponents**

$$
y(t) = b_1 e^{-k_1 t} + b_2 e^{-k_2 t}  + \cdots + b_r e^{-k_r t}
$$

then

$$
\begin{align*}
\psi(t) &= (e^{-k_1 t}, e^{-k_2 t}, \cdots e^{-k_r t})^T \\
\theta  &= \{b_1, b_2, \cdots b_r \}^T
\end{align*}
$$



The Problems can be written as follows:

$$
\begin{align*}
y(1) &= \psi^T (1) \theta \\
y(2) &= \psi^T (2) \theta \\
\cdots \\
y(N) &= \psi^T (N) \theta 
\end{align*}
$$



it is 

$$
Y = \Psi \theta, \;\;\; Y \in \mathbb{R}^N, \; \Psi \in \mathbb{R}^{N \times n}
$$

The error is 

$$
\varepsilon(t) = y(t) - \psi^T(t) \cdot \theta
$$

and the Best Predictor is (by Projection rule)

$$
\theta = (\Psi^T \Psi)^{-1} \Psi^T Y \label{RLS_01}\tag{2}
$$

It is also induced by 

$$
0 = \frac{dV}{d\theta} =  -Y^T \Psi + \theta^T (\Psi^T \Psi)
$$

where $V$ is, under the condition which the error term is defined as $\varepsilon = Y - \Psi \theta$, a object function which is defined as follows

$$
\begin{align*}
V(t) &= \frac{1}{2}\varepsilon^T \varepsilon \\
&= \frac{1}{2} (Y - \Psi \theta)^T (Y - \Psi \theta) \\
&= \frac{1}{2} (Y^T Y  - Y^T \Psi \theta + (\Psi \theta)^T Y -  \theta^T \Psi^T \Psi \theta) \\
&= \frac{1}{2} [\theta - (\Psi^T \Psi)^{-1} \Psi^T Y]^T (\Psi^T \Psi ) [\theta - (\Psi^T \Psi)^{-1} \Psi^T Y] + \frac{1}{2} [Y^TY - Y^T \Psi (\Psi^T \Psi)^{-1} \Psi^T Y]
\end{align*}
$$



## Advanced Simple Case 

Looking back the equation (1), assume that the Variance of Error $e(t)$ is 1 (neglecting $\lambda$)

$$
P(t)= (\Phi^T \Phi)^{-1} = \frac{1}{t} \;\;\; \Phi^T = (1, \cdots 1) \in \mathbb{R}^n \label{basicVariance}\tag{3}
$$



식 ($\ref{basicVariance}$) 의 역수는 간단히 Index $t$ 이므로

$$
P^{-1}(t) = P^{-1}(t-1) + 1 = t
$$

에서 $P^{-1}(t)$의 Inverse를 취하면

$$
P(t) = \frac{1}{P^{-1}(t-1) + 1} = \frac{P(t-1)}{1 + P(t-1)}
$$



Linear Regression 에서 Best Estimator ($\ref{RLS_01}$) 을 다시 생각해보면 ($\varphi(t) \in \mathbb{R}^n$)

$$
\hat{\theta}(t) = [\sum_{s=1}^{t} \varphi (s) \varphi^T (s)]^{-1}[\sum_{s=1}^{t} \varphi (s) y(s)] \label{asc_01}\tag{4}
$$

그러므로 다음과 같이 $P(t)$를 놓으면 (즉, Measurement의 Corelation / Relation 의 Inverse)

$$
P(t) = [\sum_{s=1}^{t} \varphi (s) \varphi^T (s)]^{-1}
$$

Measurement의 Inverse를 다시 Inverse 하면 it is trivially

$$
P^{-1}(t) = \sum_{s=1}^{t-1} \varphi (s) \varphi^T (s) + \varphi (t) \varphi^T (t) = P^{-1}(t-1) + \varphi (t) \varphi^T (t) \tag{5}
$$


그러면 ($\ref{asc_01}$) 은 다음과 같이 쓸 수 있다.

$$
\begin{align*}
\hat{\theta}(t) &= [\sum_{s=1}^{t} \varphi (s) \varphi^T (s)]^{-1}[\sum_{s=1}^{t} \varphi (s) y(s)] \\
&= P(t) [\sum_{s=1}^{t-1} \varphi (s) y(s) + \varphi (t) y(t)] \\
&= P(t) [P^{-1}(t-1) \hat{\theta}(t-1) + \varphi (t) y(t)] \\
&= P(t) [(P^{-1}(t) - \varphi (t) \varphi^T(t)) \hat{\theta}(t-1) + \varphi (t) y(t)] \\
&= \hat{\theta}(t-1) - P(t)\varphi (t) \varphi^T(t) \hat{\theta}(t-1) + P(t) \varphi (t) y(t)] \\
&= \hat{\theta}(t-1) + P(t) \varphi (t) (y(t) - \varphi^T(t) \hat{\theta}(t-1))
\end{align*}
$$



따라서 다음의 Recursive Regression 방정식을 얻는다.



$$
\begin{align*}
\hat{\theta}(t) &= \hat{\theta}(t-1) + K(t) \varepsilon(t) \\
K(t) &= P(t)\varphi(t) \\
\varepsilon(t) &= y(t) - \varphi^T(t) \theta(t-1)
\label{std_eqn}\tag{6}
\end{align*}
$$



그런데, $P(t)$이 Update 식이 위에는 없으므로 이를 추가하면 식(5)를 Matrix Inversion Lemma를 적용하는 것이므로 (여기서 $A = P^{-1}(t), B=\varphi(t), C=I, D=\varphi^T(t)$)

$$
P(t) = P(t-1) - P(t-1)\varphi(t) (I + \varphi^T(t) P(t-1) \varphi(t)) \varphi^T(t) P(t-1)
$$

그런데, $\varphi^T(t) \in \mathbb{R}^{1 \times n}$ 이므로 $C = 1$ 고로 



$$
P(t) = P(t-1) - P(t-1)\varphi(t) \varphi^T(t) P(t-1)/(1 + \varphi^T(t) P(t-1) \varphi(t)) \label{std_var}\tag{7}
$$



또한 식 ($\ref{std_var}$) 을 사용하여 Recursive Regression 방정식 ($\ref{std_eqn}$)을 업데이트 하면 

$$
\begin{align*}
K(t)&=P(t-1)\varphi(t) - P(t-1)\varphi(t)\varphi^T(t) P(t-1) \varphi(t)/(1 + \varphi(t) P(t-1) 
\varphi^T(t)) \\
&= P(t-1)\varphi(t) \left(1 - \varphi^T(t) P(t-1) \varphi(t)/(1 + \varphi^T(t) P(t-1) \varphi(t))\right) \\
&= P(t-1)\varphi(t)/(1 + \varphi^T(t) P(t-1) \varphi(t))
\end{align*}
$$



그러므로 다음의 두 가지 방식의 Recursive 방정식을 얻을 수 있다.



### Forgetting Factor

For **Modified Loss function **

$$
V_t (\theta) = \sum_{s=1}^{t} \lambda^{t-s}\varepsilon^2(s)
$$

Parameter Estimation Algorithm is

$$
\begin{align*}
\hat{\theta}(t) &= \hat{\theta}(t-1) + K(t) \varepsilon(t) \\
\varepsilon &= y(t) - \varphi^T(t) \hat{\theta}(t-1) \\
K(t) &= P(t)\varphi(t) = P(t-1)\varphi(t)/(\lambda + \varphi^T(t) P(t-1) \varphi(t)) \\
P(t) &= P(t-1) - P(t-1)\varphi(t) \varphi^T(t) P(t-1)/(\lambda + \varphi^T(t) P(t-1) \varphi(t))/\lambda
\end{align*}
$$



### Kalman Filter

For **State Space equation **

$$
\begin{align*}
x(t+1) &= x(t) \\
y(t) &= \varphi^T(t)x(t) + e(t) \\
\end{align*}
$$

Parameter Estimation Algorithm is

$$
\begin{align*}
\hat{\theta}(t) &= \hat{\theta}(t-1) + K(t) \varepsilon(t) \\
\varepsilon &= y(t) - \varphi^T(t) \hat{\theta}(t-1) \\
K(t) &= P(t)\varphi(t) = P(t-1)\varphi(t)/(1 + \varphi^T(t) P(t-1) \varphi(t)) \\
P(t) &= P(t-1) - P(t-1)\varphi(t) \varphi^T(t) P(t-1)/(1 + \varphi^T(t) P(t-1) \varphi(t)) + R_1
\end{align*}
$$



where $R_1$ is a state space covariance such as

$$
x(t+1) = x(t) +\nu(t), \;\; E \nu(t) \nu^T(t) = R_1 \delta_{t,s}
$$