Maximum Entropy Kalman Filter for Image Reconstruction and Compression
====================
Journal of Electronic Imaging 13(4), 738-755 (October 2004) 
Nastooh Avesta, Tyseer Aboulnasr

## Introduction

The Optimum mean square Error Equation :
$$
\hat{x}_{MS} = \int_{-\infty}^{\infty} x f_{X|Z}(x|z) dx = \frac{1}{f_Z(z)}\int_{-\infty}^{\infty}x f_{Z|X}(z|x) f_X(x) dx
$$
- $z$ : Observer data
- $x$ : State data 

To the following Baysian :
$$
f_{Z|X}(z|x) \frac{f_X(x)}{f_Z(z)}
$$

1. $f_X(x)$ is known : It is **MAP : Maximum A Posteriori Estimator**
2. $f_X(x)$ is unknown
	- $f_X(x)$ assume to be **Uniform** : It is a **ML : Maximum Likelihood** Estimator 
	- $f_X(x)$ assume to be **Exponential** : It is a **ME : Maximum Entropy** Estimator 

Let the observer signal as follows:
$$
z = s(x) + n
$$
- $x$ Ideal Image
- $s$ Arbitrary degradation function
- $n$ Noise

### MAP
다음의 목적함수를 최소화하는 Estimator
$$
J = (x - \hat{x})^T C_{xx}^{-1} (x - \hat{x}) + \left( z - s(x) \right)^T C_{nn}^{-1} \left( z - s(x) \right)
$$

이러한 경우 Near Optimum Solution은 다음과 같이 찾을 수 있다.
$$
\hat{x}_{k+1} = \hat{x}_k + \alpha \nabla J(\hat{x}_k)
$$

만일, the degradation function $s(x)$ 가 Nonlinear 이면 최적 Solution을 찾기가 어려워진다.

### ML
다음의 목적함수를 최소화하는 Estimator
$$
J = \left( z - s(x) \right)^T C_{nn}^{-1} \left( z - s(x) \right)
$$

즉 $f_X(x)$ 에 대한 A priori Knowldge가 없어서, 즉, $C_{xx}$ 정보가 없다. 
- $s(x)$ is Linear : LMS, RLS 알고리즘으로 $\hat{x}$를 찾는다.
- $s(x)$ is Nonlinear : Newton-Raphson, ME (Maximize Expecation) 알고리즘으로 찾는다.


### Maximum Entropy criterion as smoothing Measure
Entropy는 Image Processing에서는 Measure of smoothness로 많이 사용된다. 그러므로 Observation Noise나 Degradation을 줄이는데 다음과 같은 방법이 사용된다.
$$
\begin{align}
H  &= H_x + \rho H_{n'} \\
n' &= n + B|_{n' \geq 0}
\end{align}
$$
- $H_x$ is the entropy of the ideal image.
- $H_{n'}$ is the entropyof the modified noise, such that it is always positive.
- $\rho$ is a weighting scalar

The Entropy of the Image $x \in \mathbb{R}^{M \times N} $ is 
$$
H_x = - \sum_{i=1}^M \sum_{j=1}^N x(i,j) \ln x(i,j)
$$

Newoton Raphson 방법이 Entropy를 최대화 하는데 많이 사용된다.

### Maximum entropy criterion as information measure (MEIN)
MEIN 알고리즘의 전제는 **Signal sequence의 DFT sequence의 Entropy가 Maximized** 되었다는 것에 있다.

## One Dimensional MEKE(maximum Entropy Kalman Filter) for Modeling
Typical Kalman Filter
$$
\begin{align}
x_{k+1} &= A_k x_k + B_k w_{k+1} \\
y_k &= C_k x_k + v_k
\end{align}
\tag{9}
$$

- $x_k \in \mathbb{R}^n$, $y_k \in \mathbb{R}^M$, $A_k \in \mathbb{R}^{n \times n}$, $B_k \in \mathbb{R}^n$, $C_k \in \mathbb{R}^{M \times n}$
- $w_k \in \mathbb{R} \sim (0,Q) $, $v_k \in \mathbb{R}^{M} \sim (0,R)$
- $Q \in \mathbb{R}$, $R \in \mathbb{R}^{M}$

이때 방정식 (9)는 다음과 같이 다시 쓸 수 있다.
$$
\begin{align}
x_{k+1} &= A_k x_k + [0, \cdots , 1]^T w_{k+1} \\
y_k &=
\begin{bmatrix}
c_{11} & \cdots & c_{1N} \\
\cdots & \cdots & \cdots \\
c_{M1} & \cdots & c_{MN}
\end{bmatrix}
x_k + v_k
\end{align}
\tag{10}
$$

이때 일반적인 Kalman Filter는 다음과 같다.
$$
\begin{align}
\hat{x}_{k+1} &= (I - K_{k+1}C_{k+1})A_k\hat{x}_k + K_{k+1}y_{k+1} \\
&= A_k\hat{x}_k + K_{k+1} \cdot (y_{k+1} - C_{k+1}A_k\hat{x}_k)
\end{align}
\tag{11}
$$

방정식 (11)은 다음과 같이 변경 시킬 수 있다.
$$
\begin{align}
\hat{x}_{k+1} &= A_k\hat{x}_k + K_{k+1} \cdot (y_{k+1} - C_{k+1}A_k\hat{x}_k) \\
&= A_k \hat{x}_k + K_{k+1} \cdot (y_{k+1} - C_{k+1}A_k \hat{x}_k^{-}) \\
&= A_k \hat{x}_k + K_{k+1} \xi_{k+1} \\
&=\hat{x}_{k+1}^{-} + K_{k+1} \xi_{k+1}
\end{align}
$$
여기서 
- $\hat{x}_{k+1}^{-}$은 one step prediction of $\hat{x}_{k+1}$
- $\xi_{k+1}$은 Innovation process

### Idea of Maximum Entropy Estimation
Dynamics가 Stationary이고 Transient가 사라진 상태라면, $\xi_{k+1}$은 White Gaussian Noise $\mathcal{N}(0, \sigma_{\xi})$로 놓을 수 있다.
- 이때 **MEKF는 $A_k$와 $K_{k+1}$이 $\hat{x}$가 ME sequence가 되도록 만든다.**
- 이떄 $\sigma_{\xi}$는 다음과 같다.

$$
\begin{align}
\sigma_{\xi} &= E\left[ (y_{k+1} - C_{k+1}A_k \hat{x}_k)(y_{k+1} - C_{k+1}A_k \hat{x}_k)^T   \right] \\
&= E \left[ (C_{k+1}x_{k+1} +v_{k+1})(C_{k+1}x_{k+1} +v_{k+1})^T   \right] + E\left[(C_{k+1}A_k \hat{x}_k)(C_{k+1}A_k \hat{x}_k)^T   \right] \\
&= C_{k+1}R_xC_{k+1}^T + R + C_{k+1}A_kR_{\hat{x}}A_k^T C_{k+1}^T \\
&= C_{k+1}R_xC_{k+1}^T + R + C_{k+1}A_kR_{x}A_k^T C_{k+1}^T 
\end{align}
$$

- $x^{ME}$가 주어진 autocrelation $r$에 대하여 maximum entropy sequence가 될 수 있도록 **$x^{ME}$는 반드시 다음과 같은 AR process**가 되어야 한다.

$$
x_{k+1}^{ME} = - \sum_{i=0}^{N-1} a_i^{ME} x_{k-1}^{ME} + z_{k+1}
$$
satisfying the following equation
$$
\begin{align}
R^{ME}a^{ME} &= r \\
Q^{ME} &= r(0) - (a^ME)^T r
\end{align}
\tag{13}
$$
여기서 $z_k \approx (0, Q^{ME})$, and $R^{ME} \in \mathbb{R}^{N \times N}$ is the autocorrelation matrix corresponding to the given autocorrelation function $r$.

$$
\begin{align}
x_{k+1}^{ME} &= 
\begin{bmatrix}
0 & 1 & 0 & \cdot & \cdot \\
0 & 0 & 1 & 0     & \cdot \\
\cdot & \cdot & \cdot & \cdot & \cdot \\
\cdot & \cdot & \cdot & \cdot & \cdot \\
-a_1^{ME} & -a_2^{ME} & \cdot & \cdot & -a_N^{ME}
\end{bmatrix}
x_k^{ME} + [0, \cdots, 1]^T z_{k+1} \\
x_{k+1}^{ME} &= A^{ME} x_k^{ME} + [0, \cdots, 1]^T z_{k+1}
\end{align}
\tag{15}
$$

여기서 $a^{ME}$는 Levinson-Durbin 알고리즘에 의해 결정되는 계수들이다.
방정식 (12)에서 (15)까지를 살펴 보면 다음의 관계를 알 수 있다.

$$
\begin{align}
A_{k+1} &= A^{ME} \\
K_{k+1} &= [0, \cdots \sigma_{ME}]^T
\end{align}
$$
for all k, where
$$
\sigma_{ME} = \sqrt{\frac{Q^{ME}}{\sigma_{\xi}}}
$$

이때, 
$$
C_N(k+1) = \frac{\frac{1}{\sigma_{ME}} - \sqrt{\frac{1}{\sigma_{ME}^2} - 4 \frac{\alpha_N(k+1)}{\sigma_{ME}}} }{2}  \\
C_i = \frac{\alpha_i}{(1 - \sigma_{ME}C_N)} \;\;\;\; i=1, \cdots N-1
$$

여기서 $C_k \in \mathbb{R}^{1 \times N}$ 그리고 $\alpha_k$는 
$$
[C_1 - \sigma_{ME}C_1 C_N, C_2 - \sigma_{ME}C_2C_N, \cdots, C_N - \sigma_{ME} C_N C_N]_{k+1}^T = [\alpha_1, \cdots, \alpha_N]_{k+1}^T
$$



## Reference
[1] N. Avesta, T. Aboulnasr, "Maximum Entropy Kalman Filter for Image Reconstruction and Compression", Journal of Electronic Imaging 13(4), 738-755 (October 2004) 
