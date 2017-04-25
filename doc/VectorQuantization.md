#Vector Qunatization and Signal Compression

[toc]
by Allen Gersho and Roberty M. Gray

- Vector Qunatization에 관련된 Study Note.
- Evernote에 최신 파일을 저장하고 Haroopad에서 해당 내용을 작성하여 html로 만든다.

##Bit Allocation and Transform Coding


### Bit Allocation Problem

* The maena squarred error $W_i(b)$ incurred in optimally qunatizaing $X_i$ with $b$ bits of resolution
* The overall Distortion $D$ is defined as a function of bit allocation vector, $ b = (b_1, b_2, \cdots, b_k)$
$$ D = D(b) = \sum_{i=1}^k W_i (b_i)$$

* **The bit allocation Problem** 
Find $b_i$ for $i=\mathbb{Z}[1, k]$ to minimize $D(b) = \sum_{i=1}^k W_i (b_i)$ subject to the constraint that $\sum_{i=1}^k b_i \leq B$, where $B$ is a given fixed quota. 

* **How to solve** 
 - 직접법 
   - Decision Tree를 통해 직접적으로 최적 Decision Allocation을 구하는 방법 (ch. 17)
 - Using High Resolution quantization approximation
   - 그 결과로 Largrangian 을 도출할 수 있으면 이것을 풀어서 해결하는 방법 (Current Ch.)


### Example of High Resolution Technique
- from (6.3.2) we have
$$ W_i (b_i) \approx h_i \sigma_i^2 2^{-2b} $$
여기서, $h_i$는 Normalized Random Variable $X_i/\sigma_i$의 pdf $f_i(x)$에 의해 결정되는 값으로 다음과 같다. (Notmalized R.v.의 정의 $X_i/\sigma_i$를 기억하자)
$$ h_i = \frac{1}{12} \left\{ \int_{-\infty}^{\infty}[f_i(x)]^{1/3} dx \right\}^3  $$

- 만일 $f_i(x)$가 Gaussian 이면 다음과 같다.
$$ h_g \equiv \frac{1}{12} \left\{ \int_{-\infty}^{\infty}\left[ \frac{e^{\frac{-x^2}{2}}}{\sqrt{2 \pi}} \right]^{1/3} dx \right\}^3 = \frac{\sqrt{3} \pi}{2} $$

- 만일 $X_i/\sigma_i$가 identically distriburted 되어 있다면 $h_i$는 $h$ 가 되고 서로 다른 Normalized pdf로 구할 수 있으므로 Distortion은 다음과 같은 Weighted Sum의 형태가 된다.
$$
D_w = \sum_{i=1}^k g_i W_i(b_ii)
$$

##Optimal Bit Allocation Results

### Identically Distributed Normalized Random Variables
 논의를 간략화 하기 위하여 이러한 Identically Distributed Normalized PDF 를 가정하고 또한, Component Distortion은 Weighted 되어 있지 않다고 가정 한다.

Let 
$$
b_i = \bar{b} + \frac{1}{2} \log_2 \frac{\sigma^2_i}{\rho^2} \;\;\;\;\;\;\;(8.3.1)
$$
* The average number of bits per parameter $ \bar{b}=\frac{B}{k}$
* The number of parameters $k$
* The geometric mean of the variance of the random variables $ \rho^2 = \left(\prod_{i=1}^k \sigma^2_i \right)^{\frac{1}{2}}$

이 경우 최소 Distortion은 다음과 같이 주어진다. ($k$ 가 나타나는 것은 똑같은 W_i(b_i)를 $k$번 더하기 때문, 방정식은 Bit AllocationProblem에서 사용된 것이다.
$$
D =k h \rho^2 2^{-2\bar{b}}   
$$ 
따라서 $ W_i(b_i) = h \rho^2 2^-{2 \bar{b}}$ 이다.
* 이 방정식에서 알 수 있듯이, Optimal bit Allocation은 Normalized random Variable의 pdf 에 대하여 independent 하다.
	* 쉽게 말해서  pdf 함수가 보이지 않는다.
* 방정식에서 만일, Normalized random variable의 표준편차 $\sigma$ 가 너무 작으면, 필요 비트는 마이너스가 되는 문제점이 발생하기도 한다. 이 경우에는 SKIP 과 같은 것이 발생한다고 생각해야 한다. 따라서 $b_i = 0$ 이된다고 보아야 한다.
* 몇 가지 비 실용적인 문제점들, 즉, log 함수의 존재로 인해 정수 비트가 할당 되지 않는 문제들이 있으나 이 모델은 충분히 Bit Allocation 문제를 서술하는데 있어 부족함이 없다.

### proof of the bit allocation Solution
if $ N_i = 2^{b_i}; i=1, 2, \cdots, k$, and

$$
\sum_{i=1}^k \log N_i = B
$$

then

$$
\sum_{i=1}^k h \sigma^2_i 2^{-2b_i} \geq k h \sigma^2_i 2^{-2\bar{b}_i}
$$

$\bar{b}_i$ 는 Optimal bits. 등식의 경우는 식 (8.3.1)의 경우가 성립할 때이다. This is equivalnet to the inequality

$$
\frac{1}{k} \sum_{i=1}^k \sigma^2_i 2^{-2b_i} \geq \sigma^2_i 2^{-2\bar{b}_i} = \left( \prod_{i=1}^k \sigma^2_i \right)^{\frac{1}{k}} 2^{-2\bar{b}_i}
$$

Arimetic/Geometric mean 부등식에 의하면 임의의 양수 $a_i, \; i=1, 2, \cdots, k$ 에 대하여 다음이 성립한다.

$$
\frac{1}{k} \sum_{i=1}^k a_i \geq \left( \prod_{j=1}^k a_j \right)^{\frac{1}{k}}
$$

같은 경우는 $a_i$가 모두 같은 값일 경우이다.  이를 대입하면 

$$
\frac{1}{k} \sum_{i=1}^k \sigma^2_i 2^{-2b_i} \geq \left( \prod_{i=1}^k \sigma^2_i 2^{-2b_i} \right)^{\frac{1}{k}} = \left(\prod_{i=1}^k \sigma^2_i \right)^{\frac{1}{k}} 2^{\frac{2}{k}\sum_{i=1}^k b_i} = \left(\prod_{i=1}^k \sigma^2_i \right)^{\frac{1}{k}} 2^{-2 \bar{b}}
$$

(즉, $\bar{b} = \frac{1}{k}\sum_{i=1}^{k} b_i$)
간단하게 보면 만일, $\sigma^2_i 2^{-2b_i} = C$ 이면 등식이 성립한다. 그래서, $C = \rho^2 2^{-2\bar{b}}$  이어서 $\sigma^2_i 2^{-2b_i} = \rho^2 2^{-2\bar{b}}$ 이면 

$$
2^{-2b_i} = \frac{\rho^2}{\sigma^2_i} 2^{-2\bar{b}}, \;\; -2b_i = \log_2 \frac{\rho^2}{\sigma^2_i} -2\bar{b}
$$

고로 (8.3.1) 증명

## Nonidentically Distributed Random Variables
보다 일반적인 경우로서 Nonidentically Distributed R.V. 의 경우에 대한 해석이다.
이 경우에는 결국 다음의 Distrotion을 최적화 시키는 문제가 된다.

$$
\sum_{i=1}^k h_i \sigma^2_i 2^{-2b_i}
$$

제한 조건은 bit Quota $B$ 보다 $b_i$의 총합이 작아야 하는 것이다. 
이때, 각 $\sigma_i$는 $h_i\sigma_i$ 로 Scaled 되어야 하며 이때의 bit는 

$$
b_i = \bar{b} + \frac{1}{2} \log_2 \frac{\sigma^2_i}{\rho^2} + \frac{1}{2} \log_2 \frac{h_i}{H}
$$

$H$ 는 Geometric mean of the coefficients $h_i$ 이다.
Distortion은 다음과 같다.

$$
D = k H \rho^2 2^{-2 \bar{b}}
$$

## Weighted Overall Distortion Measure
Weighted의 경우에는 위 경우를 조금 더 확장 시키면 된다 , 즉, Weighrted Parameter에 대한 항을 추가하면 된다.

$$
b_i = \bar{b} + \frac{1}{2} \log_2 \frac{\sigma^2_i}{\rho^2} + \frac{1}{2} \log_2 \frac{h_i}{H} + \frac{1}{2} \log_2 \frac{g_i}{G}
$$

where $G$ is the geometric mean of the weighted values $g_i$.
Distortion은 다음과 같다.

$$
D = k H G \rho^2 2^{-2\bar{b}}
$$

## Nonnegative Constraint 
### Segall's Solution for Nonnmegative Bit allocation (pp. 233)
The optimal allocation is given by

$$
b_i = 
\left\{\begin{matrix}
b_i^* = J\left( \frac{\theta^*}{\sigma^2_i} \omega'(0) \right) & \text{if}\;\; 0 < \theta^* < \sigma^2_i\\ 
0 & \text{if}\;\; \theta^* \geq \sigma^2_i
\end{matrix}\right.
$$

where $\theta^*$ is the unique root of the equation

$$
S(\theta) = \sum_{i:\sigma^2_i \geq \theta^*} J\left( \frac{\theta^*}{\sigma^2_i} \omega'(0) \right) = B
$$




