Limit Theorem (극한 정리)
===

## Law of large numbers

Let a stochastic process $\{ X_n\}$ be defined on a probablity space $(\Omega, \mathcal{F}, P )$.

Set a partial sum of $$\{ X_n\}$$ such that

$$
S_n = \sum_{k=1}^n X_k
$$

- Strong Law of Large Numbers

$$
\frac{S_n - a_n}{b_n} \overset{P}{\rightarrow} 0
$$

- Weak Law of Large Numbers

$$
\frac{S_n - a_n}{b_n} \overset{a.s.}{\rightarrow} 0
$$

이때, 위의 Law가 성립하는 실수열 $$\{a_n \}, \; \{b_n\}$$ (단, $$0< b_n \uparrow \infty$$) 은 무엇이며, 그 존재 조건은 무엇인가가 문제.

Here, the issues are what are the sequences $$\{a_n \}, \; \{b_n\}$$, where $$b_n$$ is satisfies $$0< b_n \uparrow \infty$$, and what is the existence condition of those sequences.

## Definitions 

### Definition 1 : Large $O(b_n), O(1)$

For $$\{a_n \}, \; \{b_n\}$$, $$\exists K > 0, \; n_K >0$$ such that

$$
\left| \frac{a_n}{b_n} \right| \leq K, \;\;\forall n \geq n_K
$$

The sequences $$\{a_n \}, \; \{b_n\}$$ are $$a_n = O(b_n)$$

Especially, $$\{a_n \}$$ is bounded, in other words, for $$n>0$$, there exists $$K_n > 0$$ such that $$ | a_n | < K_n$$ then the sequence $$\{a_n \}$$ denotes as follows:

$$
a_n = O(1)
$$

### Definition 2 : small $o(b_n), o(1)$

For $$\{a_n \}, \; \{b_n\}$$,

$$
\lim_{n \rightarrow \infty} \frac{a_n}{b_n} = 0
$$

Alternatively, For $$ \epsilon > 0, \exists n_{\epsilon}$$ 

$$
\left| \frac{a_n}{b_n} \right| \leq \epsilon, \;\; \forall n \geq n_{\epsilon}
$$

The sequences $$\{a_n \}, \; \{b_n\}$$ are $$a_n = o(b_n)$$.

Especially, $$a_n = o(1)$$ means that

$$
\lim_{n \rightarrow \infty} a_n = 0
$$

- 쉽게 말해, $$a_n = o(b_n) $$ 은 $$\{a_n \}$$이 $$\{b_n \}$$ 보다 증가 속도가 느리다는 것이다.

### Definition 3 : Large $O_p(1)$ (확률적 유계 : Bounded in Probablity)

For a stochastic process $$\{X_n \}$$, $$\forall \epsilon > 0, \exists K_{\epsilon} > 0, n_{\epsilon} > 0$$ such that

$$
P\left( | X_n | \leq K_{\epsilon} \right) \geq 1 - \epsilon, \;\; \forall n \geq N(\epsilon)
$$

i.e. $$X_n = O_p(1)$$

### Definition 4 : Large $O_p(b_n)$ 
For a stochastic process $$\{X_n \}$$ and a deterministic sequence $$\{ b_n \}$$, $$\forall  \eta > 0, \; \epsilon > 0$$, $$\exists n_{\epsilon, \eta} > 0$$ such that

$$
P\left( \left| \frac{X_n}{b_n} \right| \leq K_{\eta} \right) \geq 1 - \eta, \;\;\;\ \forall n \geq n_{\epsilon, \eta}
$$

i.e. $$X_n = O_p(b_n)$$.

### Definition 5 : Small $o_p(1)$ 
For a stochastic process $$\{X_n \}$$, $$\forall \epsilon > 0, \eta > 0,\;\; \exists n_{\epsilon, \eta} > 0 $$

$$
P \left( | X_n | \geq 0 \right) < \epsilon, \;\;\; \forall n \geq n_{\epsilon, \eta}
$$

i.e. $$X_n o_p(1)$$.

이는 $$X_n \overset{P}{\rightarrow} 0$$ 와 같다.

### Definition 5 : Small $o_p(b_n)$ 
For a stochastic process $$\{X_n \}$$ and a deterministic sequence $$\{ b_n \}$$, $$\forall  \eta > 0, \; \epsilon > 0$$, $$\exists n_{\epsilon, \eta} > 0$$ such that

$$
P \left( \left| \frac{X_n}{b_n} \right| \geq \eta \right) < \epsilon, \;\;\; \forall n \geq n_{\epsilon, \eta}
$$

i.e. $$X_n = o_p(b_n)$$.



## Weak Law of Large Numbers

### Theorem 1

For all $$ i \in \mathbf{Z}^{++} $$, suppose that the stochastric process $$\{X_n\}$$ contains the following properties

$$
\mathbb{E} X_i = \mu_i, \;\;\; Var(X_i) = \sigma^2, \;\;\; Cov(X_i, X_j) = 0, \forall i \neq j.  
$$

If 

$$
a_n =\mathbb{E} S_n = \sum_{i=1}^n \mu_i, \;\;\; b_n = \sum_{i=1}^n \sigma_i^2 = o(b_n^2)
$$

then

$$
\frac{S_n - a_n}{b_n} = \frac{S_n - \mathbb{E}S_n}{b_n} \overset{P}{\rightarrow} 0
$$

#### proof
By **the chebyshev inequality**, For all $$\epsilon > 0​$$,

$$
\begin{aligned}
P \left( |S_n - \mathbb{E} S_n | \geq \epsilon b_n \right) 
&= P \left( (S_n - \mathbb{E} S_n )^2 | \geq \epsilon^2 b_n^2 \right) \\
&\leq \frac{\mathbb{E}(S_n - \mathbb{E} S_n)^2}{\epsilon^2 b_n^2} \\
&= \frac{Var S_n}{\epsilon^2 b_n^2} = \frac{\sum_{i=1}^n \sigma_i^2}{\epsilon^2 b_n^2} \rightarrow 0
\end{aligned}
$$

as $$ n \uparrow \infty$$. Therefore, as $$ n \uparrow \infty$$,

$$
P \left( \left| \frac{S_n - \mathbb{E} S_n}{b_n} \right| \geq \epsilon b_n \right) \rightarrow 0
\label{th01}
\tag{1}
$$

**Q.E.D**

#### Note 1 : Uniform boundedness
If $$b_n = n$$, the condition $$\sum_{i=1}^n \sigma_i^2 = o(b_n^2)$$ is **the uniform boundedness of variance**.

즉, 모든 $$i$$에 대하여, $$Var X_i \leq M$$ 인 $$M$$이 존재한다. 

#### Note 2 : Same form of Theorem 1

$$
P \left( \left| \frac{S_n - \mathbb{E} S_n}{b_n} \right| \geq \epsilon b_n \right) \leq \frac{K}{\epsilon b_n^2}
\label{th02}
\tag{2}
$$

#### Note 3

A stochastic process $$\{ X_n \}​$$ contatining a uniform distribution, 은 **누이 과정 상 동일한 확률 $p​$** 를 가지고 있다고 할 수 있으며 이는 이후의 약 대수 법칙과 **중심극한 정리 : Central Limit Theorem**를 통해 이것이 **Normal Distribution**을 따른 다는 것을 증명할 수 있다.

### Corollary 1.

For a stochastic process $$\{X_n \}$$ with same probablities to each $$X_n$$,  for all $i$, if 

$$
EX_i = \mu, \;\; Var (X_i) = \sigma^2 < \infty, \;\; Cov(X_i, X_j) =0,\; i \neq j
$$

then 

$$
\frac{S_n}{n} - \mu \overset{P}{\rightarrow} 0
$$

#### proof
Let $$a_n = \mathbb{E} S_n = n \mu$$ and $$b_n = n$$. Since $$n \sigma^2 = o(n^2)$$, by the  $$\eqref{th02}$$,

$$
\frac{S_n}{n} - \mu = \frac{S_n - n \mu}{n} \overset{P}{\rightarrow} 0
$$

### Corollary 2.

For a stochastic process $$\{X_n \}$$ with same probablities to each $$X_n$$,  for all $i$, if 

$$
EX_i = \mu, \;\; Var (X_i) = \sigma^2 < \infty, \;\; Cov(X_i, X_j) =0,\; i \neq j
$$

and $$\sum_{i=1}^n \sigma^2 = o(n^2)$$ then 

$$
\frac{S_n - \sum_{i=1}^n \mu_i}{n} \overset{P}{\rightarrow} 0
$$

#### proof
It is verty simple according to the theorem 1.

### Remarks

대수의 약 법칙은 중대한 약점이 있는데, 그것은 유한 분산의 가정이다. 즉, $$\sum_{i=1}^n \sigma^2 = o(\cdot)$$.

그런데, 만일, **주어진 Stochastic process가 서로 독립**이고, **동일한 분포**를 따르는 경우에는 **이러한 제한 조건이 필요하지 않다**.

이떄에는 당연히 Chebyshev 부등식이 사용되지 못하고, **유한 평균의 조건**을 이용하게 된다.

### Theorem 2: Kintchine 

For a stochastic process $$\{X_n \}$$ with same probablities and independent to each $$X_n$$ , If $$EX_i = \mu < \infty$$,

$$
\bar{X}_n = \frac{S_n}{n} \overset{P}{\rightarrow} \mu.
$$

#### proof

Let $$F(x)$$ be a distribution of $$X_1$$, the charateristic cunction of $$X_1$$ is $$\phi_X (t)$$. For $$ n \rightarrow \infty$$,

$$
\begin{aligned}
\phi_{X_n}(t) 
&= \mathbb{E} \left[ \prod_{k=1}^{n} e^{it \frac{X_k}{n}} \right] = \left[ \phi_X \left( \frac{t}{n} \right) \right]^n \\
&= \left\{ 1 + it\frac{1}{n} \mathbb{E}X_1 + o(\frac{1}{n}) t \right\}^n
\rightarrow e^{it\mu}
\end{aligned}
$$

**Note**

1. $$\prod_{k=1}^{n} e^{it \frac{X_k}{n}}$$ 인 이유는 $$\{X_n \}$$ 가  Independent 이기 때문에  $$\{X_n \}$$의 Ensemble 은 거듭 곱의 형태가 된다.

2. Kinchine의 방법은 분산의 제한이 없다.  Chracteristic function 의 사용이 Idea.
3. 대신 평균의 제한은 존재한다.

## Strong Law of Largenumbers

### Theorem 3 : Kolmogorov

확률변수 열 $\{X_n\}$ 이 모든 $k$에 대하여 $\mathbb{E} X_k = \mu_k < \infty$, $Var X_k = \sigma_k^2 < \infty$ 이고 또 서로 독립일 때 만약, $a_n \equiv \mathbb{E}S_n = \sum_{k=1}^n \mu_k$ 이고 또 $b_n$ 이 $\sum_{k=1}^{\infty} \frac{\sigma_k^2}{b_k^2} < \infty$를 만족하면

$$
\frac{S_n - a_n}{b_n} = \frac{S_n - \mathbb{E}S_n}{b_n} \overset{a.s.}{\rightarrow} 0
$$

### Lemma 1.
확률변수 열 $\{X_n\}$ 이 모든 $k$에 대하여 $\mathbb{E} X_k = \mu_k < \infty$, $Var X_k = \sigma_k^2 < \infty$ 이고 또 서로 독립일 때 만약 $\sum_{k=1}^{\infty} \sigma_k < \infty$ 이면

$$
\sum_{k=1}^{\infty} (X_k - \mu_k) < \infty, \;\;\text{a.s.}
$$

### Lemma 2. Kronecker
If $\{a_n \}$ is monotone increasing positive sequence such as $a_n \rightarrow \infty$, and $\{b_n\}$ is a sequence such as $\sum_{k=1}^{\infty} b_k = u (< \infty)$, then

$$
\frac{1}{a_n} \sum_{k=1}^{\infty} a_k b_k \rightarrow 0
$$

### proof of Theorem 3
Since
$$
\sum_{k=1}^{\infty}\left[ \frac{X_k - \mathbb{E}X_k}{b_k}\right]^2 = \sum_{k=1}^{\infty} \frac{Var X_k}{b_k^2} = \sum_{k=1}^{\infty} \frac{\sigma_k^2}{b_k^2} < \infty,
$$

by the Lemma 1, 

$$
\sum_{k=1}^{\infty} \frac{X_k - \mu_k}{b_k} < \infty \;\;\text{a.s.}
$$
However,
$$
\frac{S_n - \mathbb{E}S_n}{b_n} = \frac{1}{b_n}\sum_{k=1}^{\infty} b_k \left( \frac{X_k - \mu_k}{b_n} \right)
$$
thus, by the Lemma 2, the above equation converges to 0.

### Note.
Strong Law of Large Numbers 는 결국, 평균이 유계 i.e. $\mathbb{E}X_i = \mu < \infty$ 이고 분산이 유계 $Var(X_i) = \sigma^2 < \infty$ 서로 독립인 확률 변수열이면

$$
\frac{S_n}{n} - \mu \overset{a.s.}{\rightarrow} 0
$$

### Corollary 1
확률변수 열 $\{X_n\}$ 이 서로 독립이고, 4차 적률이 균등 유계이면 즉, 모든 $n$에 대하여 $\mathbb{E}(X_n - \mathbb{E}X_n))^4 < M$ 을 만족하는 실수 $M > 0$이 존재하면

$$
\frac{S_n - \mathbb{E}S_n}{n} \overset{a.s.}{\rightarrow} 0
$$

### proof of Corollary 1
$a_n = \mathbb{E}S_n = n \mu, \; b_n = n$ 이면 Cauchy-Scahwartz 부등식에 의하여 

$$
Var X_n = \mathbb{E}\left[ (X_n - \mathbb{E}X_n)^2 \cdot 1 \right] \leq M^{\frac{1}{2}}
$$

그러므로

$$
\sum_{k=1}^{\infty} \frac{\sigma^2}{k^2} \leq \sum_{k=1}^{\infty} \frac{M^{\frac{1}{2}}}{k^2} = \sqrt{M }\sum_{k=1}^{\infty} \frac{1}{k^2} < \infty
$$

따라서 Theorem 3에 의해
$$
\frac{S_n - \mathbb{E}S_n}{n} \overset{a.s.}{\rightarrow} 0
$$

### Strong law of Large Numbers (For I.I.D )
확률변수 열 $\{X_n\}​$ 이 서로 독립이고, 또 같은 분포를 따르는 확률변수열로서 $\mathbb{E}X_1 = \mu < \infty​$ 이면

$$
\frac{S_n}{n} \overset{a.s.}{\rightarrow} \mu
$$

### Note

i.i.d 의 경우는 분산에 대한 제한이 없는 것이 지금까지와는 다른 점이다. 또한 굳이 $\mathbb{E}X_1 = \mu $ 가 유한일 필요가 없다, 무한의 경우에는 평균값이 무한이 되어 마찬가지로 무한의 평균을 가지게 된다. 



## Central Limit Theorem

Strong/Weak Law of Large numbers의 경우는 평균이나 분산의 유계 혹은 i.i.d의 경우에는 평균 유계 조건에 의해 확률변수들의 합의 평균을 유추하는 것이었다. 하지만, 이 경우에는, 확률변수들의 합의 분포를 근사화 하는데는 이르지 못했다.



Central limit theorem은 확률변수들의 합의 극한 분포들을 설명하는 것이다. (결국 Weak Convergence 이론과 맞닿아 있다. )

