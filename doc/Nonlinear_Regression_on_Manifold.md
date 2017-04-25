Nonlinear Regression on Manifold
======

$$
\begin{align}
\dot{x}_t &= f(x_t) + g(x_t) u + \sigma_1 dB_t \\
y &= h(x_t) + \sigma_2(x_t) dW_t
\end{align}
$$

먼저 Deterministic 한 경우를 생각해 보자

$$
\begin{align}
\dot{x}_t &= f(x_t) + g(x_t) u\\
y &= h(x_t)
\end{align}
$$

Backpropagation을 생각해 보면 맨 마지막 Layer의 경우는
$$
y = h(x_t, w_t)
$$


## Financial Ratio 

$$
\begin{align}
e &= (1+\frac{1}{n})^n \\
y(r) &= \left ((1+\frac{r}{n})^{\frac{n}{r}} \right)^r \\
\lim_{n \rightarrow \infty} y(r) &= \lim_{n \rightarrow \infty}\left ((1+\frac{r}{n})^{\frac{n}{r}} \right)^r
= \left( \lim_{n \rightarrow \infty} (1+\frac{r}{n})^{\frac{n}{r}} \right)^r = e^r
\end{align}
$$

Let $P(r) = \lim_{n \rightarrow \infty} y(r)$
$$
\begin{align}
P(r) = e^r \Rightarrow \ln P(r) = r\;\; 
\end{align}
$$
and $n = r$ 
