Stochastic Method in Statistical Machines
=========================================

## Simple Associative Memory

Assume that $x^i \in \mathbb{R}^n$ is Input or **Visible Vector** and $x^j \in \mathbb{R}^n$ is a **Hidden vector**.
In addition, the Visible and Hidden vector, in which each component has a value $+1, -1$, is normal vector such that $\left\| x^k \right\| = 1$, for instance
$$
x^k = \frac{1}{\sqrt{N}}(1, -1, 1, -1 \cdots , 1)^T. 
$$

Let Weight matrix  $W = [w_{kl}]$ such that

$$
W = x_j x_i^T \;\;\textit{where}\;\; w_{kl} = x_k^j x_l^i.
$$

Then
$$
W x_i = (x^j {x^i}^T) x^i = x^j ({x^i}^T x^i) = x^j \;\; \because \left\| x^k \right\| = 1
$$

#### Note
As a result, weight matrix $W$ is symmetric such that $W = W^T$ i.e. $w_{ji} = w_{ij}$

### For Non-Normalized Case
Introduce the following threshold to decide a output to be 1 or -1 for $W \in \mathbb{R}^{n \times n}$
$$
\theta_k \triangleq -\frac{1}{2} \sum_{l=1}^n w_{kl} \Rightarrow \Theta = [\theta_k] \in \mathbb{R}^n
$$

The result of calculation is
$$

$$
