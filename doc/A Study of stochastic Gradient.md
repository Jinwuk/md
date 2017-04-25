A Study of Stochastic Gradient
=====================

## Gradient Identity

### Lemma 1
$$
\nabla_{\mu_i} \mathcal{N}(\xi | \mu, C) = - \nabla_{\xi_i} \mathcal{N}(\xi | \mu, C)
$$
#### proof of Lemma
$$
\begin{align}
\nabla_{\mu} \mathcal{N}(\xi | \mu, C) &= \frac{\partial}{\partial \mu} \frac{1}{\sqrt{2 \pi C}} \exp \left(- \frac{(\xi -\mu)^2}{2C} \right) \\
&= \frac{1}{\sqrt{2 \pi C}} \frac{\partial}{\partial \mu}  \exp \left(- \frac{(\xi -\mu)^2}{2C} \right) \\
&= \frac{1}{\sqrt{2 \pi C}} \frac{(\xi - \mu)}{C} \exp \left(- \frac{(\xi -\mu)^2}{2C} \right) \\
\end{align}
$$
$$
\begin{align}
\nabla_{\xi} \mathcal{N}(\xi | \mu, C) &= \frac{\partial}{\partial \xi} \frac{1}{\sqrt{2 \pi C}} \exp \left(- \frac{(\xi -\mu)^2}{2C} \right) \\
&= \frac{1}{\sqrt{2 \pi C}} \frac{\partial}{\partial \xi} \exp \left(- \frac{(\xi -\mu)^2}{2C} \right) \\
&= \frac{1}{\sqrt{2 \pi C}} -\frac{(\xi - \mu)}{C} \exp \left(- \frac{(\xi -\mu)^2}{2C} \right) 
\end{align}
$$
Thus,
$$
\nabla_{\mu_i} \mathcal{N}(\xi | \mu, C) = - \nabla_{\xi_i} \mathcal{N}(\xi | \mu, C)
$$
Q.E.D.

### Bonnet's Theorem
Let $f(\xi) : \mathbb{R}^d \rightarrow \mathbb{R}$ be a integrable and twice differentiable function. The gradient of the expectation of $f(\xi)$ under a Gaussian distribution $\mathcal{N}(\xi | \mu, C)$ with respect to the mean $\mu$ can be expressed as the expectation of the gradient of $f(\xi)$.
$$
\nabla_{\mu_i} \mathbb{E}_{\mathcal{N}(\mu, C)} [f(\xi)] = \mathbb{E}_{\mathcal{N}(\mu, C)}[\nabla_{\xi_i} f(\xi)]
$$

#### proof
$$
\begin{align}
\nabla_{\mu_i} \mathbb{E}_{\mathcal{N}(\mu, C)} [f(\xi)] &= \int \nabla_{\mu_i} \mathcal{N}(\xi | \mu, C) f(\xi) d\xi \\
&= - \int \nabla_{\xi_i} \mathcal{N}(\xi | \mu, C) f(\xi) d\xi  \;\;\;\;\;\;\text{by Lemma 1 } \\
&= [\int \mathcal{N}(\xi | \mu, C) f(\xi) d\xi]_{\xi = -\infty}^{\xi = \infty} + \int \mathcal{N}(\xi | \mu, C) \nabla_{\xi_i} f(\xi) d\xi \\
&= \int \mathcal{N}(\xi | \mu, C) \nabla_{\xi_i} f(\xi) d\xi \\
&= \mathbb{E}_{ \mathcal{N}(\mu, C)} [\nabla_{\xi_i} f(\xi)]
\end{align}
$$

### Lemma 2
$$
\nabla_{C} \mathcal{N}(\xi | \mu, C) = \frac{1}{2} \frac{\partial^2 }{\partial \xi^2} \mathcal{N}(\xi | \mu, C)
$$
#### proof of Lemma
$$
\begin{align}
\nabla_{C} \mathcal{N}(\xi | \mu, C) &= \frac{\partial }{\partial C} \frac{1}{\sqrt{2 \pi C}} \exp \left(- \frac{(\xi -\mu)^2}{2C} \right) \\
&= \frac{1}{\sqrt{2 \pi C}} \cdot -\frac{1}{2} \cdot C^{-1} \cdot \exp \left(- \frac{(\xi -\mu)^2}{2C} \right) + \frac{1}{\sqrt{2 \pi C}} \cdot (\xi - \mu)^2 \cdot \frac{1}{2} \cdot C^{-2} \cdot \exp \left(- \frac{(\xi -\mu)^2}{2C} \right) 
\end{align}
$$
From Lemma 1
$$
\begin{align}
\frac{\partial^2 }{\partial \xi^2} \mathcal{N}(\xi | \mu, C) &= \frac{\partial }{\partial \xi} \frac{\partial }{\partial \xi} \mathcal{N}(\xi | \mu, C) \\
&= \frac{\partial }{\partial \xi} \frac{1}{\sqrt{2 \pi C}} -\frac{(\xi - \mu)}{C} \exp \left(- \frac{(\xi -\mu)^2}{2C} \right) \\
&= \frac{1}{\sqrt{2 \pi C}} \cdot -C^{-1} \cdot \exp \left(- \frac{(\xi -\mu)^2}{2C} \right) + \frac{1}{\sqrt{2 \pi C}} \cdot (\xi - \mu)^2 \cdot C^{-2} \exp \left(- \frac{(\xi -\mu)^2}{2C} \right)
\end{align}
$$
Thus,
$$
\nabla_{C} \mathcal{N}(\xi | \mu, C) = \frac{1}{2} \frac{\partial^2 }{\partial \xi^2} \mathcal{N}(\xi | \mu, C)
$$

Q.E.D.

### Price's Theorem
Under the same condition as the theorem of Bonnet, the gradient of the expectation of $f(\xi)$ under a Gaussian distribution $\mathcal{N}(\xi| 0, C)$ with respect to the covariance $C$ can be expressed in terms of the expectaion of the Hessian of $f(\xi)$ as
$$
\nabla_{C_{i, j}} \mathbb{E}_{\mathcal{N}(0,C)} [f(\xi)] = \frac{1}{2} \mathbb{E}_{\mathcal{N}(0,C)} [\nabla_{\xi_i, \xi_j} f(\xi)]
$$

#### proof
$$
\begin{align}
\nabla_{C_{i, j}} \mathbb{E}_{\mathcal{N}(0,C)} [f(\xi)] &= \int \nabla_{C_{i, j}} \mathcal{N}(\xi | 0,C) f(\xi) d\xi \\
&= \frac{1}{2} \int \nabla_{\xi_i, \xi_j} \mathcal{N}(\xi | 0,C) f(\xi) d\xi \;\;\;\;\;\;\text{From Lemma 2} \\
&= \frac{1}{2} \int \mathcal{N}(\xi | 0,C) \nabla_{\xi_i, \xi_j} f(\xi) d\xi \;\;\;\;\;\;\text{부분 적분 2번 적용} \\
&= \frac{1}{2} \mathbb{E}_{\mathcal{N}(0,C)} [\nabla_{\xi_i, \xi_j} f(\xi)]
\end{align}
$$

### Theorem 3
$$
\nabla_{\theta} \mathbb{E}_{\mathcal{N}(\mu,C)} [f(\theta)] = \mathbb{E}_{\mathcal{N}(\mu,C)} \left[ (\nabla_{\theta} f(\theta) )^T \frac{\partial \mu}{\partial \theta} + \frac{1}{2} Tr \left( H \frac{\partial C}{\partial \theta} \right) \right]
$$

#### proof
$$
\begin{align}
\nabla_{\theta} \mathbb{E}_{\mathcal{N}(\mu,C)} [f(\theta)] &= \frac{\partial}{\partial \theta} \mathbb{E}_{\mathcal{N}(\mu,C)} [f(\theta)] \\
&= \frac{\partial }{\partial \mu} \frac{\partial \mu}{\partial \theta} \mathbb{E}_{\mathcal{N}(\mu,C)} [f(\theta)] + \frac{\partial }{\partial C} \frac{\partial C}{\partial \theta} \mathbb{E}_{\mathcal{N}(\mu,C)} [f(\theta)] \\
&= \frac{\partial }{\partial \mu} \mathbb{E}_{\mathcal{N}(\mu,C)} [f(\theta)] \cdot \frac{\partial \mu}{\partial \theta} + \frac{\partial }{\partial C}  \mathbb{E}_{\mathcal{N}(\mu,C)} [f(\theta)] \cdot \frac{\partial C}{\partial \theta} \\
&= \mathbb{E}_{\mathcal{N}(\mu,C)} [\nabla_{\theta} f(\theta)] \cdot \frac{\partial \mu}{\partial \theta} + \frac{1}{2} \mathbb{E}_{\mathcal{N}(\mu,C)} Tr(\mathbb{E}_{\mathcal{N}(\mu,C)}[H(f(\theta))]\cdot \frac{\partial C}{\partial \theta} \\
&= \mathbb{E}_{\mathcal{N}(\mu,C)} \left[ (\nabla_{\theta} f(\theta))^T \frac{\partial \mu}{\partial \theta} + \frac{1}{2} Tr(H(f(\theta) \cdot \frac{\partial C}{\partial \theta} ) \right]
\end{align}
$$

## My Idea

$$
\nabla_{\theta} [f(\theta)] = \nabla_{\theta} \mathbb{E}_{\mathcal{N}(\mu,C)} [f(\theta)] + \nabla_{w_t} f(\theta)
$$
이면 Ito Differential 에 의해 앞 항과 뒤 항을 각각 유도하여 증명할 수 있을 것으로 생각된다.
즉,
$$
z_t = \mu(z_t) + \sigma(z_t) w_t  
$$
라 하면..
$$
dz_t = \frac{\partial \mu}{\partial z_t} dt + \frac{\partial \sigma}{\partial z_t} dw_t
$$
이므로 Ito Calculus에 의해
$$
\begin{align}
df(\theta_t) &= (\nabla_{\theta_t} f(\theta_t))^T d\theta_t + \frac{1}{2} Tr \left( H(f(\theta_t)) d\theta_t^2 \right)\\
&= (\nabla_{\theta_t} f(\theta_t) )^T \left(\frac{\partial \mu}{\partial \theta_t} dt + \frac{\partial \sigma}{\partial \theta_t} dw_t \right) + \frac{1}{2} Tr \left (H(f(\theta_t)) \left(\frac{\partial \sigma}{\partial \theta_t} \right) \left( \frac{\partial \sigma}{\partial \theta_t} \right)^T \right) dt \\
&= \left[ (\nabla_{\theta_t} f(\theta_t) )^T \frac{\partial \sigma}{\partial \theta_t} +  \frac{1}{2} Tr \left( H(f(\theta_t)) \frac{\partial C}{\partial \theta} \right) \right] dt + (\nabla_{\theta_t} f(\theta_t) )^T \frac{\partial \sigma}{\partial \theta_t} dw_t



\end{align}
$$

