Diffusions: Applications
==========================

## Kolmogorov Backward Equation. (Fokker-Plank Equation)

Let $X_t$ be an Ito diffusion in $\mathbb{R}^n$ with generator $A$
If we choose $f \in C^2_0 (\mathbb{R}^n)$ and $\tau = t$ in Dynkin's formula,이면  다음의 방정식이 성립한다.
$t$ 에 대하여
$$
u(t,x) = E^x [f(X_t)]
$$
는 Differentiable 이다. 그리고
$$
\frac{\partial u}{\partial t} = E^x [Af(X_t)].
$$

따라서 다음과 같이 Kolmogorov Backward Equation을 유도할 수 있다.