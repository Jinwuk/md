Hamilton Jacobi Bellman Equation
====

1. The state equation
$$
\dot{x} = f(t, x(t), u(t)) \tag{1}
$$

2. The performance measure
$$
J = h(x(t_f), t_f) + \int_{t_0}^{t_f} F(\tau, x(\tau), u(\tau)) d\tau 
$$

## The induction of the HJB equation


The performance measure is a function on $x(t), t$, and $u(t)$. However, we want to find a control that minimize the performance measure. Therefore, the optimal performance measure based on the optimal control $u(t)$ is as follows :


$$
J^*(x(t),t) = \underset{\underset{1 \leq \tau \leq t_f}{u(\tau)}}{\min} \left\{ J^*(x(t),t) + \frac{\partial J}{\partial t} \Delta +  \frac{\partial J}{\partial x} \left(x(t+\Delta) - x(t)\right) + \int_{t}^{t+\Delta t}  F(\tau, x(\tau), u(\tau)) d\tau \right\}
$$

by the expansion of the taylor series

Now for small $\Delta t$, since the performance measure $J$ is only dependent on $x(t)$ and $t$, we can obtain

$$
J^*(x(t),t) = \underset{\underset{1 \leq \tau \leq t_f}{u(\tau)}}{\min} \left\{ J^*(x(t),t) + \frac{\partial J^*}{\partial t} \Delta t +  \frac{\partial J^*}{\partial x} f(t,x,u) \Delta t + F \Delta t \right\}
$$

(By $x(t+\Delta) - x(t) =\Delta x = f(t, x(t), u(t)) \Delta x$))

$$
0 = \frac{\partial J^*}{\partial t} \Delta t + \underset{\underset{1 \leq \tau \leq t_f}{u(\tau)}}{\min} \left\{\frac{\partial J^*}{\partial x} f(t,x,u) \Delta t + F \Delta t \right\}
 \tag{2}
$$

Since the Hamiltonian $H$ can be defined as follows (즉 $\min$ 안에 있는 것이 Hamiltonian 실제 핵심인 이것만 최적화 시켜야 하며 그것이 Hamiltonian 최적화의 핵심 아이디어임 )

$$
H(t, x, u, \frac{\partial J^*}{\partial x}) = F(t, x, u) + \frac{\partial J^*}{\partial x} f(t,x,u)
$$

, and
$$
H(t,x,u^*(x(t), \frac{\partial J^*}{\partial x}, t), \frac{\partial J^*}{\partial x}) = \underset{\underset{1 \leq \tau \leq t_f}{u(\tau)}}{\min} H(t, x, u, \frac{\partial J^*}{\partial x}) \tag{3}
$$


The HJB equation is defined as follows from (2), (3)
$$
0 = \frac{\partial J^*}{\partial t} + H(t, x, u^*(x(t), \frac{\partial J^*}{\partial x}), \frac{\partial J^*}{\partial x})
$$
