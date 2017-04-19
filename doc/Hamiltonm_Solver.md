Hamiltonian 적용 예
=================

##### Hamiltonian 해석의 기본 방정식과 해법
$$
\text{State}\;\;\;\;   \dot{x} = \frac{\partial \mathcal{H}}{\partial \lambda} , \;\;\;
\text{Costate}\;\;\;\; \dot{\lambda}^* = -\frac{\partial \mathcal{H}}{\partial x}, \;\;\;  
\text{Control}\;\;\;\; 0 = \frac{\partial \mathcal{H}}{\partial u} 
$$
- Final 상태가 고정되어 있지 않다면 Pontragyn's Maximal Principle에 의해 횡단조건 추가.
   - 횡단조건 (Transversal Condition) : $\lambda(T) = 0$

##### 문제 
$$
\begin{align}
\textit{Object :}\;\;\; &\int_0^1 (y(t) - u^2(t))dt \\
\textit{Constraint :}\;\;\; & \dot{y}(t) = u(t), \;\; y(0)=5, \;\; y(1) \in \mathbb{R} \text{:무제약}
\end{align}
$$
##### 해법
- Hamiltonian : 
 $$\mathcal{H}(t) = (y(t) - u^2(t)) + \lambda \cdot u(t)$$
- 1계 조건 (최적 Control 조건) 
$$
\frac{\partial \mathcal{H}}{\partial u} = -2 u + \lambda = 0 \implies u(t) = \frac{1}{2} \lambda, \;\; \dot{y}(t) = \frac{1}{2} \lambda
$$
- Costate Equation : 횡단조건에서 $\lambda(1) =0$
$$
\dot{\lambda}^* = -\frac{\partial \mathcal{H}}{\partial y} = -1 \implies \lambda(t) = c_1 - t \implies \lambda(t) = 1 - t
$$
- Derive the State $y$ : Since $\dot{y}(t) = \frac{1}{2} \lambda = \frac{1}{2}(1 - t)$, and $y(0) =5$, and Optimal Control $u(t) = \frac{1}{2} \lambda$
$$
y = \frac{1}{2} t - \frac{1}{4} t^2 + 5, \;\;\;u(t) = \frac{1}{2}(1 - t)
$$


