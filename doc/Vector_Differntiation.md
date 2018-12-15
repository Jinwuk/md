Vector Differntiation
======================
$\forall x, y \in \mathbb{R}^n$

1. 미분은 Transpose 된 동일한 Parameter에 행해지고 나머지 항은 그대로 나온다.
	- 미분의 결과가 그대로 Vector가 되기를 원하기 때문
3. Transpose 않은 동일한 Parameter에 행해진 미분은 나머지 항이 Transpose 되어야 한다.

$$
\frac{\partial }{\partial x}(x^T y) = y
$$

$$
\frac{\partial }{\partial x}(x^T x) = 2x
$$

$$
\frac{\partial }{\partial x}(x^T Ay) = Ay
$$

$$
\frac{\partial }{\partial x}(y^T Ax) = \left(y^T A \right)^T = A^T y
$$

$$
\frac{\partial }{\partial x}(x^T Ax) = (A + A^T)x
$$

$$
\frac{\partial }{\partial x}(a^T(x) Qa(x)) = 2(\nabla_x a^T(x))Q a(x)
$$

