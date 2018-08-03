Differential Game
=====

## Introduction
미분 게임은 제어문제의 확장으로 볼 수 있다. 즉, Two-sided 제어 문제로 볼 수 있다. 다음 시스템을 생각해보자

#### System
$$
\dot{x}(t) = A(t)x(t)+ B_1(t)u(t) + B_2(t)v(t) \;\;\; t \in \mathbb{R}[0, T], \;\; x(0)=x_0  \tag{1}
$$

#### Cost Function
$$
J_i (u(\cdot), v(\cdot)) = \int_0^T \{ u'(t)R_i(t)u(t) + x'(t)Q_i(t)x(t) \} dt + x'(T)Q_{if}x(T)
$$
where $x, x_0 \in \mathbb{R}^n$, $u,v \in \mathbb{R}^m$ 그리고 $A, R_i, Q_i, Q_{if}, B_i, i=1,2$는 구분적 연속인 적당한 차원의 실수행렬들이다. 특히, $R_i, Q_i, Q_{if}, i=1,2$ 는 대칭행렬이고 $R_i, i=1,2$는  Positive Definite 이다. 

$J_1$ 은 $u$ 경기자의 Cost Function이고 $J_2$는 $v$ 경기자의 Cost Function이다. 
두 경기자 $u,v$는 시스템 State $x(t)$에 대하여 제어작용을 하고 그들의 제어 작용은 비용함수 $J_1, J_2$에 대하여 각각 영향을 준다. 두 경기자 $u,v$의 목적은 그들의 비용함수들 $J_1, J_2$를 최소화 하는 것이다. 

## 미분 게임의 구성
### Strategies
결정론적 최적제어이론이나 변분론에 있어서는 비용함수를 최적으로 만드는 최적제어함수 $u(t), t\in [0, T]$ 을 적당한 함수공간에서 찾아내는 것이 문제. 그러나 미분게임에서는 비용함수들의 값이 논으의 대상이 되며 최적제어함수 $u(t), v(t) t \in [0,T]$를 안다는 것이 떄로는 유용하다. 이때, 우리는 그 게임에 대하여 경기를 진행할 것인지, 아닌지를 결정할 수 있다. 
만일, 한 편의 경기자가 어떤 제어를 택하고 있다면, 다른 편의 경기자는 ** 최적제어가 아니라 전략을 필요로 하게 된다.** 즉, 시스템의 반응은 두 경기자의 행동선택(각각의 제어)에 다 같이 영향을 받기 때문에 전략에 의하여 경기자들은 상대방의 실수를 즉각적으로 유리하게 이용할 수 있는 기회를 포착하는 것이다. 
LQDG (Linear Quadratic Dynamic Game) 문제에 있어서 우리는 Feedback 전략에 관심을 두기로 한다.즉,
$$
u(t) = F_1 (t, x(t)), \;\; v(t) = F_2(t, x(t)), 0 \leq t < T
$$
그러나 일반적으로 미분게임에 있어서의 전략은 Feedback 전략만 있는 것이 아니다.

### Concept of  Solution
#### Definition : Nash equilibrium pair
모든 전략 $u,v$ 에 대하여 미분 방정식 조건 (1)에 대하여,
$$
J_1 (u, v_N^*) \geq J_1 (u_N^*, v_N^*) \;\;\textit{and}\;\; J_2 (u, v_N^*) \geq J_2 (u_N^*, v_N^*)
$$
이 성립하면, 전략쌍 $(u_N^*, v_N^*)$ 를 **(최적)Nash 평형쌍 (Nash equilibrium pair)**라 한다.

#### Definition : Pareto Stratage Pair
적당한 전략 $u,v$에 대하여 미분 방정식 조건 (1)하에서,
$$
J_i (u, v) \geq J_i (u_P^*, v_P^*) \;\; i=1,2 
$$
이면 반드시 
$$
J_i (u, v) = J_i (u_P^*, v_P^*) \;\; i=1,2 
$$
이면 전략쌍 $(u_P^*, v_P^*)$ 를 **최적 Pareto 전략쌍** 이라고 한다.

여기서, 다음과 같은 사실을 유추해 볼 수 있다. 즉, 양측 경기자가 다 같이 Pareto 최적 전략에 따른다면 양측 모두 차선책을 쓰는 경기라 할 수 있다. 그러나 이 경우 상대편에 의한 속임수를 당하지 않는다는 보장이 없다. Nash 최적전략에 따른다면 양측 경기자는 누구도 일방적으로 평형상태를 벗어나고자 하지는 않는다. (이를 포로 또는 Arms Limitation의 딜레마라고 한다.) Nash 혹은 Pareto 최적전략외에도 여러가지 최적전략이 있다. 그래서 전략분석 연구자는 그때의 문제에 따라서 최선의 개념을 선택해야 한다. $J = J_1 - J_2$ 라 두면 우리는 LQDG에 의한 특수한 문제를 얻는다. (이때 ** 경기자 $u$를 Minimizer, 경기자 v를 Maximizer라고 한다.** )

#### Definition : Saddle Point
$$
\forall u, v, \;\; \max_v \min_u J(u,v) = \min_u \max_v J(u,v)=J(u^*, v^*)
$$
이면 전략쌍 $(u_P^*, v_P^*)$를 안장전이라고 한다.

명백히 LQDG에서 **안정점은 Nash 평형점과 동일**하다. 특히 안장점의 존재를 보이기 위해서는 다음 부등식을 보이면 된다. 
$$
\max_v \min_u J(u,v) \geq \min_u \max_v J(u,v)
$$
$\max-\min$이 $(u, \bar{v})$ 에서 일어났다고 가정하자. 그러면 모든 전략 $u$에 대해서, 
$$
\max_v \min_u J(u,v) \leq J(u, \bar{v})
$$
이다. 그리고 $\min-\max$가 $(\bar{u}, v)$에서 일어났다고 가정하면 모든 전략 $v$에 대하여
$$
\min_u \max_v J(u,v) \geq J(\bar{u}, v)
$$
여기서 안장점의 존재를 보이는 것은 최소지표(minimum performance)의 보장을 의미한다. 위 3개의 부등식으로 부터 다음을 얻는다.
모든 전략 $u,v$ 에 대하여
$$
J(u^*, v) \leq J(u^*, v^*) \leq J(u, v^*)
$$
(최소값 보다 안장점은 크고 최대값보다 안장점은 작다.)
특히 위식에서 안장점의 값 $J(u^*, v^*)$은 유일하다는 것을 알 수 있다. 그러나, 이 부등식이 안장점의 값보다 작거나 큰 비용함수를 가질 수 없다는 것을 의미하는 것은 아니다. 
특, 최소자 $u$ (혹은 최대자 $v$)가 상대방 $v$ (혹은 $u$)의 전략을 미리 알고 있다면, $J(u^*, v^*)$보다 작은  (혹은 큰)값을 비용함수가 가질 수 있도록 할 수 있다. 끝으로, 만약 Open loop 안장점 $(u,v)$ 이것을 순수전략 : Pure Stratage 라 하자.)이 존재한다면 이것들을 계산해 낼 수 있게 된다. 일반적으로 우리는 다음을 알고 있다.
$$
\max_{v(t)} \min_{u(t)} J(u(\cdot), v(\cdot)) < \min_{u(t)} \max_{v(t)} J(u(\cdot), v(\cdot)), \;\; 0 \leq t \leq T
\tag{3}
$$
양변의 값 차이를 없애는 것은 다음 절에서 생각해 보자.

### Viewpoint of Game theory
(이산적인) 행렬 게임에서 maximum과 minimum의 차이를 줄이기 위하여 (즉, 식(3)에 의하여 주어진 안장점을 얻기 위하여 각 경기자는 반드시 복합전략 (mixed stratage)를 사용해야 한다는 것을 Von Neumann과 Morgenstern이 증명하였다. 즉, 여기서 복합전략은 각 경기자(Row 경기자와 Column경기자)가 그의 Row 혹은 Column을 선택하는 빈도를 나타내는 확율분포를 의미한다. 
어떤, 특수한 미분게임, 예를 들어, 시스템과 비용함수가 상태변수에 대하여 선형인 비분 게임에서는 '이완된' 제어라 불리우는 방법에 의하여 안장점을 구할 수 있다는 것을 Elliot, Kalton, Markus 가 보였다. 더우기 만약 미분게임의 Hamiltonian이 $u$와 $v$의 혼합된 항을 가지고 있지 않다면 안장점이 있ㅇ므을 보일 수 있다.
Flemming, Varayia-Lin, Friedman 에 의하여 미분게임에 대한 또 다른 시도가 이루어졌다, 그들은 시간구간 $[0, T]$를 n 등분하여 다음과 같은 두 개의 최적제어 문제를 생각했다.
$$
V_n = \max_{\bar{v}_1} \min_{\bar{u}_1} \max_{\bar{v}_2} \min_{\bar{u}_2} \cdots \max_{\bar{v}_n} \min_{\bar{u}_n} J(u,v)
$$
$$
V^n = \min_{\bar{u}_1} \max_{\bar{v}_1} \min_{\bar{u}_2} \max_{\bar{v}_2}  \cdots \min_{\bar{u}_n} \max_{\bar{v}_n}  J(u,v)
$$
단, $\bar{u}_i$ 와 $\bar{v}_i$, $i=1,2, \cdots n$ 은 i 번째 부분 구간에서 정의된 제어 함수들이고 $u$는 $\bar{u}_1, \bar{u}_2, \cdots, \bar{u}_n$ 을 연결한 $[0, T]$에서의 함수이고 $v$는 $\bar{v}_1, \bar{v}_2, \cdots, \bar{v}_n$를 연결한 함수이다. 여기서 **$V^n$은 Upper value, $V_n$은 Lower value** 이다. 그러면 $V^n \geq V_n, \;\; n=1,2,3 \cdots $ 임을 쉽게 알 수 있다.
이때 다음 조건을 생각한다.
$$
\lim_{n \rightarrow \infty} V^n = \lim_{n \rightarrow \infty} V_n = V
$$
이는 LQDG의 하나이다.

### 미분 게임 해법의 특수성 
미분 게임에 있어 Open-Loop 최적 제어 함수 $u(t), v(t), \;\; 0 \leq t \leq T$를 구하는 것 보다, Close-Loop Feedback제어법칙 $u(t,x(t)), v(t, x(t))$를 구하는 것이 중요하다.
미분게임의 최적제어 문제에 대하여 변분적 방법이나 적당한 함수공간에 대하여 최적제어 $u^*(t), v^*(t), 0 \leq t \leq T$를 찾고자 하는 람수해석적 방법을 사용하기는 어렵다. 더우기 최적제어 $u^*(t), v^*(t), 0 \leq t \leq T$를 찾고자 하는 함수 해석적 방법을 사용하기는 어렵다. 더우기 최적제어 $u^*, v^*$에 관심이 있다고 하더라도 이분 게임에 있어서 변분적 방법은 매우 취약하다, (최적 제어 함수의 특성을 나타내는데 있어서)즉, 변법에 있어 1차적인 필요조건들을 양측 최적제어 문제인 $\max_{v(\cdots)} \min_{u(\cdots)} J(u,v)$ 와 $\min_{u(\cdots)} \max_{v(\cdots)} J(u,v)$를 구별하지 못한다. 그래서 안장점 조건을 조사할 수 없다. 더우기 선형화 (예를 들어 Gradient  형태의 방법들)에 의해서 제시되는 수치해석적 방법들은  미분게임 문제에 일반적으로 적용할 수 없다.
즉, 상대편이 그의 제어를 (안장점에서)조금만 변화 시키더라 하더라도 (Perturbation) 시스템의 궤도는 (제어들이 안장점일때의 궤도에서) 크게 벗어날 수 있다. 따라서 미분게임에 있어서 선형화는 적합하지 않다. 따라서 미분게임을 풀기 위해서는 일반적으로 **대역적 Hamilton-Jacobi Dynamic Programming/Regular Synthesis** 방법을 사용하는데, 이방법은 해를 구성하기 위한 충분조건을 제공한다. 대략적으로 이 방법은 **비선형 Hamilton-Jacobi 편미분 방정식 **을 푸는 것이고 또한 이 방법은 합성한 최적해가 있음을 보이는 것이다.
그러나, 편미분 방정식이 정의되는 영역에서 해가 특이점을 가질 수도 있다는 어려움이 있다. 이때 불연속 곡면을 따라서 여러개로 분리된 Hamilton-Jacobi 편미분 방정식을 풀어서 이들 해들을 하나로 붙여야 (Patch Together) 된다. 다행히도, 선형 이차형식 미분게임에서는 단지 행렬 Riccati 미분 방정식을 푸는 문제로 귀결된다.

## 유한시간 LQDG 
Cost Function $J$를 $T > 0$에 대하여 다음과 같이 둔다.
$$
J(u(\cdot), v(\cdot)) = \int_0^T [u^T(t)R_1(t)u(t) - v^T(t) R_2(t)V(t)+x^T(t)Q(t)x(t)] dt + x^T(T)Q_f x(T)
$$
where $ Q, Q_f$ is symmetry $n \times n$ real Matrix. $Q(\cdot)$ 는 구분적 연속인 행렬함수이다. 

이떄 최적 Feedback 전략은 다음과 같다.

$$
\begin{align}
u^*(t,x(t)) &= -R_1^{-1} (t) B_1^T P(t)x(t) \tag{3-1}\\
v^*(t,x(t)) &= -R_2^{-1} (t) B_2^T P(t)x(t) \tag{3-2}
\end{align}
$$

where $P(t), 0 \leq t \leq T$ is the solution of the follwoing Ricatti equation.
$$
\begin{align}
-\dot{P}(t) &= A^T(t)P(t)+P(t)A(t)-P(t)\left(B_1^T(t) R_1^{-1}(t) B_1(t) - B_2^T(t) R_2^{-1}(t) B_2(t)\right) P(t) +Q(t) \\
P(T) &= Q_f   \tag{4}
\end{align}
$$

Moreover, the minimum oost is given as follows:
$$
J(u^*(\cdot), v^*(\cdot)) = x_0^T P(0) x_0
$$
여기서 방정식(4)의 해가 존재한다고 가정하면 (즉, $T$가 방정식 (4)의 유한 탈출 시간 보다 작다고 가정하면)
LQDG 문제를 다음 두개의 최적제어문제로 생각할 수 있다.
$J(u,v)$는 2차 형식 비용함수이고 $P(\cdot)$는 Riccati 방정식의 해라고 하자.
$$
\begin{align}
(P_1) &: \max_{v(\cdot)} J(u(\cdot), v(\cdot)) \\
(P_2) &: \min_{u(\cdot)} J(u(\cdot), v(\cdot)) 
\end{align}
$$
문제 $(P_1), (P_2)$, 는 다 같은 전형적인 이차형식 최적제어 문제이다 따라서 $(P_1)$에 대하여 다음의 feedback 제어를 얻을 수 있다.
$$
v(t) = R_2^{-1}(t)B_2^T(t)P_1(t)x(t)  \tag{4-1}
$$
여기서 $P_1$는 다음 행렬 Riccati 미분방정식의 해이다.
$$
\begin{align}
-\dot{P}(t) &= \left(A(t) - B_1(t) R_1^{-1}(t) B_1^T(t) P(t)\right)^T P_1(t) + P_1(t)\left(A(t) - B_1(t) R_1^{-1}(t) B_1^T(t) P(t)\right) \\
&+ P_1(t)B_2(t)R_2^{-1}(t)B_2^T(t)P_1(t) + Q(t) + P(t)B_1(t)R_1^{-1}(t)B_1(t)P(t) \\
P_1(T) &= Q_f   \label{eq_14_3_8} \tag{5}
\end{align}
$$
그리고 with same way, we can obtain the following control for $(P_2)$:
$$
u(t) = -R_1^{-1}(t) B_1^T(t)P_2(t)x(t)    \tag{5-1}
$$

여기서 $P_2(t)$는 다음 행렬 Riccati 방정식의 해이다:

$$
\begin{align}
-\dot{P}(t) &= \left(A(t) + B_2(t) R_2^{-1}(t) B_2^T(t) P(t)\right)^T P_2(t) + P_2(t)\left(A(t) - B_2(t) R_2^{-1}(t) B_2^T(t) P(t)\right) \\
&+ P_2(t)B_1(t)R_2^{-1}(t)B_1^T(t)P_2(t) + Q(t) + P(t)B_2(t)R_2^{-1}(t)B_2(t)P(t) \\
P_2(T) &= Q_f   \tag{5}
\end{align}
$$

따라서 방정식 (4), (5), (6) (모두 Riccati EQ) 으로 부터 $[0, T]$구간에서 $P_1(t)=P_2(t)=P(t)$ 라는 것을 알 수 있고 경기자 $u$의 Feeddback 전략 (3-1)과 (5-1)은 결국 같은 것임을 알 수 있다.이는 경기자 $v$에 대해서도 마찬가지다.

비용함수 $J$의 복잡성을 피하기 위해 
$$
J(u(\cdot), v(\cdot)) = \int_0^T \left( u^T(t) R_1 (t) u(t) - v^T(t) R_2 (t) v(t) \right)dt + x^T(T) Q_f x(T)   \tag{6}
$$
즉, $Q(t)=0$ 
이떄 간단한 최적 제어문제 $(P_3)$을 생각해보면 
시스템 (1)과 (6)에서 주어진 제어
$$
u^0(t) = -R_1^{-1}(t) B_1^T(t) \Psi^T(0, t)P(0)x_0   \tag{7}
$$
하에서
$$
(P_3): \max_{v(\cdot)} J(u^0(\cdot), v(\cdot))
$$
단, $\Psi(t,\tau)$는 제차 방정식
$$
\dot{x}(t) = A(t)x(t)
$$
의 Fundamental Matrix이다. Adjoint variable $\lambda(t)$를 고려할 때 즉,
$$
\lambda(t) = p(t)x(t), u^0=-R_1^{-1}(t)B_1^T \lambda(t)
$$
와 
$$
\lambda(t) = \Psi(0,t)\lambda(0)
$$
식 (7)의 $u^0(t)$는 식(3-1)의 Optimal feedback control에 대한 Open-Loop OPtimal control function이다. 따라서 문제 $P(3)$은 시스템 (1)에서 식(7)에 의하여 Perturbed 된 $v(\cdot)$에 관한 선형 이차 형식의 최적 제어 문제이다.
최적제어문제 $P(3)$의 해는 다음과 같다.
$$
v(t) = R_2^{-1}(t)B_2^T(t)\left(P_3(t)x(t) + a(t) \right)
$$
여기서
$$
\begin{align}
-\dot{P}_3(t) &= A^T(t)P_3(t) + P_3(t)A(t)+ P_3(t)B_2(t)R_2^{-1}(t)B_2^T(t)P_3(t) \\
P_3(t) &= Q_f \\
\dot{a}(t) &= -A^T(t)a(t) - P_3(t)B_2(t)R_2^{-1}(t)B_2^T(t)a(t) - P_3(t)B_1(t)u^0(t) \\
a(T) &= 0
\end{align}
$$

같은 방법으로 $u(\cdot)$ 에 대한 선형 이차형식 문제를 생각하자:
$J$를 식 (6)에서 주어진 비용함수라고 하고  Perturbation $v^0(t)$를 
$$
v^0 (t) = R_2^{-1}(t) B_2^T(t) \Psi^T (0,t) P(0) x_0
$$
라 하자. 최적제어문제 $(P_4)$를 다음과 같이 둔다.
$(P_4)$ : Under the system (1) $\min_{u(\cdot)} J(u(\cdot), v^0(\cdot))$
그러면 최적 제어 문제 $(P_4)$의 해는 다음과 같다:
$$
u(t) = -R_1^{-1}(t)B_1^T(t) \left( P_4(t) x(t) + b(t) \right)
$$
여기서
$$
\begin{align}
-\dot{P}_4(t) &= A^T(t)P_4(t) + P_4(t)A(t)- P_4(t)B_1(t)R_1^{-1}(t)B_1^T(t)P_4(t) \\
P_4(t) &= Q_f \\
\dot{b}(t) &= -A^T(t)b(t) + P_4(t)B_1(t)R_1^{-1}(t)B_1^T(t)b(t) + P_4(t)B_2(t)v^0(t) \\
b(T) &= 0
\end{align}
$$

다음을 쉽게 알 수 있다:

$$
P_3(t) \neq P_1(t) = P
$$

and

$$
P_4(t) \neq P_3(t) = P^T
$$

## Example
추격자 (Pursuer, interceptor) $P$와 도망자 (Evader) $E$에 대한 수학적 모델은 다음과 같다.
$$
\left\{\begin{matrix}
 \dot{v}_p (t) = g + a_p (t), &\dot{v}_e (t) = g + a_e(t) \\ 
 \dot{r}_p (t) = v_p (t) , &\dot{r}_e (t) = v_e(t) 
\end{matrix}\right.
$$
where $v(t)$ is a velocity vector in a 3-dimensional space, $r(t)$ is a location vector in a 3-dimensional space, and $g$ is a gravity vectpr to a material, $a(t)$ is a acceleration vector as a control function.

The cost functionis given as follows : for $T>0$
$$
J = \alpha \| r_p(T) - r_a (T) \|_2^2 + \int_0^T \left( c_p^{-1} \| a_p (t) \|_2^2 - c_e^{-1} \| a_e (t) \|_2^2 \right) dt \tag{8}
$$

Constant values $\alpha, c_p, c_e > 0$,

Cost Function 



