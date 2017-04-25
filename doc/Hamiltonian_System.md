Hamiltonian System
===================
[TOC]
## The Least Action Principle - 최소작용 원리
시간 $t_1$ 에서 $t_2$ 에서의 시스템의 동역학 상태가 각각, A, B 라고 하면 상태 A 에서 B 로의 진화는 다음 적분의 값이 최소가 되도록 진화한다.

$$
S \equiv \int_{t_1}^{t_2} \mathcal{L}(x, \dot{x}, t) dt
$$

이 적분을 시스템의 작용 옥은 Action 이라고 정의하며 적분안의 라그랑지안은 경로에 따라 그 값이 달라지는 경로의 함수이다. 이를 **최소 작용 원리, 혹은 해밀턴의 원리**라고 한다.

## Euler-Lagrange 방정식
방정식
$$
S \equiv \int_{t_1}^{t_2} \mathcal{L}(x, \dot{x}, t) dt
$$

에 대하여 어떤 특정한 경로  가 위 적분 값을 가장 작게 만든다고 가정할 때, 그러한 경로를 만족시키는 미분방정식은
$$
\frac{\partial \mathcal{L}}{\partial x} - \frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{x}} \right ) = 0
$$

### Memorize
$\mathcal{L}$ 을 $x$로 미분후 $\dot{x}$ 미분의 시간에 대한 미분을 뺸다.

### Proof
Let $x(t)$ as follows
$$
x(\alpha, t) = x(t) + \alpha \eta(t)
$$
where $\alpha$ is a parameter when it is 0, $x(t)$  is the minimum phase such as $x(\alpha, t) = x(t)$, $\eta(t)$  is a differentiable function w,r,t time t, that is 0 at end of time , such that $\eta(t_1) = \eta(t_2) = 0$

$\alpha = 0$ 일때 최소값을 가진다면, 모든 임의의 함수에 대하여 다으므이 필요조건이 만족된다.

$$
\frac{\partial S}{\partial \alpha} |_{\alpha=0} = 0
$$

$\alpha$의 미분값은
$$
\frac{\partial S}{\partial \alpha} = \int_{t_1}^{t_2} \frac{d \mathcal{L}}{d \alpha} dt = \int_{t_1}^{t_2} \left( \frac{\partial \mathcal{L}}{\partial x}\frac{\partial x}{\partial \alpha } + \frac{\partial \mathcal{L}}{\partial \dot{x}}\frac{\partial \dot{x}}{\partial \alpha} \right ) dt
$$
여기에서
$$
\frac{\partial x}{\partial \alpha} = \eta, \;\; \frac{dx}{dt} = \frac{\partial x}{\partial t} + \alpha \frac{\partial \eta}{\partial t} \Rightarrow \frac{d}{d \alpha}\left( \frac{dx}{dt} \right)= \frac{\partial \eta}{\partial t}
$$
이므로

$$
\frac{\partial S}{\partial \alpha} = \int_{t_1}^{t_2} \left( \frac{\partial \mathcal{L}}{\partial x}\eta + \frac{\partial \mathcal{L}}{\partial \dot{x}}\frac{\partial \eta}{\partial t} \right ) dt
$$

또한 부분적분에서 
$$
\int \frac{\partial \mathcal{L}}{\partial \dot{x}}\frac{\partial \eta}{\partial t} dt = \frac{\partial \mathcal{L}}{\partial \dot{x}} \eta - \int \frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{x}} \right) \eta dt
$$

$$
\frac{\partial S}{\partial \alpha} = \int_{t_1}^{t_2} \left( \frac{\partial \mathcal{L}}{\partial x}\eta  \right ) dt + \frac{\partial \mathcal{L}}{\partial \dot{x}} \eta |^{t=t_2}_{t=t_1} - \int_{t_1}^{t_2} \frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{x}} \right) \eta dt
$$

이때 중간 항 $\frac{\partial \mathcal{L}}{\partial \dot{x}} \eta |^{t=t_2}_{t=t_1} $ (i.e. \eta(t_0) = \eta(t_1) = 0) $ 이므로 이를 정리하면

$$
\frac{\partial S}{\partial \alpha} = \int_{t_1}^{t_2} \left( \frac{\partial \mathcal{L}}{\partial x}\eta  \right ) dt - \int_{t_1}^{t_2} \frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{x}} \right) \eta dt = 0
$$

** Q.E.D. **

## Largrange 운동 방정식
전체 운동 에너지를  $\mathcal{L}$ 라고 하고 일반화된 좌표계 위에서의 힘을 $Q_0$ 라고 하면


### Lagrange 운동 방정식
$$
\frac{\partial \mathcal{L}}{\partial x} - \frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{x}} \right ) = Q_0
$$

### 보존계의 Largrange 운동 방정식
$$
\frac{\partial \mathcal{L}}{\partial x} - \frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{x}} \right ) = 0
$$

이때, 만일 보존계의 운동 에너지가 $\mathcal{L} = F + \lambda \dot{x}$ 이라고 하면 보존계의 Largrange 운동 방정식에 의해
$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \dot{x}} \equiv \lambda \;\;\;\;&: \;\;\;\; \frac{\partial \mathcal{L}}{\partial \dot{x}} = \frac{\partial F}{\partial \dot{x}} (x,t) + \lambda \\
\frac{\partial \mathcal{L}}{\partial x} = \dot{\lambda} \;\;\;\;&: \;\;\;\; \frac{\partial \mathcal{L}}{\partial x} - \frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{x}} \right) = \frac{\partial \mathcal{L}}{\partial x} - \frac{d \lambda}{dt} = 0
\end{align*}
$$
(위에서 $\frac{\partial F}{\partial \dot{x}} (x,t) = 0 $)

### Note
위 공상태 방정식을 기업하는 방법은 Dot 가 어디에 붙어 있는 가 이다.

## Hamiltonian 운동 방정식
Largrangian $\mathcal{L}(x, \dot{x}, t)$ 의 전미분이 다음과 같을 때 (그리고 $\frac{\partial \mathcal{L}}{\partial x} - \frac{d}{dt}\left( \frac{\partial \mathcal{L}}{\partial \dot{x}} \right)) = 0$
$$
d\mathcal{L} = \frac{\partial \mathcal{L}}{\partial x} dx + \frac{\partial \mathcal{L}}{\partial \dot{x}} d{\dot{x}} + \frac{\partial \mathcal{L}}{\partial t} dt
$$
그런데, Largrangian 운동 방정식에서 $\frac{\partial \mathcal{L}}{\partial \dot{x}}=\lambda$ (Largrangian에서 Largange Multiplier는 보통 dx/dt=0 제한조건에서 ) $\frac{\partial \mathcal{L}}{\partial x}= \dot{\lambda}$ (Largrangian 운동 방정식에서 자연스럽게 유도) 이므로

$$
d\mathcal{L} = \dot{\lambda} dx + \lambda d{\dot{x}} + \frac{\partial \mathcal{L}}{\partial t} dt
$$
위 식에 다음을 대입한다.
$$
\lambda d\dot{x} = d(\lambda \dot{x}) - \dot{x} d\lambda
$$
그러면

$$
\begin{align*}
d\mathcal{L} &= \dot{\lambda} dx + d(\lambda \dot{x}) - \dot{x} d\lambda + \frac{\partial \mathcal{L}}{\partial t} dt \\
d(\lambda \dot{x} -\mathcal{L}) &= -\dot{\lambda} dx +  \dot{x} d\lambda - \frac{\partial \mathcal{L}}{\partial t} dt
\end{align*}
$$
여기에서 Hamiltonian 을 다음과 같이 정의한다.
$$
H = \lambda \dot{x} -\mathcal{L}
$$

이렇게 정의하면 위 Hamiltonian 운동 방정식은 다음과 같다.
$$
dH = -\dot{\lambda} dx +  \dot{x} d\lambda - \frac{\partial \mathcal{L}}{\partial t} dt
$$

### Note 
살펴보면 Hamiltonian $H$ 와 적분 경로 $\mathcal{L}$ 은 서로 부호가 반대이다.
이 반대 부호 때문에 공상태 방정식에 있어서 부호가 반대로 나타나는 경우가 발생한다.
(적분 경로에 대해서는 부호가 동일하였다)

## Hamiltonian 운동 방정식 해석
### Largrangian Multiplier 와 Hamiltonian

Hamiltonian $H = \lambda \dot{x} -\mathcal{L}$ 에서 
1. Largrangian Multiplier 에 대하여 Hamiltonian을 미분한 경우
2. 상태 변수 $x$ 에 대하여 Hamiltonian을 미분한 경우
를 각각 살펴보자.

1의 경우는 다음과 같다. (매우 쉽다)
$$
\frac{\partial H}{\partial \lambda} = \dot{x}
$$

2의 경우는 다음과 같다. 다음과 같이 유도된다.
$$
\frac{\partial H}{\partial x} = \frac{\partial}{\partial x}(\lambda \dot{x}) - \frac{\partial \mathcal{L}}{\partial x}
$$
여기서 우변의 첫항은 
$$
\frac{\partial}{\partial x}(\lambda \dot{x}) = \lambda \frac{\partial \dot{x}}{\partial x} =  \lambda \frac{\partial }{\partial x} \frac{\partial x}{\partial t} = \lambda \frac{\partial }{\partial t} \frac{\partial x}{\partial x} = 0
$$
우변의 두번쨰 항은
$$
\frac{\partial \mathcal{L}}{\partial x} = \dot{\lambda}
$$
그러므로 다음과 같다.
$$
\frac{\partial H}{\partial x} = - \dot{\lambda}
$$
이를 ** 공상태 방정식** 이라 한다.


### Hamiltonian과 Lagrangian 의 차이
결론적으로 보면 일반적인 Largrangian에 추가로 시간에 따른 상태 변화 (시간에 따른 상태의 1계 미분) 에 대한 추가적인 Largrangian 이 붙은 것으로 볼 수 있다.

즉, 다음과 같다.
$$
H = \lambda \dot{x} - \mathcal{L}
$$

그런데 보존계의 운동 에너지가
$$
\mathcal{L} = F + \lambda \cdot \dot{x}
$$

로 주어지면 이는 $H = F$이 된다. 다시말해, Hamiltonian은 보존계의 어떤 에너지가 되어 버린다. 

만일, 시간에 따른 상태 변화가 없는 정상 상태가 된다고 가정하면 ($\dot{x} = 0$) Hamiltonian은 자연스럽게 Largrangian의 운동 에너지가 된다. 그런데, Largrangain 이 well defined 되어 있다고 가정하면, 여기에 상태 변화에 따른 Largrangian Multiplier가 추가된 것으로 볼 수 있다. 

그러므로 만일, **Largrangian**이 well defined 되어 어떤 **상태 변화에 대한 제한 조건이 잡혀 있지 않는 상태**라고 가정하고
**상태 변화에 대한 항만 추가** 되면 이것이 **Hamiltonian** 이 되는 것이다.

이러한 생각을 기반으로 최적 제어 분야에서 Hamiltonian이 적용 된다. (Hamiltonian과 제어 이론)

### Note
정리하면 일반적인 보존계 에너지에 **상태변화에 대한 등식 제한 조건**이 추가되면 **Hamiltonian** 이다.

## Hamiltonian과 제어 이론
상태 방정식이 상태 $x(t)$와 제어 입력 $u(t)$ 그리고 시간 $t$에 의해 다음과 같이 정의된다고 가정하자.
$$
\dot{x}(t) = f(x(t), u(t), t)
$$

이때 Admissable trajectory $x^*$ 가 있어 다음의 Performance Measure를 최소화 시킨다고 가정하자.
$$
J(u) = h(x(t_f), t_f) + \int^{t_f}_{t_0} g(x(t), u(t), t) dt
$$
이와 관련한 Hamiltonian을 유도하는 과정은 다음을 참조한다.[^1]

하지만, 지금까지의 논의를 생각해본다면, 보존계의 에너지를 $g(x(t), u(t), t)$ (왜냐하면 이 값을 최소화 시켜야 하므로) 로 놓고 시간에 따른 상태 변화를 등식 제한조건으로 놓으면 다음과 같이 Hamiltoian을 정의 할 수 있다
$$
\mathcal{H}(x(t),u(t),\lambda, t) = g(x(t), u(t), t) + \lambda f(x(t),u(t), t)
$$

이때 Hamiltonian $\mathcal{H}$의 최적 제어는 다음의 3 조건에서 유도된다.

#### 상태 방정식 조건
$$
\dot{x}^*(t)=\frac{\partial \mathcal{H}}{\partial \lambda}(x(t),u(t),\lambda, t) \\
$$

#### 공상태 방정식 조건
$$
\dot{\lambda}^*(t) = -\frac{\partial \mathcal{H}}{\partial x}(x(t),u(t),\lambda, t)
$$

#### 최적 제어조건
$$
0 = -\frac{\partial \mathcal{H}}{\partial u}(x(t),u(t),\lambda, t)
$$

### Hamiltonian 운동방정식과의 비교
동일하다. 단, 입력 u 에 대한 항이 더 추가 되었다.
1. 상태 방정식 조건
$$
\dot{x} = \frac{\partial H}{\partial \lambda}
$$
 
2. 공상태 방정식 조건
$$
\dot{\lambda} = - \frac{\partial H}{\partial x}
$$

해당 이론은 최종적으로 Pontryagins's minimum (maximum) Principle 를 통해 완성된다.

#### Note 
항상 기억하자 공상태 방정식 - 상태 방정식과 Dot의 위치는 그대로 있고 편미분의 분모가 $x$ 냐 $\lambda$ 냐에 따라 달라진다. 
기본의 상태방정식이다. 
Hamiltonian 공상태 방정식은 마이너스 부호가 더 붙는다. 
보존계 상태방정식은 $\dot{x}$ 로 편미분하는 것이며 이때 그 결과인 $\lambda$는 당연하게도 dot가 붙지 않는다. 이는 Hamiltonian에서도 동일하다.

## Pontryagins's minimum (maximum) Principle
최적 제어는 반드시, **Hamiltonian을 최소 (혹은 최대) 로 만드는 것이다.** ** 다시말해 다음의 조건을 만족해야 한다.

### Minimum Principle
$$
\mathcal{H}(x^*(t), u^*(t), \lambda^*, t) \leq \mathcal{H}(x^*(t), u(t), \lambda^*, t)
$$
상태와 Largrangian multiplier가 최적이라 하더라도 **입력이 최소값을 만들 수 있어야 한다.**

### Maximum Principle
$$
\mathcal{H}(x^*(t), u^*(t), \lambda^*, t) \geq \mathcal{H}(x^*(t), u(t), \lambda^*, t)
$$
어떤 효용을 극대회 시키는 것으로 해석할때 이렇게 된다.

### Note
결론적으로
$$
\frac{\partial \mathcal{H}}{\partial u} = 0
$$

이어야 한다는 의미이다. 그런데 이는 Necessary Condition이기 때문에 극대와 극소를 모두 포함한다.

그러므로 Pontryagins's minimum (maximum) Principle 에 의해 최소(최대)화 시키는 제어는 반드시 그렇지 않은 제어값보다 작거나(크거나) 해야 하는 것이다.

==================================================
[^1]:Donald E. Kirk, 'Optimal Control Theory : An Introduction', Prentice Hall, pp185 - 188
