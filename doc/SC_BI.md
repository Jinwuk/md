Short Course of Baysian Inference
====================

Baysian Inference의 용어
$$
p(\theta | \mathcal{D}) = p(\mathcal{D} | \theta) \frac{p(\theta)}{p(\mathcal{D})} \tag{1}
$$
- A Priori (the prior) probability $p(\theta)$
- Likelihood $p(\mathcal{D} | \theta)$
- A Posteriori (the posterior) probability $p(\theta | \mathcal{D})$

#### Comment
- $\mathcal{D}$ 로 표현한 것은 Data $D_t$ 로 이루어진 $\sigma$-Field 라는 의미이다. 즉, 입력 데이터들은 $\mathcal{D}$ adapted stochastic process $D_t$로 볼 수 있다.
- 따라서 입력 데이터를 서로 구별하기 위하여 index $t$를 도입한다면 $\sigma$-Field $\mathcal{D}_t$ 로 표현하여야 옳다.
- 이렇게 표현하면, 식(1)의 경우는 좀 더 실제 알고리즘 구현에 알맞는 다음과 같은 형식으로 표현해 줄 수 있다. 먼저
$$
\begin{align}
p(\mathcal{D}_t) &= p(D_t, \mathcal{D}_{t-1}) = p(D_t | \mathcal{D}_{t-1}) p(\mathcal{D}_{t-1}) \\
\frac{p(\theta, D_t, \mathcal{D}_{t-1})}{p(\theta, \mathcal{D}_{t-1})} &= \frac{p(\theta, D_t)}{p(\theta)} = p(D_t | \theta)
\end{align}
$$
그러므로
$$
\begin{align}
p(\theta_t|D_t, \mathcal{D}_{t-1}) &= p(\theta_t, D_t, \mathcal{D}_{t-1}) \frac{1}{p(D_t, \mathcal{D}_{t-1})} = \frac{p(\theta_t, D_t, \mathcal{D}_{t-1})}{p(D_t | \mathcal{D}_{t-1}) p(\mathcal{D}_{t-1})} \\
&= \frac{p(\theta_t, D_t, \mathcal{D}_{t-1})}{p(D_t | \mathcal{D}_{t-1}) p(\mathcal{D}_{t-1})} \frac{p(\theta_t, \mathcal{D}_{t-1})}{p(\theta_t, \mathcal{D}_{t-1})} = \frac{p(\theta_t, D_t, \mathcal{D}_{t-1})}{p(\theta_t, \mathcal{D}_{t-1})} \cdot \frac{1}{p(D_t | \mathcal{D}_{t-1})} \cdot \frac{p(\theta_t, \mathcal{D}_{t-1})}{p(\mathcal{D}_{t-1})}
\end{align}
$$
정리하면
$$
p(\theta_t|\mathcal{D}_t) = p(\theta_t|D_t, \mathcal{D}_{t-1}) = p(D_t|\theta_t) \frac{p(\theta_t|\mathcal{D}_{t-1})}{p(D_t| \mathcal{D}_{t-1})}  \tag{2}
$$
- A Priori (the prior) probability $p(\theta_t|\mathcal{D}_{t-1})$, 바로 직전의 데이터 Set ($\mathcal{D}_{t-1}$)으로 학습시킨 신경망 $\theta_t$의 성능 
- Likelihood $p(D_t | \theta_t)$ : 신경망 $\theta_t$이 데이터 $D_t$를 잘 만들어 낼 수 있는 가에 대한 Likelihood 
- A Posteriori (the posterior) probability $p(\theta_t | \mathcal{D}_t)$ : 현재까지의 데이터 Set $\mathcal{D}_{t-1}$ 으로 신경망 $\theta_t$ 학습

## Baysian and Filter Theory
제어가 없는 실제 시스템에 대하여 우리가 원하는 특성을 시스템이 나타내게 하기 위해 예전에는, 우측 그림 처럼 PID 제어 혹은 시스템의 Dynamics를 알고 있으면 Pole Placement 제어기를 설계하여 Feedback 제어를 실현 하였다. 그런데 1950년대말, 미,소간 우주개발이 활발해 지면서 Rudolf Kalman의 Kalman Filter 이론이 NASA의 Apollo 계획에 적용되기 시작하면서 다음과 같이 System의 실제 출력을 통해 시스템에 대한 근사 모델의 State를 역으로 유추한 후, 이를 바탕으로 제어를 수행하는 Observer based Control 이론이 확립된다.
이때, Observer가 하는 일ㅇ

