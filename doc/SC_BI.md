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
<<<<<<< HEAD
- Posteriori (the posterior) probability $p(\theta_t | \mathcal{D}_t)$ : 현재까지의 데이터 Set $\mathcal{D}_{t-1}$ 으로 신경망 $\theta_t$ 학습

## Baysian and Filter Theory
만일, 실제 시스템이 다음과 같은 동역학 방정식을 가지고 있다고 가정하자.
$$
\begin{align}
x_{t+1} &= F x_t + G u_t + w_t\\
y_{t+1} &= H x_{t+1} + v_t
\end{align}
$$
| 시간 종속 파라미터 | 설명 | Constant 파라미터 | 설명 |
|---|---|---|----|
|$x_{\tau} \in \mathbb{R}^n$ | State, 상태 변수 | $F \in \mathbb{R}^{n \times n}$ | 시스템 동역학 기술 Matrix (전달함수) |
|$u_{\tau} \in \mathbb{R}$   | 시스템의 입력 값  | $G \in \mathbb{R}^n $           | 시스템 입력 전달 함수 |
|$y_{\tau} \in \mathbb{R}$   | 시스템의 출력 값  | $H \in \mathbb{R}^{1 \times n}$  | 출력 전달 함수 (State -> 출력) |

- $w_t \in \mathbb{R}^n$ State에 인가되는 잡음, i.i.d White Noise. Variance 는 $Ew_t w_t^T = Q_t$, zero mean mean 
- $v_t \in \mathbb{R}$ State에 인가되는 잡음, i.i.d White Noise. Variance 는 $Ev_t v_t^T = R_t$, zero mean

### Filtering 과정
Filtering 과정은 다음의 두 Stage를 거친다.
- A Priori (Prediction) 과정
   - Filter Parameter (신경망의 Weight 같은 존재) $\theta_{t-1|t-1}$
   - 입력 $u_{t-1}$을 가지고 
   - Prediction 데이터 $\theta_{t|t-1}$을 만든다.
- Posteriori (Update) 과정   
   - Prediction 데이터 $\theta_{t|t-1}$와 
   - Kalman Gain(신경망에서 학습계수) 를 가지고 
   - Filter Parameter를 업데이트 한다. 
   - $\theta_{t-1|t-1} \rightarrow \theta{t|t}$

### 최적화 해야 하는 함수
원하는 목적은 이것이다. 
- **$x_t$와 $\theta_{t|t}$ 가 같았으면 좋겠다.** 
   - 그런데, $x_t$와 $\theta_{t|t}$은 둘다 Random Variable
   - 따라서, Randome Variable의 거리, 혹은 Metric인 Covariance를 목적함수로 놓는다.
   $$
   P_{t|t} = \mathbb{E}(x_t - \theta_{t|t})(x_t - \theta_{t|t})^T = cov(x_t - \theta_{t|t}) 
   $$
- 그래서 이것을 최소화 시키는 Kalman Gain을 찾는다. (신경망에서 Gradient Descent 와 유사하게...)
$$
K_t = arg \min_{K_t} cov(x_t - \theta_{t|t}) \Rightarrow \frac{\partial tr P_{t|t}}{\partial K_t} = 0
$$
   - 그리고, Minimum Variancve는 $\theta_{t|t}$가 $x_t$의 평균 값일 떄 얻어지므로 ($\mathcal{D}_t$ 대신 $\mathcal{Y}_t$ 사용, $p(X_t = x_t)$를 $p(x_t)$로 표시)
$$
\begin{align}
p(x_t|\mathcal{Y}_t) &= \frac{1}{Z} \exp \left(-\frac{1}{2} (x_t - \theta_{t|t})^T P_{t|t}^{-1} (x_t - \theta_{t|t}) \right) \\
p(x_t|\mathcal{Y}_{t-1}) &= \frac{1}{Z} \exp \left(-\frac{1}{2} (x_t - \theta_{t|t-1})^T P_{t|t-1}^{-1} (x_t - \theta_{t|t-1}) \right) 
\end{align}
$$

특별히 만일, 제어 입력 $G = 0$ 이라고 가정하면, Filter의 Baysian은 다음과 같이 명확하다.
$$
p(X_t|\mathcal{Y}_t) = p(X_t|Y_t, \mathcal{Y}_{t-1}) = p(Y_t|X_t) \frac{p(X_t|\mathcal{Y}_{t-1})}{p(Y_t| \mathcal{Y}_{t-1})} 
$$
- A priori (the prior) probability (Prediction) $p(X_t|\mathcal{Y}_{t-1})$
- posteriori (the posterior) probablity (Update)$p(X_t|\mathcal{Y}_t)$
- Likelihood $p(Y_t|X_t)$

##### 왜 이렇게 표현하는가?

### Estimator의 종류
- **Minimum Mean Square Error Estimation** : $\hat{X}^{\text{MMSE}}_t= E(X_t |\mathcal{Y}_t ) = \int_t x_t \cdot p(x_t |\mathcal{Y}_t)dx_t$
- **Maximum a Posteriori Estimation (MAP)**: $\hat{X}^{\text{MAP}}_t= \arg \max_{x_t} p(x_t|\mathcal{Y}_t)$
- **Maximum Likelihood (ML) Estimator** : $\frac{\partial f_{Y_t|X_t}}{\partial x}|_{x = \theta_t} = 0$, $f$는 $\log p(Y_t|X_t)$ 등을 사용하여 보통, 다음 함수를 최소화 시킨다. 
$$
J = \frac{1}{2}(y_t - H \theta_t)^TR_t^{-1}(y_t - H \theta_t)
$$
- Minimum Conditional KL Divergence, Minimum Conditional Accuracy Estimator등이 있다.
- Kalman Filter는 결국 평균을 구하기는 하지만, 기본적으로 **Variance를 최소화 시켜 확률을 극대화** 시키는 것이므로 MAP Estimator 이다.
   - 여기서 $\hat{X}^{\text{MMSE}}_t = \hat{X}^{\text{MAP}}_t = \theta_{t|t}$ 
   - 평균 개념을 사용하거나 Log-Likelihood 개념을 사용하여 업데이트 Rule을 유도한다. 

### Neural Network은 왜 이렇게 표현하지 못하는가?
- 목적함수가 Estimator와 다르기 때문이다. 
   - 그렇다고 못하는 것은 아니다. 
   - Filter 알고리즘과 유사한 Competitive Learning 계열은 동일하게 사용 가능하다.
   - 기본적으로 Hyper Plane을 직접 기술하는 방식에서는 직접 적용이 어려울 수 있다.
- Estimator 는 Covariance를 비교적 쉽게 정의할 수 있는 반면, Neural Network은 목적함수가 다종 다양하다.
   - Supervised Learning의 경우 (Desired Value가 있는 경우) : 일반적인 Covariance로 정의하는 대신, $P$ Momentum Distance를 정의하고 $P$를 업데이트 하는 방식은 가능하다.
      - 이 경우 $P$가 Covariance를 대신할 수 있다.
   - 만일, Random Variable이 되도록, Update 자체에 White Noise 항을 정의하게 되면, 어떠한 목적함수를 사용하더라도 MAP 방식의 Learning이 가능하다. (연구과제)

- 평균 개념대신 직접적으로 Weight를 업데이트 할 수 있는 기법이 있다. (가장 큰 이유)
   - Markov Chain Monte-Carlo 등. 
   - 실제 예가 Boltzmann Machine 

## Baysian 기반 Neural Network
ANN(Artificial Neural Network)에서는 오랫동안 MLE 방식의 알고리즘을 사용해왔다.
- A priori 확률을 알고 있는 것이 없기 떄문, 오직 데이터의 Measurement만 알고 있기 때문
$$
\theta_{t+1} = \theta_t - \eta_t \nabla_{\theta} J(y_t, \theta_t) = \theta_t + \eta_t H^T R_t^{-1}(y_t - H \theta_t)
$$
   - 위 알고리즘은 다음의 Measurement 확률을 최대화 시킨다. ($\mathcal{Y}_t$가 Gaussian, i.i.d White Noise에 의해 만들어진 $y_t$에 의해 생성된 $\sigma$-algebra)
$$
p(Y_t|X_t) = \frac{1}{Z} \exp(-\frac{1}{2}(y_t - H \theta_t)^TR_t^{-1}(y_t - H \theta_t)) 
$$
   - Baysian 관점에서 보면 다음과 같다. (Since we don't know correctly a priori probability $p(X_t|\mathcal{Y}_{t-1})$ and $p(Y_t|\mathcal{Y}_{t-1})$ is a uniform or meanigless probabblity such as normal factor, )
$$
\begin{align}
\theta_t &= \max_{x_t} \int_x x_t p(X_t|\mathcal{Y}_t) dx_t = \max_{x_t} \int_x x_t p(Y_t|X_t) \frac{p(X_t|\mathcal{Y}_{t-1})}{p(Y_t|\mathcal{Y}_{t-1})} dx_t \\
&= \int_x x_t \max_{x_t} p(Y_t|X_t) \frac{dP(X_t|\mathcal{Y}_{t-1})}{dP(Y_t|\mathcal{Y}_{t-1})} = \int_x x_t \max_{x_t} f(X_t) dx_t
\end{align}
$$

### Idea
- **A priori 확률을 정의한다.** 
   - Normalization Factor 로 $p(Y_t|\mathcal{D}_{t-1})$ 을 간주하자. 
   - 그러면 자연스럽게 **MAP** Estimator로서 Baysian Neural Network이 정의된다.
- State $x_t$를 정의하여 알고리즘을 만들 것인가, 아니면 **Weight 자체를 직접 Update 할 것인가.** 
   - State $x_t$를 정의하게 되면 미분 기반의 알고리즘을 만들 수는 있다. (예 : Hopfield Network)
   - Monte Carlo 계열의 알고리즘으로 직접 Weight를 Update 하면 굳이 미분을 쓰지 않아도 된다. (예: Boltzmann Machine)

### Fundamental Baysian
Let $w \in \mathbb{R}^n$ is weight, $\mathcal{D}$ is a $\sigma$-algebra generated by data set. since $p(X)= \int p(X|Y)p(Y) dy$,
$$
p(w|\mathcal{D}) = p(\mathcal{D}|w) \frac{p(w)}{p(\mathcal{D})} = \frac{p(\mathcal{D}|w)p(w)}{\int p(\mathcal{D}|w)p(w) dw}    \tag{3}
$$

#### A priori Probablity $p(w)$
##### Assumption 
- A priori Probability $p(w)$는 Boltzman 분포 (혹은 Gibb's 분포)를 따른다고 가정한다. 
- Energy 함수는 Weight가 어떤 형태를 가지는 것이 좋을 것인가로 놓는다.

##### Construction of A priori Probablity
###### Definition of Energy Function
- Weight의 수가 작았으면 좋겠다. (1)
   - Regularization 문제이기 때문 (함수 근사화 하는데 Overfitting 문제를 해결하기 위해)
$$
E_w = \frac{1}{2} \| w \|^2 = \frac{1}{2} \sum_{i=1}^{W} w_i^2, 
$$

###### Definition of A priori Probablity
A priori Probability $p(w)$는 Boltzman 분포를 따른다고 했으므로 온도계수 $T$ 에 대하여  
$$
p(w) = \frac{1}{Z_w} \exp (-\frac{E_w}{T_w}) = \frac{1}{Z_w} \exp (-\alpha \cdot E_w) \tag{4}
$$
여기서 $Z_w$는 Normalization Factor로 $Z_w = \int_{\mathbb{R}^n} \exp (-\alpha \cdot E_w) dw$. 
- 다시말해, $w$가 가질 수 있는 모든 경우에 대하여 나누어 주어야 한다.
- 실제로 $Z_w$를 구하여 $p(w)$를 구한다는 것은 경우의 수가 너무 많기 때문에 어려운 일이지만 다음과 같이 구하는 것이 일반적이다. 
   - 그래서 많은 경우 Gaussian 분포를 가정하거나 다음 식을 사용하여 정규화 해야 한다. 
   $$\int_{-\infty}^{\infty} \exp(-x^2) dx = \sqrt{\pi}$$
- 그러나, Baysian을 더 분석하면 Normalization Factor가 사라지거나 대치 될 수 있으므로 일단 개념적으로 놓아둔다.

##### Construction of Likelihood
Likelihood는 **실제로 최적화 하려는 Cost를 Boltzman 분포의 Energy로 놓는다**. Neural Network의 출력을 입력 $x$와 weight에 의해 $y(x;w)$ 라 놓고 n번쨰 Training Data를 $t^n$ 이라 놓자. 그러면 Likelihood는 다음과 같다.
$$
\begin{align}
E_{\mathcal{D}} &= \frac{1}{2} \sum_{n=1}^N \left( y(x; w) - t^n \right)^2 \\
p(\mathcal{D}|w) &= \frac{1}{Z_{\mathcal{D}}} \exp (- \frac{E_{\mathcal{D}}}{T_{\mathcal{D}}}) = \frac{1}{Z_{\mathcal{D}}} \exp (- \beta E_{\mathcal{D}}) =\prod_{n=1}^{N} p(t^n|x^n ,w)  \tag{5}
\end{align}
$$
where $Z_{\mathcal{D}} = \int \exp (-\beta E_{\mathcal{D}}) dw$

##### Posterior over the weights
Since the equation (3), (4), (5) are hold,
$$
\begin{align}
p(w|\mathcal{D}) &= p(\mathcal{D}|w) \frac{p(w)}{p(\mathcal{D})} = \frac{1}{Z_{\mathcal{D}}} \exp(-\beta E_{\mathcal{D}}) \cdot \frac{1}{Z_w} \exp(-\alpha E_w) \cdot \frac{1}{\int \frac{\exp(-\beta E_{\mathcal{D}} - \alpha E_w)}{Z_{\mathcal{D}} Z_w}dw}  \\
&= \frac{\exp(-\beta E_{\mathcal{D}} - \alpha E_w)}{\int \exp(-\beta E_{\mathcal{D}} - \alpha E_w) dw} = \frac{1}{Z_S}\exp(-S(w))
\end{align}
$$
- 문제를 살펴보면 결국 다음 목적함수를 최적화 시키는 문제로 귀결된다. 
$$
S(w) = \frac{1}{2} \left(\beta \sum_{n=1}^N \left( y(x; w) - t^n \right)^2 + \alpha \|w\|^2 \right) = \frac{\beta}{2} \left( \sum_{n=1}^N \left( y(x; w) - t^n \right)^2 + \frac{\alpha}{\beta} \|w\|^2 \right)
$$
- 이때, $\frac{\alpha}{\beta} = \lambda$ 로 놓으면 1계 제한 조건을 가지는 Largrangian 문제로 변형된다.
$$
S(w) = \frac{\beta}{2} \left( \sum_{n=1}^N \left( y(x; w) - t^n \right)^2 + \lambda \|w\|^2 \right)
$$
- Khun-Tucker Condition에 의해 다음이 $S(w)$를 위한 하나의 minimizer 필요조건이다.
$$
0 = \nabla_w E_{\mathcal{D}} + \lambda \nabla_w E_w, \;\; \lambda = -\frac{\nabla_w E_w^T \nabla_w E_{\mathcal{D}}}{\nabla_w E_w^T \nabla_w E_w}
$$

### Output from Baysian Neural Network.
Baysian Neural Network을 출력은 두 가지 방식이 모두 가능하다. 
- Weight Matrix로 생각하고 Fermi-Dirac 통계를 통해 출력을 내는 방법
- Weight Vector로 생각하고 정통적인 MAP Estimation으로 출력을 내는 방법 (Radial Basis Function 방법)

#### Distribution over outputs
신경망 출력의 분포함수는 어떠할 지 생각해 본다.
$$
p(y|x, \mathcal{D}) = \int p(y|x, w)p(w|\mathcal{D})dw 
$$
   - 이때,, Training 데이터에 의해 MAP로 추정된 Weight, $\hat{w}$가 존재하고 그로 인한 신경망 출력이 $\hat{y} = y(\hat{w}, x)$ 라고 하면, 임의의 데이터 $x$ 가 들어왔을 때, 출력 $p(y|x, \mathcal{D})$을 Gaussian 분포 확률로 나타내고자 한다. 
$$
p(y|x, \mathcal{D}) \approx \frac{1}{\sqrt{2 \pi }\sigma_y} \exp \left( -\frac{(y - \hat{y})^2}{2 \sigma_y^2} \right)
$$

여기서 Variance는 다음과 같다. 먼저, Gaussian 분포의 Baysian 특성은 다음과 같다.
$$
\begin{align}
p(x) &= \mathcal{N}(x, \mu, \Lambda^{-1}) \\
p(y|x) &= \mathcal{N}(y, Ax+b, L^{-1}) \\
p(y) &= \mathcal{N}(y, A\mu+b, L^{-1} + A  \Lambda^{-1} A^T )
\end{align}
$$
이므로 $p(w|\mathcal{D})$를 Gaussian 분포 $q(w|\mathcal{D})$ 로 생각하면 $p(y|x, \mathcal{D}) = p(y|x, w) q(w| \mathcal{D})$ 로 볼 수 있다. 이때, $q(w| \mathcal{D})$의 mean은 $\hat{w}$, Variance $\sigma^2 = \Lambda^{-1}$는 $S(w)$의 $w$에 대한 Hessian $A=\alpha I + \beta H$ 이다. 여기서 $y(w, x) = \hat{y} + \nabla_w^T y(w, x)|_{w=\hat{w}}(w - \hat{w})$ 로 근사화 하면 $p(y|x, w) \approx \mathcal{N}(y, \hat{y}+\nabla_w^T y(w, x)|_{w=\hat{w}}(w - \hat{w}), (\beta =\frac{\partial^2 S}{\partial y^2})^{-1})$ 그러므로 $p(y|x, \mathcal{D}) = \mathcal{N}(y, \hat{y}, \sigma^2)$ . 
$$
\sigma^2 = \beta^{-1} + \nabla_w y(w,x)^T |_{w=\hat{w}} A^{-1}\nabla_w y(w,x)|_{w=\hat{w}}
$$
=======
- A Posteriori (the posterior) probability $p(\theta_t | \mathcal{D}_t)$ : 현재까지의 데이터 Set $\mathcal{D}_{t-1}$ 으로 신경망 $\theta_t$ 학습

## Baysian and Filter Theory
제어가 없는 실제 시스템에 대하여 우리가 원하는 특성을 시스템이 나타내게 하기 위해 예전에는, 우측 그림 처럼 PID 제어 혹은 시스템의 Dynamics를 알고 있으면 Pole Placement 제어기를 설계하여 Feedback 제어를 실현 하였다. 그런데 1950년대말, 미,소간 우주개발이 활발해 지면서 Rudolf Kalman의 Kalman Filter 이론이 NASA의 Apollo 계획에 적용되기 시작하면서 다음과 같이 System의 실제 출력을 통해 시스템에 대한 근사 모델의 State를 역으로 유추한 후, 이를 바탕으로 제어를 수행하는 Observer based Control 이론이 확립된다.
이때, Observer가 하는 일ㅇ
>>>>>>> 0bf2dc419bed46060879033840f1bb761c66cb44

