# Simulated Annealing using Sampling Trick in Motion Estimation
Motion Estimation에서 Sampling에 대한 조작으로 Annealing Effect를 만들 수 있을 것인가에 대한 연구이다.
예를 들어 4K UHD 에서는 적어도 256 by 256의 Search Range를 가져야 충분히 최적화된 Motion Estimation 결과를 얻을 수 있다고 가정한다면,
만일, 이것의 1/16 영상 즉, 해상도 960x540 영상에서는 64 by 64 Search Range 에서 충분히 최적화된 Motion Estimation 결과를 얻을 수 있을 것이다. 
이것이 정말로 가능한 것인지, 그리고, 이를 Simulated Annealing의 측면에서 보면 어떠한 것인지를 생각해 보자.

## Simple Survey
Annealing Probabiliuty가 다음과 같다고 하자

$$
P(X_j|X_i) = \exp\left(-\frac{\Delta E}{k_b T} \right)
$$

여기서 $\Delta E = E(X_j) - E(X_i)$ 로 정의한다. 
