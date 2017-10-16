Mathematical Note
======

## $\frac{\varepsilon}{1 + \varepsilon}$을 사용한 확률 공간에서의 Convexity

Convex 조건은, Domain이 만일 Convex 일 경우 함수 $f$가 contimuity 이면 (최소 Lipschitz Continuous 이면), Compact 조건을 만족 시킬 수 있기 떄문에 알고리즘의 수렴성 증명에 있어 매우 중요한 조건이다.

그러나 일반적인 Convex 조건에 의한 해석은 Euclidean Space $\mathbb{R}^n$ 이나, Manifold $M^n$ 에서 유효하다.
따라서 확률 변수의 경우에는 적분이 무한대 적분으로 나타내야 하는 경우가 많으므로 Convexity를 체크하기 위한 변수가 무한대에서 잡을 수 있도록 해야 한다.
따라서 이러한 경우, Convexity 변수를 $s \in \mathbb[0,1]$ 대신 $\varepsilon \in \mathbb{R}$로 놓아야 한다. 이때 Convexity Check를 위해 다음과 같이 놓아보자.

$$
\lim_{\varepsilon \rightarrow 0} \frac{\varepsilon}{1 + \varepsilon} = 0, \;\;\; \lim_{\varepsilon \rightarrow \infty} \frac{\varepsilon}{1 + \varepsilon} = 1 
$$

그러므로 어떤 확률 변수 $z$가 1과 z에서 변한다고 가정해 보자. 즉 Deterministic 한 경우에
$$
h = x + s(y - x), \;\;\; s \in \mathbb{R}[0,1]
$$

이것을 1과 z과 바꾸어 보면
$$
h = z + s(1 - z) = (1 -s)z + s = \left( 1 - \frac{\varepsilon}{1 + \varepsilon} \right) z + \frac{\varepsilon}{1 + \varepsilon} = \frac{1}{1 + \varepsilon}z + \frac{\varepsilon}{1 + \varepsilon}
$$

z와 1의 위치를 바꾸어 보면
$$
h = 1 + s(z - 1) = (1 -s) + sz = \left( 1 - \frac{\varepsilon}{1 + \varepsilon} \right) + \frac{\varepsilon}{1 + \varepsilon}z = \frac{1}{1 + \varepsilon} + \frac{\varepsilon}{1 + \varepsilon}z
$$
