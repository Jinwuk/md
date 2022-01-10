Neural Tangent Kernel
===

[toc]

 인공지능에서 말하는 Kernel method의 정의는 좀더 정확하게 다르면 다음과 같다. 
 해당 내용은 다음 포스팅의 내용을 그대로 가져온 것이다. 

 https://process-mining.tistory.com/95

## Kernel method

어떤 데이터의 분포를 파악하는 것은 데이터 분석에 있어서, 나아가 머신 러닝에 있어서 굉장히 중요한 단계이다. 하지만 실제 데이터는 대부분의 경우에 정규 분포와 같은 분포 형태를 따르지 않고, 어떤 분포를 따르는지 전혀 알 수가 없다. 이런 경우에 데이터의 분포를 추정하여 확률 밀도(probability density)를 추정하는 방법에는 histogram, kernel density estimation, K-nearest neighbor 등 다양한 방법이 있다.  

여기에서는 **kernel density estimation (kernel method)**, 그리고 그 대표적인 방법인 parzen window에 대해 알아볼 것이다.

### Motivation

- 목표는 확률 밀도(probability density), 즉 $p(x)$를 구하는 것
- 이는 일정한 단위 안에 우리의 데이터가 몇 개나 포함되는지의 값과 같다. 
- 그러므로 다음과 같다고 하자.
  - $N$은 데이터 전체의 크기, 고정 값
  - $V$는 대산 Region, 즉 부피, 크기 
$$
p(\mathbf{x}) \approx \frac{K}{N \cdot  V}
\label{kernel-eq01}
$$

- **Kernel method는 region의 부피(크기) $V$ 를 고정**하고 **그 안에 몇 개가 들어갈 수 있는지($K$)를 찾아냄으로써 확률 밀도**를 구하는 방식
- 반대로 갯수를 고정하고 부피를 결정하는 것을 K-nearest method

### Parzen WIndow

- 다음과 같은 평면이 있다고 하자.
<img src="http://jnwhome.iptime.org/img/research/2021/NTK-01.png" style="zoom: 80%;" />

- 위 그림에서 사각형의 한 변을 $h$ 라고 하면  vector $\mathbf{u} \in \mathbf{R}^2$에 대하여 다음과 같이 사각형 안에 들어가면 1, 그렇지 않으면 0이 되는 어떤 Characteristic function $k(\mathbf{u})$ 를 잡으면
$$
k(\mathbf{u}) = 
\begin{cases}
1, & |u_i | \leq \frac{1}{2} h, \\
0, & else
\end{cases}
\forall i \in \mathbf{Z}[1, d=2]
\label{kernel-eq02}
$$

- 이러한 식을 데이터 포인트 $\mathbf{u} \in \mathbf{R}^d$ 에 대한 **kernel function** 이라고 한다.
- 이를 일반화 하여 $\forall \mathbf{x}, \mathbf{x}_n in \mathbf{R}^d$ 에 대하여 다음과 같이 $K$와 $V$를 정의하자.
$$
K = \sum_{n=1}^N k(\mathbf{x} - \mathbf{x}_n), \quad V= \int k(\mathbf{u}) d\mathbf{u} = h^d
\label{kernel-eq03}
$$
- 식 $\eqref{kernel-eq03}$를 식 $\eqref{kernel-eq01}$에 대입하면 다음의 식을 얻는다. 
$$
p(\mathbf{x}) \approx \frac{K}{N \cdot V} = \frac{1}{N \cdot h^d} \sum_{n=1}^N k( \mathbf{x} - \mathbf{x}_n)
\label{kernel-eq04}
$$

- 하지만 이러한 parzen window 방식은 각 큐브의 경계값 위에 데이터가 있을 때는 이를 처리하는 것이 모호하다 (discontinuity)는 단점을 가진다. 이의 해결을 위해 우리는 다른 kernel function을 이용할 수 있다. 

### Gaussian kernel (function)
- kernel function으로 Gaussian Kernel을 이용한다고 하자. 그러면 kernel function은 다음과 같다. 
$$
k(\mathbf{u}) = \frac{1}{\sqrt{2 \pi} h} \exp \left( -\frac{\mathbf{u}^2}{2 h^2} \right)
\label{kernel-eq05}
$$
  - 즉, Variance가 Kernel function의 영역 범위의 개념이 된다. 
- 식 $\eqref{kernel-eq05}$ 를 kernel function으로 할 경우 $K$와 $V$는 다음과 같다. 
$$
K = \sum_{n=1}^N k(\mathbf{x} - \mathbf{x}_n), \quad V = \int k(\mathbf{u}) d\mathbf{u} = 1
\label{kernel-eq06}
$$

그러므로 Probability density estimation을 하면 다음과 같다. 
$$
p(\mathbf{x}) \approx \frac{K}{N \cdot V} = \frac{1}{N} \sum_{n=1}^N \frac{1}{\sqrt{2 \pi} h} \exp \left( -\frac{\| \mathbf{x} - \mathbf{x}_n \|^2}{2 h^2} \right)
\label{kernel-eq07}
$$

쉽게말해 
- **Kernel function**은 데이터 1개에 대한 발생확률 이고
- Estimated Probability Deensity는 Empirical mean of data occurring 이라 보면 된다.










