Idea Note (From 2018-09-27)
====

## QP Adapted Decoding (Post Processing)
만일,  QP가 rate control에 의해 계속 가변하는 상태라 가정하자.  
이때, Frame의 Contents가 변화하지 않는 상황이라면 rate Control에 의해 QP가 상승하는 경우, 이를 낮은 QP에서의 Contents를 유추하여 디코딩 할 수도 있을 것이다.

~~~mermaid
graph LR;

A[QP=26] --> B[QP=28]
B-->C[QP=30]
C-->D[QP=32]
A-->E[QP=26]
E-->F[QP=26]
F-->G[QP=26]
~~~

##  Mutiple manifold based nonlinear optimization 

DNN의 ReLu function $g(x)$ 은 다음과 같다. 
$$
g(x) = \frac{1}{1 + \exp(-\lambda x)} \cdot x
$$

이것의 plot는 다음과 같다. 

이를 다시 생각해보면,
어떤 Hyper-Plane으로 구별 혹은, 어떤 Open Ball $B^o(\hat{x}, \rho)$ 로 규정되어지는 Convex and Compact 집합 $S_n$에 대하여 다음을 만족하고자 하는 것이다.
$$
g_n (x) = 
\begin{cases}
x & x \in S_n \\
0 & x \notin S_n
\end{cases}
$$

그러므로, 만일,  $x \in A$ 에서 $A$가 Convex and Compact가 아니고, $A$의 진 부분집합인 $S_n \subset A$ 는 
$$
\cup_{k=1}^n S_k \neq A
$$

이 상태에서 최대한, Convex and Compact 집합 $S_k$로 데이커를 변환시켜 Mapping 하는 것이다. 문제는 선형 변환이기 때문에 꼭 해당 집합이 Convex and Compact라고 볼 수 없다는 점이고 이를 복수층의 Layer를 통해 Metric Conversion으로 최대한 Convex and Compact 집합으로 보낸다.


그런데, 이것이 Fermi-Dirac Function의 형태가 되기 위해서는 