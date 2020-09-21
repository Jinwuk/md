Tensor Algebra and Analysis 2020-09-17
===
[toc]

본 내용은 Tensor Algebra와 Analysis에 관련된 내용을 요약 정리한 것이다.

## Curves and Surfaces in Three-Dimensional Euclidean Space 
### Curves and Surfaces in Three-Dimensional Euclidean Space 

먼저 다음과 같은 vector field를 정의하자.
$$
\mathbf{r} = \mathbf{r}(t), \quad \mathbf{r} \in \mathbb{E}^3
\label{3.1}
$$
그리고 $\mathbf{r}$ 는 differentiable 로서 $\frac{d \mathbf{r}}{dt} \neq 0$ over the whole definition domain.

An arbitrary coordinate system을 다음과 같이 놓자.
$$
\theta^i = \theta^i (\mathbf{r}), \quad i=1, 2, 3,
\label{3.3}
$$
그러면 $\eqref{3.3}$ 은 다음과 같이 표시할 수 있다.
$$
\theta^i = \theta^i (t), \quad i=1, 2, 3,
$$

#### Example : Straight Line 
$$
\mathbf{r}(t) = \mathbf{a} + \mathbf{b} t, \quad \mathbf{a}, \mathbf{b} \in \mathbb{E}^3
\label{3.5}
$$
Basis $\mathcal{ \mathbf{h}_1, \mathbf{h}_2, \mathbf{h}_3 }$ 에 대한 linear coordinate 에 대해 $\eqref{3.5}$ 는 다음과 같다.
$$
r^i(t) = a^i + b^i (t), \quad i=1, 2, 3
$$

where $\mathbf{r} = r^i \mathbf{h}_i$ , $\mathbf{a} = a^i \mathbf{h}_i, \; \mathbf{b} = b^i \mathbf{h}_i$ .

#### Example : Circular Helix 

<img src="http://jnwhome.iptime.org/img/research/2020/tensor_002.png" style="zoom:50%;" />
$$
\mathbf{r}(t) = R \cos (t) \mathbf{e}_1 + R \sin (t) \mathbf{e}_2 + ct \mathbf{e}_3
\label{3.7}
$$
$\mathbf{e}_i$ 는 Orthonormal basis in $\mathbf{E}^3$ 이다.  여기에 대하여 $r=R, \varphi=t, z = ct$ 라 잡는다.

이때, $\eqref{3.7}$ 로 정의된 curve $\eqref{3.1}$ 에 대한 tangent vector를 구하면 다음과 같다. 
$$
\mathbf{g}_t = \frac{d \mathbf{r}}{dt}
\label{3.9}
$$

- **Length**

이를 사용하여 먼저 curve의 길이를 알아내자. curve의 길이를 $s(t)$ 라 하면 다음과 같다.
$$
s(t) = \int_{\mathbf{r}(t_1)}^{\mathbf{r}(t_2)} \sqrt{d\mathbf{r} \cdot d\mathbf{r}}
\label{3.10}
$$
$\eqref{3.9}$에서 $d\mathbf{r} = \mathbf{g}_t dt$  이므로 $\eqref{3.10}$ 은 
$$
s(t) 
= \int_{\mathbf{r}(t_1)}^{\mathbf{r}(t_2)} \sqrt{\mathbf{g}_t \cdot \mathbf{g}_t } dt 
= \int_{\mathbf{r}(t_1)}^{\mathbf{r}(t_2)} \| \mathbf{g}_t \| dt 
= \int_{\mathbf{r}(t_1)}^{\mathbf{r}(t_2)} \sqrt{g_{tt}(t)}dt
\label{3.11}
$$
$\eqref{3.11}$ 의 의미는 $\frac{ds(t)}{dt} = \sqrt{g_{tt} (t)} \neq 0$  이 방정식을 그대로 이용하면 다음과 같이 $s$ 에 대한  시간의 방정식을 얻을 수 있다.
$$
t(s) = \int_{s(t_1)}^{s} \| g_t \|^{-1} ds = \int_{s(t_1)}^{s} \frac{ds}{\sqrt{g_{tt}(t)}}
$$

- **Curvature**

  curve $\eqref{3.1}$ 을 arc length $s$를 사용하여 다음과 같이 재정의하자.
  $$
  \mathbf{r} = \mathbf{t(s)} = \hat{\mathbf{r}}(s)
  \label{3.14}
  $$
  - 즉, 길이에 따라 시간을 정의할 수 있다.  
  - 당연히 $\hat{\mathbf{r}}(s)$ 의 $s$ 에 대한 **tangent vector**를 구하면 

  $$
  \mathbf{a}_1 = \frac{d \hat{\mathbf{r}}}{ds} = \frac{d \hat{\mathbf{r}}}{dt}\frac{dt}{ds} = \frac{\mathbf{g}_t}{\| \mathbf{g}_t \|} \; \Rightarrow \| \mathbf{a}_1 \| = 1
  $$

  - unit normal vector로 길이에 대한 tangent vector가 도출되어 효용이 더 낫다. -> Covarinat diffentiation으로 이어진다. 

  - $\| \mathbf{a}_1 \| = 1$ 에서 이를 $s$ 에 대하여 다시 미분을 하면
  $$
  1 = \mathbf{a}_1 \cdot \mathbf{a}_1 \Rightarrow 0 = 2 \mathbf{a}_1 \cdot \mathbf{a}_{1, s}, \quad \therefore  \mathbf{a}_{1, s} \cdot \mathbf{a}_1 = 0
  $$

  - 여기에서 the length of the vector $\mathbf{a}_1, s$ 를 다음과 같이 정의한다. ($s$는 scalar 이므로 Dimension이 그대로 보존된다.)
  $$
  \kappa (s) = \| \mathbf{a}_{1, s} \| 
  \label{3.18}
  $$
  - 식 $\eqref{3.18}$ 를 **curvature** 라고 한다. 
  - 결국 curvature는 curve의 2계 미분이며 2계 미분의 norm 이다.  (가속도의 크기)
    - $s$가 scalar 이기 때문에 2계 미분을 하더라도 Parametric curve vector가 그대로 유지된다. 
    - 최적화론에서도 이 부분이 중요하다. (2계 미분이므로 )
  - Curvature의 Inverse value는 
  $$
  \rho (s) = \frac{1}{\kappa(s)}
  $$
  - 이것은 refered to as the radius of curvatire of the curve at the point $\hat{\mathbf{r}}(s)$ 

  	- 따라서 curvature가 0이 되면 radius of curvature가 정의되지 않으므로 (평평하므로) 무의미하다. 
   	- 또한 그러므로, non-zero curvature에 대해서만 생각할 것이다.

- **Normal, Bimormal** 
	
	- **Principal normal vector**
		
		- **The unit vector in the direction of $\mathbf{a}_{1,s}$**
    $$
    \mathbf{a}_2 = \frac{\mathbf{a}_{1,s}}{\| \mathbf{a}_{1,s} \|} = \frac{\mathbf{a}_{1,s}}{\kappa(s)}
	  \label{3.20}
		$$
		
		- 즉, curvature $\kappa(s)$는 $\mathbf{a}_2$ 방향 (**Principal normal vector 방향**) 으로의  $\mathbf{a}_{1,s}$ 의  값 (Inner product)
		
	- **unit binormal vector** 
	  
	  - $\mathbf{a}_1, \mathbf{a}_2$ 는 상호 orthonormal (미분의 정의에 따라 당연). 따라서 또 하나의 Orthonormal basis $\mathbf{a}_3 \in \mathbb{E}^3$을 다음과 같이 정의
	$$
	\mathbf{a}_3 = \mathbf{a}_1 \times \mathbf{a}_2
	$$
	- 기본적인 접근 방법은 다음 방정식에서 출발한다.
	$$
	\mathbf{a}_{i,s} = \Gamma_{is}^k \mathbf{a}_k, \quad i=1,2,3
	$$
	where $\Gamma_{is}^k = \mathbf{a}_{i,s}\mathbf{a}_k$.
	- 여기서 다음을 알 수 있다. 
	$$
	\begin{aligned}
	\Gamma_{1s}^2 &= \kappa, \quad \because \kappa = \mathbf{a}_{1,s} \mathbf{a}_2 \; \text{by } \eqref{3.20} \\
	\Gamma_{1s}^1 &= 0, \quad \because \Gamma_{1s}^1 = \mathbf{a}_{1,s} \mathbf{a}_1 = 0\; \text{by } \eqref{3.20} \\
	\Gamma_{1s}^3 &= 0, \quad \because \Gamma_{1s}^3 = \mathbf{a}_{1,s} \mathbf{a}_3 = 0\; \text{by } \eqref{3.20}
	\end{aligned}
	$$
	
	- 즉 Principal normal vector에 대하여 orthonormal인 $\mathbf{a}_1,\; \mathbf{a}_3$ 	

- Torsion

  $\mathbf{a}_1$ 은 Tangent Vector, $\mathbf{a}_2$ 는 Principal normal vector, 그럼 여기에 orthonormal인 $\mathbf{a}_3$ 은? Torsion.  

  - 정의에 의해 $\mathbf{a}_3 = \mathbf{a}_1 \times \mathbf{a}_2$ 에서 orthonormal 이므로 $\mathbf{a}_3 \cdot \mathbf{a}_3 = 1$ 에서 $s$로 미분하므로
  $$
  0 = \mathbf{a}_{3,s} \cdot \mathbf{a}_3
  $$
  - 그리고 $\mathbf{a}_1 \cdot \mathbf{a}_3 = 0$ 에서 
  $$
  \mathbf{a}_{1, s} \cdot \mathbf{a}_3 + \mathbf{a}_1 \cdot \mathbf{a}_{3, s} = 0
  $$
  - 그러므로
  
  $$
  \mathbf{a}_1 \cdot \mathbf{a}_{3, s} = - \mathbf{a}_{1, s} \cdot \mathbf{a}_3 = -\kappa(s) \mathbf{a}_{2} \cdot \mathbf{a}_3  = 0
  $$
  
  - 따라서 $\mathbf{a}_{3, s} \perp \mathbf{a}_1, \mathbf{a}_3 \Rightarrow \mathbf{a}_{3, s} // \mathbf{a}_{2}$ 이므로 다음과 같이  놓을 수 있다, 
  $$
  \mathbf{a}_{3, s} = \tau(s) \mathbf{a}_{2}, \quad \tau(s) = - \mathbf{a}_{3, s} \cdot \mathbf{a}_{2}, \quad \Gamma_{3s}^2 = -\tau(s)
  $$
  
- **Frenet Formula**

  - 이들을 모두 정리하면 다음과 같다.
  $$
\Gamma_{1s}^2 = \kappa(s), \;  \Gamma_{2s}^1 = -\kappa(s), \; \Gamma_{2s}^3 = \tau(s), \; \Gamma_{3s}^2 = -\tau(s)
  $$
  - $\Gamma_{ij}^k$ 에서 입력은 $k$,  출력은 $i$ 그리고 $j$로 미분하는 것이므로 (미분에서 분모)
  $$
  [\Gamma_{is}^j] = \left[  
  \begin{matrix}
  0         & \kappa(s) & 0 \\
  -\kappa(s) & 0          & \tau(s) \\
  0         & -\tau(s)    & 0
  \end{matrix}
  \right]
  $$
  - 이에 의해
  $$
  \begin{aligned}
  \mathbf{a}_{1,s} &=   &  \kappa \mathbf{a}_2 &   \\
  \mathbf{a}_{2,s} &= -\kappa \mathbf{a}_1 & + & \tau \mathbf{a}_3 \\
  \mathbf{a}_{3,s} &=   &  -\tau \mathbf{a}_3   &  
  \end{aligned}
  $$
  
  



<2020-09-18>

​	