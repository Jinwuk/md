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
  
## Note $\otimes$ 
$$
\mathbf{A} = A^{ij} \mathbf{g}_i \otimes \mathbf{g}_j
$$


## Surface in Three-Dimensional Euclidean Space

A surface in three dimensional Euclidean space
$$
\mathbf{r} = \mathbf{r}(t^1, t^2), \quad \mathbf{r} \in \mathbb{E}^3
$$
이떄, coordinate system은 다음과 같다고 하자.
$$
\theta^i = \theta^i (t^1, t^2), \quad i = 1, 2, 3
$$
마찬가지로 $\mathbf{r}$은 $t^i$ 에 대하여  over all definition domain에 대하여 다음과 같이 differentiable 하다고 가정하자.
$$
\frac{d \mathbf{r}}{dt^{\alpha}} \neq 0, \quad \alpha=1, 2
$$

#### Example 1 : plane
3 Linearly independent vectors - 3 point ($\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2$) 를 가진 Plane에 대하여 다음과 같이 $\mathbf(r)(t^1, t^2)$ 이 정의된다.

$$
\mathbf(r)(t^1, t^2) = \mathbf{x}_0 + t^1(\mathbf{x}_1 - \mathbf{x}_0) + t^2(\mathbf{x}_2 - \mathbf{x}_0)
$$

#### Example 2 : Cylinder
반지름 $R$을 가지고 주축이 $e_3$ 을 따라 존재하는 Cylinder는 다음과 같다.

$$
\mathbf(r)(t^1, t^2) = R \cos t^1 \mathbf{e}_1 + R \sin t^1 \mathbf{e}_2 + t^2 \mathbf{e}_3
$$

여기서 $\mathbf{e}_i, \; i=1, 2,3$은 Orthonormal basis in $\mathbb{E}^3$.
특히 극좌표계를 사용하면 다음과 같이 표시 가능하다.
$$
\varphi = t^1, \quad z = t^2, \quad r = R
$$

#### Example 3 : Sphere
반지름 $R$ with the center at $\mathbf{r} = 0$ 를 가진 sphere는
$$
\mathbf{r}(t^1, t^2) = R \sin t^1 \sin t^2 \mathbf{e}_1 +
$$

구 좌표계에서는 

$$
\varphi = t^1, \quad \phi = t^2, \quad r=R
$$

### 일반이론 

Parameter $t^1, t^2$가 다음과 같은 parametric representation을 가진다고 하면
$$
t^1 = t^1(t), \quad t^2 = t^2(t)
\label{3.59}
$$
여기에 대하여 일반적인 평면은 다음 그림과 같다.

<img src="http://jnwhome.iptime.org/img/research/2020/tensor_004.png" alt="center 80%" style="zoom: 67%;" />

#### Tangent vector

$\eqref{3.59}$ 와 같이 parametric representation이 되어 있다면.
$$
\mathbf{g}_t = \frac{d \mathbf{r}}{dt} = \frac{\partial \mathbf{r}}{\partial t^1} \frac{dt^1}{dt} + \frac{\partial \mathbf{r}}{\partial t^2} \frac{dt^2}{dt} = \mathbf{g}_1 \frac{d t^1}{dt} + \mathbf{g}_2 \frac{d t^2}{dt}
$$
where 
$$
\mathbf{g}_{\alpha} = \frac{\partial \mathbf{r}}{\partial t^{\alpha}} = \mathbf{r}_{, \alpha}, \quad \alpha=1, 2
$$

- **Length of infinitesimal elements of the curve**
  $$
  (ds)^2 
  = d \mathbf{r} \cdot d \mathbf{r} 
  = \mathbf{g}_t dt \cdot \mathbf{g}_t dt
  = (\mathbf{g}_1 dt^1 + \mathbf{g}_2 dt^2) \cdot (\mathbf{g}_1 dt^1 + \mathbf{g}_2 dt^2)
  \label{3.63}
  $$
  with the aid of the abbreviation 
  $$
  g_{\alpha \beta} = g_{\beta \alpha} = \mathbf{g}_{\alpha} \cdot \mathbf{g}_{\beta}
  $$
  그러면 $\eqref{3.63}$  은 
  $$
  (ds)^2 = g_{11} (dt^1)^2 + 2 g_{12}dt^1 dt^2 + g_{22} (dt^2)^2
  \label{3.65}
  $$
  $\eqref{3.65}$를 **first fundamental form of the surface** 라 한다.  이를 다음과 같이 간단하게 표시할 수도 있다.
  $$
  (ds)^2 = g_{\alpha \beta} dt^{\alpha} dt^{\beta}
  \label{3.66}
  $$
  이때, $\alpha, \beta = 1, 2 $가 된다. 

  - n-Dimensional Euclidean space 에서 $g_{\alpha \beta}$ 는 평면에서의 **metric** 이다. 

  - 식 $\eqref{3.66}$과 같이 differential quadratic form 으로 이루어진 metric을 **Riemannian metric** 이라 한다.

    - 즉, Tangent vector의 Inner product가 Riemannian metric 이며, 이를 통해 metric Tensor가 이루어진다. 

- **principal normal vector**
  $$
  \mathbf{g}_3 = \frac{\mathbf{g_1} \times \mathbf{g_2}}{\| \mathbf{g_1} \times \mathbf{g_2} \|}
  \label{3.67}
  $$
앞에서 구한 tangent vector $\mathbf{g}_1,  \mathbf{g}_2$ 와 principal normal vector를 통해 $\mathbb{E}^3$ basis를 구성하게 된다.
  
  - **Normal Section** : normal space를 의미한다. 이를 통해 평면에서의 curvature를 생각해 본다

### Curvature on surface 

#### Gauss Formula 

곡선에서의 curvature를 생각할 수 있겠으나, 그렇게 되면 너무나 많은 평면상의 곡선을 모두 생각해야 할 것이다. 그러므로 Tangent vector와 수직인 Normal section 상에서 curvature를 생각한다.
- 곡선의 방정식에서 $\kappa(s) = \mathbf{a}_{1, s} \mathbf{a}_2$ 에서 $\mathbf{a}_2 \perp \mathbf{g}_1$ 이고 $\mathbf{a}_{1, s}$ 의 $\mathbf{a}_2$ 를 curvature로 정의된다는 점에서. (원래 curvature $\kappa(s) = \frac{1}{\| \mathbf{a}_{1, s} \|}$ 이지만 $\mathbf{a}_2$의 정의에서 유도됨)

- Christoffel 기호를 사용하여 basis vector $\mathbf{g}_i$ 의 평면 좌표에 대하여 생각해 보면
  $$
  \mathbf{g}_{i, \alpha} 
  = \frac{\partial \mathbf{g}_i}{\partial t^{\alpha}} 
  = \Gamma_{i \alpha k} \mathbf{g}^k 
  = \Gamma_{i \alpha}^k \mathbf{g}_k
  \label{3.68}
  $$
  여기서 Christoffel 기호의 정의에 따라
  $$
  \Gamma_{i \alpha k} = \mathbf{g}_{i, \alpha} \mathbf{g}_k, \quad
  \Gamma_{i \alpha}^k = \mathbf{g}_{i, \alpha} \mathbf{g}^k, \quad i=1,2,3, \; k = 1, 2
  $$
  이때, $\eqref{3.67}$ 에 의해 $\mathbf{g}_3 = \mathbf{g}^3$ 이므로 $\Gamma_{i \alpha 3} = \Gamma_{i \alpha}^3$  이다. 
  
  먼저 $\mathbf{g}_{\alpha} \perp \mathbf{g}_3$ 이고 $\|  \mathbf{g}_3 \| = 1$ 이므로 
  $$
  \mathbf{g}_{\alpha} \cdot \mathbf{g}_3 = 0 
  \Rightarrow \; \mathbf{g}_{\alpha, \beta} \cdot \mathbf{g}_3 + \mathbf{g}_{\alpha} \cdot \mathbf{g}_{3, \beta} = 0 
  \Rightarrow \; \mathbf{g}_{\alpha, \beta} \cdot \mathbf{g}_3 = -\mathbf{g}_{\alpha} \cdot \mathbf{g}_{3, \beta}
  \label{3.72-a}
  $$
  그리고
  $$
  \mathbf{g}_{3, \alpha} \cdot \mathbf{g}_3 = 0, \quad \alpha, \beta =1, 2
  \label{3.72-b}
  $$
  식 $\eqref{3.72-a}$ 에서 $\Gamma_{\alpha \beta}^3 = - \Gamma_{3 \beta \alpha}$, 식 $\eqref{3.72-b}$ 에서 $\Gamma_{3 \alpha}^3 = 0, \; \alpha. \beta = 0$  (뒤에 $\alpha$가 밑으로 간 이유는 $\mathbf{g}_\alpha \cdot \mathbf{g}_{3 \beta} = \mathbf{g}_{3 \beta} \cdot \mathbf{g}^{\alpha }$  에서  $\mathbf{g}_{3 \beta}$ 는 Matrix 이기 때문이다. ) 
  
  식 $\eqref{3.68}$ 에서 다음과 같은 축약형을 얻을 수 있다.
  $$
  b_{\alpha \beta} = b_{\alpha \beta} = \Gamma_{\alpha \beta}^3 = -\Gamma_{3 \alpha \beta} = \mathbf{g}_{\alpha \beta} \cdot \mathbf{g}_3, \quad \alpha, \beta = 1,2 
  $$
  
  
  이를 통해 다음과 같은 Gauss Formula를 얻을 수 있다.
  $$
  \mathbf{g}_{\alpha, \beta} = \Gamma_{\alpha \beta}^{\rho} \mathbf{g}_{\rho} + b_{\alpha \beta} \mathbf{g}_3, \quad \alpha, \beta, \rho = 1,2
  \label{3.75}
  $$
  Gauss Formula $\eqref{3.75}$ 는 단순하게 생각하면 $\rho = 1,2,3$ 으로 생각하면 다음과 같이 볼 수 있다. 결국 index $3$ 만 다르게 빼 놓는 것.
  $$
  \begin{aligned}
  \mathbf{g}_{\alpha, \beta} 
  &= \Gamma_{\alpha \beta}^k \mathbf{g}_k, \quad &k=1,2,3,\; \alpha, \beta=1,2 \\
  &= \Gamma_{\alpha \beta}^{\rho} \mathbf{g}_{\rho} + b_{\alpha, \beta} \mathbf{g}_3, \quad &\rho=1,2 \; \alpha, \beta=1,2 \\
  \end{aligned}
  $$
  



<2020-09-21>

​	