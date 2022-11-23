Tensor Algebra and Analysis -2
===
[toc]

본 내용은 Tensor Algebra와 Analysis에 관련된 내용 중 2절 37P 이후의 것들 중 중요한 내용을 요약한 것이다.

## Coordinates inEuclidean Space, Tangent Vectors
Definition 2.1 에 의해 

$$
\begin{equation}
x^i = x^i (\mathbf{r})  \Leftrightarrow \mathbf{r} = \mathbf{r}(x^1, x^2, \cdots , x^n)
\end{equation}
$$
where $\mathbf{r} \in \mathbb{E}^n$ and $x^i \in \mathbb{R} (i=1, 2, \cdots , n)$ 
따라서 $x^i = x^i (r) $ 과 $r = r(x^1, x^2, \cdots , x^n)$ 는 sufficiently differentiable 이라 가정한다.

- 예 : Cylindrical coordinates in $\mathbb{E}^3$ 

$$
\begin{equation}
\mathbf{r} = \mathbf{r} (\psi, z, r) = r \cos \varphi e_1 + r \sin \varphi e_2 + z e_3
\label{eq02_01}
\end{equation}
$$

<img src="http://jnwhome.iptime.org/img/research/2020/tensor_001.png" style="zoom:50%;" />

- 그림에서 $\mathbf{r} = x^i h_i$ 의 형태로 또 다른 basis $\mathcal{H} = \{ h_1, h_2, \cdots , h_n \}$ 로 표시할 수 있다.

- 그러므로 관심이 있는 것은,  $x^i =x^i(\mathbf{r})$ and $y^i = y^i (\mathbf{r})$ 의 두개의 임의의 $\mathbb{E}^n$ 기반 좌표계가 주어졌을 때
  $$
  \begin{equation}
  x^i = \hat{x} (y^1, y^2, \cdots, y^n) \Leftrightarrow y^i = \hat{y}^i(x^1, x^2, \cdots x^n)
  \end{equation}
  $$
  과 같이 invertible 이면 다음이 성립한다.
  $$
  \begin{equation}
  \frac{\partial y^i}{\partial y^j} = \delta_{ij} = \frac{\partial y^i}{\partial x^k}\frac{\partial x^k}{\partial y^j}
  \end{equation}
  $$
  
- 그러므로 $\| \delta_{ij} \|$ 를 생각해 보면 Jacobian $J = [\frac{\partial y^i}{\partial x^k} ]$ 은 0이 아니어야 하며 그, inverse $K$가 존재해야 한다.

- 여기에서 다음의 임의의 curvilinear coordinate system 을 생각해보면
  $$
  \begin{equation}
  \theta^i = \theta^i (\mathbf{r}) \rightarrow \mathbf{r} = \mathbf{r} (\theta^1, \theta^2, \cdots \theta^n)
  \end{equation}
  $$
  where $\mathbf{r} = \mathbb{E}^n$ and $\theta^i \in \mathbb{R} (i=1, 2, \cdots n)$. 
  
  그리고 (k만 빠졌다)
  $$
  \begin{equation}
  \theta^i = const, \quad i=1, 2, \cdots, k-1, k+1, \cdots n
  \end{equation}
  $$
  이면 $\theta^k$ coordinate line 이라고 하는 curve in $\mathbb{E}^n$ 을 다음과 같이 정의할 수 있다.  
  
  The vectors
  $$
  \begin{equation}
  \mathbf{g}_k = \frac{\partial \mathbf{r}}{\partial \theta^k}, \quad k=1, 2, \cdots n
  \end{equation}
  $$
  are called the **Tangent vectors to the cooresponding $\theta^k$ coordinate line** .
  
- Example : Tangent vectors and metric coefficients of cylindricla coordinates in $\mathbb{E}^n$ . $\mathbf{r}=(\varphi, z, r) = r \cos \varphi e_1 + r \sin \varphi e_2 + z e_3$, 즉 $\eqref{eq02_01}$ 
    $$
    \begin{equation}
    \begin{aligned}
    g_1 &= \frac{\partial \mathbf{r}}{\partial \varphi} = -r \sin \varphi e_1 + r \cos \varphi e_2 \\
    g_2 &= \frac{\partial \mathbf{r}}{\partial z} = e_3\\
    g_3 &= \frac{\partial \mathbf{r}}{\partial r} =  \cos \varphi e_1 + \sin \varphi e_2
    \end{aligned}
    \end{equation}
    $$

그러므로 metric coefficient는 
$$
    \begin{equation}
    [g_{ij}]= [g_i \cdot g_j ] = \left[
    \begin{matrix}
    r^2 & 0 & 0 \\
    0   & 1 & 0 \\
    0   & 0 & 1
    \end{matrix} \right]
    ,\quad  [g^{ij}] = [g_{ij}]^{-1} = \left[
    \begin{matrix}
    r^{-2} & 0 & 0 \\
    0   & 1 & 0 \\
    0   & 0 & 1
    \end{matrix} \right]
    \end{equation}
$$
따라서, $g^{1} = g^{-1} g_1$ 에서
$$
    \begin{equation}
    \begin{aligned}
    g^1 &= \frac{1}{r^2}(-r \sin \varphi e_1 + r \cos \varphi e_2) = \frac{1}{r} (-\sin \varphi e_1 + \cos \varphi e_2)\\
    g^2 &= g_2 = e_3\\
    g^3 &= g_3 = \cos \varphi e_1 + \sin \varphi e_2
    \end{aligned}
    \end{equation}
$$

## Coordinatye Transformation, Co-, Contra- and Mixed variant Components

Let $\theta^i = \theta^i(\mathbf{r})$ and $\bar{\theta}^i = \bar{\theta}^i(\mathbf{r}), \; (i=1, 2, \cdots, n)$ 을 임의의 두 coordinaye system in $\mathbb{E}^n$. It holds
$$
\begin{equation}
\bar{\mathbf{g}}_i 
= \frac{\partial \mathbf{r}}{\partial \bar{\theta}^i} 
= \frac{\partial \mathbf{r}}{\partial \theta^j} \frac{\partial \theta^j}{\partial \bar{\theta}^i} 
= \mathbf{g}_j \frac{\partial \theta^j}{\partial \bar{\theta}^i},
\quad i = 1, 2, \cdots , n.
\label{eq-2.31}
\end{equation}
$$
만일, $\mathbf{g}^i$ 가 $\mathbf{g}_i$ ($i = 1, 2, \cdots , n$)의 dual basis 이면, then we can write
$$
\begin{equation}
\bar{\mathbf{g}}^i 
= \mathbf{g}^j \frac{\partial \bar{\theta}^i}{\partial \theta^j},
\quad i = 1, 2, \cdots , n.
\label{eq-2.32}
\end{equation}
$$

- 여기에서 base가 되는 Coordinate는 $\bar{\theta}_k$ 이다. Upper index이면 분자에, Lower index 이면 분모에 있다.

Indeed, 
$$
\begin{equation}
\begin{aligned}
\bar{\mathbf{g}}^i \cdot \bar{\mathbf{g}}_j 
&= \left( \mathbf{g}^k \frac{\partial \bar{\theta}^i}{\partial \theta^k} \right) 
\cdot \left( \mathbf{g}_l \frac{\partial \theta^l}{\partial \bar{\theta}^j}\right)
= \mathbf{g}^k \cdot \mathbf{g}_l\left( \frac{\partial \bar{\theta}^i}{\partial \theta^k} \frac{\partial \theta^l}{\partial \bar{\theta}^j}\right)
= \delta_l^k\left( \frac{\partial \bar{\theta}^i}{\partial \theta^k} \frac{\partial \bar{\theta}^l}{\partial \theta^j}\right) \\

&= \frac{\partial \bar{\theta}^i}{\partial \theta^k} \frac{\partial \theta^k}{\partial \bar{\theta}^j}
= \frac{\partial \bar{\theta}^i}{\partial \bar{\theta}^j}
= \delta_j^i, \quad i, j=1,2, \cdots , n
\end{aligned}
\end{equation}
$$

- Covariant Basis : Lower index  (vector)   $\bar{\mathbf{g}}_i= \mathbf{g}_j \frac{\partial \theta^j}{\partial \bar{\theta}^i}$
- Contravariant Basis : Upper index (covector or differential )  $\bar{\mathbf{g}}^i 
  = \mathbf{g}^j \frac{\partial \bar{\theta}^i}{\partial \theta^j}$
- Mixed variant Basis : Lower and Upper index denotes mixed.

$$
\begin{equation}
\begin{aligned}
\mathbf{x} &= x_i \mathbf{g}^i = x^i \mathbf{g}_i = \bar{x}_i \bar{\mathbf{g}}_i \\

\mathbf{A} &= A_{ij} \mathbf{g}^i \otimes \mathbf{g}^j 
= A^{ij} \mathbf{g}_i \otimes \mathbf{g}_j  
= A_{.j}^i \mathbf{g}_i \otimes \mathbf{g}^j \\
&= \bar{A}_{ij} \bar{\mathbf{g}}^i \otimes \bar{\mathbf{g}}^j  
= \bar{A}^{ij} \bar{\mathbf{g}}_i \otimes \bar{\mathbf{g}}_j  
= \bar{A}_{.j}^i \bar{\mathbf{g}}_i \otimes \bar{\mathbf{g}}^j  
\end{aligned}
\end{equation}
$$

- Equation (1.28)
  $$
  \begin{equation}
  x^i = \mathbf{x} \cdot \mathbf{g}^i, \quad x_i = \mathbf{x} \cdot \mathbf{g}_i, \quad i=1, 2, \cdots, n
  \label{eq-1.28}
  \end{equation}
  $$

- Equation (1.88)
  $$
  \begin{equation}
  A^{ij}   = \mathbf{g}^i \mathbf{A} \mathbf{g}^j, \; A_{ij}   = \mathbf{g}_i \mathbf{A} \mathbf{g}_j, \;
  A_{.j}^i = \mathbf{g}^i \mathbf{A} \mathbf{g}_j, \; A_{.i}^j = \mathbf{g}_i \mathbf{A} \mathbf{g}^j, \;
  \quad i, j = 1, 2, \cdots n.
  \label{eq-1.88}
  \end{equation}
  $$

- $\eqref{eq-1.28}, \eqref{eq-1.88}, \eqref{eq-2.31}, \eqref{eq-2.32}$ 에서 다음을 얻는다. 
  $$
  \begin{equation}
  \begin{aligned}
  \bar{x}_i 
  &= \mathbf{x} \cdot \bar{\mathbf{g}}_i 
  = \mathbf{x} \cdot \left( \mathbf{g}_j \frac{\partial \theta^j}{\partial \bar{\theta}^i}\right) 
  = x_j \frac{\partial \theta^j}{\partial \bar{\theta}^i} \\
  
  \bar{x}^i 
  &= \mathbf{x} \cdot \bar{\mathbf{g}}^i 
  = \mathbf{x} \cdot \left( \mathbf{g}^j \frac{\partial \bar{\theta}^i}{\partial \theta^j} \right) 
  = x^j \frac{\partial \bar{\theta}^i}{\partial \theta^j} \\
  
  \bar{A}_{ij}
  &= \bar{\mathbf{g}}_i \mathbf{A} \bar{\mathbf{g}}_j 
  = \left( \mathbf{g}_k \frac{\partial \theta^k}{\partial \bar{\theta}^i}\right) 
  \mathbf{A} \left( \mathbf{g}_l \frac{\partial \theta^l}{\partial \bar{\theta}^j}\right) 
  = \left( \frac{\partial \theta^k}{\partial \bar{\theta}^i}\right) 
  \mathbf{g}_k \mathbf{A} \mathbf{g}_l \left( \frac{\partial \theta^l}{\partial \bar{\theta}^j}\right) 
  = \frac{\partial \theta^k}{\partial \bar{\theta}^i} \frac{\partial \theta^l}{\partial \bar{\theta}^j} A_{kl} \\
  
  \bar{A}^{ij}
  &= \bar{\mathbf{g}}^i \mathbf{A} \bar{\mathbf{g}}^j 
  = \left( \mathbf{g}^k \frac{\partial \theta^i}{\partial \bar{\theta}^k}\right) 
  \mathbf{A} \left( \mathbf{g}^l \frac{\partial \theta^j}{\partial \bar{\theta}^l}\right) 
  = \left( \frac{\partial \theta^i}{\partial \bar{\theta}^k}\right) 
  \mathbf{g}^k \mathbf{A} \mathbf{g}^l \left(  \frac{\partial \theta^j}{\partial \bar{\theta}^l}\right) 
  = \frac{\partial \theta^i}{\partial \bar{\theta}^k} \frac{\partial \theta^j}{\partial \bar{\theta}^l} A^{kl} \\
  
  \bar{A}_{.j}^{i}
  &= \bar{\mathbf{g}}^i \mathbf{A} \bar{\mathbf{g}}_j 
  = \left( \mathbf{g}^k \frac{\partial \theta^i}{\partial \bar{\theta}^k}\right) 
  \mathbf{A} \left( \mathbf{g}_l \frac{\partial \theta^l}{\partial \bar{\theta}^j}\right) 
  = \left( \frac{\partial \theta^i}{\partial \bar{\theta}^k}\right) 
  \mathbf{g}^k \mathbf{A} \mathbf{g}_l \left( \frac{\partial \theta^l}{\partial \bar{\theta}^j}\right) 
  = \frac{\partial \theta^i}{\partial \bar{\theta}^k} \frac{\partial \theta^l}{\partial \bar{\theta}^j} A_{.l}^{k}
  \end{aligned}
  \end{equation}
  $$
  
- For a higher-order tensors 
  $$
  \begin{equation}
  \bar{A}_{ijk} 
  = \frac{\partial \theta^r}{\partial \bar{\theta}^i} \frac{\partial \theta^s}{\partial \bar{\theta}^j} \frac{\partial \theta^t}{\partial \bar{\theta}^k} A_{rst}
  , \quad 
  \bar{A}^{ijk} 
  = \frac{\partial \bar{\theta}^i}{\partial \theta^r} \frac{\partial \bar{\theta}^j}{\partial \theta^s}\frac{\partial \bar{\theta}^k}{\partial \theta^t} A^{rst}
  \end{equation}
  $$

- **The differentials of the coordinates** really transform according to the **contravariant law**, for a coordinate system $\bar{\theta}^i = \bar{\theta^i}(\theta^1. \theta^2, \cdots, \theta^n)$, such that

$$
\begin{equation}
d \bar{\theta}^i = \frac{\partial \bar{\theta}^i}{\partial \theta^k} d\theta^k, \quad i=1, 2, \cdots, n
\end{equation}
$$



## Gradient, Covariant and Contravariant Derivatives 

Let $\Phi = \Phi(\theta^1, \theta^2, \cdots, \theta^n), \mathbf{x} = \mathbf{x}(\theta^1, \theta^2, \cdots, \theta^n)$, and $\mathbf{A} = \mathbf{A}(\theta^1, \theta^2, \cdots, \theta^n)$ be a scalar, a vector, and a tensor valued differentiable functionof the coordinates $\theta^i \in \mathbb{R} (i=1, 2, \cdots, n)$.

- 이러한 함수들의 coordinates들은 일반적으로 **field** 로 불린다 (refered). 그래서 scalar field, vector field, tensor field..

- 다음과 같이 represented 될 수 있다.
  $$
  \begin{equation}
  \Phi = \Phi(\mathbf{r}), \quad \mathbf{x} = \mathbf{x}(\mathbf{r}), \quad \mathbf{A} =\mathbf{A}(\mathbf{r}).
  \end{equation}
  $$
  
- 이들 Field의 Directional Derivative는 다음과 같다.  For all $\mathbf{a} \in \mathbb{E}^n$ , 
  $$
  \begin{equation}
  \begin{aligned}
  \frac{d}{ds} \Phi(\mathbf{r} + s \mathbf{a}) \bigg\vert_{s=0} 
  &= \lim_{s \rightarrow 0}\frac{\Phi(\mathbf{r} + s \mathbf{a}) - \Phi(\mathbf{r})}{s} \\
  
  \frac{d}{ds} \mathbf{x}(\mathbf{r} + s \mathbf{a}) \bigg\vert_{s=0} 
  &= \lim_{s \rightarrow 0}\frac{\mathbf{x}(\mathbf{r} + s \mathbf{a}) - \mathbf{x}(\mathbf{r})}{s} \\
  
  \frac{d}{ds} \mathbf{A}(\mathbf{r} + s \mathbf{a}) \bigg\vert_{s=0} 
  &= \lim_{s \rightarrow 0}\frac{\mathbf{A}(\mathbf{r} + s \mathbf{a}) - \mathbf{A}(\mathbf{r})}{s} 
  \end{aligned}
  \end{equation}
  $$

- 또한 미분이므로 Directional Derivatives는 Linear 함수이다. 즉,

$$
\begin{equation}
\frac{d}{ds} \Phi(\mathbf{r} + s(\mathbf{a} + \mathbf{b})) \bigg\vert_{s=0} 
= \frac{d}{ds} \Phi(\mathbf{r} + s_1\mathbf{a} + s_2\mathbf{b}) \bigg\vert_{s=0}
= \frac{d}{ds} \Phi(\mathbf{r} + s\mathbf{a}) \bigg\vert_{s=0}  \frac{d}{ds} \Phi(\mathbf{r} + s\mathbf{b}) \bigg\vert_{s=0} 
\end{equation}
$$

, where $s_1=s_2=s$.  그리고,
$$
\begin{equation}
\frac{d}{ds} \Phi(\mathbf{r} + s \alpha \mathbf{a}) \bigg\vert_{s=0}
= \frac{d}{d(\alpha s)} \Phi(\mathbf{r} + s \alpha \mathbf{a}) \frac{d(\alpha s)}{s} \bigg\vert_{s=0}
= \alpha \frac{d}{ds} \Phi(\mathbf{r} + s \alpha \mathbf{a}) \bigg\vert_{s=0}, \forall \mathbf{a} \in \mathbb{E}^n, \forall \alpha \in \mathbb{R}
\end{equation}
$$

- **Definition of Gradient** of $\Phi$, $\nabla \Phi \in \mathbb{E}^n$
  $$
  \begin{equation}
  \frac{d}{ds} \Phi(\mathbf{r} + s \mathbf{a}) \bigg\vert_{s=0} = \nabla \Phi \cdot \mathbf{a}, \quad \forall \mathbf{a} \in \mathbb{E}^n, 
  \end{equation}
  $$
  
- Definition of Gradient for a vector field, and a tensor field.
    $$
    \begin{equation}
    \begin{aligned}
    \frac{d}{ds} \mathbf{x}(\mathbf{r} + s \mathbf{a}) \bigg\vert_{s=0} 
    &= (grad \,\mathbf{x}) \mathbf{a}, \quad \forall \mathbf{a} \in \mathbb{E}^n \\

    \frac{d}{ds} \mathbf{A}(\mathbf{r} + s \mathbf{a}) \bigg\vert_{s=0} 
    &= (grad \,\mathbf{A}) \mathbf{a}, \quad \forall \mathbf{a} \in \mathbb{E}^n \\ 
    \end{aligned}
    \end{equation}
    $$
	- $(grad \,\mathbf{x})$ : Second order Tensor, $(grad \,\mathbf{A})$  : Third order Tensor
	

이를 Tensor Notation으로 표현해 보자.

A Fixed basis $\mathcal{H} = \{ \mathbf{h}_1, \mathbf{h}_2, \cdots, \mathbf{h}_n\}$ 에 대하여 $\mathbf{r} = x^i \mathbf{h}_i, \; \mathbf{a} = a^i \mathbf{h}_i$ 일 때,  $\Phi$ 의 Directional Derivative를 다시 써보면
$$
\begin{equation}
\begin{aligned}
\frac{d}{ds} \Phi(\mathbf{r} + s \mathbf{a}) \bigg\vert_{s=0} 
&= \frac{d}{ds} \Phi(x^i \mathbf{h}_i + s a^i\mathbf{h}_i) \bigg\vert_{s=0} 
= \frac{d}{ds} \Phi(x^i + s a^i) \mathbf{h}_i\bigg\vert_{s=0} \\
&= \frac{\partial \Phi}{\partial (x^i + s a^i)} \frac{d (x^i + s a^i)}{d s} \bigg\vert_{s=0} 
\quad \because \text{the result of derivation is scalar}\\
&= \frac{\partial \Phi}{\partial x^i} a^i \\
&= \frac{\partial \Phi}{\partial x^i} a^i (\mathbf{h}^i \cdot \mathbf{h}_i) \\
&= \frac{\partial \Phi}{\partial x^i} \mathbf{h}^i \cdot a^i \mathbf{h}_i \\
&= \frac{\partial \Phi}{\partial x^i} \mathbf{h}^i \cdot \mathbf{a}
\end{aligned}
\end{equation}
$$
그러므로 $\Phi$의 Gradient 표현은
$$
\begin{equation}
grad \; \Phi = \nabla_x \Phi = \frac{\partial \Phi}{\partial x^i} \mathbf{h}^i
\end{equation}
$$
만약 Gradient of $\Phi$를 임의의 curvilinear coordinates $\mathbf{r} = \mathbf{r}(\theta^1, \theta^2, \cdots \theta^n)$ 와 이에 대한 Tangent vectors $\mathbf{g}_k = \frac{\partial \mathbf{r}}{\partial \theta^k}$ 로 다시쓰게 되면 
$$
\begin{equation}
grad \; \Phi 
= \frac{\partial \Phi}{\partial x^i} \mathbf{h}^i 
= \frac{\partial \Phi}{\partial \theta^k} \frac{\partial \theta^k}{\partial x^i} \mathbf{h}^i 
= \frac{\partial \Phi}{\partial \theta^k} \mathbf{g}^k, \quad \ \mathbf{g}^k = \frac{\partial \theta^k}{\partial x^i} \mathbf{h}^i
\end{equation}
$$
즉, **Gradient** 는**coordinate system의 선택에 대해 Independent** 하다.

그래서 만약 임의의 coordinate system $\bar{\theta}^i = \bar{\theta}^i (\theta^1, \theta^2, \cdots \theta^n), \; (i=1, 2, \cdots, n)$ 에 대하여, Gradient를 다시쓰면
$$
\begin{equation}
grad \, \Phi 
= \frac{\partial \Phi}{\partial \theta^i} \mathbf{g}^i 
= \frac{\partial \Phi}{\partial \bar{\theta}^j} \frac{\partial \bar{\theta}^j}{\partial \theta^i} \mathbf{g}^i 
= \frac{\partial \Phi}{\partial \bar{\theta}^j} \bar{\mathbf{g}}^j 
\end{equation}
$$


-  Vector Field와 Tensor Field에 대한 Gradient도 마찬가지로 정의할 수 있다.
  $$
  \begin{equation}
  grad \; \mathbf{x}= \frac{\partial \mathbf{x}}{\partial \mathbf{\theta}^i} \otimes \mathbf{g}^i, \quad
  grad \; \mathbf{A}= \frac{\partial \mathbf{A}}{\partial \mathbf{\theta}^i} \otimes \mathbf{g}^i
  \end{equation}
  $$
  Henceforth, the derivarives of the functions $\Phi = \Phi(\theta^1, \theta^2, \cdots \theta^n), \; \mathbf{x} = \mathbf{x}(\theta^1, \theta^2, \cdots \theta^n), \; \mathbf{A} = \mathbf{A}(\theta^1, \theta^2, \cdots \theta^n)$  with respect to curilinear coordinates $\theta^i$ will be denoted shortly, by 
  $$
  \begin{equation}
  \Phi_{, i}       = \frac{\partial \Phi}{\partial \theta^i}, \; 
  \mathbf{x}_{, i} = \frac{\partial \mathbf{x}}{\partial \theta^i}, \;
  \mathbf{A}_{, i} = \frac{\partial \mathbf{A}}{\partial \theta^i}
  \label{eq-2.61}
  \end{equation}
  $$
  

  -  여기서 comma가 $i$ 앞에 붙은 것은 $\theta^i$로 편미분 하였다는 의미이다. 


## Christoffel Symbols, Representation of the Covariant Derivatives 





## Note

- Partial Differential에서 Scalar Component의 index는 모두 Upper index이다.   $\eqref{eq-2.31}, \eqref{eq-2.32}$를 살펴볼 것.
- 대체로 **scalar  값은 모두 upper index** 라고 생각하자. 
  - 왜냐하면 기본 Basis Vector $\mathbf{e}_k \in \mathbb{E}^n$ 은 **vector이므로 Lower index** 이므로.

