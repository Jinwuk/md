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
  \label{eq-2.27}
  \end{equation}
  $$
  are called the **Tangent vectors to the cooresponding $\theta^k$ coordinate line** .
  
### Example : Tangent vectors and metric coefficients of cylindricla coordinates in $\mathbb{E}^n$. 
$\mathbf{r}=(\varphi, z, r) = r \cos \varphi e_1 + r \sin \varphi e_2 + z e_3$, 즉 $\eqref{eq02_01}$ 
$$
\begin{equation}
\begin{aligned}
    g_1 &= \frac{\partial \mathbf{r}}{\partial \varphi} = -r \sin \varphi e_1 + r \cos \varphi e_2 \\
    g_2 &= \frac{\partial \mathbf{r}}{\partial z} = e_3\\
    g_3 &= \frac{\partial \mathbf{r}}{\partial r} =  \cos \varphi e_1 + \sin \varphi e_2
\end{aligned}
\label{eq08}
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
  - 즉,  $\frac{\partial \mathbf{x}}{\partial \mathbf{\theta}^i}$ 자체는 Vector 이다.  마찬가지로 $\frac{\partial \mathbf{A}}{\partial \mathbf{\theta}^i}$ 자체는 Tensor 이다. Vector or Tensor에 scalar $\theta^i$ 로 편미분을 하였으므로.
  
  Henceforth, the derivarives of the functions $\Phi = \Phi(\theta^1, \theta^2, \cdots \theta^n), \; \mathbf{x} = \mathbf{x}(\theta^1, \theta^2, \cdots \theta^n), \; \mathbf{A} = \mathbf{A}(\theta^1, \theta^2, \cdots \theta^n)$  with respect to curilinear coordinates $\theta^i$ will be denoted shortly, by 
  $$
  \begin{equation}
    \Phi_{, i}       = \frac{\partial \Phi}{\partial \theta^i}, \; 
    \mathbf{x}_{, i} = \frac{\partial \mathbf{x}}{\partial \theta^i}, \;
    \mathbf{A}_{, i} = \frac{\partial \mathbf{A}}{\partial \theta^i}
    \label{eq-2.61}
    \end{equation}
  $$
  

  -  여기서 **comma가 $i$ 앞에 붙은 것은 $\theta^i$로 편미분** 하였다는 의미이다. 
  
- **Gradient**는 **Covariant Transformation rule**을 따른다. 다음과 같이
    $$
    \begin{equation}
    \frac{\partial \Phi}{\partial \theta^i} = \frac{\partial \Phi}{\partial \bar{\theta}^k} \frac{\partial \bar{\theta}^k}{\partial \theta^i}, \quad

    \frac{\partial \mathbf{x}}{\partial \theta^i} = \frac{\partial \mathbf{x}}{\partial \bar{\theta}^k} \frac{\partial \bar{\theta}^k}{\partial \theta^i}, \quad

    \frac{\partial \mathbf{A}}{\partial \theta^i} = \frac{\partial \mathbf{A}}{\partial \bar{\theta}^k} \frac{\partial \bar{\theta}^k}{\partial \theta^i}
    \label{eq-2.62}
    \end{equation}
    $$

### Differential Operator를 Index에 표기하는 경우, Representation 

- $(\cdot) \vert_i$ denotes some differential operator on the components of the vector $\mathbf{x}$ or the tensor $\mathbf{A}$. 
- differential operator $(\cdot) \vert_i$를 사용할 경우 위에서 vector나 Tensor에는 comma를 붙였다. 이를 사용하여 먼저 Vector의 Gradient를표현해보면
	$$
	\begin{equation}
	\begin{aligned}
	\mathbf{x}_{, i} = x^j\vert_i \mathbf{g}_j = x_j\vert_i \mathbf{g}^j
	\end{aligned}
	\end{equation}
	$$

- 이것은 이렇게 생각하면 된다.  식 $\eqref{eq-2.61}$ 에서, $\mathbf{x}= x^j \mathbf{g}_j$ 라 놓으면  
    $$
    \begin{equation}
    \mathbf{x}_{, i} 
    = \frac{\partial \mathbf{x}}{\partial \theta^i}
    = \frac{\partial x^j}{\partial \theta^i} \mathbf{g}_j
    \triangleq x^j \vert_{i} \mathbf{g}_j
    \label{eq-my01}
    \end{equation}
    $$

- 그러므로 Tensor $\mathbf{A}$ 의 경우 
    $$
    \begin{equation}
    \mathbf{A}_{, i} 
    = A^{kl}\vert_{i} \mathbf{g}_k \otimes \mathbf{g}_l 
    = A_{kl}\vert_{i} \mathbf{g}^k \otimes \mathbf{g}^l
    = A^{k}_{.l}\vert_{i} \mathbf{g}_k \otimes \mathbf{g}^l
    \end{equation}
    $$

### Covariant Derivative 

- Index $i$ 에 대한 covariant rule $\eqref{eq-2.62}$ 을 **Covariant Derivative** 라고 한다. 즉 $\eqref{eq-my01}$ 를 말한다.
  $$
  \mathbf{x}_{, i} 
  = \frac{\partial \mathbf{x}}{\partial \theta^i}
  = \frac{\partial x^j}{\partial \theta^i} \mathbf{g}_j
  \triangleq x^j \vert_{i} \mathbf{g}_j
  $$
  
- **Covariant derivative**는 위 식에서 보듯이 $(\cdot)\vert_{i}$ 에 대한 미분, 즉,  **coordinate system에서 아래쪽 index**에 대한 Derivetive이다.

- Contravariant derivative는 반면에 위쪽 index 즉, $(\cdot) \vert^i$ 에 대한 미분이다. 즉, **Contravariant Transformation** $(\cdot) \vert^{i} = g^{ij} (\cdot) \vert_j$ 를 도입한 미분이다. 
    $$
    \begin{equation}
    \begin{aligned}
    &x^j \vert^i    = g^{ik} x^j \vert_k    &  x_j   \vert^i = g^{ik} x^j \vert_k    &    \\ 
    &A^{kl} \vert^i = g^{im} A^{kl} \vert_m & A_{kl} \vert^i = g^{im} A_{kl} \vert_m &  A_{.l}^k \vert^i = g^{im} A_{.l}^k \vert_m 
    \end{aligned}
    \end{equation}
    $$

- Scalar Function의 경우 Covariant와 Contravariant 미분이 동일하다. (그래서 굳이 최적화 이론에서 이를 구별하지 않았던 것) 따라서,
    $$
    \begin{equation}
    \Phi \vert_i = \Phi \vert^i = \Phi_{,i}
    \end{equation}
    $$

- Scalar Component에 대한 미분 표시를 사용하여 Gradient를 표기하면 다음과 같다.
	- 첫 표시가 기본형이라 생각하자.
	- Scalar $\Phi$ 에 대한 것은 Scalar 이므로 Component index 없이 Covariant 미분 index $(\cdot) \vert_i$  혹은 contravariant 미분 index $(\cdot) \vert^i$ 만을 사용한다. 
	-  Vector $\mathbf{x}$ 의 경우 Component index는 $x^j$ 를 기본으로 한다.
	- Tensor $\mathbf{A}$의 경우 Component index는 $A^{kl}$ 를 기본으로 한다.
	$$
	\begin{equation}
	\begin{aligned}
	grad \; \Phi 
	&= \Phi \vert_i \mathbf{g}^i = \Phi \vert^i \mathbf{g}_i \\
	
	grad \; \mathbf{x} 
	&= x^j \vert_i \mathbf{g}_j \otimes \mathbf{g}^i 
	 = x_j \vert_i \mathbf{g}^j \otimes \mathbf{g}^i 
	 = x^j \vert^i \mathbf{g}_j \otimes \mathbf{g}_i 
	 = x_j \vert^i \mathbf{g}^j \otimes \mathbf{g}_i \\
	 
	grad \; \mathbf{A} 
	&= A^{kl} \vert_i   \mathbf{g}_k \otimes \mathbf{g}_l \otimes \mathbf{g}^i
	 = A_{kl} \vert_i   \mathbf{g}^k \otimes \mathbf{g}^l \otimes \mathbf{g}^i
	 = A_{.l}^k \vert_i \mathbf{g}_k \otimes \mathbf{g}^l \otimes \mathbf{g}^i \\
	&= A^{kl} \vert^i 	\mathbf{g}_k \otimes \mathbf{g}_l \otimes \mathbf{g}_i
	 = A_{kl} \vert^i   \mathbf{g}_k \otimes \mathbf{g}_l \otimes \mathbf{g}_i
	 = A_{.l}^k \vert^i \mathbf{g}_k \otimes \mathbf{g}^l \otimes \mathbf{g}_i  
	\end{aligned}
	\end{equation}
	$$


## Christoffel Symbols, Representation of the Covariant Derivatives 

- Differential Operator for Covariant Derivatives를 만드는 것. 

  - 다시말해, Tensor 표기법으로 covarinat derivatives를 표현하겠다는 것

  - 임의의 **곡선 좌표계 (curvilinear sysytem)** $\theta^i = \theta^i (\mathbf{r})$ 에 대하여 ($\mathbf{r} \in \mathbb{E}^n$ 인 벡터, $\theta^i \in \mathbb{R}$ ) 

  - 이떄, Tangent Vector $\mathbf{g}_i$를 식 $\eqref{eq-2.27}$ 과 같이 정의한다.
    $$
    \begin{equation}
    \mathbf{g}_i = \frac{\partial \mathbf{r}}{\partial \theta^i}
    \end{equation}
    $$
  - 한편 dual vector $\mathbf{g}^i$ 를 정의한다. 둘 다, $\mathbb{E}^n$ 의 Bases 이다. 

### Christoffel Symbol $\Gamma_{ijk}, \; \Gamma_{ij}^k$ 

- 위와 같이 Tangent vector 및 그 Dual vector로 $\mathbb{E}^n$의 bases를 놓을 때, Christoffel Symbol을 다음과 같이 정의한다. 
	$$
  \begin{equation}
	\mathbf{g}_{i, j} = \Gamma_{ijk} \mathbf{g}^k = \Gamma_{ij}^k \mathbf{g}_k
  \label{eq-2.67}
  \end{equation}
	$$
	- 이것의 의미는 $\mathbf{g}_i$ 를 $\theta^j$ 에 대하여 편미분 한 것을 의미한다. 즉, Tangent Vector를 미분 했으니, Tensor가 나오고, 이를 $\mathbb{E}^n$의 Bases인 Tangent vector $\mathbf{g}_k$ 로 표시하겠다는 것이다.

- Relation $\mathbf{g}^k = g^{kl} \mathbf{g}_l$ 에 의해, $\eqref{eq-2.67}$의 Christoffel 기호는 다음의 관계가 있다.
  $$
  \Gamma_{ij}^k = g^{kl} \Gamma_{ijl}, \quad i,j,l = 1,2, \cdots n 
  $$
  
- Christoffel symbol의 특성을 살펴보기 위해 먼저 $\mathbf{g}_i$를 $\theta^j$ 로 편미분 하면 
	$$
	\begin{equation}
	\mathbf{g}_{i, j} 
	= \frac{\partial \mathbf{g}_i}{\mathbf \theta^j} 
	= \frac{\partial}{\partial \theta^j} \frac{\partial \mathbf{r}}{\partial \theta^i} 
	= \frac{\partial^2 {\mathbf{r}}}{\partial \theta^j \theta^i}
	\end{equation}
	$$
	- 위 식만 보면 마치 3-order Tensor 같이 보이지만, 실제로는 vector $\mathbf{r}$ 에 대하여 스칼라 $\theta^i$에 대하여 미분하여 벡터가 그대로 유지되고 여기에 또 스칼라 $\theta^j$가 미분하였기 때문에 $\mathbf{g}_{i, j}$ 는 벡터다. 
	- 그러므로
	$$
	\begin{equation}
	\mathbf{g}_{i, j} = \mathbf{r}_{, ij} = \mathbf{r}_{, ji} = \mathbf{g}_{j, i}
	\label{eq-2.69}
	\end{equation}
	$$
- 이에 의해 Christoffel 기호는 다음과 같이 2차 미분과, Tangent Vector에 의해 정의되는 **order-3 Tensor index** 를 가진, **Scalar Operator**가 된다.
	- 살펴보면, $i$번쨰 Tangent vector $\mathbf{g}_{i}$의 $\theta^j$ 에 대한 **편미분 $\mathbf{g}_{i, j}$은 벡터**이다. Component가 $\theta$ 전체가 아닌 $\theta^j$ 하나에 대해서이므로 Vector가 유지된다. ($\theta$ 전체라면 당연히 Matrix:Order-2 Tensor)
	$$
	\begin{equation}
	\begin{aligned}
	\Gamma_{ijk} = \Gamma_{jik} = \mathbf{g}_{i, j} \cdot \mathbf{g}_k = \mathbf{g}_{j, i} \cdot \mathbf{g}_k\\
	\Gamma_{ij}^k =\Gamma_{ji}^k= \mathbf{g}_{i, j} \cdot \mathbf{g}^k = \mathbf{g}_{j, i} \cdot \mathbf{g}^k
	\end{aligned}
	\label{eq-2.70}
	\end{equation}
	$$
	
	- 식 $\eqref{eq-2.70}$ 에서, Christoffel 기호의 **맨 마지막 index**가 유지 되면 나머지는 바뀌어도 된다.  (2차 편미분의 분모항이므로)
	
- Dual basis $\mathbf{g}^i$ 에 대하여 살펴보면, 먼저 $\mathbf{g}^i \cdot \mathbf{g}_j = \delta_j^i$ 에서 이를 $\theta^k$ 에 대하여 미분하면 0 이므로
	$$
	\begin{equation}
	0 
	= (\delta_j^i)_{, k} 
	= (\mathbf{g}^i \cdot \mathbf{g}_j)_{, k}
	= \mathbf{g}_{, k}^i \cdot \mathbf{g}_j + \mathbf{g}^i \cdot \mathbf{g}_{j, k} 
	= \mathbf{g}_{, k}^i \cdot \mathbf{g}_j + \mathbf{g}_i \cdot \Gamma_{jk}^l \mathbf{g}_l 
	= \mathbf{g}_{, k}^i \cdot \mathbf{g}_j + \Gamma_{jk}^i
	\end{equation}
	$$
	- **Notice !!** 그러므로 다음의 관계가 성립한다. 
	$$
	\begin{equation}
	\Gamma_{jk}^i 
	= - \mathbf{g}_{, k}^i \cdot \mathbf{g}_j 
	= - \mathbf{g}_{, j}^i \cdot \mathbf{g}_k
	= \Gamma_{kj}^i, \quad i, j, k= 1, 2,  \cdots , n
	\label{eq-2.72}
	\end{equation}
	$$
	- 그러므로 
	$$
	\begin{equation}
	\Gamma_{jk}^i = - \mathbf{g}_{, k}^i \cdot \mathbf{g}_j 
	\implies
	\mathbf{g}_{, k}^i \cdot \mathbf{g}_j \cdot \mathbf{g}^j= - \Gamma_{jk}^i \mathbf{g}^j
	\implies
	\mathbf{g}_{, k}^i = - \Gamma_{jk}^i \mathbf{g}^j = - \Gamma_{kj}^i \mathbf{g}^j
	\label{eq-2.73}
	\end{equation}
	$$
- 또한 $g_{ij} \mathbb{R}$ 은 $g_{ij} = \mathbf{g}_i \cdot \mathbf{g}_j$ 에서
    $$
    \begin{equation}
	{g_{ij}}_{, k}
	= (\mathbf{g}_i \cdot \mathbf{g}_j)_{, k} 
	= \mathbf{g}_{i, k} \cdot \mathbf{g}_j + \mathbf{g}_i \cdot \mathbf{g}_{j, k}
	= \Gamma_{ikj} + \Gamma_{jki}
    \label{eq-2.74}
    \end{equation}
    $$
	- 또한 $\eqref{eq-2.74}$ 에서 보듯이 $g_{ij, k}$는 미분 index $k$를 사이에 두고 두개의 Christoffel 기호가 더해져서 구해진다. 이를 사용하여 $\Gamma_{ijk}$ 를 구해보면 
	$$
	\begin{equation}
	\begin{aligned}
	g_{ij, k} &= \Gamma_{ikj} + \Gamma_{jki}  = \Gamma_{ikj} + \Gamma_{kji}\\
	g_{kj, i} &= \Gamma_{kij} + \Gamma_{jik}  = \Gamma_{ikj} + \Gamma_{ijk}\\	
g_{ki, j} &= \Gamma_{kji} + \Gamma_{ijk}  \\	
	\end{aligned}
	\implies 
	\begin{aligned}
	(g_{ki, j} + g_{kj, i} - g_{ij, k}) 
	&= \Gamma_{kji} + \Gamma_{ijk} + \Gamma_{ikj} + \Gamma_{ijk} -\Gamma_{ikj} - \Gamma_{kji} \\
	&= 2 \Gamma_{ijk}  \end{aligned}
	\end{equation}
	$$
	
	- 그러므로
	$$
	\begin{equation}
    \begin{aligned}
	\Gamma_{ijk}  &= \frac{1}{2} (g_{ki, j} + g_{kj, i} - g_{ij, k}) \\
	\Gamma_{ij}^k &= \frac{1}{2} g^{kl} (g_{li, j} + g_{lj, i} - g_{ij, l}) = g^{kl} \Gamma_{ijl}
	\end{aligned}
	\label{eq-2.75}
	\end{equation}
	$$
	
	- 식 $\eqref{eq-2.75}$ 에서 Cartesian coordinate 에서 Christoffel 기호는 Vanish 되어 버린다. 왜냐하면, Cartesian basis는
	  $$
	  \begin{equation}
	  \mathbf{r} = x^i \mathbf{e}_i
	  \end{equation}
	  $$
	  이므로  정의에 의해 Cartesian coordinate system에서 Tangent vector는 
	  $$
	  g_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j = \delta_{ij}
	  $$
	  따라서 $g_{ik} = \delta_{ik}$를 $\theta_j$  로 미분해야 얻을 수 있는 $g_{ik, j} = 0$ 이 되므로 
	  $$
	  \Gamma_{ijk} = \Gamma_{ij}^k = 0, \quad i,j,k=1, 2, \cdots n
	  $$
	  
### Example : Christoffel symbol of Cylidrical coordinate
앞의 예제를 다시 생각해 보면 
$\mathbf{r}=(\varphi, z, r) = r \cos \varphi e_1 + r \sin \varphi e_2 + z e_3$ 에서 
$$
\begin{equation}
\begin{aligned}
    g_1 &= \frac{\partial \mathbf{r}}{\partial \varphi} = -r \sin \varphi e_1 + r \cos \varphi e_2 \\
    g_2 &= \frac{\partial \mathbf{r}}{\partial z} = e_3\\
    g_3 &= \frac{\partial \mathbf{r}}{\partial r} =  \cos \varphi e_1 + \sin \varphi e_2
\end{aligned}
\label{eq55}
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
\label{eq56}    
\end{equation}
$$

결국 $g_{11, 3}$ 만 유의미한 상태이다. $g_{11, 3} = \frac{\partial r^2}{\partial r} = 2r$ , 고로,  $\Gamma_{ijk} = \frac{1}{2} (g_{ki, j} + g_{kj, i} - g_{ij, k})$ 에서,
$$
\begin{equation}
\begin{aligned}
\Gamma_{113} &= \frac{1}{2}(g_{31, 1} + g_{31,1} - g_{11, 3}) = -r \\
\Gamma_{131} &= \frac{1}{2}(g_{11, 3} + g_{13,1} - g_{13, 1}) = r = \Gamma_{311}
\end{aligned}
\end{equation}
$$

또한, $g_{11, 3}$ 만 미분에 대해 유의미하므로, $\Gamma_{ij}^k = \frac{1}{2} g^{kl}(g_{li, j} + g_{lj, i} - g_{ij, l})$ 에서 

- $l=1$ 에서 $k=1=l$ 자동 선택,  이때 $i, j =(1, 3), (3, 1)$, 이때는  

$$
\begin{equation}
\Gamma_{13}^1 = \frac{1}{2} g^{11} g_{11, 3}= \frac{1}{2} r^{-2} \cdot 2r = \frac{1}{r} = \Gamma_{31}^1
\end{equation}
$$

- $l=2$ 에서 $k=2$ 에서는 없음
- $l=3$ 에서 $k=3$  이때, $i, j = (1, 1)$ $\Gamma_{11}^3 = -r$
- 위와 같이 구할 수 있으나, $\Gamma_{113}$ 과 $\Gamma_{131}$ 이 0이 아니고 $g^{kl}$ 이 0이 아닌 Component는 $k=l$ 이므로 
  - $\Gamma_{11}^{3} = g^{33} \Gamma_{113} = -r$
  - $\Gamma_{13}^1 = \Gamma_{31}^{1} = \frac{1}{2} g^{11} g_{11, 3} = \frac{1}{r}$ 



#### Christoffel Symbol 을 사용한 covariant derivation 정의
For the vector valued function $\mathbf{x} = \mathbf{x}(\theta^1, \theta^2, \cdots, \theta^n)$, 이를 $\theta^j$ 에 대하여 covariant derivate 하면 ($\mathbf{x} = x^i \mathbf{g}_i$)
$$
\begin{equation}
\mathbf{x}_{, j}
=(x^i \mathbf{g}_i)_{, j}
= x_{, j}^i \mathbf{g}_i + x^i \mathbf{g}_{i, j}
\label{eq-2.58-1}
\end{equation}
$$

$\Gamma_{ij}^k = \mathbf{g}_{i, j} \cdot \mathbf{g}^k$ 에 의해,
$$
\begin{equation}
\mathbf{g}_k \Gamma_{ij}^k
= \mathbf{g}_{i, j} \cdot \mathbf{g}^k \cdot \mathbf{g}^k
= \mathbf{g}_{i, j}
\label{eq-2.58-2}
\end{equation}
$$

$\eqref{eq-2.58-2}$ 를 $\eqref{eq-2.58-1}$ 에 대입하면,
$$
\begin{equation}
\begin{aligned}
x_{, j}^i \mathbf{g}_i + x^i \mathbf{g}_{i, j}
&= x_{, j}^i \mathbf{g}_i + x^i \mathbf{g}_k \Gamma_{ij}^k \\
&= x_{, j}^i \mathbf{g}_i + x^k \Gamma_{kj}^i \mathbf{g}_i \\
&= \left( x_{, j}^i + x^k \Gamma_{kj}^i \right)\mathbf{g}_i,
\end{aligned}
\quad \therefore \mathbf{x}_{, j} = \left( x_{, j}^i + x^k \Gamma_{kj}^i \right)\mathbf{g}_i
\label{eq-2.83}
\end{equation}
$$

Covariant tangent vector $\mathbf{g}^i$ 에 대하여 생각해 보면, $\Gamma_{kj}^i = -\mathbf{g}_{, j}^i \cdot \mathbf{g}_k$
$$
\begin{equation}
\begin{aligned}
\mathbf{x}_{, j}
&=(x_i \mathbf{g}^i)_{, j}
= x_{i, j} \mathbf{g}^i + x_i \mathbf{g}_{, j}^i \\
&= x_{i, j} \mathbf{g}^i - x_i \Gamma_{kj}^i \mathbf{g}^k \\
&= x_{i, j} \mathbf{g}^i - x_k \Gamma_{ij}^k \mathbf{g}^i \\
&= \left( x_{i, j} - x_k \Gamma_{ij}^k \right) \mathbf{g}^i
\label{eq-2.84}
\end{aligned}
\end{equation}
$$

이것을 $\eqref{eq-2.63}$ 과 비교해 보면
$$
\begin{equation}
x^i \vert_j = \left( x_{, j}^i + x^k \Gamma_{kj}^i \right), \quad
x_i \vert_j = \left( x_{i, j} - x_k \Gamma_{ij}^k \right), \quad i, j=1, 2, \cdots n
\label{eq-2.85}
\end{equation}
$$

- **Notice!!** 여기서 중요한 사실은 Christoffel 기호와 연동되는 Tangent vector가 covariant냐, contra variant냐에 따라, 부호가 바뀌어 진다.
$$
\begin{equation}
\Gamma_{kj}^i \mathbf{g}_i = -\Gamma_{ij}^k \mathbf{g}^i
\label{note-2.84}
\end{equation}
$$

- 번외로 $\Gamma_{ij}^k = \Gamma_{kj}^i$ 임을 증명해 보면, $\mathbf{g}_k = \frac{\partial \mathbf{r}}{\partial \theta^k}, \; \mathbf{g}^k = g_i^k \mathbf{g}^i$ 에서
$$
\begin{equation}
\Gamma_{ij}^k
= \mathbf{g}_{i, j} \cdot \mathbf{g}^k
= \mathbf{g}_{i, j} \cdot g_i^{k}\mathbf{g}^i
= g_i^{k} \mathbf{g}_{i, j} \cdot \mathbf{g}^i
= \mathbf{g}_{k, j} \cdot \mathbf{g}^i
= \Gamma_{kj}^i
\end{equation}
$$

#### Christoffel Symbols for Tensor Valued Function
Let a tensor valued function $\mathbf{A} = \mathbf{A}(\theta^1, \theta^2, \cdots \theta^n)$:

$$
\begin{equation}
\begin{aligned}
\mathbf{A}_{, k}
&= \left( A^{ij} \mathbf{g}_i \otimes \mathbf{g}_j \right)_{, k} \quad	\because \text{기본은 covariant tangent vector} \\
&=  A^{ij}_{, k} \mathbf{g}_i \otimes \mathbf{g}_j + A^{ij} \mathbf{g}_{i, k} \otimes \mathbf{g}_j + A^{ij} \mathbf{g}_i \otimes \mathbf{g}_{j, k} \\
&=  A^{ij}_{, k} \mathbf{g}_i \otimes \mathbf{g}_j + A^{ij} \left( \Gamma_{ik}^l \mathbf{g}_l \right) \otimes \mathbf{g}_j + A^{ij} \mathbf{g}_i \otimes \Gamma_{jk}^l \mathbf{g}_l\\

&=  A^{ij}_{, k} \mathbf{g}_i \otimes \mathbf{g}_j + A^{lj} \Gamma_{lk}^i \mathbf{g}_i \otimes \mathbf{g}_j + A^{il} \mathbf{g}_i \otimes \Gamma_{lk}^j \mathbf{g}_j\\

&= \left(A^{ij}_{, k} + A^{lj} \Gamma_{lk}^i + A^{il} \Gamma_{lk}^j \right)  \mathbf{g}_i \otimes \mathbf{g}_j
\end{aligned}
\end{equation}
$$

Thus,
$$
\begin{equation}
A^{ij} \vert_k = A^{ij}_{, k} + A^{lj} \Gamma_{lk}^i + A^{il} \Gamma_{lk}^j , \quad i,j,k= 1, 2, \cdots, n
\label{eq-2.87}
\end{equation}
$$

Notice $\eqref{note-2.84}$ 에서 볼 수 있듯이 $\Gamma_{kj}^i \mathbf{g}_i = -\Gamma_{ij}^k \mathbf{g}^i$ 이므로
$$
\begin{equation}
\begin{aligned}
A_{ij} \vert_k   &= A_{ij, k}    - A_{lj} \Gamma_{ik}^l - A_{il} \Gamma_{jk}^l \\
A_{.j}^i \vert_k &= A_{j, k}^{i} + A_j^l \Gamma_{lk}^i  - A_l^i  \Gamma_{jk}^{l}
\end{aligned}
\label{eq-2.88}
\end{equation}
$$


#### Note 1. What is the Christoffel symbol?

- $\Gamma_{ijk}$ : **Scalar**, Inner product of tangent vector $\mathbf{g}^k$ and the partial differential of one tangent vector $\mathbf{g}_i$ to one coordinate component $\theta^j$ such that
$$
\begin{equation}
\Gamma_{ijk}
= \mathbf{g}_{i, j} \cdot \mathbf{g}_k
= \mathbf{g}_{j, i} \cdot \mathbf{g}_k = \Gamma_{jik}

,\quad \therefore
\Gamma_{ijk} =
\frac{\partial \mathbf{g}_i}{\partial \theta^j} \cdot \mathbf{g}_k
\label{note-001}
\end{equation}
$$

- 만일, contra derivative Tangent vector $\mathbf{g}^k$ 를 따라 (inner product 하여) 정의한다면, 다음과 같이 covariant derivative용  Christoffel symbol $\Gamma_{ij}^k$은 쉽게  정의할 수 있다.
$$
\Gamma_{ij}^k = \mathbf{g}_{i, j} \cdot \mathbf{g}^k
$$

- For covariant derivative, we require a Christoffel symbol $\Gamma_{ij}^k$ along to covarint tangent vector $\mathbf{g}_k$ so that
$$
\begin{equation}
0
= (\delta_j^i)_{,k}
= (\mathbf{g}^i \cdot \mathbf{g}_j)_{, k}
= \mathbf{g}_{, k}^i \cdot \mathbf{g}_j + \mathbf{g}^i \cdot \mathbf{g}_{j, k}
= \mathbf{g}_{, k}^i \cdot \mathbf{g}_j + \Gamma_{jk}^i, \\
\quad \therefore
\Gamma_{jk}^i
= - \mathbf{g}_{, k}^i \cdot \mathbf{g}_j
= - \mathbf{g}_{, j}^i \cdot \mathbf{g}_k
= \Gamma_{kj}^i,
\quad
\Gamma_{jk}^i = -\frac{\partial \mathbf{g}^i}{\partial \theta^k} \cdot \mathbf{g}_j
\label{note-002}
\end{equation}
$$

- $\Gamma_{ijk}$ : Contravariant $\mathbf{g}^k$ compatible, 순서대로 해석이 된다. $i$  번째 벡터를 $j$ 번째 Component로 미분해서 $k$ 번째 Tangent vector와 Inner-product.

- $\Gamma_{jk}^i$ : Covariant $\mathbf{g}_j$ compatible. Covariant는 행렬의 수학적 표시대로 왼쪽 아래 $j$ 번째 Tangent vector와 **minus Inner-product**.  미분은 $i$  번째 벡터를 $k$ 번째 Component로 미분.  **minus 가 있어서 $k, j$를 바꾸어진다** 즉, **inner product로 붙는 index가 아래앞으로 온다**. 라고 기억해도 된다. 그러면 위의 것과 같은 해석

- 즉, 아래 식 처럼, $j$로 미분한다는 것만 fix가 된다는 의미가 된다.

$$
\begin{equation}
\Gamma_{ij}^k = \mathbf{g}_{i, j} \cdot \mathbf{g}^k = - \mathbf{g}_{,j}^k \cdot \mathbf{g}_i
\end{equation}
$$



## Note

- Partial Differential에서 Scalar Component의 index는 모두 Upper index이다.   $\eqref{eq-2.31}, \eqref{eq-2.32}$를 살펴볼 것.
- 대체로 **scalar  값은 모두 upper index** 라고 생각하자. 
  - 왜냐하면 기본 Basis Vector $\mathbf{e}_k \in \mathbb{E}^n$ 은 **vector이므로 Lower index** 이므로.

