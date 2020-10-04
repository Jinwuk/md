Tensor Algebra and Analysis -2
===
[toc]

본 내용은 Tensor Algebra와 Analysis에 관련된 내용 중 2절 37P 이후의 것들 중 중요한 내용을 요약한 것이다.

## Coordinates inEuclidean Space, Tangent Vectors
Definition 2.1 에 의해 

$$
x^i = x^i (r)  \Leftrightarrow r = r(x^1, x^2, \cdots , x^n)
$$
where $r \in \mathbb{E}^n$ and $x^i \in \mathbb{R} (i=1, 2, \cdots , n)$ 
따라서 $x^i = x^i (r) $ 과 $r = r(x^1, x^2, \cdots , x^n)$ 는 sufficiently differentiable 이라 가정한다.

- 예 : Cylindrical coordinates in $\mathbb{E}^3$ 

$$
r =r (\psi, z, r) = r \cos \varphi e_1 + r \sin \varphi e_2 + z e_3
\label{eq02_01}
$$



<img src="http://jnwhome.iptime.org/img/research/2020/tensor_001.png" style="zoom:50%;" />

- 그림에서 $r = x^i h_i$ 의 형태로 또 다른 basis $\mathcal{H} = \{ h_1, h_2, \cdots , h_n \}$ 로 표시할 수 있다.

- 그러므로 관심이 있는 것은,  $x^i =x^i(r)$ and $y^i = y^i (r)$ 의 두개의 임의의 $\mathbb{E}^n$ 기반 좌표계가 주어졌을 때
  $$
  x^i = \hat{x} (y^1, y^2, \cdots, y^n) \Leftrightarrow y^i = \hat{y}^i(x^1, x^2, \cdots x^n)
  $$
  과 같이 invertible 이면 다음이 성립한다.
  $$
  \frac{\partial y^i}{\partial y^j} = \delta_{ij} = \frac{\partial y^i}{\partial x^k}\frac{\partial x^k}{\partial y^j}
  $$
  
- 그러므로 $\| \delta_{ij} \|$ 를 생각해 보면 Jacobian $J = \[\frac{\partial y^i}{\partial x^k} \]$ 은 0이 아니어야 하며 그, inverse $K$가 존재해야 한다.

- 여기에서 다음의 임의의 curvilinear coordinate system 을 생각해보면
  $$
  \theta^i = \theta^i (r) \rightarrow r = r (\theta^1, \theta^2, \cdots \theta^n)
  $$
  where $r = \mathbb{E}^n$ and $\theta^i \in \mathbb{R} (i=1, 2, \cdots n)$. 
  
  그리고 (k만 빠졌다)
  $$
  \theta^i = const, \quad i=1, 2, \cdots, k-1, k+1, \cdots n
  $$
  이면 $\theta^k$ coordinate line 이라고 하는 curve in $\mathbb{E}^n$ 을 다음과 같이 정의할 수 있다.  
  
  The vectors
  $$
  g_k = \frac{\partial r}{\partial \theta^k}, \quad k=1, 2, \cdots n
  $$
  are called the **tangent vectors to the cooresponding $\theta^k$ coordinate line** .
  
- Example : Tangent vectors and metric coefficients of cylindricla coordinates in $\mathbb{E}^n$ . $\mathbf{r}=(\varphi, z, r) = r \cos \varphi e_1 + r \sin \varphi e_2 + z e_3$, 즉 $\eqref{02_01}$ 
$$
\begin{aligned}
g_1 &= \frac{\partial \mathbf{r}}{\partial \varphi} = -r \sin \varphi e_1 + r \cos \varphi e_2 \\
g_2 &= \frac{\partial \mathbf{r}}{\partial z} = e_3\\
g_3 &= \frac{\partial \mathbf{r}}{\partial r} =  \cos \varphi e_1 + \sin \varphi e_2
\end{aligned}
$$

그러므로 metric coefficient는 
$$
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
$$

따라서, $g^{1} = g^{-1} g_1$ 에서
$$
\begin{aligned}
g^1 &= \frac{1}{r^2}(-r \sin \varphi e_1 + r \cos \varphi e_2) = \frac{1}{r} (-\sin \varphi e_1 + \cos \varphi e_2)\\
g^2 &= g_2 = e_3\\
g^3 &= g_3 = \cos \varphi e_1 + \sin \varphi e_2
\end{aligned}
$$

## Coordinatye Transformation, Co-, Contra- and Mixed variant Components

Let $