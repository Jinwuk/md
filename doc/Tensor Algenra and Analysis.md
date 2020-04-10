Tensor Algebra and Analysis
===
[toc]

본 내용은 Tensor Algebra와 Analysis에 관련된 내용을 요약 정리한 것이다.

## Basic Notation (pp 8)
Let $\mathcal{G} = \{ g_1, g_2, \cdots , g_n \} $ be a basis in n-dimensional Euclidean space $\mathbb{E}^n$.  
우리는 이러한 경우 즉, **아래 첨자 이면 Column Vector**로 인식한다.

Then, a basis $\mathcal{G'} = \{ g^1, g^2, \cdots , g^n \} $ of $\mathbb{E}^n$ is called **dual to $\mathcal{G}$** if 

$$
g_i \cdot g^j = \delta_i^j, \quad i,j =1,2, \cdots n
$$

위에서 이렇게 생각하면 된다. 즉,  위치와 상관없이  $\langle g^j , g_i \rangle$.  반대로 생각해도 상관은 없지만 이렇게 보는 것이 좋다. 



### Theorem 1.6 (pp 9)

모든 Basis는 그것의 Dual Basis가 있다는 정리인데, 이 증명 중에 중요한 것은 Inverse Matrix의 Notation이다.  

Let ${\mathbf{g}}^i$ be a basis dual to ${\mathbf{g}}_j$. Then 
$$
{\mathbf{g}}^i = g^{ij}{\mathbf{g}}_j, \quad {\mathbf{g}}_i = g_{ij}{\mathbf{g}}^j
$$
Therefore,
$$
{\mathbf{g}}^i = g^{ij}g_{jk}{\mathbf{g}}^k
$$
Multiplying scalarly with the vectors ${\mathbf{g}}_l$
$$
\delta_l^i = g^{ij}g_{jk} \delta_l^k
$$
그러므로 Matrix $[ g_{kj}]$  와 $[g^{kj}]$ 는 Inverse 이다. 즉,
$$
[g^{kj}] = [g_{kj}]^{-1}
\label{eq_th01}
$$

#### 몇가지 추가 사항

- For all $\quad g_k \in \mathbb{E}^n, \; g^{ij} \in \mathbf{R}$

$$
g^{ij} = g^{ji} = g^i \cdot g^j, \quad g_{ij} = g_{ji} = g_i \cdot g_j
$$

- The orthonormal  $e_k \in \mathbb{E}^n$ is **self dual** , so that

$$
e_i = e^i, \; e_i \cdot e^j = \delta_i^j
$$

- $\mathbf{x} = x^i \mathbf{g}_i = x_i \mathbf{g}^i$ 이므로 $x_i = \mathbf{x} \cdot \mathbf{g}_i, \; x^i = \mathbf{x} \cdot \mathbf{g}^i$  이것은 

$$
\mathbf{x} \cdot \mathbf{g}^i = x^j \mathbf{g}_j \cdot \mathbf{g}^i = x^j \delta_j^i = x^i
\label{eq01}
$$

식 $\eqref{eq01}$을 사용하여 일반적인 벡터의 Inner Product를 살펴보면 $\mathbf{x} \cdot \mathbf{y}$는  각각 $\mathbf{x} = x_i \mathbf{g}^i = x^i \mathbf{g}_i, \mathbf{y} = y_i \mathbf{g}^i = y^i \mathbf{g}_i$ 로 놓으면 
$$
\mathbf{x} \cdot \mathbf{y} = x^i \mathbf{g}_i \cdot y^j \mathbf{g}_j = x^i y^j g_{ij} = x^i y_i = x_i y^j
\label{eq02}
$$
식 $\eqref{eq01}, \eqref{eq02}$의 경우 Vector/Scalar 구별을 확실하게 하기 위하여 Vector는 굵게 표시 나머지의 경우는 Vector로 일반적으로 정의 하였음

####  Vector product 

For $a, b, c \in \mathbb{E}^n$.  Let $[abc] = (a \times b) \cdot c = (b \times c) \cdot a = (c \times a ) \cdot b$ 

Let  $\mathcal{G} = \{ \mathbf{g}_1, \mathbf{g}_2 , \mathbf{g}_3\}$, and $\mathbf{g}_k \in \mathbb{E}^3$. 

Set $g = [\mathbf{g}_1, \mathbf{g}_2 , \mathbf{g}_3]$

Consider the following set of vectors
$$
\mathbf{g}^1 = g^{-1} \mathbf{g}_2 \times \mathbf{g}_3, \quad 
\mathbf{g}^2 = g^{-1} \mathbf{g}_3 \times \mathbf{g}_1, \quad 
\mathbf{g}^3 = g^{-1} \mathbf{g}_3 \times \mathbf{g}_1
\label{eq_vec01}
$$

- proof
  $$
  \begin{aligned}
  g &= \mathbf{g}_2 \times \mathbf{g}_3 \cdot \mathbf{g}_1 \\
  g \mathbf{g}^1 &= \mathbf{g}_2 \times \mathbf{g}_3 \cdot \mathbf{g}_1 \cdot \mathbf{g}^1 \\
  g \mathbf{g}^1 &= \mathbf{g}_2 \times \mathbf{g}_3 \quad \because \mathbf{g}_1 \cdot \mathbf{g}^1 = \delta_1^1 = 1 \\
  \mathbf{g}^1 &= g^{-1} \mathbf{g}_2 \times \mathbf{g}_3
  
  \end{aligned}
  $$
  

#### Determinant

##### proof of $g^2 = | g_{ij} |$

Let $\mathbf{g}_k = \beta_k^i  \mathbf{e}_i$ , we obtain
$$
g 
= [\mathbf{g}_1, \mathbf{g}_2, \mathbf{g}_3] 
= [\beta_1^i \mathbf{e}_i, \beta_2^j \mathbf{e}_j, \beta_3^k \mathbf{e}_k] 
= \beta_1^i \beta_2^j \beta_3^k [\mathbf{e}_i, \mathbf{e}_j, \mathbf{e}_k] 
= \beta_1^i \beta_2^j \beta_3^k e_{ijk} = |\beta_j^i|
\label{eq_det01}
$$
즉,  $[\mathbf{e}_i, \mathbf{e}_j, \mathbf{e}_k] = e_{ijk}$  vector이기 때문에 이렇게 쓸 수 있으며 $\beta_1^i \beta_2^j \beta_3^k e_{ijk} = |\beta_j^i|$ 는 역시 Determinant의 정의에 의해 (부피 이므로 정 입방체의 부피) 이렇게 쓸 수 있다. 

여기서 $e_{ijk}$는 permutation symbol로서 **Levi-Civita symbol** 로 알려져 있으며 다음과 같이 정의된다.
$$
e_{ijk} = e^{ijk} = [\mathbf{e}_i, \mathbf{e}_j, \mathbf{e}_k] =
\begin{cases}
1  &\text{if } i,j,k \text{ is an even permuataion of 123}\\
-1 &\text{if } i,j,k \text{ is an odd permuataion of 123} \\
0  &Otherwise
\end{cases}
$$
이 경우에 right hand system에서 $[\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3] = 1$ 이 된다. 

$g_{ij}$ 의 정의에 의해
$$
g_{ij} 
= \mathbf{g}_i \cdot \mathbf{g}_j 
= \beta_i^k \mathbf{e}_k \cdot \beta_j^k \mathbf{e}_k 
= \sum_{k=1}^3 \beta_i^k \beta_j^k {e}_k {e}_k 
= \sum_{k=1}^3 \beta_i^k \beta_j^k
$$
그러므로 이는 
$$
[g_{ij}] = [\beta_i^k][\beta_j^k]^T
$$
따라서
$$
|g_{ij}| = |\beta_i^k| | \beta_j^k |= |\beta_i^k|^2 = g^2
$$

**Notice**

-  $g = [\mathbf{g}_1, \mathbf{g}_2 , \mathbf{g}_3]$ 는 단순히 $\mathbf{g}_k$ 로 구성된 





##### Vector Product의 특성 

- General Determinant 

식 $\eqref{eq_det01}$ 에서  ($g= [\mathbf{g}_1, \mathbf{g}_2, \mathbf{g}_3] $   이므로 i,j,k이면 permutation 항이 필요하다.)
$$
[\mathbf{g}_i, \mathbf{g}_j, \mathbf{g}_k] = e_{ijk} g
$$
식 $\eqref{eq_vec01}$ 에서
$$
\mathbf{g}_i \times \mathbf{g}_j = e_{ijk}g \mathbf{g}^k
\label{eq_vp02}
$$

$$
[ \mathbf{g}_i, \mathbf{g}_j, \mathbf{g}_k ] 
= \mathbf{g}_i \times \mathbf{g}_j \cdot \mathbf{g}_k = e_{ijk} g \\

\mathbf{g}_i \times \mathbf{g}_j \cdot \mathbf{g}_k \cdot \mathbf{g}^k = e_{ijk} g \mathbf{g}^k
$$

- Inverse Notation 

마찬가지로 $[\mathbf{g}^1, \mathbf{g}^2, \mathbf{g}^3]$ 를 생각해 보면 $g= [\mathbf{g}_1, \mathbf{g}_2, \mathbf{g}_3] $ 에서  $\eqref{eq_th01}$ 를 통해 이 값은 $g^{-1}$ 을 유추할 수 있다. 

Let $\mathbf{g}^k = \alpha_i^k e^i$  라 놓으면 마찬가지로 
$$
[\mathbf{g}^1, \mathbf{g}^2, \mathbf{g}^3] = \alpha_i^1 \alpha_j^2 \alpha_k^3 [\mathbf{e}^i, \mathbf{e}^j, \mathbf{e}^k] = \alpha_i^1 \alpha_j^2 \alpha_k^3 e^{ijk}
\label{eq_vp03}
$$
그런데 다음과 같으므로 
$$
\mathbf{g}_k \cdot \mathbf{g}^k = \beta_k^i \alpha_i^k \mathbf{e}_i \cdot \mathbf{e}^i = \delta_k^k \implies \alpha_i^k = (\beta_k^i)^{-1}
\label{eq_vp04}
$$
식 $\eqref{eq_vp04}$를 식 $\eqref{eq_vp03}$에 대입하면 
$$
[\mathbf{g}^1, \mathbf{g}^2, \mathbf{g}^3] 
= \alpha_i^1 \alpha_j^2 \alpha_k^3 e^{ijk}
= (\beta_i^1 \beta_j^2 \beta_k^3)^{-1}e^{ijk} = |\beta_i^1 |^{-1} = g^{-1}
$$
그러므로
$$
|g^{ij}| = g^{-2}
$$
따라서
$$
[\mathbf{g}^i, \mathbf{g}^j, \mathbf{g}^k] = \frac{e^{ijk} }{g}
$$
식 $\eqref{eq_vp02}$ 에 대한  Analogy는 다음과 같다.
$$
\mathbf{g}^i \times \mathbf{g}^j = \frac{e_{ijk} }{g} \mathbf{g}^k
$$

- 일반적인 Vector Product 

Let $\mathbf{a} = a^i \mathbf{g}_i = a_i \mathbf{g}^i$ , $\mathbf{b} = b^j \mathbf{g}_j = b_j \mathbf{g}^j$ 
$$
\mathbf{a} \times \mathbf{b} = (a^i \mathbf{g}_i) \times (b^j \mathbf{g}_j) = a^i b^j \mathbf{g}_i \times \mathbf{g}_j = a^i b^j e_{ijk} g \mathbf{g}^k 
= g 
\begin{vmatrix}
a^1 & a^2 & a^3 \\
b^1 & b^2 & b^3 \\
\mathbf{g}^1 & \mathbf{g}^2 & \mathbf{g}^3
\end{vmatrix} \\

\mathbf{a} \times \mathbf{b} = (a_i \mathbf{g}^i) \times (b_j \mathbf{g}^j) = a_i b_j \mathbf{g}^i \times \mathbf{g}^j = a_i b_j e^{ijk} g \mathbf{g}_k 
= \frac{1}{g} 
\begin{vmatrix}
a_1 & a_2 & a_3 \\
b_1 & b_2 & b_3 \\
\mathbf{g}_1 & \mathbf{g}_2 & \mathbf{g}_3
\end{vmatrix}
$$
만일 Orthonormal Basis 라면 $\mathbf{e}_i \times \mathbf{e}_j = e_{ijk} \mathbf{e}^k = e^{ijk} \mathbf{e}_k$

 그러므로 
$$
\mathbf{a} \times \mathbf{b} =  
\begin{vmatrix}
a_1 & a_2 & a_3 \\
b_1 & b_2 & b_3 \\
\mathbf{e}_1 & \mathbf{e}_2 & \mathbf{e}_3
\end{vmatrix} \\
$$


## Tensor Product $\otimes$

Tensor product는 두 개의 vetor에서 2nd order tensor를 만들때 유용하다.

For $\mathbf{a}, \mathbf{b} \in \mathbb{E}^n$ and an arbitrary vector $\mathbf{x} \in \mathbb{E}^n$ , 여기서, $\mathbf{x}$를 $\mathbf{b}$에 Projection된 값으로 $\mathbf{a}$로 나타낸다.
$$
(\mathbf{a} \otimes \mathbf{b}) \mathbf{x} = \mathbf{a}(\mathbf{b} \cdot \mathbf{x})
$$

즉, 출력이 $\mathbf{a}$ , 입력은 $\mathbf{x}$  시스템은 $\mathbf{b}$ 이다. 

