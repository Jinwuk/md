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

\mathbf{a} \times \mathbf{b} = (a_i \mathbf{g}^i) \times (b_j \mathbf{g}^j) = a_i b_j \mathbf{g}^i \times \mathbf{g}^j = a_i b_j e^{ijk} g^{-1} \mathbf{g}_k 
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

#### Amendment

Vector Product에서 특별히 다음과 같이 $\hat{\cdot}$ 를 정의한다.
$$
\mathbf{w} \times \mathbf{x} = \hat{\mathbf{w}} \mathbf{x}
$$
이는 일종의 Matrix로 볼 수 있으며 $\mathbf{x}$를 어떤 특정한 



#### Matrix 연산

$\mathcal{G} = \{ \mathbf{g}_1,  \mathbf{g}_2, \cdots  \mathbf{g}_n \}$ 으로 Basis가 주어지고 Dual Basis는 $\mathcal{G} = \{ \mathbf{g}^1,  \mathbf{g}^2, \cdots  \mathbf{g}^n \}$ 으로 주어진다고 가정하자.  그리고 $\mathbf{x}, \mathbf{y} \in \mathbb{E}^n$  에 대하여 $\mathbf{x} = g_i \mathbf{g}^i$, $\mathbf{y} = g_j \mathbf{g}^j$ 라 놓으면

- **Right Mapping**  (of $\mathbf{x}$ ) : $ \mathbf{y} = \mathbf{A} \mathbf{x}$  $x$ 에서 $y$로 mapping 되므로 
- **Left Mapping** (of $\mathbf{y}$) : $\mathbf{y} \mathbf{A} \cdot \mathbf{x} = \mathbf{y} \cdot (\mathbf{A} \mathbf{x})$  즉, 출력이 오른쪽으로 가능 Mapping 이기 때문에, 그리고 이를 조금만 변형 시키면 Right Mapping과 동일한 결과가 나오도록 해야한다. 

먼저 Left Mapping을 생각해보면 
$$
\mathbf{y} \mathbf{A} = y_i\mathbf{g}^i \mathbf{A} = y_i[\mathbf{g}^i (\mathbf{A} \mathbf{g}^j)]\mathbf{g}_j
\label{eq_mat01}
$$
일단, Matrix이므로 이렇게 쓴다.  즉, $\eqref{eq_mat01}$ 에서 Matrix는 $[\mathbf{g}^i (\mathbf{A} \mathbf{g}^j)]$ 이렇게 생각한다. 모두 Upper에 있는 것으로 생각한다.  그래서 원래 $\mathbf{g}^i \rightarrow \mathbf{g}_j $ 로 보내는 Transform이 된다 .(Left Transform) 

그러므로
$$
\mathbf{y} \cdot (\mathbf{A} \mathbf{x}) 
= \mathbf{y} \cdot (x_j \mathbf{A} \mathbf{g}^j) 
= y_i x_j [\mathbf{g}^i \mathbf{A} \mathbf{g}^j]
$$


## Tensor Product $\otimes$

Tensor product는 두 개의 vetor에서 2nd order tensor를 만들때 유용하다.

- Right Mapping

For $\mathbf{a}, \mathbf{b} \in \mathbb{E}^n$ and an arbitrary vector $\mathbf{x} \in \mathbb{E}^n$ , 여기서, $\mathbf{x}$를 $\mathbf{b}$에 Projection된 값으로 $\mathbf{a}$로 나타낸다.
$$
(\mathbf{a} \otimes \mathbf{b}) \mathbf{x} = \mathbf{a}(\mathbf{b} \cdot \mathbf{x})
$$

즉, 출력이 $\mathbf{a}$ , 입력은 $\mathbf{x}$  시스템은 $\mathbf{b}$ 이다.  즉 입력이 오른쪽에 있으면 오른쪽에 대한 Mapping이 Right Mapping이다. 그리고 당연히 출력 기준 Basis를 사용하게 된다.

- Left Mapping

$$
\mathbf{y}(\mathbf{a} \otimes \mathbf{b}) = (\mathbf{y} \cdot \mathbf{a}) \mathbf{b}
$$

즉, 출력이 $\mathbf{b}$ , 입력은 $\mathbf{y}$  시스템은 $\mathbf{a}$ 이다.  즉 입력이 왼쪽에 있으면 왼쪽에 대한 Mapping이 Left Mapping이다. 그리고 당연히 출력 기준 Basis를 사용하게 된다.

- Matrix는 Left Mapping이든 Right Mapping 이든 Dimension만 맞으면 같은 결과가 나오도록 해야한다.
- Tensor Product를 통해 Matrix가 정의될 수 있음을 보인다.

### Theorem 1.7

Let $\mathcal{F} = \{ \mathbf{f}_1, \mathbf{f}_2, \cdots \mathbf{f}_n \}$  , $\mathcal{G} = \{ \mathbf{g}_1, \mathbf{g}_2, \cdots \mathbf{g}_n  \} $   be two arbitrary bases of $\mathbb{E}^n$ . Then , the tensors $\mathbf{f}_i \otimes \mathbf{g}_j$ represent a basis of $\mathbf{L}in^n$ . The dimension of the vector space $\mathbf{L}in^n$ is thus $n^2$ . 

$\mathbf{L}in^n$ 의 element는 Second order Tensor를 의미한다. 즉, Matrix. 

- 일반적인 Notation에서 $a \otimes b$ 는 $a b^T$ 로 생각하면 된다. 따라서 

$$
\begin{aligned}
(\mathbf{a} \otimes \mathbf{b}) \mathbf{x} 
&= \mathbf{a} \mathbf{b}^T \mathbf{x} = \mathbf{a}(\mathbf{b} \cdot \mathbf{x}) \\

\mathbf{y} (\mathbf{a} \otimes \mathbf{b}) 
&= \mathbf{y}^T (\mathbf{a} \mathbf{b}^T) = (\mathbf{y}^T \mathbf{a}) \mathbf{b}^T 
= (\mathbf{y} \cdot \mathbf{a}) \mathbf{b}
\end{aligned}
$$

#### proof

Let $\mathbf{A}' = (\mathbf{f}^i \mathbf{A} \mathbf{g}^j) \mathbf{f}_i \otimes \mathbf{g}_j$

The tensors $\mathbf{A}$ and $\mathbf{A}'$ coincide if and only if $\mathbf{A}'\mathbf(x) = \mathbf{A} \mathbf{x} \quad \forall x \in \mathbb{E}^n$   그러므로
$$
\mathbf{A}' \mathbf{x} 
= (\mathbf{f}^i \mathbf{A} \mathbf{g}^j) \mathbf{f}_i\otimes \mathbf{g}_j (x_k \mathbf{g}^k) 
= (\mathbf{f}^i \mathbf{A} \mathbf{g}^j) \mathbf{f}_i\otimes x_k \mathbf{g}_j \mathbf{g}^k 
= (\mathbf{f}^i \mathbf{A} \mathbf{g}^j) \mathbf{f}_i x_k \delta_j^k
= x_j (\mathbf{f}^i \mathbf{A} \mathbf{g}^i) \mathbf{f}_j
\label{eq_th1.7_01}
$$
또한 

$$
\mathbf{A} \mathbf{x} = \mathbf{A} x_j \mathbf{g}^j = x_j  \mathbf{A} \mathbf{g}^j
\label{eq_th1.7_02}
$$

그런데,  $\mathbf {x} = \mathbf{g}^i \mathbf{x} \mathbf{g}_i$  이므로  $\eqref{eq_th1.7_02}$ 는

$$
\mathbf{A} \mathbf{g}^j = \mathbf{f}^i (\mathbf{A} \mathbf{g}^j ) \mathbf{f}_i = (\mathbf{f}^i \mathbf{A} \mathbf{g}^j ) \mathbf{f}_i
$$
그러므로 
$$
\mathbf{A} \mathbf{x} = x_j (\mathbf{f}^i \mathbf{A} \mathbf{g}^j) \mathbf{f}_i=\mathbf{A}' \mathbf{x}
$$
따라서 $\eqref{eq_th1.7_01}$ 을 사용하여 1차독립을 증명할 수 있다. 구체적인 증명은 pp18을 본다.

- Theorem 1.7 에 따라 Matrix or Second order tensor는 다음과 같이 쓸 수 있다. 

$$
\mathbf{A} 
= A^{ij} \mathbf{g}_i \otimes \mathbf{g}_j 
= A_{ij} \mathbf{g}^i \otimes \mathbf{g}^j 
= A_{\cdot j}^i \mathbf{g}_i \otimes \mathbf{g}^j 
= A_{i \cdot}^{j} \mathbf{g}^i \otimes \mathbf{g}_j
$$

- $A_{\cdot j}^i$ 에서 $\cdot j$ 는 $j$가 뒷 편 Index라는 의미이다.  $\cdot$ 에 해당하는 부분은 위쪽 인덱스 이므로.
- $A_{i \cdot}^j$ 에서 $i \cdot $ 는 $i$가 앞 편 Index라는 의미이다.  $\cdot$ 에 해당하는 부분은 위쪽 인덱스 이므로. 
- 즉, $\cdot$ 는 아래에만 쓰인다.

그러므로 
$$
\mathbf{A} \mathbf{g}^j
= A^{ij} \mathbf{g}_i \otimes \mathbf{g}_j \mathbf{g}^j = A^{ij} \mathbf{g}_i
\implies
\mathbf{g}^i \mathbf{A} \mathbf{g}^j = A^{ij} \mathbf{g}^i \mathbf{g}_i = A^{ij}
$$
**$i, j$ 위치가 그대로 $\mathbf{g}$ 의 index**가 된다. 그러므로 다음이 성립한다.  
$$
A^{ij} = \mathbf{g}^i \mathbf{A} \mathbf{g}^j \quad 
A_{ij} = \mathbf{g}_i \mathbf{A} \mathbf{g}_j \quad 
A_{i \cdot}^{j} = \mathbf{g}_i \mathbf{A} \mathbf{g}^j \quad 
A_{\cdot j}^{i} = \mathbf{g}^i \mathbf{A} \mathbf{g}_j \quad
$$


## Change of Basis 

기본적으로 Basis $\mathbf{g}_k, \bar{\mathbf{g}}_k$ 에 대하여 다음이 성립한다.
$$
\mathbf{g}_i = a_i^j \bar{\mathbf{g}}_j
\label{eq_cb01}
$$
따라서, 임의의 벡터 $\mathbf{x}$ 에 대하여
$$
\mathbf{x} = x^i \mathbf{g}_i = x^i a_i^j \bar{\mathbf{g}}_j
$$
그러므로 Matrix $\mathbf{A}$ 에 대하여는 
$$
\mathbf{A} = A^{ij} \mathbf{g}_i \otimes \mathbf{g}_j = A^{ij} (a_i^k \bar{\mathbf{g}}_k) \otimes (a_j^l \bar{\mathbf{g}}_l) = A^{ij} a_i^k a_j^l \bar{\mathbf{g}}_k \otimes \bar{\mathbf{g}}_l \implies \bar{A}^{kl} = A^{ij} a_i^k a_j^l
$$



## Matrix Tranposition

Tranposition은 다음과 같이 정의된다.
$$
(\mathbf{a} \otimes \mathbf{b})^T = \mathbf{b} \otimes \mathbf{a}
$$
그러므로 $\mathbf{A} = A^{ij} \mathbf{g}_i \otimes \mathbf{g}_j$ 에 대하여 
$$
\mathbf{A}^T = (A^{ij} \mathbf{g}_i \otimes \mathbf{g}_j)^T = A^{ij} (\mathbf{g}_i \otimes \mathbf{g}_j)^T = A^{ij} \mathbf{g}_j \otimes \mathbf{g}_i 
$$
즉, index가 정상적인 경우에는 가까운 것이 가깝게 멀리있는 것이 멀리 이렇게 되었으나 Transpotion이 되면 가까이 index는 먼 Basis를 먼 index는 가까운 Basis를 선택하도록 되어있다.
$$
(\mathbf{A}^T)_{ij} = A_{ji}, \quad (\mathbf{A}^T)^{ij} = A^{ji} \\
(\mathbf{A}^T)_{i \cdot}^j = A_{\cdot i}^j=g^{jk}A_{k \cdot}^{l}g_{li}, \quad 
(\mathbf{A}^T)_{\cdot j}^i = A_{\cdot j}^i=g_{jk} A_{\cdot l}^{k} g^{li}
$$
위 식의 유도 예는 다음과 같다.
$$
(\mathbf{A}^T)_{\cdot j}^i = \mathbf{g}^i \mathbf{A}^T \mathbf{g}_j = \mathbf{g}_j \mathbf{A} \mathbf{g}^i = \mathbf{g}_j A_{\cdot l} ^k \mathbf{g}_k \otimes \mathbf{g}^l  \mathbf{g}^i = g_{jk} A_{\cdot l}^k g^{li}
$$

## Vector Identity 

$$
(\mathbf{a} \times \mathbf{b}) \times \mathbf{c} = (\mathbf{a} \cdot \mathbf{c}) \mathbf{b} - (\mathbf{b} \cdot \mathbf{c}) \mathbf{a}
\label{eq_vi01}
$$


$$
\hat{\mathbf{a} \times \mathbf{b}} = (\mathbf{b} \otimes \mathbf{a}) - (\mathbf{a} \otimes \mathbf{b})
\label{eq_vi02}
$$

Vector Identity는 식 $\eqref{eq_vi01}$ 를 의미한다. 

이를 증명하기 위해서는 연습문제 1.14에서 제시한 다음 4개의 명제를 증명해야 한다.

For $n=3$ 

- $\delta^{ij} e_{ijk} = 0$

$$
\delta^{ij} e_{ijk} = \delta^{ij} (e_{ij1} + e_{ij2} + e_{ij3}) = e_{iik} \vert_{k=1,2,3} = 0
$$

- $e^{ikm} e_{jkm} = 2 \delta_j^i $

	- $k, m$ 이 fix 이면 총 9개의 $i, j$ index가 나온다. 그런데, $k, m$ fix에서 0이 아니려고 하면 $i, j $는 같아야 한다. 고로 $\delta_j^i$.  그리고 하나가 결정되면 Even 방향과 Odd 방향의 Permutation이 되므로 $ 1 \cdot 1 + (-1) \cdot (-1) = 2$ 가 된다. 따라서 $2 \delta_j^i$. 

- $e^{ijk} e_{ijk} = 6$ 
	
- n= 3인 경우 Permutation 종류는 Even의 경우 $(1,2,3), (2,3,1), (3,1, 2)$ 고로 위의 결과에서 6
	
- $e^{ijm} e_{klm} = \delta_k^i \delta_l^j - \delta_l^i \delta_k^j$ 

  - $m$만 같고 $i, j$ 가 다른 경우이므로 $(i, k), (j, l)$ 의 index가 같은 경우와 다른 경우 ($(i, k), (j, l)$) 를 생각하면 위와 같다. 

  

식 $\eqref{eq_vi01}$ 를 이 결과를 통해 증명하면 

$\mathbf{a} = a^i \mathbf{g}_i, \; \mathbf{b} = b^j \mathbf{g}_j, \; \mathbf{c} = c_l \mathbf{g}^l$ 라 하자. 그러면 
$$
\mathbf{a} \times \mathbf{b} 
= a^i b^j \mathbf{g}_i \times \mathbf{g}_j = a^i b^j e_{ijk} g \mathbf{g}^k \\

(\mathbf{a} \times \mathbf{b}) \times \mathbf{c} 
= c_l(a^i b^j e_{ijk} g \mathbf{g}^k) \times \mathbf{g}^l
= c_l a^i b^j g e_{ijk} (\mathbf{g}^k \times \mathbf{g}{^l})
= c_l a^i b^j g g^{-1} e_{ijk} e^{klm} \mathbf{g}_m
\label{eq_pfvi01}
$$

따라서, 이 경우는 

$$
e_{ijk} e^{klm} = e_{ijk} e^{lmk} = \delta_i^l \delta_j^m - \delta_i^m \delta_j^l
$$
에 해당한다.  그러므로 식 $\eqref{eq_pfvi01}$ 은 다음과 같이 쓸 수 있다,. 
$$
c_l a^i b^j e_{ijk} e^{klm} \mathbf{g}_m = a^i b^j c_l e_{ijk} e^{klm} \mathbf{g}_m = a^i b^j c_l (\delta_i^l \delta_{j}^m - \delta_i^m \delta_j^l) \mathbf{g}_m
\label{eq_pfvi02}
$$
따라서
$$
a^i b^j c_l (\delta_i^l \delta_{j}^m - \delta_i^m \delta_j^l) \mathbf{g}_m
= a^i c_i b^j \mathbf{g}_j - b^j c_j a^i \mathbf{g}_i 
= (\mathbf{a}\cdot \mathbf{c} ) \mathbf{b} - (\mathbf{b}\cdot \mathbf{c} ) \mathbf{a}
\label{eq_pfvi03}
$$
그러므로  $\eqref{eq_vi01}$ 이 증명되었다. $\eqref{eq_vi02} $의 경우는 간단히 증명된다.



## Scalar Product of Second-Order Tensors

- Definition of Matrix Scalar Product

$$
(\mathbf{a} \otimes \mathbf{b}) : (\mathbf{c} \otimes \mathbf{d}) = (\mathbf{a} \cdot \mathbf{c})(\mathbf{b} \cdot \mathbf{d}), \quad \mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d} \in \mathbb{E}^n
$$

이는 다음을 유도한다.
$$
\mathbf{c} \otimes \mathbf{d} : \mathbf{A} = \mathbf{c} \mathbf{A} \mathbf{d} = \mathbf{d} \mathbf{A}^T \mathbf{c}
\label{eq_sp01}
$$
간단히 $\mathbf{A} = \mathbf{a} \otimes \mathbf{b}$ 라 놓으면 정의에 의해 
$$
\mathbf{c} \otimes \mathbf{d} : \mathbf{A} 
= (\mathbf{c} \cdot \mathbf{a})(\mathbf{d} \cdot \mathbf{b} )
= (\mathbf{c} \cdot \mathbf{a})(\mathbf{b} \cdot \mathbf{d} )
= \mathbf{c} (\mathbf{a} \otimes \mathbf{b}) \mathbf{d}  
= \mathbf{c} \mathbf{A} \mathbf{d} 
= (\mathbf{c} \mathbf{A} \mathbf{d})^T = \mathbf{d} \mathbf{A}^T \mathbf{c}
$$
그러므로 임의의 두 Tensor $\mathbf{A}, \mathbf{B}$ 의 scalar product는 
$$
\mathbf{A}:\mathbf{B} = A_{ij} B^{ij} = A^{ij}B_{ij} = A_{\cdot j}^i B_{i \cdot }^j = A_{i \cdot }^j B_{\cdot j}^i
$$
그리고 이는 Scalar 이기 떄문에 기존의  벡터 scalar product의 특성을 모두 가지고 있다.

식 $\eqref{eq_sp01}$ 의 특성으로 인해 다음이 성립한다.
$$
\mathbf{A} : (\mathbf{B}\mathbf{C}) = (\mathbf{B}^T\mathbf{A}):\mathbf{C}= (\mathbf{A}\mathbf{C}^T):\mathbf{B}
$$

- Trace는 다음과 같이 Scalar Product가 된다. 

$$
tr \mathbf{A} = \mathbf{A} : \mathbf{I}
$$

따라서 다음의 특성을 가진다.
$$
\begin{aligned}
tr(\mathbf{a} \otimes \mathbf{b}) &= \mathbf{a} \cdot \mathbf{b} \\
tr(\mathbf{A} \mathbf{B}) &= \mathbf{A} : \mathbf{B}^T = \mathbf{A}^T : \mathbf{B} \\
tr(\mathbf{A}\mathbf{B}) &= tr(\mathbf{B}\mathbf{A})
\end{aligned}
$$




