Differential Forms Essential
=============

## Diffential forms in $\mathbb{R}^n$

### Defintion 1.
A field of linear forms (or exterior form of degree 1) in $\mathbb{R}^3$ is a map $w$ that associatesa to each $p \in \mathbb{R}^3$ an element $w(p) \in (\mathbb{R}_p^3)^*$ ; $w$ can be written as
$$
w(p) = a_1(p)(dx_1)_p + a_2(p)(dx_2)_p + a_3(p)(dx_3)_p \\
w = \sum_{i=1}^3 a_i(p)dx_i
$$

Let $\varphi: \mathbb{R}^3 \times \mathbb{R}^3 \rightarrow \mathbb{R}$ be a **bilinear (i.e. $\varphi$ is a linearin each variable) ** and **alternate (i.e. $\varphi(v_1, v_2) = - \varphi(v_2, v_1)$)** then for $\varphi_1 \wedge \varphi_2 \in \Lambda^2(\mathbb{R}_p^3)^*$ by setting
$$
(\varphi_1 \wedge \varphi_2)(v_1, v_2) = \det(\varphi_i(v_j))
$$

- 일반적으로 $\varphi \in \Lambda^k(\mathbb{R}_p^n)^*$ 에 대하여 $\varphi : \underset{\text{k times}}{\underbrace{\mathbb{R}_p^n \times \cdots \mathbb{R}_p^n}} \rightarrow \mathbb{R}$

$$
(\varphi_1 \wedge \cdots \wedge \varphi_k)(v_1, \cdots v_k) =\det(\varphi_i(v_j)), \;\;\; i,j = 1, \cdots, k
$$


- Differential form의 중요한 기능은 Differential Maps하에서 하나의 연산으로 기능한다는 것이다.
Let $f : \mathbb{R}^n \rightarrow \mathbb{R}^m$ be a differentiable map.
Then **$f$ induces a map $f^*$ that takes k-forms in $\mathbb{R}^m$ into a k-forms in $\mathbb{R}^n$** 
	- 마치 **역함수** 처럼 기능한다. i.e. $f : \mathbb{R}^n \rightarrow \mathbb{R}^m$, $f^* : (\mathbb{R}^m)^* \rightarrow (\mathbb{R}^n)^*$
	- 위 첨자에 별표가 붙는 것으로서 역함수 처럼 기능한다는 것을 알 수 있다.

$$
\begin{align}
(f^*w)(p)(v_1, \cdots v_k) &= w(f(p))(df_p(v_1), \cdots, df_p(v_k)) 
\end{align}
$$

 - 즉, 다시말해,

$$
\begin{align}
f^*(\varphi_1 \wedge \cdots \wedge \varphi_k)(v_1, \cdots v_k) &= (\varphi_1 \wedge \cdots \wedge \varphi_k)(df(v_1), \cdots, df(v_k)) \\
&= \det(\varphi_i (df(v_j))) \\
&= \det(f^* \varphi_i (v_j)) \\
&= (f^* \varphi_1, \wedge, \cdots, \wedge f^* \varphi_k)(v_1, \cdots, v_k)
\end{align}
$$

### Note
$v(f)$는 표기이고 실제는 $df(v) = \langle \nabla f, v \rangle$ 인 것 처럼 Differential Map에 대한 Form 연산도 마찬가지이다.
즉, $f^* w$는 표기이고 실제는 $w (df)$ 즉 Differential Form의 Inner Product 다시말해 Determinent를 의미한다. 


- 표기법
	- $p \in \mathbb{R}^n$, $ v_1 \cdots v_k \in \mathbb{R}_p^n$, and $df_p : \mathbb{R}_{p}^n \rightarrow \mathbb{R}_{f(p)}^m$ is the differential of the map $f$ at $p$ 그리고 $g$가 0-form 이면

$$
f^* (g) = g \circ f = g(f)
$$

- An exteriror k-form in $\mathbb{R}^n $ is a map $w$ that associates to each $p \in \mathbb{R}^n$  an element $w(p) \in \Lambda^k(\mathbb{R}_p^n)^*$ 는 다음과 같이 표시된다. **Differential k-form**

$$
w(p) = \sum_{i_1 < \cdots i_k} a_{i_1 \cdots i_k} (p) (dx_{i_1} \wedge \cdots \wedge dx_{i_k})_p, \;\; i_j \in \{ 1, \cdots , n \}
$$
위 식의 간략한 표기는 다음과 같다.
$$
w = \sum_I a_I dx_I
$$

- Differentiable 0-form은 다음과 같은 Differentiable function 을 의미한다.
$$
f:\mathbb{R}^n \rightarrow \mathbb{R}
$$

- elemental differential form such as $dx_i, dy_j \in (\mathbb{R}^n)^*$ 에 대한 $f^* dx_i$ 의 경우는 ($y$로 치환해도 같다.)

$$
f^* dx_i (v) = dx_i (df(v)) = d(x_i \circ f)(v) = df_i (v)
$$


### Proposition 
Let $f:\mathbb{R}^n \rightarrow \mathbb{R}^m$ be a differentiable map, 
$w$ and $\varphi$ be k-forms on $\mathbb{R}^m$ and 
$g : \mathbb{R}^m \rightarrow \mathbb{R}$ be a 0-form on $\mathbb{R}^m$. Then:
1. $f^*(w + \varphi) = f^* w + f^* \varphi$
2. $f^*(gw)= f^*(g) f^*(w)$
3. If $\varphi_1 \cdots \varphi_k$ are 1-forms in $\mathbb{R}^m$ $f^*(\varphi_1 \wedge \cdots \wedge \varphi_k) = f^* \varphi_1 \wedge \cdots \wedge f^*\varphi_k$

### Example : compute $f^* w$
Let $w$ be the 1-form in $\mathbb{R}^2 - \{0, 0 \}$ by
$$
w = - \frac{y}{x^2 + y^2}dx + \frac{y}{x^2 + y^2}dy
$$

Let $U$ be the set in the plane $(r,\theta)$ given by
$$
U = \{ r > 0 : 0 < \theta < 2 \pi \}
$$

and let $f:U \rightarrow \mathbb{R}^2$ be the map
$$
f(r, \theta) = 
\begin{cases}
x &= r \cos \theta \\
y &= r \sin \theta
\end{cases}
$$
그러므로 $f^* w$ 는 $\mathbb{R}^2 - \{0, 0 \}$ 를 $U$로 가져온다.
#### How to
Let $w = \sum_i a_i dx_i$ then
$$
a_1 = - \frac{y}{x^2 + y^2}, \;\; a_2 = \frac{y}{x^2 + y^2}, \;\; dx_1 =dx, \;\; dx_2 = dy
$$
Thus
$$
f^* w = \sum_i f^* a_i f^* dx_i = \sum_i (a_i \circ f) (dx_i (df ))
$$
$f$ 는 Corrdination Transform이므로 $a_i$의 $(x,y)$를 $(r, \theta)$로 변경시켜야 한다.
$$
\begin{align}
a_1 \circ f &= (- \frac{y}{x^2 + y^2}) \circ f(r, \theta) = -\frac{1}{r} \sin \theta \\
a_2 \circ f &= (\frac{x}{x^2 + y^2}) \circ f(r, \theta) = \frac{1}{r} \cos \theta 
\end{align}
$$
그리고 Elementary Differential의 변화는 $dx_i(df) = d(x_i \circ f) = df_i$ 이므로
$$
\begin{align}
df_1 &= \cos \theta dr - r \sin \theta d\theta \\
df_2 &= \sin \theta dr + r \cos \theta d\theta 
\end{align}
$$
따라서
$$
\begin{align}
f^* w &= (a_1 \circ f) df_1 + (a_2 \circ f) df_2 \\
&= -\frac{1}{r} \sin \theta (\cos \theta dr - r \sin \theta d\theta) + \frac{1}{r} \cos \theta (\sin \theta dr + r \cos \theta d\theta ) \\
&= d \theta
\end{align}
$$

### Proposition 4
Let $f:\mathbb{R}^n \rightarrow \mathbb{R}^m$ be a differential map. Then
- $f^*(w \wedge \varphi) = (f^*w) \wedge (f^* \varphi)$
- $(f \circ g)^* w = g^* (f^* w)$ , where $g: \mathbb{R}^p \rightarrow \mathbb{R}^n$
... 다시말해 $(f \circ g)$는 $\mathbb{R}^p \rightarrow \mathbb{R}^n \rightarrow \mathbb{R}^m$ , 그러므로 $$(f \circ g)^* w$ 는 $\mathbb{R}^p \rightarrow \mathbb{R}^n$ 
