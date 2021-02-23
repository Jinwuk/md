Tensor Algebra and Analysis -6
===
[toc]

본 내용은 Tensor Algebra와 Analysis에 관련된 내용 중 6절  중 중요한 내용을 요약한 것이다.

## Scalar-valued Isotropic Tensor Functions 

Scalar fied $f(\mathbf{A}_1, \mathbf{A}_2, \cdots \mathbf{A}_l)$ 과 $\mathbf{A}_k \in \mathbf{L}in^n$ 에서 다음 관계가 만족되면 **Isotropic**  이라 한다. 
$$
f(\mathbf{Q}\mathbf{A}_1\mathbf{Q}^T, \mathbf{Q}\mathbf{A}_2\mathbf{Q}^T, \cdots \mathbf{Q}\mathbf{A}_l\mathbf{Q}^T) = f(\mathbf{A}_1, \mathbf{A}_2, \cdots \mathbf{A}_l), 
\quad \forall \mathbf{Q} \in \mathbf{O}trh^n 
$$

- Isotropic 은 Symmetry Matrix : Orthogonal Matrix의 특성으로 이루어진다.  
  - Symmetry Matrix의 Eigen vector들로 이루어진 Second-order Tensor (Orthogonal Matrix) 의 특성을 본다.
  - 복소수로 확장하면 Skew Symmetry 특성이 된다.



## Scalar-valued Anisotropic Tensor Functions 

Scalar fied $f(\mathbf{A}_1, \mathbf{A}_2, \cdots \mathbf{A}_l)$ 과 $\mathbf{A}_k \in \mathbf{L}in^n$ 에서 다음 관계가 만족되면 **Anisotropic**  이라 한다. 
$$
f(\mathbf{Q}\mathbf{A}_1\mathbf{Q}^T, \mathbf{Q}\mathbf{A}_2\mathbf{Q}^T, \cdots \mathbf{Q}\mathbf{A}_l\mathbf{Q}^T) = f(\mathbf{A}_1, \mathbf{A}_2, \cdots \mathbf{A}_l), 
\quad \forall \mathbf{Q} \in \mathbf{S}orth^n \subset \mathbf{O}rth^n
$$

- $\mathbf{S}orth^n$ 는 **Symmetry Group**



## Derivatives of Scalar-Valued Tensor Functions

A scalar valued function $f(\mathbf{A}) : \mathbf{L}in^n \mapsto \mathbb{R}$  is **differentiable ** in a neighborhood of $\mathbf{A}$ if there exists a **tensor** $f(\mathbf{A}),_{\mathbf{A}} \in \mathbf{L}in^n$ such that 
$$
\frac{d}{dt} f(\mathbf{A} + t\mathbf{X}) \Bigg\vert_{t=0} = f(\mathbf{A}),_{\mathbf{A}} : \mathbf{X}
$$

- **Directional Derivative** : **Gateaux derivatives ** 라 한다. $\mathbf{A}$ 에서 연속이다.

- $f(\mathbf{A}),_{\mathbf{A}}$는 **Derivative** 혹은 **Gradient** of the tensor function $f(\mathbf{A})$ 라고 한다. 

- 이를 임의의 basis $\mathbf{g}_i \otimes \mathbf{g}_j$ 를 사용하여 표현하면 
  $$
  \frac{d}{dt}f(\mathbf{A}+t\mathbf{X}) \Bigg\vert_{t=0} = \frac{d}{dt}f[(A^i_{\cdot j} + tX^i_{\cdot j}) \mathbf{g}_i \otimes \mathbf{g}^j ] \Bigg\vert_{t=0} = \frac{\partial f}{\partial A^i_{\cdot j}} X^i_{\cdot j}
  $$

- 여기에서 $\mathbf{A} \in \mathbf{S}lin^n \subset \mathbf{L}in^n$ 즉,  Special Linear group에 속하는 것으로서 특별히 Symmetric tensor function에 대하여 
  $$
  f(\mathbf{M}),_{\mathbf{M}} \in\mathbf{S}ym^n \quad for\; \mathbf{M} \in \mathbf{S}ym^n
  $$
  에 대하여 논하는 것이다. 

  

#### Note

$$
\mathbf{A} : \mathbf{A} = (A_{ij} \mathbf{g}^i \otimes \mathbf{g}^j)(A_{kl} \mathbf{g}^k \otimes \mathbf{g}^l) = A_{ij}A_{kl} g^{ik}g^{jl}
$$

즉,  위에서 $\mathbf{g}^i$ 는 $\mathbf{g}^k$ 와  $\mathbf{g}^j$ 는 $\mathbf{g}^l$  과 Coincide 되어 있다고 볼 수 있다. 



## Derivatives of Tensor valued Tensor Functions

A function $g(\mathbf{A}) : \mathbf{L}in^n \mapsto \mathbf{L}in^n$ is differentiable in a neighborhood of $\mathbf{A}$ ,  만일  a fourth order tensor $g(\mathbf{A})_{, \mathbf{A}} \in \mathcal{L}in^n$ 이 존재하여 다음을 만족하면  
$$
\frac{d}{dt} g(\mathbf{A} + t\mathbf{X}) \Bigg\vert_{t=0} = g (\mathbf{A})_{, \mathbf{A}} : \mathbf{X}, \quad \forall \mathbf{X} \in \mathbf{L}in^n
\label{dtt_eq01}
$$
이때, $\mathbf{G} = g(\mathbf{A}) \in \mathbf{L}in^n$의 basis 를 $\mathbf{g}_i \otimes \mathbf{g}^j, \; (i, j = 1, 2, \cdots , n)$  라 하면 Chain Rule에 의하여 
$$
\begin{aligned}
\frac{d}{dt} g(\mathbf{A} + t\mathbf{X}) \Bigg\vert_{t=0} 

&= \frac{d}{dt} \left\{G^i_{.j} \left[ (A^k_{.l} + t X^k_{.l}\right) \mathbf{g}_k \otimes \mathbf{g}^l]  \mathbf{g}_i \otimes \mathbf{g}^j \right\} \\

&= \frac{\partial G^i_{.j}}{\partial A^k_{.l}} X^k_{.l} \mathbf{g}_i \otimes \mathbf{g}^j

\end{aligned}
$$

- 즉,  $\frac{d g}{dt} = \frac{dg}{dA} \frac{dA}{dt}$ 로 보고, $A = A^k_{.l} + t X^k_{.l}$ 로 놓으면 기본적인 미분의 Chain Rule이 설명된다. 
- 그 다음, 출력은 $G^i_{.j}$  이므로 당연히 Basis는 $\mathbf{g}_i \otimes \mathbf{g}^j$ 이다.
- 미분의 중간단계에서 $k, l$ index는 사라지므로 $\mathbf{g}_k \otimes \mathbf{g}^l$ 는 사라진다. 

그러므로 식 $\eqref{dtt_eq01}$ 식에서 ${g}_{, \mathbf{A}}$ 는 다음과 같이 정의된다.
$$
g_{, \mathbf{A}} 
= \frac{\partial G^i_{.j}}{\partial A^k_{.l}} \mathbf{g}_i \otimes \mathbf{g}^k \otimes \mathbf{g}_l \otimes \mathbf{g}^j
$$

- 즉 분모 $\partial A^k_{.l}$의 경우 윗 첨자는 Basis index가 위에 아래는 아래에 해당한다.
- 또한, Tensor 표기는 $i$ - $k$- $l$-$j$ 의 순이 된다.  
  - 즉, **분자 , 출력**에 해당 하는 index는 **맨 앞과 맨 뒤** (**첫번쨰와 네번째**)에 
  - **분모 , 입력**에 해당하는 index는 **두번째와 세번째**에 

- Symmetry에 해당되면 이것의 Basis는 다음과 같이 표현 할 수 있다.

  - 윗 첨자 혹은 아래 첨자는 일치해야 한다.

  $$
  \frac{1}{2}(\mathbf{g}^k \otimes \mathbf{g}^l + \mathbf{g}^l \otimes \mathbf{g}^k)
  $$

  - Symmetry Tensor $\mathbf{M} \in \mathbf{S}ym^n \subset \mathbf{L}in^n$ 에 대하여 다음과 같다. 
    $$
    \begin{aligned}
    g(\mathbf{M})_{, \mathbf{M}} 
    
    &= \frac{1}{2} \sum_{k, l=1, l \leq k}^n \frac{\partial G^i_{.j}}{\partial M^{kl}} \mathbf{g}_i \otimes (\mathbf{g}^k \otimes \mathbf{g}^l + \mathbf{g}^l \otimes \mathbf{g}^k) \otimes \mathbf{g}^j \\
    
    &= \frac{1}{2} \sum_{k, l=1, l \leq k}^n \frac{\partial G^i_{.j}}{\partial M_{kl}} \mathbf{g}_i \otimes (\mathbf{g}_k \otimes \mathbf{g}_l + \mathbf{g}_l \otimes \mathbf{g}_k) \otimes \mathbf{g}^j \\
    
    \end{aligned}
    $$
    

