Tensor Algebra and Analysis -5
===
[toc]

본 내용은 Tensor Algebra와 Analysis에 관련된 내용 중 5절 3이후의 것들 중 중요한 내용을 요약한 것이다.

## Fundamental Definition 

$$
\mathbf{A} \otimes \mathbf{B} : \mathbf{X} = \mathbf{A} \mathbf{X} \mathbf{B},  \quad
\mathbf{A} \odot \mathbf{B} : \mathbf{X}   = \mathbf{A} (\mathbf{B} : \mathbf{X})
$$

$$
\mathbf{Y} : \mathbf{A} \otimes \mathbf{B}   = \mathbf{A}^T \mathbf{Y} \mathbf{B}^T,  \quad
\mathbf{Y} : \mathbf{A} \odot \mathbf{B}     = (\mathbf{Y} :\mathbf{A}) \mathbf{B}  
$$

- 그런데 식 (2)에서 다음의 Tensor dot가 Composition Mapping의 정의에 의해 가능하다.
  $$
  (\mathbf{A} \odot \mathbf{B}) : \mathbf{X}  = (\mathbf{B} : \mathbf{X}) \mathbf{A}
  $$

  - 즉, $(\mathbf{B} : \mathbf{X})$ 는 Mapping 이므로 좌우 어느 쪽에 놓여도 상관 없다.  
  - Tensor outer product ($\otimes$ ) 는 Matrix 연산이므로 Left, Right 연산 결과가 다르다.
  - 그러나, Tensor dot product는 dot product의 정의에 따라 Left, Right 연산 결과가 같아야 한다. 
    - 다시말해 Composition Mapping은 차원을 축소한다. 그러므로 Second Order Tensor = Matrix의 Composition Mapping 결과는 Scalar가 된다.  
      - 2-Order씩 줄어든다고 생각하면 된다. (4th order -> 2nd order: 2nd order -> 0 order)



## Special Operations with Fourth-Order Tensors 

$$
(\mathbf{A} \otimes \mathbf{B}) : (\mathbf{C} \otimes \mathbf{D}) = (\mathbf{A} \mathbf{C}) \otimes (\mathbf{D} \mathbf{B})
$$

- proof 
  $$
(\mathbf{A} \otimes \mathbf{B}) : (\mathbf{C} \otimes \mathbf{D}) : \mathbf{X} 
= (\mathbf{A} \otimes \mathbf{B}) : (\mathbf{C} \mathbf{X} \mathbf{D})
= \mathbf{A} (\mathbf{C} \mathbf{X} \mathbf{D}) \mathbf{B}
= (\mathbf{A} \mathbf{C}) \mathbf{X} (\mathbf{D} \mathbf{B})
= (\mathbf{A} \mathbf{C}) \otimes (\mathbf{D} \mathbf{B}) : \mathbf{X}
  $$
   **Q.E.D**

$$
(\mathbf{A} \otimes \mathbf{B}) : (\mathbf{C} \odot \mathbf{D}) 
= (\mathbf{A} \mathbf{C} \mathbf{B}) \odot \mathbf{D}
$$

- proof 
  $$
(\mathbf{A} \otimes \mathbf{B}) : (\mathbf{C} \odot \mathbf{D}) : \mathbf{X}
= (\mathbf{A} \otimes \mathbf{B}) : \mathbf{C} (\mathbf{D} : \mathbf{X}) 
= (\mathbf{A} \mathbf{C} \mathbf{B}) (\mathbf{D} : \mathbf{X}) 
= (\mathbf{A} \mathbf{C} \mathbf{B}) \odot \mathbf{D} : \mathbf{X}
  $$
  **Q.E.D**

  - Another Approach
  
  $$
  (\mathbf{A} \otimes \mathbf{B}) : (\mathbf{C} \odot \mathbf{D}) 
  = \mathbf{A} (\mathbf{C} \odot \mathbf{D}) \mathbf{B}
  = (\mathbf{A} \mathbf{C} \mathbf{B} )\odot \mathbf{D}
  $$
  

$$
(\mathbf{A} \odot \mathbf{B}) : (\mathbf{C} \otimes \mathbf{D}) 
= \mathbf{A} \odot (\mathbf{C}^T \mathbf{B} \mathbf{D}^T)
$$

- proof
  $$
  \mathbf{Y} : (\mathbf{A} \odot \mathbf{B}) : (\mathbf{C} \otimes \mathbf{D}) 
  = (\mathbf{Y} : \mathbf{A}) \mathbf{B} : (\mathbf{C} \otimes \mathbf{D})
  = (\mathbf{Y} : \mathbf{A}) (\mathbf{C}^T \mathbf{B} \mathbf{D}^T)
  = \mathbf{Y} : \mathbf{A} \odot (\mathbf{C}^T \mathbf{B} \mathbf{D}^T)
  $$

  - Another Approach
  
  $$
  (\mathbf{A} \odot \mathbf{B}) : (\mathbf{C} \otimes \mathbf{D}) 
  = \mathbf{C}^T (\mathbf{A} \odot \mathbf{B}) \mathbf{D}^T
  = \mathbf{A} \odot (\mathbf{C}^T \mathbf{B} \mathbf{D}^T )
  $$
  

$$
(\mathbf{A} \odot \mathbf{B}) : (\mathbf{C} \odot \mathbf{D})  = (\mathbf{B} : \mathbf{C}) \mathbf{A} \odot \mathbf{B}
$$

- proof
  $$
  (\mathbf{A} \odot \mathbf{B}) : (\mathbf{C} \odot \mathbf{D}) : \mathbf{X} 
  = (\mathbf{A} \odot \mathbf{B}) : \mathbf{C} (\mathbf{D} : \mathbf{X})
  = \mathbf{A} (\mathbf{B} : \mathbf{C}) (\mathbf{D} : \mathbf{X})
  = (\mathbf{B} : \mathbf{C}) \mathbf{A} (\mathbf{D} : \mathbf{X})
  = (\mathbf{B} : \mathbf{C}) (\mathbf{A} \odot \mathbf{D}) : \mathbf{X}
  $$
  

#### Note 1

$$
\mathcal{A} : \mathbf{X}
= (\mathcal{A}^{ijkl} \mathbf{g}_i \otimes \mathbf{g}_j \otimes \mathbf{g}_k \otimes \mathbf{g}_l) 
: (X_{qp} \mathbf{g}^q \otimes \mathbf{g}^p)
= \mathcal{A}^{ijkl} X_{jk} \mathbf{g}_i \otimes \mathbf{g}_l \\

\mathbf{X} : \mathcal{A} 
= (X_{qp} \mathbf{g}^q \otimes \mathbf{g}^p) : (\mathcal{A}^{ijkl} \mathbf{g}_i \otimes \mathbf{g}_j \otimes \mathbf{g}_k \otimes \mathbf{g}_l) 
= \mathcal{A}^{ijkl} X_{il} \mathbf{g}_i \otimes \mathbf{g}_l \\
$$

- Fourth order tensor와 Second order tensor의 Composition 연산의 경우 

  - Left의 두번째와 Right의 첫번쨰

  - Left의 세번쨰와 Right의 끝번쨰  index의 product로 정의한다. 

    

#### Note 2

$$
\mathcal{A} : \mathcal{B} 
= (\mathcal{A}^{ijkl} \mathbf{g}_i \otimes \mathbf{g}_j \otimes \mathbf{g}_k \otimes \mathbf{g}_l) 
: (\mathcal{B}_{pqrt} \mathbf{g}^p \otimes \mathbf{g}^q \otimes \mathbf{g}^r \otimes \mathbf{g}^t)
= \mathcal{A}^{ijkl} \mathcal{B}_{jqrk} \mathbf{g}_i \otimes \mathbf{g}^q \otimes \mathbf{g}^r \otimes \mathbf{g}_l
$$

- $j \leftrightarrow p, \; k \leftrightarrow t$  를 연산하게 된다.  즉, Left의 두번쨰 index가  Right의 첫번쨰 index,  그리고 Left의 세번째 index가 Right의 끝번쨰(마지막 index) 에 대응하여 연산하게 된다. 
- 이는 Note 1에서의 정의에 따르는 것이다. 



#### Simple Composition with second-order tensors

Let $\mathcal{D}$ be a fourth-order tensor and $\mathbf{A, B}$ are two second order tensors.

##### Define 

$$
(\mathbf{A} \mathcal{D} \mathbf{B}) : \mathbf{X} = \mathbf{A} (\mathcal{D} : \mathbf{X}) \mathbf{B}
$$

- $\mathbf{A} \mathcal{D} \mathbf{B} : \mathbf{X} =  (\mathbf{A} \otimes \mathbf{B}) : \mathcal{D} : \mathbf{X} = (\mathbf{A} \otimes \mathbf{B}) : (\mathcal{D} : \mathbf{X}) = \mathbf{A} (\mathcal{D} : \mathbf{X}) \mathbf{B}$ 



#### Transpoitions

- 단, Large/small Transposition $T, t$는 오직, Fourth order Tensor에만 적용된다. 
  - 고로 Second order Tensor에서는 Large/small Transposition이 구별되지 않으며   동일한 Transposition이다.  이때는 Large $T$를 사용하여 표시한다.

$$
\mathcal{A}^T : \mathbf{X} = \mathbf{X} : \mathcal{A}, \quad \mathcal{A}^t : \mathbf{X} = \mathcal{A} : \mathbf{X}^T
$$

- Left Large Transposition은 Tensor Compliance의 순서를 바꾼다.
- Left small Transposition은 Tensor Compliance의 순서는 그대로이고 Right Large Transposition이 된다. 
- 즉,  Right Large Transposition은 Left small Transposition  이에 따라

$$
\mathbf{Y} : \mathcal{A}^t = (\mathbf{Y} : \mathcal{A})^T
$$

- proof 
  $$
  (\mathbf{Y} : \mathcal{A}^t) : \mathbf{X} = \mathbf{Y} : (\mathcal{A}^t : \mathbf{X}) = \mathbf{Y} : (\mathcal{A} : \mathbf{X}^T) = (\mathbf{Y} : \mathcal{A}) : \mathbf{X}^T = (\mathbf{Y} : \mathcal{A})^T : \mathbf{X}
  $$
  
- 즉,  Fourth order tensor에서 Large Transposition은 연산의 순서를 바꾸어야 하지만,  small Transposition은 대응되는 Second order Tensor에 작용하여 이를 Transposition 시키는 것이다. 

##### Symmetrization 

$$
\mathcal{F}^s = \frac{1}{2} (\mathcal{F} + \mathcal{F}^t)
$$



- 즉, Fourth order Tensor에서 Second order Tensor를 Transposition 시키기 위한  small Transposition을 작용시킨 것. 따라서

$$
\mathcal{F}^s : \mathbf{X} = \mathcal{F}: sym \mathbf{X}, \quad \mathbf{Y} : \mathcal{F}^s = sym(\mathbf{Y} : \mathbf{F})
$$

##### Transposotion and Tensor Product

$$
(\mathbf{A} \otimes \mathbf{B})^T = \mathbf{A}^T \otimes \mathbf{B}^T, \quad (\mathbf{A} \odot \mathbf{B})^T = \mathbf{B} \odot \mathbf{A}, \quad (\mathbf{A} \odot \mathbf{B})^t = \mathbf{A} \odot \mathbf{B}
$$

##### Order of Transposition 

$$
\begin{aligned}
(\mathbf{a} \otimes \mathbf{b} \otimes \mathbf{c} \otimes \mathbf{d})^T 
&= \mathbf{b} \otimes \mathbf{a} \otimes \mathbf{d} \otimes \mathbf{c} \\

(\mathbf{a} \otimes \mathbf{b} \otimes \mathbf{c} \otimes \mathbf{d})^t 
&= \mathbf{a} \otimes \mathbf{c} \otimes \mathbf{b} \otimes \mathbf{d}
\end{aligned}
$$

- 즉 Large $T$는 Transposition의 정의에 충실한 방식이고, Small $t$는 Transpodition 연산 효과에 충실한 방식이다. 
- 그 결과는 다음과 같다.

$$
(\mathcal{A} : \mathcal{B})^T = \mathcal{B}^T : \mathcal{A}^T, \quad (\mathcal{A} : \mathcal{B})^t = \mathcal{A} : \mathcal{B}^t
$$

- 또한 Transposition 이므로 다음과 같다. 
  $$
  \mathcal{A}^{TT} = \mathcal{A}, \quad \mathcal{A}^{tt} = \mathcal{A}, \quad \forall \mathcal{A}\in \mathcal{L}in
  $$
  
- 이를 사용하면 다음과 같은 Relation이 가능하다.
  $$
  (\mathbf{A} \otimes \mathbf{B})^t : (\mathbf{C} \otimes \mathbf{D}) = \left[ (\mathbf{A} \mathbf{D}^T) \otimes (\mathbf{C}^T \mathbf{B})\right]^t \\
  (\mathbf{A} \otimes \mathbf{B})^t : (\mathbf{C} \odot \mathbf{D}) = (\mathbf{A} \mathbf{C}^T \mathbf{B}) \odot \mathbf{D}
  $$

  - proof
    $$
    \begin{aligned}
    (\mathbf{A} \otimes \mathbf{B})^t : (\mathbf{C} \otimes \mathbf{D}) : \mathbf{X} 
    &= (\mathbf{A} \otimes \mathbf{B})^t : (\mathbf{C} \mathbf{X} \mathbf{D}) \\
    &= (\mathbf{A} \otimes \mathbf{B}) : (\mathbf{C} \mathbf{X} \mathbf{D})^T \\
    &= (\mathbf{A} (\mathbf{C} \mathbf{X} \mathbf{D})^T \mathbf{B}) \\
    &= \mathbf{A} \mathbf{D}^T \mathbf{X}^T \mathbf{C}^T \mathbf{B} \\
    &= (\mathbf{A} \mathbf{D}^T \otimes \mathbf{C}^T \mathbf{B}) : \mathbf{X}^T \\
    &= (\mathbf{A} \mathbf{D}^T \otimes \mathbf{C}^T \mathbf{B})^t : \mathbf{X}
    \end{aligned}
    $$

    $$
    \begin{aligned}
    (\mathbf{A} \otimes \mathbf{B})^t : (\mathbf{C} \odot \mathbf{D}) : \mathbf{X}
    &= (\mathbf{A} \otimes \mathbf{B})^t : \mathbf{C} (\mathbf{D} : \mathbf{X})  \\
    &= (\mathbf{A} \otimes \mathbf{B}) : [\mathbf{C} (\mathbf{D} : \mathbf{X})]^T \\
    &= (\mathbf{A} \otimes \mathbf{B}) : [(\mathbf{D} : \mathbf{X}) \mathbf{C}^T ] \\
    &= \mathbf{A} [(\mathbf{D} : \mathbf{X}) \mathbf{C}^T ] \mathbf{B}  \\
    &= (\mathbf{D} : \mathbf{X})\mathbf{A} \mathbf{C}^T \mathbf{B}  \\
    &= (\mathbf{A} \mathbf{C}^T \mathbf{B})(\mathbf{D} : \mathbf{X}) \\
    &= (\mathbf{A} \mathbf{C}^T \mathbf{B}) \odot \mathbf{D} : \mathbf{X} \\
    \end{aligned}
    $$

  

##### Scalar Product  

Tensor product에서도 다음과 같이 Scalar Product를 정의할 수 있다. 
$$
(\mathbf{A} \odot \mathbf{B}) :: (\mathbf{C} \odot \mathbf{D}) = (\mathbf{A} : \mathbf{C})(\mathbf{B} : \mathbf{D})
$$
따라서 다음과 같다.
$$
(\mathbf{a} \otimes \mathbf{b} \otimes \mathbf{c} \otimes \mathbf{d}) :: (\mathbf{e} \otimes \mathbf{f} \otimes \mathbf{g} \otimes \mathbf{h}) = (\mathbf{a} \cdot \mathbf{e})(\mathbf{b} \cdot \mathbf{f})(\mathbf{c} \cdot \mathbf{g})(\mathbf{d} \cdot \mathbf{h})
$$
그러므로 Fourth order Tensor의 scalar product는 다음과 같이 간단히 정의된다.
$$
\mathcal{A} :: \mathcal{B} = (\mathcal{A}^{ijkl} \mathbf{g}_i \otimes \mathbf{g}_j \otimes \mathbf{g}_k \otimes \mathbf{g}_l \otimes) :: (\mathcal{B}_{pqrt} \mathbf{g}^p \otimes \mathbf{g}^q \otimes \mathbf{g}^r \otimes \mathbf{g}^t) = \mathcal{A}^{ijkl} \mathcal{B}_{ijkl}
$$



## Super Symmetric Fourth-order Tensors

- Major Symmetry 
  $$
  \mathcal{E}^T = \mathcal{E}, \quad \forall \mathcal{E} \in \mathcal{L}in^n
  $$

- Minor Symmetry
  $$
  \mathcal{E}^t = \mathcal{E}
  $$

- Super Symmetry는 Major 및  Minor  Symmetry 모두 지원되는 경우이다. 

- Super Symmetry  space는 일반적인 Tensor space $\mathcal{L}in^n$을 Super symmetry space $\mathcal{S}sym^n$ 로 표기한다. 

### Super Symmetry Fourth-order tensor의 특성

- Super symmetry Fourth-order tensor는 임의의 Matrix와의 Mapping 결과가 Transposition과 무관하게 나타나도록 한다. 즉,
  $$
  (\mathcal{E} : \mathbf{X})^T = \mathcal{E} : \mathbf{X}, \quad \forall \mathcal{E} \in \mathbf{S}sym^n, \; \mathbf{X} \in \mathbf{L}in^n
  $$

  - Proof
    $$
    \begin{aligned}
    (\mathcal{E} : \mathbf{X})^T 
    &= (\mathbf{X} : \mathcal{E}^T)^T 
    \quad \because \mathcal{E} : \mathbf{X} = \mathcal{E}^{TT} : \mathbf{X} = \mathbf{X} : \mathcal{E}^T \\
    &= (\mathbf{X} : \mathcal{E})^T \\
    &= \mathbf{X} : \mathcal{E}^t \quad \because (\mathbf{X} : \mathcal{E})^T = \mathbf{X} : \mathcal{E}^t \\
    &= \mathbf{X} : \mathcal{E} \\
    &= \mathbf{X} : \mathcal{E}^T \\
    &= \mathcal{E}: \mathbf{X}  \quad \because \mathcal{E}: \mathbf{X} = \mathbf{X} : \mathcal{E}^T
    \end{aligned}
    $$

- Let $\mathcal{F} = \{ \mathbf{F}_1, \mathbf{F}_2, \cdots \mathbf{F}_n^2 \}$ 를 임의의 $\mathbf{L}in^n$ 의 Basis라 하고 $\mathcal{F}' = \{ \mathbf{F}^1, \mathbf{F}^2, \cdots \mathbf{F}^{n^2} \}$ 를 Dual Basis라고 하자. 그러면
  $$
  \mathbf{F}_p :\mathbf{F}^q = \delta_p^q, \quad p,q = 1,2, \cdots n^2
  $$

  그러므로  Theorem 5.1에 의하ㅣ여 다음과 같다. 
  $$
  \mathcal{E} = \mathcal{E}^{pq} \mathbf{F}_p \odot  \mathbf{F}_q
  $$

  - $\mathbf{F}_p \odot  \mathbf{F}_q$는 Fourth order Tensor이다.  $\mathbf{F}_p \odot  \mathbf{F}_q : \mathbf{X} = \mathbf{F}_p  (\mathbf{F}_q : \mathbf{X})$ 에 의해 $\mathbf{F}_p$의 방향과 $\mathbf{F}_q$의 고유 값이 들어 있는 형태라고 생각하면 된다.  

  - 또한 Super symmetry 이므로 
    $$
    \mathcal{E}^{pq} = \mathcal{E}^{qp}
    $$

- Super symmetry fourth tensor의 Eigen Value 형식 
  $$
  \mathcal{E}: \mathbf{M} = \Lambda \mathbf{M}, \quad \mathcal{E} \in \mathbf{S}sym^n, \; \mathbf{M} \neq 0
  $$
  where $\Lambda, M \in \mathcal{S}sym^n, \mathbf{M} \neq 0$. 
  
    - Spectral Decomposition 
      $$
      \mathcal{E} = \sum_{p=1}^m \Lambda_p \mathbf{M}_p \odot \mathbf{M}_p, \;\; m = \frac{1}{2}n(n+1)
      $$
      and $\mathbf{M}_p : \mathbf{M}_q = \delta_{pq}, \;\; p,q = 1, 2, \cdots, m$. 
  

## Special Fourth-Order Tensors
- Identity Tensor
  $$
  \mathcal{I} : \mathbf{X} = \mathbf{X}, \;\; \forall \mathbf{X} \in \mathbf{L}in^n
  $$

  - 다음의 특성을 가지게 된다.
    $$
    \begin{aligned}
    \mathbf{X} : \mathcal{I} &= \mathbf{X}, \;\;\forall \mathbf{X} \in \mathbf{L}in^n \\
    \mathcal{I} &= \mathbf{I} \otimes \mathbf{I} \\
    \mathcal{I} &= \mathbf{g}_i \otimes \mathbf{g}^i \otimes \mathbf{g}_j \otimes \mathbf{g}^j \\
    \mathcal{I} : \mathcal{A} &= \mathcal{A} : \mathcal{I}, \quad \forall \mathcal{A} \in \mathcal{L}in^n
    \end{aligned}
    $$

    - Eigen PRojection $\mathbf{P}_i, \; i=1, 2, \cdots, s$ 가 존재하는 경우 다음과 같다. 
      $$
      \mathcal{I} = \sum_{i,j=1}^s \mathbf{P}_i \otimes \mathbf{P}_j
      $$
      

- Transpotion Tensor

  - Matrix의 Transposition은 Linear mapping으로 볼 수 있으며 이는 Fourth order tensor에 의한 변환으로 볼 수 있다.이 텐서를 $\mathcal{T}$ 라고 하면
    $$
    \begin{aligned}
    \mathcal{T} : \mathbf{X} &= \mathbf{X}^T, \quad \forall \mathbf{X} = \mathbf{L}in^n \\
    \mathbf{Y} : \mathcal{T} &= \mathbf{Y}^T, \quad \forall \mathbf{Y} = \mathbf{L}in^n
    \end{aligned}
    $$

  - 정의에 의해 $\mathcal{T} = \mathcal{T}^T$ 그리고, $\mathcal{T} = \mathcal{I}^T$ 

- Spherical, Deviation and Trace projection Tensors 
  - Spherical, Deviation Tensors
    $$
    sph\mathbf{A} = \mathcal{P}_{sph}: \mathbf{A}, \quad dev\mathbf{A} = \mathcal{P}_{dev}: \mathbf{A}
    $$
    
   - Trace Projection Tensors
     $$
     \mathbf{I} \odot \mathbf{I} : \mathbf{X} = \mathbf{I} tr \mathbf{X}, \quad \forall \mathbf{X} \in \mathbf{L}in^n
     $$
     
   - Spherical, Deviation projection Tensors 
     $$
     \mathcal{P}_{sph} = \frac{1}{n} \mathbf{I} \cdot \mathbf{I} \quad
     \mathcal{P}_{dev} = \mathcal{I} - \frac{1}{n} \mathbf{I} \cdot \mathbf{I}
     $$
     
   - 연산특징
     $$
     \mathcal{P}_{sph} : \mathcal{P}_{sph} = \mathcal{P}_{sph}, \quad 
     \mathcal{P}_{dev} : \mathcal{P}_{dev} = \mathcal{P}_{dev}, \quad
     \mathcal{P}_{dev} : \mathcal{P}_{sph} = \mathcal{P}_{sph} : \mathcal{P}_{dev} = \mathcal{O}
     $$
     
     



 













