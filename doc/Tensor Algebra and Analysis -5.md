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

