#Vector Qunatization and Signal Compression

[toc]
by Allen Gersho and Roberty M. Gray

- Vector Qunatization에 관련된 Study Note.
- Evernote에 최신 파일을 저장하고 Haroopad에서 해당 내용을 작성하여 html로 만든다.

## Transform Coding 
Let $X$ denote the sample vector

$$
X = (X_1, X_2, \cdots , X_k)^T
$$

- Tranform Coding의 아이디어는 적절한 Linear Transform을 사용하여 입력 Vector $X$를 $k$ component (Transform coefficient)를 가진 $Y$ 벡터로 변환 시키는 것이다.
- 이러한 변환의 결과로 Redundancy를 제거하여 최적의 bit allocation을 만드는 것이다.
- 이절의 도입부는 Transform 코딩과 관련되어 중요한 Fundamental Theory in Information Theory 의 결과와 Reference들을 포함하고 있다. 특히 Shannon's  rate-distortion theory는 중요하다,

### Definition : Overall squarred error distortion $D_{tc}$

$$
D_{tc} = \sum_{i=1}^k E(|X_i - \hat{X}_i|^2) = E (||X - \hat{X}||^2)
$$

- 양자화 에러는 $T^{-1}$ 에 의해 나타나기 때문에 적절한 Transform Matrix 의 선택도 매우 중요하다.
- 양자화 에러가 Inverse Transform 에 의해 증촉될 수 있기 때문이다.
- 특히 Transform 이 Orthogonal Matrix i.e. $T^{-1} = T^t $ 인 경우 다음이 성립한다.

$$
R_Y = E(YY^t)=TE(XX^t)T^t = T R_X T^t 
$$

$$
det R_Y = det T \cdot det R_X \cdot det T^t = det T \cdot det T^t \cdot det R_X = det R_X
$$

## Karhunen-Loeve Transform
or **Hotelling Transform**

만일 $X$가 Correlated with one another components 이지만, $Y = TX $는 Uncorrelated 이면 최고의 변환이다.
이러한 이상적인 Transform이 Karhunen-Loeve Transform이다. 

- Let $u_i$를 $R_X$의 Eigenvector라고 하고 $\lambda_i$를 이에 대한 Eigen Value라고 하자.
- Autocorrelation Matrix는 Symmetric이고 Positive semi Definite 이다.
- 또한 Self Adjoint 이다. i.e.
	- Let $(V, <,>)$ be a Euclidean vector space. An operator or endormorphism $f:V \rightarrow V $ is said to be self adjoint if $\left< f(v), w \right> = \left< v, f(w) \right>$ 
	- 즉, $ v^t R^t \cdot w =v^t R^t w, v^t R w = \left( w^t R^t v \right)^t = v^t R w = v^t R^t w$
	- Self Adjopint 이면 $\left< f(v), w \right> = \left< v, f(w) \right>$ 에서 
$$
\left< \lambda_v v, w \right> = \left< v, \lambda_w w \right> \Rightarrow (\lambda_v - \lambda_w)\cdot \left< v , w\right> = 0
$$ 
	- 즉, 임의의 서로 다른 Eigen Vector는 Orthogonal 하다.



