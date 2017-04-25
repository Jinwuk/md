Riemannian Geometry (Very Essential)

=====================

\#\# Covariant Derivative

Surface : $S \\subset \\mathbb{R}^3$ , Parameterized curve : $c : I \\rightarrow S$, and a Vector field along $c$ tangent to $S$ : $V : I \\rightarrow \\mathbb{R}^3$

이때, Vector $\\frac{dV}{dt}(t), t \\in I$ 는 일반적으로 \*\*Tangent space\*\* $T\_{c(t)}S$ 에 존재하지 않는다.

그러므로 $\\frac{dV}{dt}(t), t \\in I$ 이 \*\*Tangent space\*\* $T\_{c(t)}S$ 에 존재하도록 Orthogonal Projection 된 미분이 필요하며 그것이 \*\*Covariant Derivative\*\* $\\frac{DV}{dt}$ 이다.

1. Covariant Derivative는 the First fundamental form of $S$에 의존적

2. Covariant Derivative는 속도벡터 $c$의 미분으로서 $S$ 위의 $c$의 가속도

3. 무가속도는 \*\*Geodesic of\*\* $S$, 그리고 \*\*Gaussian Curvature of\*\* $S$는 \*\*Covariant Derivative\*\*로 표현

\#\#\#\# Definition : Affine Connection $\\nabla$

An Affine connection $\\nabla$ on a differtential manifold $M$ is a mapping

$$

\\nabla : \\mathcal{X}(M) \\times \\mathcal{X}(M) \\rightarrow \\mathcal{X}(M)

$$

\#\#\#\# Proposition : The Covariant Derivative

If $V$ is induced by a vector field $Y \\in \\mathcal{X}(M)$ i.e. $V(t) = Y(C(t))$, then

$$

\\frac{DV}{dt} = \\nabla\_{\\frac{dc}{dt}} Y

$$

\#\#\#\# Remark 1

Choosing a system of coordinates $(x\_1, \\cdot, x\_n)$ about $p$ and writing

$$

X = \\sum\_i x\_i X\_i, \\;\\;\\; Y=\\sum\_j y\_j X\_j \\;\\;\\;\\text{where}\\;\\; X\_i = \\frac{\\partial}{\\partial x\_i}

$$

then

$$

\\nabla\_X Y = \\sum\_k \\left( \\sum\_{ij} x\_i y\_j \\Gamma\_{ij}^k + X(y\_k) \\right) X\_k

$$

\#\#\#\# Remark 1-1

where \*\*Christoffel symbol\*\* $\\Gamma\_{ij}^k$ is defined as

$$

\\nabla\_{X\_i}X\_j = \\sum\_k \\Gamma\_{ij}^k X\_k

$$

1. Christoffel symbol은 결국 Tensor 이며 각 Component는 그냥 \*\*Scalar\*\* 값이다.

2. 하단의 $i, j$는 원래 입력 벡터의 Component index, 상단의 $k$는 새로운 벡터의 Component index

3. 결국 Affine Connection에 의해 만들어지는 새로운 벡터필드에 대한 Parameter 이다.

\#\#\#\#\# proof

$$

\\nabla\_X Y = \\sum\_i x\_i \\nabla\_{X\_i} \\left( \\sum\_j y\_j X\_j \\right) = \\sum\_i x\_i \\left( \\sum\_j X\_i(y\_j) X\_j + y\_j \\nabla\_{X\_i} X\_j\\right)

$$

Thus,

$$

\\nabla\_{X\_i}X\_j = \\sum\_k \\Gamma\_{ij}^k X\_k, \\;\\;\\; \\sum\_i \\sum\_j x\_i X\_i(y\_j) X\_j = \\sum\_j X(y\_j) X\_j = \\sum\_k X(y\_k) X\_k

$$

$X(y\_k)$ 는 결국 $ \\langle X, dy\_k \\rangle \\in \\mathbb{R}$ 따라서,

$$

\\sum\_{ij}x\_i X\_i(y\_j)X\_j + \\sum\_{ij}x\_i y\_j \\nabla\_{X\_i} X\_j = \\sum\_k \\left( X(y\_k) + \\sum\_{ij} x\_i y\_j \\Gamma\_{ij}^k \\right)X\_k

$$

\*\* Q.E.D \*\*

\#\#\#\# Total Covariant (Essential)

Let

$$

V = \\sum\_j v^j X\_j , \\;\\; v^j = v^j(t), \\;\\; X\_j = X\_j(c(t))

$$

Then

$$

\\frac{DV}{dt} = \\frac{D}{dt} \\left( \\sum\_j v^j X\_j \\right) = \\sum\_j \\frac{dv^j}{dt} X\_j + \\sum\_j v^j \\frac{DX\_j}{dt}

$$

이떄

$$

\\frac{DX\_j}{dt} = \\nabla\_{\\frac{dc}{dt}}X\_j, \\;\\;\\; \\frac{dc}{dt} = \\sum\_i \\frac{dx\_i}{dt}X\_i

$$

이므로

$$

\\frac{DX\_j}{dt}= \\nabla\_{\\frac{dc}{dt}}X\_j= \\nabla\_{\\sum\_i \\frac{dx\_i}{dt}X\_i } X\_j = \\sum\_i \\frac{dx\_i}{dt} \\nabla\_{X\_i}X\_j = \\sum\_i \\frac{dx\_i}{dt} \\sum\_k \\Gamma\_{ij}^k X\_k

$$

Index 정리를 하면

$$

\\begin{align}

\\frac{DV}{dt} &= \\sum\_j \\frac{dv^j}{dt} X\_j + \\sum\_j v^j \\sum\_i \\frac{dx\_i}{dt} \\sum\_k \\Gamma\_{ij}^k X\_k \\\\

&= \\sum\_j \\frac{dv^j}{dt} X\_j + \\sum\_{i,j} v^j \\frac{dx\_i}{dt} \\sum\_k \\Gamma\_{ij}^k X\_k \\\\

&= \\sum\_k \\left( \\frac{dv^k}{dt} + \\sum\_{i,j} v^j \\frac{dx\_i}{dt} \\Gamma\_{ij}^k \\right)X\_k

\\end{align}

$$

특히 \*\*Parallel Transportation\*\*이 되기 위해서는 위 Covariant Derivative가 0이 되어야 하므로

$$

\\frac{dv^k}{dt} + \\sum\_{i,j} v^j \\frac{dx\_i}{dt} \\Gamma\_{ij}^k = 0

$$

\#\# Riemannian Connection

\#\#\# Definition : Symmetric Connection

An affine connection on a smooth manifold $M$ is said to be symmetric when

$$

\\nabla\_X Y - \\nabla\_Y X = \[X, Y\], \\;\\; X, Y \\in \\mathcal{X}(M)

$$

If $X = X\_i, Y= X\_j, X\_k = \\frac{\\partial}{\\partial x\_i}$ 이면

$$

0 = \[X\_i, X\_j\] = \\nabla\_{X\_i} X\_j - \\nabla\_{X\_j} X\_i = \\sum\_k (\\Gamma\_{ij}^k - \\Gamma\_{ji}^k) X\_k

$$

에서 $\\Gamma\_{ij}^k = \\Gamma\_{ji}^k$ 이므로 Symmetric

또한 Lie Bracket이 이렇게 정의된다.

\#\#\# Levy-Civita Theorem

Given a Riemannian manifold $M$, there exists a unique affine connection $\\nabla$ satisfying the conditions :

1. is symmetric

2. is compatible with the Riemannian metric

\#\#\#\# Remark

$$

\\begin{align}

X \\langle Y, Z \\rangle &= \\langle \\nabla\_X Y, Z \\rangle + \\langle Y, \\nabla\_X Z\\rangle \\;\\;\\;\\text{....1} \\\\

Y \\langle Z, X \\rangle &= \\langle \\nabla\_Y Z, X \\rangle + \\langle Z, \\nabla\_Y X\\rangle \\;\\;\\;\\text{....2} \\\\

Z \\langle X, Y \\rangle &= \\langle \\nabla\_Z X, Y \\rangle + \\langle X, \\nabla\_Z Y\\rangle \\;\\;\\;\\text{....3}

\\end{align}

$$

1 + 2 - 3 (Very Important)

$$

\\begin{align}

X \\langle Y, Z \\rangle + Y \\langle Z, X \\rangle - Z \\langle X, Y \\rangle &= \\langle \\nabla\_X Y, X \\rangle + \\langle Y, \\nabla\_X Z \\rangle \\\\

&+ \\langle \\nabla\_Y Z, X \\rangle + \\langle Z, \\nabla\_Y X \\rangle - \\langle \\nabla\_Z X, Y \\rangle - \\langle X, \\nabla\_Z Y \\rangle \\\\

&= \\langle Y, \\nabla\_X Z \\rangle - \\langle Y, \\nabla\_Z X \\rangle + \\langle X, \\nabla\_Y Z \\rangle - \\langle X, \\nabla\_Z Y \\rangle \\\\

&+ \\langle Z, \\nabla\_X Y \\rangle - \\langle Z, \\nabla\_Y X \\rangle + 2 \\langle Z, \\nabla\_Y X \\rangle \\\\

&= \\langle \[X, Z\], Y \\rangle + \\langle \[Y,Z\], X \\rangle + \\langle \[X, Y\], Z \\rangle + 2 \\langle Z, \\nabla\_Y X \\rangle

\\end{align}

$$

$$

\\langle Z, \\nabla\_Y X \\rangle = \\frac{1}{2} \\left( X \\langle Y, Z \\rangle + Y \\langle Z, X \\rangle - Z \\langle X, Y \\rangle - \\langle \[X, Z\], Y \\rangle - \\langle \[Y,Z\], X \\rangle - \\langle \[X, Y\], Z \\rangle\\right) ...\\text{4}

$$

Assume that $X = \\frac{\\partial}{\\partial x\_i}, Y = \\frac{\\partial}{\\partial x\_j}, Z = \\frac{\\partial}{\\partial x\_k}$ then, $\[X, Y\]=\[Y,Z\]=\[Z, X\] = 0$

Symmetry and

$$

\\langle \\nabla\_Y X, Z \\rangle = \\langle \\sum\_l \\Gamma\_{ij}^l X\_l, X\_k \\rangle= \\sum\_l \\Gamma\_{ij}^l \\langle X\_l, X\_k \\rangle = \\sum\_l \\Gamma\_{ij}^l g\_{lk}

$$

Thus, The equation 4 is

$$

\\sum\_l \\Gamma\_{ij}^l g\_{lk} = \\frac{1}{2} \\left( \\frac{\\partial}{\\partial x\_i} g\_{jk} + \\frac{\\partial}{\\partial x\_j} g\_{ki} - \\frac{\\partial}{\\partial x\_k} g\_{ij} \\right)

$$

이때, the matrix $(g\_{km})$ admits an inverse $(g^{km})$ 이면.

$$

\\sum\_k \\sum\_l \\Gamma\_{ij}^l g\_{lk} g^{km} = \\frac{1}{2} \\sum\_k \\left( \\frac{\\partial}{\\partial x\_i} g\_{jk} + \\frac{\\partial}{\\partial x\_j} g\_{ki} - \\frac{\\partial}{\\partial x\_k} g\_{ij} \\right) g^{km}

$$

$l = m $ 일때만 1 이고 나머지 경우는 0 이고 좌항의 $k$와 $m$이 맞추어져야 하므로 즉, $g\_{mk} = g^{km}$ 의 경우 1이고 나머지는 0이된다. (대각성분만 남게 된다.) 그러므로

$$

\\Gamma\_{ij}^m = \\frac{1}{2} \\sum\_k \\left( \\frac{\\partial}{\\partial x\_i} g\_{jk} + \\frac{\\partial}{\\partial x\_j} g\_{ki} - \\frac{\\partial}{\\partial x\_k} g\_{ij} \\right) g^{km}

$$

\#\#\# Definition of Christoffel Symbol

$$

\\Gamma\_{ij}^m = \\frac{1}{2} \\sum\_k \\left( \\frac{\\partial}{\\partial x\_i} g\_{jk} + \\frac{\\partial}{\\partial x\_j} g\_{ki} - \\frac{\\partial}{\\partial x\_k} g\_{ij} \\right) g^{km}

$$

하지만, 실제 Christoffel 기호는 위 식으로 구할 수 없으며..(단지 정의일 뿐) 및에서와 같이 Reimannian Metric 과 Affine Connection을 사용하여야 정상적으로 풀 수 있다.

\#\#\#\# Exercise (How to get the christoffel symbol)

Consider the upper half-plane

$$

\\mathbb{R}\_{+}^2 = \\{ (x,y) \\in \\mathbb{R}^2; y &gt; 0 \\}

$$

with the metric given by $g\_{11} = g\_{22} = \\frac{1}{y^2}, g\_{12} = 0$

일때, $\\Gamma\_{11}^1 = \\Gamma\_{12}^2 = \\Gamma\_{22}^1 = 0, \\Gamma\_{11}^2 = \\frac{1}{y}, \\Gamma\_{12}^1 = \\Gamma\_{22}^2 = -\\frac{1}{y}$ 임을 보여라. (1 이 $x$ , 2가 $y$를 의미한다.)

\#\#\#\#\# Solve

$$

\\begin{align}

0 &= \\nabla\_{X\_1} g\_{11} = \\nabla\_{X\_1} \\langle X\_1, X\_1 \\rangle = 2\\nabla\_{X\_1} X\_1 \\cdot X\_1 = 2(\\Gamma\_{11}^1 X\_1 \\cdot X\_1 + \\Gamma\_{11}^2 X\_2 \\cdot X\_1) \\;\\;\\;X\_2 \\cdot X\_1 = 0 , \\;\\;\\Gamma\_{11}^1 = 0 \\\\

0 &= \\nabla\_{X\_1} g\_{12} = \\nabla\_{X\_1} \\langle X\_1, X\_2 \\rangle = \\nabla\_{X\_1} X\_1 \\cdot X\_2 + \\nabla\_{X\_1} X\_2 \\cdot X\_1 \\\\

&= \\Gamma\_{11}^1 X\_1 \\cdot X\_2 + \\Gamma\_{11}^2 X\_2 \\cdot X\_2 + \\Gamma\_{12}^1 X\_1 \\cdot X\_1 + \\Gamma\_{12}^2 X\_2 \\cdot X\_1 = \\Gamma\_{11}^2 X\_2 \\cdot X\_2 + \\Gamma\_{12}^1 X\_1 \\cdot X\_1 = \\Gamma\_{11}^2 + \\Gamma\_{12}^1 \\\\

0 &= \\nabla\_{X\_1} g\_{22} = \\nabla\_{X\_1} \\langle X\_2, X\_2 \\rangle = 2\\nabla\_{X\_1} X\_2 \\cdot X\_2 = 2(\\Gamma\_{12}^1 X\_1 \\cdot X\_2 + \\Gamma\_{12}^2 X\_2 \\cdot X\_2) \\;\\;\\because \\Gamma\_{12}^2 = 0 \\\\

0 &= \\nabla\_{X\_2} g\_{12} =\\nabla\_{X\_2} \\langle X\_1 , X\_2 \\rangle = \\nabla\_{X\_2} X\_1 \\cdot X\_2 + \\nabla\_{X\_2} X\_2 \\cdot X\_1 \\\\

&= \\Gamma\_{21}^1 X\_1 \\cdot X\_2 + \\Gamma\_{21}^2 X\_2 \\cdot X\_2 + \\Gamma\_{22}^1 X\_1 \\cdot X\_1 + \\Gamma\_{22}^2 X\_2 \\cdot X\_1 = \\Gamma\_{21}^2 + \\Gamma\_{22}^1 \\because \\Gamma\_{21}^2 = \\Gamma\_{12}^2 = 0, \\;\\; \\Gamma\_{22}^1 = 0 \\\\

\\nabla\_{X\_2} g\_{11}&= \\frac{\\partial}{\\partial y}\\left( \\frac{1}{y^2}\\right) = -2 \\frac{1}{y^3} = 2 \\cdot \\nabla\_{X\_2} X\_1 \\cdot X\_1 = 2 \\cdot \\left( \\Gamma\_{21}^1 X\_1 \\cdot X\_1 + \\Gamma\_{21}^2 X\_2 \\cdot X\_1\\right) \\\\

&= 2 \\cdot \\Gamma\_{21}^1 X\_1 \\cdot X\_1 = 2 \\cdot \\Gamma\_{21}^1 \\langle X\_1, X\_1 \\rangle = 2 \\cdot \\Gamma\_{21}^1 g\_{11} = 2 \\cdot \\Gamma\_{21}^1 \\frac{1}{y^2} \\;\\;\\therefore \\Gamma\_{21}^1 = \\Gamma\_{12}^1 = -\\frac{1}{y} \\\\

\\nabla\_{X\_2} g\_{22}&= \\frac{\\partial}{\\partial y}\\left( \\frac{1}{y^2}\\right) = -2 \\frac{1}{y^3} = 2 \\cdot \\nabla\_{X\_2} X\_2 \\cdot X\_2 = 2 \\cdot \\left( \\Gamma\_{22}^1 X\_1 \\cdot X\_2 + \\Gamma\_{22}^2 X\_2 \\cdot X\_2\\right) \\\\

&= 2 \\cdot \\Gamma\_{22}^2 X\_2 \\cdot X\_2 = 2 \\cdot \\Gamma\_{22}^2 \\langle X\_2, X\_2 \\rangle = 2 \\cdot \\Gamma\_{22}^2 g\_{22} = 2 \\cdot \\Gamma\_{22}^2 \\frac{1}{y^2} \\;\\;\\therefore \\Gamma\_{22}^2 = - \\frac{1}{y} \\\\

&\\because \\;\\;\\Gamma\_{11}^2 + \\Gamma\_{12}^1 = 0, \\;\\;\\Gamma\_{11}^2 = -\\Gamma\_{12}^1 = \\frac{1}{y}

\\end{align}

$$
