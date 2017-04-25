Introduction of Mathematics to Machine Learning
==================

![bg original](http://jnwhome.iptime.org/img/ETRI_BG.png)

##### 방송미디어연구소 실감AV연구그룹 
###### 석 진욱 (jnwseok@etri.re.kr)
###### 2017.04.xx 

---
# Very Short Course of Linear Algebra
<!-- page_number: true -->
기본적인 개념과 용어를 중심으로

---
## Vector and Matrix
### Vector Notations
Vector 표시할 때 보통, $\mathbf{x}$ 를 사용하나, 그렇지 않은 경우
- n-Dimensional Column Vector : $x \in \mathbb{R}^n$ 
- n-Dimensional Row Vector    : $x^T$ or $x'$, $\forall x \in \mathbb{R}^n$

### Matrix
$n$ by $m$ Matrix : $A \in \mathbb{R}^{n \times m}$
- 입력의 차원은 $m$, 출력의 차원은 $n$ 
- 입력은 $m$-Dimensional Row Vector를 받아 $n$-Dimensional Column Vector로 나간다는 의미 
- Alternative Form (Definition of Mapping) $A : \mathbb{R}^m \rightarrow \mathbb{R}^n$
---
![VS01](http://jnwhome.iptime.org/img/AI/VS01.png)

---
### Semigroup
- 결합법칙이 만족되는 Mapping을 "반군(Semigroup" 이라 한다.
- Semigroup의 연산은 일반적인 4칙연산이 아닌 **Operator**를 사용해야만 Closed 된 연산을 정의할 수 있다.
##### Schrodinger Equation
$\imath \hbar \frac{\partial}{\partial t} \Psi(r,t) = -\frac{\hbar}{2m}\nabla^2 \Psi(r,t)+V(r,t)\Psi(r,t)$
$\imath \hbar \frac{\partial}{\partial t} \Psi(r,t) = \hat{H} \Psi(r,t)$
where $V(r,t)$ is a potential Energy, $\Psi(r,t)$ is a wave function, and $\hat{H}$ is a **Hamiltonian Operator**
##### Fokker Plank Equation
$\frac{\partial p}{\partial t}(y,t,x,s) = -\frac{\partial }{\partial y} \mu(y,t) p(y,t,x,s) + \frac{1}{2} \frac{\partial^2}{\partial y^2} \sigma(y,t) p(y,t,x,s)$
$\frac{\partial p}{\partial t}(y,t,x,s) = \mathcal{L}_{y,t}^* p(y,t,x,s)$

---
##### Semigroup Operator의 특성
$\mathcal{L}_{y, t+s}^* = \mathcal{L}_{y, t}^*\mathcal{L}_{y, s}^*$
$\mathcal{L}_{y, t+(s+u)}^* = \mathcal{L}_{y, t}^*\mathcal{L}_{y, s}^*\mathcal{L}_{y, t}^*\mathcal{L}_{y, u}^*=\mathcal{L}_{y, (t+s)+u}^*$
- 교환법칙이 성립하면 Commutative Operator : $\mathcal{L}_{y, t+s}^* = \mathcal{L}_{y, s+t}^*$
- Semigroup은 대체로 실함수의 경우 **확률밀도 함수**의 형태로 나타나거나 Wave 함수의 형태로 나타난다. 
- wave 함수의 경우 직교특성이 있으면 좌표계를 줄 수 있다.
   - Fourier Transform 
##### Group의 특성
- **Group** 혹은 **군**:일반적인 형태의 **연산** 정의할 수 있다.
- 연산표등을 통해 Group의 특성을 알 수 있으며 이를 통해 유사한 군의 특성을 대역적으로 알 수 있다.
- 대표적인 Group 
   - Linear Algebraric Group
   - Lie Group
---
##### General Linear Group $GL(n,F)$
- $n \times n$ Matrix로 Inverse Matrix가 정의되는 행렬
- Identity Matrix (G2), Inverse Matrix (G3) 존재
##### Special Linear Group $SL(n,F)$
- $GL(n,F)$의 subgroup 으로 Determinent가 1, -1 인 행렬
- Permutation,회전 변환, Eigen Vector로 이루어진 행렬
##### Orthogonal Group $O(n,F)$
$O(n,F) = \{Q \in GL(nF)|Q^TQ = QQ^T = I\}$
- $SL(n,F)$의 Subgroup으로 Eigen Vector로 이루어진 행렬, 회전변환
##### Special Orthogonal Group $SO(n,F)$
- 회전변환, $SO(1)=1, SO(2) = S, SO(3) =S^2$
---
##### Ring and Field
- 곱셈에 대해 G0, G1(Close and Semigroup)까지 만족되면 **Ring**
  - 덧셈에 대해서는 G0, G1, G2, G3 만족 & F1, F2 만족 
- Ring 중에서 곱셈에 대해 G2(Identity), G3(Inverse) 존재하면 **Field**
##### Vector Field
- For a vector valued function $v_x : S \rightarrow \mathbb{R}^n$ in a standard Euclidean coordinates $(x_1, x_2, \cdots x_n)$
- Vector는 Function이 아닌 단순한 Scalar value의 모임이며 Vector space는 반드시 원점을 포함한다.
$\forall V, W \in C^k$ on ring $S$, and $\exists f \in \mathbb{R}$ such that $f:\mathbb{R}^n \rightarrow \mathbb{R}$
$(fV)(p) = f(p)V(p)$
$(V+W)(p) = V(p) + W(p)$
---
##### Hilbert Space
###### Inner Product Axiom
$\forall u, v \in V$ and $\alpha, \beta \in K$, where $V$ is vector space, $K$ is (scalar) field
$\langle u, v \rangle = \langle v,u \rangle$, 
$\langle u, u \rangle \geq 0$ with equality if  and only if (iff) $u = 0$
$\langle \alpha u + \beta v, w \rangle = \alpha \langle u,w \rangle + \beta \langle v, w \rangle$
- Inner Product Axiom을 만족하는 Vector Space를 Inner Product Space라고 한다.
- Inner Product Axiom을 통해 Norm을 정의할 수 있는 공간을 거리공간(Metric Space)라고 한다.

---
- 임의의 Cauchy Sequence $\{x_n, 1 \leq n < \infty \}$에 대해 $\sum_{n=1}^{\infty} x_n^2 < \infty$ 이면 $l_2$ (거리)공간이라고 한다. 
	- $l_2$ 공간에서 Inner Product는 $\langle x, y \rangle = \sum_{n=1}^{\infty} x_n y_n < \infty$
- 임의의 closed interval $[a,b]$에 대해 **Lebesgue Measurable** i.e. $\int_a^b f^2(x) dx < \infty$ 이면 $L_2[a,b]$ (Lebesgue Squared Measurable)  Space on the interval [a,b] 라고 한다.
    - $\langle  f, g \rangle = (b-a)^{-1} \int_a^b f(x)g(x) dx$
- $\mathbb{R}$ 에 대하여 **Lebesgue Measurable** i.e. $\int_{-\infty}^{\infty} f^2(x) dx < \infty$ 이면 $L_2$ (Lebesgue Squared Measurable) Space  
    - $\langle  f, g \rangle = \int_{-\infty}^{\infty} f(x)g(x) dx$

위의 3가지 조건 중 하나를 만족하는 Complete Metric Space (완비 거리공간)을 **Hilbert Space**라고 한다.

---
## Matrix Computation

###### Vector Computation  
$\forall x, y \in \mathbb{R}^n$, $\langle x, y \rangle = \sum_{k=1}^{n} x_k y_k$ 
###### Matrix Computation 
$\forall x, \in \mathbb{R}^n$ and $A \in \mathbb{R}^{m \times n}$ $Ax \in \mathbb{R}^m$ 의 m번째 성분 값 $(A x)_m$
$$
(A x)_m = \sum_{k=1}^{n} A_{m, k}x_k
$$
- Convolution ?
$f = \sum_{k=1}^{n} A(m - k) x(k)$ 

---
##### Affine Transform
For $A \in \mathbb{R}^{n \times m}$, **1차 변환으로서 평행이동을 보존**하는 변환. 
$$f = Ax + b, \;\; \forall x \in \mathbb{R}^n, b \in \mathbb{R}^m$$
##### Tensor
2-Dimension 이상의 Matrix 라 생각하면 된다. (0차 텐서는 스칼라, 1차텐서는 벡터, 2차텐서는 Matrix, 보통은 리만 기하에서 곡률텐서)

##### Tensor Flow
Tensor Field로 이루어진 Bundle을 Tensor Flow라고 한다.

---
#### Vector Analysis (Short Course)
- Vector 미분의 정의는 항상 Scalar(분모)를 Vector(분자)로 편미분 하는 것 : $\forall x,y \in \mathbb{R}^n$, $a(x), b(x) \in \mathbb{R}^m$
![eq002](http://jnwhome.iptime.org/img/AI/EQ002.svg)



---
### Matrix Notation
- Matrix는 결국 입력과 출력의 인과 관계를 적은 것 
- i 번째 입력이 들어와 j 번째 출력으로 나간다면?
   - 2 번째 입력값 $\alpha$가 들어와 3번째 출력값 $\beta$로 만들고 싶다.

$$
\begin{bmatrix}
0 \\
0 \\
\beta \\
0
\end{bmatrix} =
\begin{bmatrix}
* & 0 & * & * \\ 
* & 0 & * & * \\ 
* & \frac{\beta}{\alpha} & * & * \\ 
* & 0 & * & *  
\end{bmatrix}
\begin{bmatrix}
0 \\
\alpha \\
0 \\
0
\end{bmatrix}
$$
- i 번째 입력에 대한 j 번째 출력사이의 비용은 $\gamma$ 

$$
\gamma = [0, 0, 1, 0]
\begin{bmatrix}
* & * & * & * \\ 
* & * & * & * \\ 
* & \gamma & * & * \\ 
* & * & * & *  
\end{bmatrix}
\begin{bmatrix}
0 \\
1 \\
0 \\
0
\end{bmatrix}
$$

---
#### Othogonality and Symmetry

---
#### Example : Linear Regression

---
# Dynamic Programming
Matrix 에서 사용법에서 i 번째 입력에 대한 j 번째 출력사이의 비용을 정의하는 방법에서 출발해 본다.
- Reference
  - D. Kirk, '**Optimal Contro Theory : An Introduction**', Prentice Hall, 1970 
  - G. Chen, etc, '**Linear Stochastic Control System**', CRC, 1995 

---

- 경로문제
![Optimal_Path.png](http://jnwhome.iptime.org/img/AI/Optimal_Path.png)

---
- 경로문제의 다양한 접근방법
  - Continuous Case 
     - Stage에 있는 Destination이 무한개가 있다고 하면? 
     - 각 경로의 Cost를 어떤 Parameter의 함수로 놓는다. 
![Optimal_Path_2.png](http://jnwhome.iptime.org/img/AI/Optimal_Path_03.png)

---
   - Euler-Largrane 방정식
     - Continuous 경우에 최적경로를 찾기 위한 고전역학적 방법     
![Optimal_Path_4.png](http://jnwhome.iptime.org/img/AI/Optimal_Path_04.png)

---
- 경로문제의 응용
   - 고전역학에서 Largrangian & Hamiltonian
     - Contiuous Case 에서 $\mathcal{L} = (F + \lambda g)(x, \dot{x}, t)$ 
     - 특별히, 위 Largrangian에서 제한조건이 $\dot{x} = f(x,t)$ 이면 Hamiltonian : $\mathcal{H} = F + \lambda \dot{x}$
         - Hamiltonian의 Cost Function : $J = h(x(t_f), t_f) + \int_{t=0}^{t=t_f} F(x(t), u, t) dt$
   - 배치문제 
     - 경로문제에서 Stage에 중요도를 주던가, Cost를 Stage에 따라 다르게 주게 되면 배치문제가 된다.
         - Unit Commit 문제, 반도체 Cell 배치 문제, 등등
   - Q-Learning
     - Reward/Penalty : Stage간 Cost가 극단적인 경우 
--- 
   - Dynamic Programming
      - 위 경로 문제가 실은 Dynamic Programming 기본 개념
      - 경로 문제를 응용하면 다양한 최적화 문제를 모두 DP 문제로 만들 수 있다. -> Wikipedia 참조
      - Bellman's Optimality Principle (DP 문제 해결의 기본 원칙) 
         - 초기상태가 어떠했든 남은 경로는 반드시 최적해야 함
![BE](http://jnwhome.iptime.org/img/AI/Bellman01.png)
      - Dynamic Programming 문제 해결을 위한 방법
      	 - Discrete Case : Bellman Equation 
      	    - Necessity Condition 
      	    - HJB 방정식 보다 일반적인 해법 
         - Continuous Case : Hamilton-Jacobi-Bellman Equation 
            - Necessity and Sufficient Condition
            - Hamiltonian이 개입된다.
---
- Bellman (Recurrence) Equation 
   - Problem   
![BE](http://jnwhome.iptime.org/img/AI/EQ003.svg)
![BE](http://jnwhome.iptime.org/img/AI/EQ004.svg)   
   - Necessity Condition for Optimality
<a href="http://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\LARGE&space;J_{N-k,&space;N}^*&space;=&space;\min_{u(N-k)}&space;\{F_D(x(N-k),&space;u(N-k))&space;&plus;&space;J_{N-(K-1),&space;N}^*&space;(a_D(x(N-k)),&space;u(N-k))\}" target="_blank"><img src="http://latex.codecogs.com/svg.latex?\dpi{150}&space;\LARGE&space;J_{N-k,&space;N}^*&space;=&space;\min_{u(N-k)}&space;\{F_D(x(N-k),&space;u(N-k))&space;&plus;&space;J_{N-(K-1),&space;N}^*&space;(a_D(x(N-k)),&space;u(N-k))\}" title="\LARGE J_{N-k, N}^* = \min_{u(N-k)} \{F_D(x(N-k), u(N-k)) + J_{N-(K-1), N}^* (a_D(x(N-k)), u(N-k))\}" /></a>
   - Note : To Understand the EQ easily
![BE](http://jnwhome.iptime.org/img/AI/EQ005.svg)       

--- 
- Hamilton-Jacobi-Bellman Equation
   - Constraint
        ![BE](http://jnwhome.iptime.org/img/AI/EQ006.svg)
   - Problem
        ![BE](http://jnwhome.iptime.org/img/AI/EQ007.svg)
   - Necessity and Sufficient Condition
      - Bellman Style
        ![BE](http://jnwhome.iptime.org/img/AI/EQ008.png)
      - Hamiltonian Style
        ![BE](http://jnwhome.iptime.org/img/AI/EQ009.svg)
        ![BE](http://jnwhome.iptime.org/img/AI/EQ010.png)
---
- How to apply the Dynamic Programming to Learning 
   - Bellman 방정식을 보면, 적분항/SUM 항 내부의 Cost를 매 순간 최소화 시켜야 한다.
      - $J$를 Reward($\max$)/Penalty($\min$)로 두면 강화학습
      - Cost 함수만 잘 정의하면 비 지도 학습 
         -  오차 최소화 : Model과 실제값의 차이 줄이기
         -  Activation/Non-Activation 
         -  원하는 출력/원치 않는 출력 <- 결국 Cost 안에 Desire값을 함수 형태로 구현  
   - 신경망은 DP문제를 비 정통적인 방법으로 풀겠다는 것
     - 정통적인 방법으로 풀기 힘든 경우..
         - 미분으로 풀기 어려울 때 (Monte-Carlo ?)
         - 기존 방법이 잘 안될때 (응?)
---
![Optimal_Path_4.png](http://jnwhome.iptime.org/img/AI/Optimal_Path_05_02.png)

---

# Beginning Of Neural Network (Alternative Aspect)
기존, Perceptron 기반의 접근이 아니라, B. Kosko의 Associative Memory 및 Quadratic Optimization의 관점에서 신경망이 어떻게 발전했는가를 알아본다.
- Reference
  - S. Haykin, '**Neural networks and Learning Machines**', 3rd ed. Pearson, 2009 
  - A. Cichocki, etc, '**Neural networks for Optimization and Signal Processing**', Wiley, 1993 
  - S. Kung, '**Digital Neural Networks**', Prentice Hall, 1993

---
## Associative Memory
Fax에 이미지 전송하는 과정에서 이미지가 깨지는 경우가 있었음
- 깨진 이미지를 복원하는 문제 : MRF (Markov Random Field) 
   - 주변 데이터 연관성과 Baysian추정으로 영상 데이터 생성
   - 연산이 복잡하고, 연관성 모델 경우의 수가 많다.
      - 기본적으로 Monte Carlo / MCMC 연산
      -  후에 Boltzmann Machine으로 발전
- 좀 더 쉽게 영상 복원하는 방법은 없을까?
   - 주변 영상/혹은 원래 영상을 학습하여 깨진 영상 복원?
   - Associative Memory의 기본 아이디어 
---
![AAM](http://jnwhome.iptime.org/img/AI/aam001.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/aam002.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/aam003.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/aam004.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/aam005.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/aam006.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/aam007_1.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/aam007_2.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/aam009.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/aam010.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/aam011.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/aam012.png)

---
## From Associative Memory to Neural Networks
### Diagram of Operation in Associative Memory
![AAM](http://jnwhome.iptime.org/img/AI/0007.png)
- 만일 입력 데이터의일부를 데이터의 Label이라 한다면?
---
![AAM](http://jnwhome.iptime.org/img/AI/0008.png)
- 새로운 Neural Network이 탄생 !!!
![AAM](http://jnwhome.iptime.org/img/AI/0009.png)

---
### Short Conclusion
- 그 후, 신경망 응용은 입력데이터와 Object Function을 어떻게 만드느냐의 문제라는 것을 사람들이 인식함
   - 입력데이터 문제는 결정론적 전처리를 적용하다가 이 자체도 신경망으로 처리하겠다. (LeCun 교수의 CNN)
   - Object Function 을 어떻게 놓느냐에 따라, 여러가지 형태의 신경망 도출 (게임이론, DP, MRF...)
- Hopfield Network은 Neural Network이 기존 비선형 최적화 문제와 같다는 것을 보이고 사라짐. 
- 미분 기반 최적화 vs 확률론적 최적화
   - 직관적으로, 그리고 확실한 답과, 연산속도는 미분기반
   - Global Optimization은 확률론적 최적화 but 많은 연산
      - RISC 기반 GPU 발전으로 상당히 해소되고 있음

---
# Stochastic Approach
- Short Course of Stochastic Process 
- Short Course of Baysian Estimation
- Monte-Carlo Method
- Kullback-Leibler Divergence
- Boltzmann Machine
- and More.. (Martingale and Stochastic Differnetial Analysis)

---
![SP](http://jnwhome.iptime.org/img/AI/SP000.png)

---
## Short Course of Stochastic Process 
![SP](http://jnwhome.iptime.org/img/AI/SP001.png)

---
![SP](http://jnwhome.iptime.org/img/AI/SP002.png)

---
![SP](http://jnwhome.iptime.org/img/AI/SP003.png)

---
![SP](http://jnwhome.iptime.org/img/AI/SP004.png)

---
![SP](http://jnwhome.iptime.org/img/AI/SP005.png)

---
![SP](http://jnwhome.iptime.org/img/AI/SP007.png)

---
![SP](http://jnwhome.iptime.org/img/AI/SP008.png)

---
![SP](http://jnwhome.iptime.org/img/AI/SP009.png)

---
![SP](http://jnwhome.iptime.org/img/AI/SP010.png)

---
## Short Course of Baysian Estimation
![BI](http://jnwhome.iptime.org/img/AI/BI001.png)

---
![BI](http://jnwhome.iptime.org/img/AI/BI002_01.png)

---
![BI](http://jnwhome.iptime.org/img/AI/BI003.png)

---
![BI](http://jnwhome.iptime.org/img/AI/BI004.png)

---
![BI](http://jnwhome.iptime.org/img/AI/BI005.png)

---
![BI](http://jnwhome.iptime.org/img/AI/BI006.png)

---
![BI](http://jnwhome.iptime.org/img/AI/BI007.png)

---
![BI](http://jnwhome.iptime.org/img/AI/BI008.png)

---
![BI](http://jnwhome.iptime.org/img/AI/BI009.png)

---
## Monte-Carlo Method
### Brief
$$
I = \int F(w)p(w|\mathcal{D}) dw = \mathbb{E}(F(w)|\mathcal{D})
$$
이렇게 구하고 싶은데..실제로 $p(w|\mathcal{D})$는 잘 모르겠고 하니.. 그냥 밑에 처럼 $L$개의 Sample을 뽑고 $F(w_i)$를 근사화 시키는 방법이다.
Strong/Weak Law of Large Numbers에 기반한다. 
$$
I \approx \frac{1}{L}\sum_{i=L}^L F(w_i)
$$
- 대표적인 예 : 원주율 구하기 (Wiki등 여러군데에 설명되어 있음)

---
![BI](http://jnwhome.iptime.org/img/AI/MC001.png)

---
![BI](http://jnwhome.iptime.org/img/AI/MC002.png)

---
![BI](http://jnwhome.iptime.org/img/AI/MC003.png)

---
![BI](http://jnwhome.iptime.org/img/AI/MC004.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/MCMC01.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/MCMC02.png)
 - Simulated Annealing은 이후 Global Optimization 알고리즘과 연결된다.

---
## Kullback-Leibler Divergence
![KL](http://jnwhome.iptime.org/img/AI/KL001.png)

---
![KL](http://jnwhome.iptime.org/img/AI/KL002.png)

---
![KL](http://jnwhome.iptime.org/img/AI/KL-Gauss-Example.png)

---
![KL](http://jnwhome.iptime.org/img/AI/KL003.png)

---
![KL](http://jnwhome.iptime.org/img/AI/KL004.png)

---
![KL](http://jnwhome.iptime.org/img/AI/KL005.png)

---
## Boltzmann Machine (Application of Stochastic Approach)


---
# Nonlinear Programming
- Convex and the First, Second Order Necessity Condition
- Basic Concept of Gradient Descent 
- How to make Optimization Problem
- Global Analysis
- Reference
  - David Luenberger, '**Linear and Nonlinear Programming**', Springer, 2003 
  - A.L. Pressini, etc, '**The Mathematics of Nonlinear Programming**', Springer, 1988 
  - M. Aoki, '**Introduction to Optimization Techniques**', Macmillan, 1971
  - J. Seok, http://jnwhome.iptime.org/?cat=17
---
![NLP](http://jnwhome.iptime.org/img/AI/NLP001.png)

---
![NLP](http://jnwhome.iptime.org/img/AI/NLP002.png)

---
![NLP](http://jnwhome.iptime.org/img/AI/NLP003.png)

---
## Basic Concept of Gradient Descent 
![NLP](http://jnwhome.iptime.org/img/AI/NLP004.png)

---
![NLP](http://jnwhome.iptime.org/img/AI/NLP005.png)

---
![NLP](http://jnwhome.iptime.org/img/AI/NLP006.png)

---
![NLP](http://jnwhome.iptime.org/img/AI/NLP007.png)

---
![NLP](http://jnwhome.iptime.org/img/AI/NLP008.png)

---
![NLP](http://jnwhome.iptime.org/img/AI/NLP009.png)

---
![NLP](http://jnwhome.iptime.org/img/AI/NLP010.png)

---
![NLP](http://jnwhome.iptime.org/img/AI/NLP011.png)

---
## How to make Optimization Problem
### Object Function의 설계
- 문제가 정적인 것인가 동적인 것인가를 생각한다. 
  - 시간에 따라 문제의 형태가 변하는 경우 
    - Dynamic Programming
  - 정적인 형태로 입력 데이터의 변화에 출력이 종속인 경우 
    - 일반 최적화
- 무엇을 최소화/최대화 시킬 것인가?
  - 최소화/최대화 시켜야 할 것을 정한다. 
  - Metric/Measure 함수를 정한다.
    - Norm : 단순 거리 
    - Inner Product : Matrix를 통해 Vector간 Cost
    - Probability : Bianry로 출력이 나오는 경우 등 
---
- 문제에 적용되는 제한조건을 생각한다.
  - 등식 제한 조건인가 부등식 제한 조건인가.
    - 등식 제한 조건의 경우 일반 장정식 형태로 제한 조건이 나타난다.
  - 시간에 종속적인 제한 조건인가?
    - 시간에 따른 시스템 변화가 있는 제한 조건인가?
    - 시간 종속적인 문제는 Hamiltonian으로 푼다.
    - 시간 비 종속적인 문제는 Largrangian으로 푼다.
  - Largrangian/Penalty Function
    - Largrangian을 사용하게 되면 수학적 해석
    - Penalty Function을 사용하면 보다 수치 해석적
---
- 문제에 사용되는 Object Function
  - 일반적으로 Quadratic 함수가 되도록 한다.
     - Norm의 경우도 마찬가지.(Banach, Frechet Space상의 문제가 정의되기도 한다.)
     - 2차 함수는 보통 에너지의 개념
        - 운동 에너지(시간 종속)인가
        - 위치 에너지(데이터 종속)인가? 
  - 확률 알고리즘을 사용하려면? 
  	- Quadratic 함수가 exp 안에 들어간다.
  	- 최소화 문제의 경우 이론적 최소값이 1이 되고, Marginal 값이 0이 되도록 한다.
  	    - 예 : $\max_{x} \frac{1}{Z} \exp (- \lambda (x^T Q X + bx))$
---
### Object Function을 잘 설계하는 방법
- Object Function 설계 자체가 Idea, 논문 작성의 Thema
   - Optimization text 혹은 타 분야의 관련 Text/논문을 본다.
   - 경제학 논문/Text에 많은 최적화 문제들이 제시되어 있다.
   - 산업공학의 OR의 경우 선형계획법, Particle 최적화 문제 
   - 전력분야 문제들은 각종 최적화 알고리즘을 어떻게 적용할 것인지에 대한 많은 Idea를 제공한다. 
   - 경영학/경제학 분야: Game 이론의 실제 적용사례
      - 미분게임의 경우에는 최적 제어 문제에서 나타난다.
- Object Function을 잘 설계하는 것은 결국 다른 학문 분야의 논문을 통해 아이디어를 얻는 것이 가장 빠른 길이다.
---
## Global Analysis
### Simulated Annealing의 Stochastic Differential Equation
![AAM](http://jnwhome.iptime.org/img/AI/SGD01.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/SGD02.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/SGD03.png)

---
![AAM](http://jnwhome.iptime.org/img/AI/SGD04.png)


---
![SGD06](http://jnwhome.iptime.org/img/AI/SGD06.png)

---
![SGD06](http://jnwhome.iptime.org/img/AI/SGD07.png)

---
![SGD06](http://jnwhome.iptime.org/img/AI/SGD08.png)

---
# Thank You