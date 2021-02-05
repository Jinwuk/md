## Application to Shell Theory 

###  Geometry of the shell continuum
일단 기존과 같이 다음과 같은 3차원 Euclidean space를 가정하자.

$$
\mathbf{r} = \mathbf{r}(t^1, t^2), \quad \mathbf{r}\in \mathbb{E}^3
\label{3.102}
$$

그리고 다음 그림과 같이 Closed curve $C$에 bounded 되어 있다고 가정하자.,
아래 그림처럼 Shell Continuum이 정의되어 있다고 가정하면 

<img src="http://jnwhome.iptime.org/img/research/2020/tensor_006.png" style="zoom: 50%;" />
$$
\mathbf{r}^* = \mathbf{r}^*(t^1, t^2, t^3) = \mathbf{r}(t^1, t^2) + \mathbf{g}_3 t^3
\label{3.103}
$$

여기서 $\mathbf{g}_3$ 는 $\mathbf{g}_3 = \frac{\mathbf{g}_1 \times \mathbf{g}_2}{\| \mathbf{g}_1 \times \mathbf{g}_2 \|}$ 으로 정의되며, $-h/2 \leq t^3 \leq h/2$ 이다. 

- 식 $\eqref{3.102}$ 은 Shell의 중간에 있는 (그림상의 노란색 부분)의 surface를 의미한다 

식 $\eqref{3.102}, \eqref{3.103}$을 사용하여 thickness coordinate 를 계산한다.

$$
\mathbf{g}_{\alpha}^* = \mathbf{r}_{, \alpha}^* = \mathbf{g}_{\alpha} + t^3 \mathbf{g}{3, \alpha} = (\delta_{\alpha}^{\rho} - t^3 b_{\alpha}^{\rho}) \mathbf{g}_{\rho}, \quad \alpha=1, 2
$$

where (3.79)에서 
$$
b_{\alpha}^{\beta} = b_{\alpha \rho} g^{\rho \beta} = - \Gamma_{3 \alpha \rho} g^{\rho \beta} = - \Gamma_{3 \alpha}^{\beta}, \quad \alpha, \beta = 1, 2
$$

$$
\begin{aligned}
\mathbf{g}_3^{*} 
&= \frac{\mathbf{g}_1^* \times \mathbf{g}_2^*}{\| \mathbf{g}_1^* \times \mathbf{g}_2^* \|} 
= \mathbf{r}^*_{, 3} 
= \mathbf{g}_3 \\

g_{\alpha \beta}^* 
&= \mathbf{g}_{\alpha}^* \cdot \mathbf{g}_{\beta}^* 
= (\delta_{\alpha}^{\rho} - t^3 b_{\alpha}^{\rho})(\delta_{\beta}^{\rho} - t^3 b_{\beta}^{\rho}) \mathbf{g}_{\rho} \cdot \mathbf{g}_{\rho} \\
&= \delta_{\alpha}^{\rho} \delta_{\beta}^{\rho} \mathbf{g}_{\rho} \cdot \mathbf{g}_{\rho}  - t^3 (b_{\alpha}^{\rho} \delta_{\beta}^{\rho} + b_{\beta}^{\rho} \delta_{\alpha}^{\rho}) \mathbf{g}_{\rho} \cdot \mathbf{g}_{\rho} + (t^3)^2 b_{\alpha}^{\rho} b_{\beta}^{\rho} \mathbf{g}_{\rho} \cdot \mathbf{g}_{\rho} \\
&= \delta_{\alpha}^{\rho} \mathbf{g}_{\rho} \cdot \delta_{\beta}^{\rho} \mathbf{g}_{\rho}  - t^3 (b_{\alpha \beta} + b_{\beta \alpha}) + (t^3)^2 b_{\alpha \rho} g^{\rho \rho} b_{\beta \rho} g^{\rho \rho} , \quad \because b_{\alpha}^{\rho} \delta_{\beta}^{\rho} = b_{\alpha \beta}, \; \rho = 1, 2\\
&= g_{\alpha \beta}  - 2 t^3 b_{\alpha \beta} + (t^3)^2 b_{\alpha \beta} b_{\beta}^{\alpha} 
\end{aligned}
$$

위 식에서 $(b_{\alpha}^{\beta})^T = b_{\alpha \beta}$  로 생각하면 된다. 즉, **위 첨자는 Vector 표시의 Column  표기, 아래 첨자는 Row 표기로 생각하면 된다. 그러므로 Inner Product에 의해 하나는 Row 표기 하나는 Column 표기가 된다.**
$$
g^* = [\mathbf{g}_1^* \mathbf{g}_2^* \mathbf{g}_3^*] = [(\delta_1^{\rho} - t^3 b_1^{\rho}) \mathbf{g}_{\rho} (\delta_2^{\gamma} - t^3 b_2^{\gamma}) \mathbf{g}_{\gamma} \mathbf{g}_{3}]
\label{3.106}
$$
식 $\eqref{3.106}$ 에서 $\rho$ 는 1, 2, $\gamma$는 1, 2 이나 $\rho$ 와 겹치지 않는 값이다. 고로 $\eqref{3.106}$ 는
$$
g^* 
= (\delta_1^{\rho} - t^3 b_1^{\rho}) (\delta_1^{\rho} - t^3 b_1^{\rho}) g e_{\rho \gamma 3}
= g |\delta_{\beta}^{\alpha} - t^3 b_{\beta}^{\alpha}|
= g [1 - 2t^3 H + (t^3)^2 K]
\label{3.107}
$$

- 식 $\eqref{3.107}$ 에서 $| \cdot |$는 Determinent를 의미. 
- **Shell Shifter**  : 식 $\eqref{3.107}$ 에서 유도

$$
\mu = \frac{g^*}{g} = 1 - 2t^3 H + (t^3)^2 K
$$

### Internal Force variables 

<img src="C:\Users\Admin\OneDrive\문서\Work_Fig\Research_picture\2020\tensor_3-6.png" style="zoom:80%;" />



그림 3.6과 같이 $t^{\alpha}$에서 $t^{\alpha} +\Delta t^{\alpha}$ 만큼 변화가 있는 surface에서 Internel Force 를 생각한다. 

- Force vector $\mathbf{f}^{\alpha}$ and Couple vector $\mathbf{m}^{\alpha}$를 Surface 중앙에 다음과 같이 정의된다고 가정하자.

$$
\mathbf{f}^{\alpha} = \int_{-h/2}^{h/2} \mu \mathbf{\sigma} \mathbf{g}^{*\alpha} dt^3, \quad \mathbf{m}^{\alpha} = \int_{-h/2}^{h/2} \mu \mathbf{r}^* \times (\mathbf{\sigma} \mathbf{g}^{*\alpha}) dt^3, \quad \alpha=1, 2
\label{3.110}
$$

- $\sigma$는 Coordinate line $t^3$ 에서 $t^{\beta}$ 로의 Boundary Surface $A^{(\alpha)}$ 에서의 Cauchy Stress Tensor 이다. 
- Unit normal  to this boundary surface 

$$
\mathbf{n}^{\alpha} 
= \frac{\mathbf{g}^{*\alpha}}{\|\mathbf{g}^{*\alpha}\|} 
= \frac{\mathbf{g}^{*\alpha}}{\sqrt{g^{*\alpha \alpha}}} 
= \frac{g^{*}}{\sqrt{g^{*}_{\beta \beta}}} \mathbf{g}^{*\alpha}. \quad \beta \neq \alpha = 1, 2
$$

여기에서,  $\mathbf{g}^{* \alpha} \cdot \mathbf{g}^{*}_{\alpha}  = \mathbf{g}^{* \alpha} \cdot \mathbf{g}_{3}  = 0$ 이므로
$$
g^{*\alpha \alpha} = \frac{g^*_{\beta \beta}}{g^{*2}}, \quad \beta \neq \alpha = 1, 2
$$
이를 Cauchy Theorem  $ \mathbf{t} = \mathbf{ \sigma}  \mathbf{n}$ 에 따라 식 $\eqref{3.110}                                                                                                                                                                                                                                                                                                                                                                                                                                  $ 에 대입하여 풀면 









