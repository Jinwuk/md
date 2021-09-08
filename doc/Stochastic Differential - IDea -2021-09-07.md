Stochastic Differential - Idea -2021-09-07
===

## Motivation 



Suppose that an objective function $f(w_t, x_t) \in \mathbf{R}$ .  Since we focus on the differentiation of $f(w_t, x_t)$ with respect to $w_t$,  we abbreviate $f(w_t, x_t)$ as $f(w_t, x_t)$ by assumption of fixed $x_t$ 



대부분의 Stochastic Algorithm을 보면 다음과 같은 형태로 미분 형식을 정의한다.

In most learning equations based on the stochastic gradient descent,  many researchers define the differential form to weight as follows:
$$
\begin{aligned}
w_{t+1} = w_t - \varepsilon \nabla f(w_t) 
&\Rightarrow \frac{w_{t+1} - w_t}{\varepsilon} = -\nabla f(w_t) \\
&\Rightarrow \lim_{\varepsilon \rightarrow 0} \frac{w_{t+1} - w_t}{\varepsilon} = -\nabla f(w_t) \\
&\Rightarrow dw_t = -\nabla f(w_t)
\end{aligned}
$$

그러나 이것은 너무 뜬금 없는 너무 심각한 비약이라고 생각한다.  정확하게는 다음과 같이 되어야 한다.

However, I think that it is a serious logical jump. I suggest the following relations illustrating the differential form more correctly.
$$
\begin{aligned}
&w_{t+1} = w_t - \varepsilon \nabla f(w_t) \\
&\Rightarrow w_{t+1} = w_t - \int_{0}^{1} \varepsilon \nabla f(w_{t+\tau}) d\tau \\
&\Rightarrow w_{t+1} = w_t - \int_{t}^{t+1} \varepsilon \nabla f(w_{s}) ds, \quad \because s = t+1 \\
&\Rightarrow w_{\bar{t}} = w_{\bar{t}-1} - \varepsilon \int_{\bar{t}-1}^{\bar{t}} \varepsilon \nabla f(w_{s}) ds \\
&\Rightarrow \frac{dw_{\bar{t}}}{d \bar{t}} = \frac{d}{d\bar{t}} w_{\bar{t}-1} - \varepsilon \int_{\bar{t}-1}^{\bar{t}} \varepsilon \nabla f(w_{s}) ds \\
&\Rightarrow \frac{dw_{\bar{t}}}{d \bar{t}} = -\varepsilon \frac{d}{d\bar{t}} \int_{\bar{t}-1}^{\bar{t}} \varepsilon \nabla f(w_{s}) ds \\
&\Rightarrow dw_{\bar{t}} = -\varepsilon \nabla f(w_{\bar{t}}) d \bar{t}
\label{eq02}
\end{aligned}
$$

그러므로 다음과 같아야 한다.
$$
dw_t = -\varepsilon \nabla f(w_t) dt
\label{eq03}
$$
만약,  $\eqref{eq02}$ 에서  0과 1사이에  $\tau+ k \cdot \Delta \in \mathbf{R}[\tau, \tau+1], \; k \in \mathbf{Z}[0, N-1], N > 0$ 을 정의할 수 있고 $\Delta = N$ 하 하고  여기에 어떤 Noise $B_{\tau}$가 존재하여 $\mathbb{E}_{\tau, \tau+1} B_{\tau} = 0$  라고 하자. 그러면 위 식 $\eqref{eq02}$ 는 다음과 같다.

Let $N > 0$ such that $\tau+ k \cdot \Delta \in \mathbf{R}[\tau, \tau+1], \; k \in \mathbf{Z}[0, N-1]$  and  there exists a noise $B_{\tau}$ such that $\mathbb{E}_{\tau, \tau+1} B_{\tau} = 0$ , then  We can obtain the following alternative form of $\eqref{eq02}$ :
$$
\begin{aligned}
&w_{t+1} = w_t - \varepsilon \nabla f(w_t) \\
&\Rightarrow w_{t+1} = w_t - \varepsilon \nabla f(w_t) + \mathbb{E}_{\tau, \tau+1} B_{\tau}, \quad \because \mathbb{E}_{\tau, \tau+1} B_{\tau} = 0\\
&\Rightarrow w_{\bar{t}} = w_{\bar{t}-1} - \varepsilon \int_{\bar{t}-1}^{\bar{t}} \varepsilon \nabla f(w_{s}) ds + \int_{\bar{t}-1}^{\bar{t}} B_{\bar{\tau}} p(x_{\bar{\tau}}, \bar{\tau}) d\bar{\tau}
\label{eq04}
\end{aligned}
$$
, where $p(x_{\bar{\tau}}, \bar{\tau})$ is the probability density to $B_{\bar{\tau}}$

Assume that the probability density of $B_{\bar{\tau}}$ is a constant value $\sigma$ , the n we can rewrite the $\eqref{eq04}$ such that
$$
\begin{aligned}
&w_{\bar{t}} = w_{\bar{t}-1} - \varepsilon \int_{\bar{t}-1}^{\bar{t}} \varepsilon \nabla f(w_{s}) ds + \int_{\bar{t}-1}^{\bar{t}} B_{\bar{\tau}} p(x_{\bar{\tau}}, \bar{\tau}) d\bar{\tau} \\

&\Rightarrow 
dw_{\bar{t}} = -\varepsilon \nabla f(w_{\bar{t}}) d \bar{t} + \sigma B_{\bar{\tau}} d\bar{\tau}
\end{aligned}
$$
Let $B_{\bar{\tau}} d\bar{\tau}$ be a standard stochastic differential $dB_{\bar{t}}$ , and substitute $\bar{t}$ with $t$ , we can get the following stochastic differential equation for a learning equation 
$$
dw_t = -\varepsilon \nabla f(w_t) dt + \sigma dB_t
\label{eq06}
$$
그렇다면, 이러한 추론은 타당한 것인가? 

Is such an inference valid? 



Suppose that the $f(w_t, x_t)$ is a quadratic function, and the input $x_t$ is factorized as $x_t = \mathbb{E} x_t  + \eta_t$  for all $t$, where $\eta_t$ is a random variable with a zero expectation such that $\mathbb{E} \eta = 0$. 

Since $f(w_t, x_t)$ is a quadratic function, we can let it as $ v_t^T Q v_t + b v_t$ where $v_t = w_t + x_t$ . 

The gradient of $f$ is $Q v_t + b$ . 
$$
\frac{d f}{dw_t} = \frac{d f}{dv_t} \frac{dv_t}{dw_t} = (Q v_t + b) \left(
\begin{matrix}
I \\
0
\end{matrix}
\right) = Q v_t + b
$$
Therefore,
$$
Q v_t + b = Qw_t + Q x_t + b = Qw_t + Q(\mathbb{E} x_t + \eta_t) + b = Q(w_t + \mathbb{E} x_t) + b + Q \eta_t
$$
By assumption, we set $f(w_t, x_t)$ is defined on a fixed $x_r$ and we regard $\mathbb{E} x_t$ as a fixed $x_t$, we can obtain the gradient of $f(w_t)$ to $w_t$ as $Q(w_t + \mathbb{E} x_t) + b $ . 

Since $\mathbb{E}x_t$ is a constant value corresponding to $t$, it is valid inference.  Thereby, 
$$
\frac{df}{dw_t}(w_t, x_t) = \nabla f(w_t) + Q \eta_t
\label{eq09}
$$
, for a quadratic objective function.

식 $\eqref{eq09}$ 에서,  $Q \eta_t = B_t$ 로 놓으면,  관련 추론은 모두 성립한다.

하지만, $Q = \frac{B_t}{\eta_t}$ 혹은  $\eta_t= \frac{B_t}{Q}$   즉, $\eta_t = Q^{-1} B_t$ 가 된다.   그렇다면.  $\mathbb{E} \eta_t^2 = Q^{-2}\mathbb{E} B_t^2 = Q^{-2}$    관련하여 조금 더 생각해보자. 

 

##  Another Inference 

만일, $\eta_t = \sqrt{Q^{-1}} B_t$ 라고 하면,   $\sqrt{Q} B_t$ 형식으로 나타나게 될 것이며 $\sigma = \sqrt{Q}$ 로 놓을 수 있다.

이렇게 되면, 모든 부분에서 Stochastic Differential 의 Formation이 만족된다.  

그런데 이것은, $\eta_t$의 Variance가 $\sqrt{Q^{-1}}$ 이어야 한다는 것이다. 딱 그만큼만,  정확하게 Cover 된다는 의미이다. 











