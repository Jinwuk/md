Diffusions for Global Optimization
====
~~~bibtex
@article{geman1986diffusions,
  title={Diffusions for global optimization},
  author={Geman, Stuart and Hwang, Chii-Ruey},
  journal={SIAM Journal on Control and Optimization},
  volume={24},
  number={5},
  pages={1031--1043},
  year={1986},
  publisher={SIAM}
}
~~~

## Introduction 

A function $U : \mathbf{R}^n \rightarrow \mathbf{R}$ , and $x_t \in \mathbf{R}^n$ 에 대하여 다음 방정식은 Local minima를 만족한다.

$$
\frac{dx_t}{dt} = - \nabla U(x_i)
$$

그런데 이 방정식이 Global minima를 만족하기 위해서는 "climb hill"을 해야 하며 이를 위해서는 random fluctuation이 필요하여 다음의 방정식의 형식이 된다.

$$
dx_i = -\nabla U(x_i) dt + \sqrt{2 T} dw_t
$$

여기서 $w_t$는 **Standard Brownian motion** 이고 $T$는 temperature로서 random fluctuation의 크기를 제어하는 항이다. 

이 경우, under suitable conditions on $U$에서 $x_t$는 weakly 하게 다음 density의 Gibb's 분포를 가진 어떤 equilibrium으로 수렴한다.

$$
\pi_T (x) = \frac{1}{Z_T} \exp \left( \frac{-U(x)}{T} \right) \;\;\; \text{where} \;\;\; Z_t = \int_{\mathbf{R}^n} \exp \left( \frac{-U(x)}{T} \right) dx
$$

As $T \rightarrow 0$, $\pi_{T}$는 the global minima of $U$에 수렴한다.  그러므로, 낮은 temperature에서 equillibrium은 global minima 근방에서 찾을 수 있을 것으로 예상한다. 

하지만, 불행히도, 수렴을 위한 시간은 $\frac{1}{T}$에 대하여 exponentially 하게 증가한다. 결국 $T=T(t) \downarrow 0$. 

본 논문에서 제공하는 것은$x_t$의 $U$의 global minima로의 Weak convergence에 대한  $U$와 $T(t)$ sufficient condition이다.  보다 정확히 말하면,
$U:[0,1]^n \rightarrow \mathbf{R}$이 **unique global minimum** at $x = \xi$이기 위해서는 $T(t) = c/\log(2+t)$ 임을 sufficiently smooth $U$에 대하여 밝히는 것이며 이떄 다음과 같이 수렴하는 것이다.
$$
P(| x_t - \xi | < \varepsilon) \rightarrow 1
$$
for all $\varepsilon > 0$ and **all starting point**.

Previous Work  에서 다음과 같은 사실을 얻는다.
- fixed $T$에서 Metropolis algorithm은 Equlibrium에 대하여 Gibbs Distribution을 가진다.[1]

## Statement of Result
Given a real-valued function $U$ **on the unit cube** 

$$
U : [0,1]^n \rightarrow \mathbf{R}
$$

and an annealing schedulr $T(t) \downarrow 0$, we define a diffusion $x_t$

$$
dx_t = -\nabla U(x_t) dt + \sqrt{2T(t)} dw_t
$$
where $w_t$ is Standard Brownian motion



## References
[1] https://www.bibsonomy.org/bibtex/25bdc169acdc743b5f9946748d3ce587b/lopusz
~~~bibtex
@article{Metropolis1953,
  added-at = {2010-08-02T15:41:00.000+0200},
  author = {Metropolis, Nicholas and Rosenbluth, Arianna W. and Rosenbluth, Marshall N. and Teller, Augusta H. and Teller, Edward},
  biburl = {https://www.bibsonomy.org/bibtex/25bdc169acdc743b5f9946748d3ce587b/lopusz},
  doi = {10.1063/1.1699114},
  interhash = {b67019ed11f34441c67cc69ee5683945},
  intrahash = {5bdc169acdc743b5f9946748d3ce587b},
  journal = {The Journal of Chemical Physics},
  keywords = {MonteCarlo},
  number = 6,
  pages = {1087-1092},
  publisher = {AIP},
  timestamp = {2010-08-02T15:41:00.000+0200},
  title = {Equation of State Calculations by Fast Computing Machines},
  url = {http://link.aip.org/link/?JCP/21/1087/1},
  volume = 21,
  year = 1953
}
~~~



