Quantized ADAM : Based on Riemmanian Geometry 
===
[TOC]

## Fundamental Idea 

- 현재, 일반적인 ADAM을 분석해보면 $\beta$ 값이 너무커서 사실상 Momentum 효과는 사라지고 현재 Gradient에 대한 Approximated RMS와 비슷한 수준이다.
- 그렇다면, 좀 더 분석을 하여 $v_t$ 자체를 Quantization 하는 것은 어떠할까? 
- 물론 Squart 부분으로 나누어야 하는 문제점이 있지만 현재 $v_t$에 대한 RMS는 사실상  Gradient vecrto의 길이와 거의 유사하다.
- 따라서,  Riemann 기하 기반의 일반적 Gradient 를 적용하면 ADAM optimizer는 굳이 Gradient의 자승의 Expectation value - momentum를 사용하지 않아도 된다. 

### Object Function 

- Pytorch Loss 함수는 Cross Entropy 함수를 사용한다.
  - https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropy#torch.nn.CrossEntropyLoss

- 현재 Object function은 다음과 같다, 
$$
\mathcal{L}(x, k) = -\log \frac{\exp x(k)}{\sum_j \exp x(j)}
\label{eq01}
$$
where $k$ is class index. 

- 어쩄든 Loss Function은 Cross Entropy Function으로 생각하면 된다. 
  - 즉, KL - Divergence로 생각하면 된다. 
  - https://theeluwin.postype.com/post/6080524  등 

### 일반적인 ADAM Optimizer
- 일반적인 ADAM Optimizer는 다음과 같다. 




###  ADAM Optimizer의 Riemmanian Analysis



### ADAM Optimizer vs Quantized ADAM Optimizer 





## Note

### Sigmoid 함수의 적분

- Sigmoid 함수 $f(x) = \frac{1}{1 + \exp(-x)}$ 미분은 잘 알려진 대로 다음과 같다. 

$$
\frac{df}{dx} = \frac{-\exp(-x)}{(1+ \exp(-x))^2} = f(x)(1 - f(x))
$$

- 적분에 대하여 생각해 보면 
  $$
  \frac{d}{dx} \log (1 + \exp (-x)) = \frac{d\log (1 + \exp (-x))}{d(1 + \exp (-x))} \frac{d(1 + \exp (-x))}{dx} = \frac{- \exp(-x)}{1 + \exp (-x)} = f(x) - 1
  \label{sig-02}
  $$
  
- 식 $\eqref{sig-02}$를 다시 정리하고 적분하면
  $$
  \begin{aligned}
  d\log (1 + \exp (-x)) = (f(x) - 1) dx \\
  \int d\log (1 + \exp (-x)) = \int (f(x) - 1) dx 
  \end{aligned}
  $$

- 그러므로 Sigmoid 함수의 적분은 다음과 같다.

$$
\int f(x) dx  = \log (1 + \exp (-x)) + x
$$



