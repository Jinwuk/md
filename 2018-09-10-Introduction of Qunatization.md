Introduction of Qunatization
====

Mendeley 에 있는 Signal Processing 논문들 중, Qunatization 관련 파일들만 모아놓았다.

다음 Wikipedia의 Site를 참조한다.

[Wikipedia Link](https://en.wikipedia.org/wiki/Quantization_(signal_processing)#cite_note-Bennett-4)

Qunatization의 기본 모델은 1948년도 Benette논문을 기본으로 한다.
여기서 Qunatization의 Error 모델은 White Noise로 기본적인 Modeling을 한다.
그리고 관련 논문들 중 특히 Karlsson 논문이 수학적인 Method를 다양하게 갖추고 있다.

```math
\begin{aligned}
d_{x_i}\bar{f}(w_{t_i},x_i )
&= \nabla_{x_i} \bar{f} (w_{t_i},x_i ) dX_i + \frac{1}{2} \frac{\partial^2 f}{\partial x_i^2} (w_{t_i},x_i ) dX_i \cdot dX_i \\

&= \nabla_{x_i} \bar{f} (w_{t_i},x_i ) dB_i + \frac{1}{2} \sigma^2 Tr \left(\frac{\partial^2 f}{\partial x_i^2} (w_{t_i},x_i ) \right)dt_i
\end{aligned}
```

$$
\begin{aligned}
d_{x_i}\bar{f}(w_{t_i},x_i )
&= \nabla_{x_i} \bar{f} (w_{t_i},x_i ) dX_i + \frac{1}{2} \frac{\partial^2 f}{\partial x_i^2} (w_{t_i},x_i ) dX_i \cdot dX_i \\

&= \nabla_{x_i} \bar{f} (w_{t_i},x_i ) dB_i + \frac{1}{2} \sigma^2 Tr \left(\frac{\partial^2 f}{\partial x_i^2} (w_{t_i},x_i ) \right)dt_i
\end{aligned}
$$

 


[1] Bennett, W. R. (1948), "Spectra of Quantized Signals". Bell System Technical Journal, 27: 446-472. doi:10.1002/j.1538-7305.1948.tb01340.x
[2] Karlsson, Rickard, Gustafsson, Fredrik. "Filtering and estimation for quantized sensor information", 2018 
