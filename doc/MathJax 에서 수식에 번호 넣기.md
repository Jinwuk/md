#MathJax 에서 수식에 번호 넣기

LaTeX 과는 달리 Mathjax에서 수식 번호는 일일이 따로 넣어 주어야 한다. 물론 css를 통해 Style을 지정하여 만들어 줄 수도 있으나 때로는 수식에 번호가 없는 편이 편할 수도 있기 때문에 수식 번호를 일일이 넣어 주는 방법으로 해결을 해보도록 한다. 만일, 이후에 Style을 통하여 수식 번호를 넣게 되면 그 방법을 추가로 포시팅 한다. 

다음과 같은 경우가 있다고 가정하자
- - - 
기본 동역학 방정식이 다음과 같이 주어졌을 때

$$
dX_t = a(x,t)dt + \sigma^2(x,t) dW_t \label{basic01}\tag{1}
$$

함수 $f(X_t, t)$에 대한 방정식은 ($\ref{basic01}$) 에 대하여 다음과 같다.

$$
df(X_{ t },t)=\frac { \partial f }{ \partial t } +\left( a(x,t)\frac { \partial f }{ \partial x } +\frac { 1 }{ 2 } \sigma^{ 2 } (x,t) \frac { \partial^{ 2 } f }{ \partial x^{ 2 } }  \right) dt+\sigma (x,t)\frac { \partial f }{ \partial x } dW_{ t } \tag{2}
$$

- - - 

이때 LateX 코드는 다음과 같다.
~~~tex
dX_t = a(x,t)dt + \sigma^2(x,t) dW_t \label{basic01}\tag{1}

df(X_{ t },t)=\frac { \partial f }{ \partial t } +\left( a(x,t)\frac { \partial f }{ \partial x } +\frac { 1 }{ 2 } \sigma^{ 2 } (x,t) \frac { \partial^{ 2 } f }{ \partial x^{ 2 } }  \right) dt+\sigma (x,t)\frac { \partial f }{ \partial x } dW_{ t } \tag{2}
~~~

즉 번호는 **\\tag{1}** 에서 일일이 증가 시켜주고 \\label 도 붙이는 방식이면 된다. 문제는 HarooPad에서는 \\label이 제대로 출력 되지 않는다는 점이다 때때로 제대로 출력되기는 하지만, 이는 명백히 HarooPad의 Bug 이다. 
조만간 수정이 되는대로 관련 내용을 업데이트 한다.