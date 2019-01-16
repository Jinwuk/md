#  LaTex, EPS, Markdown and Others

## Draw.io 에서 EPS 만들기

Draw.io 는 매우 가볍게 만들어진 훌륭한 SW 이기는 하나, **EPS** 파일을 만드는데는 제한과 문제가 많다.

### SVG 파일 생성시 문제점 

해당 Solution에서 SVG 파일을 만들경우, Diagram에서 사용한 LaTeX 수식들이 제대로 Diagram에포함되지 못한다. 이것이 결정적이다.  따라서 많은 경우 Windows에서 제공하는 폰트를 사용하여 직접 수식을 만들어야 가장 근접한 SVG 파일을 Export하여 받을 수 있다.

문제는 그렇게 하여도 만들어진 SVG 파일이  Chrome 등 최신의 HTML-5 지원 브라우저에서만 제대로 보이고, IE에서는 여전히 깨져서 보인다는 점이다. 

더 큰 문제는 대부분의 SVG 파일을 EPS로 전환시켜 주는 솔루션들이 대부분 IE 기준의 SVG 파일 출력으로 EPS 파일을 생성한다는 것이다.  (Adobe Distiller등은 사용해 보지 않아서 알 수 없다.)

### 대안은 고해상도 PNG에서 EPS변환

1. 일단 별도의 Draw.io를 만든다. 
2. 변환을 필요로 하는 Diagram block을 새로운 Draw.io Diagram으로 만든다.
3. File -> Export 에서 Advanced를 선택하고 거기에서 Zoom을 필요한 만큼 확대시킨다.
  * 300% 정도가 좋은 결과를 얻었다. (파일이 너무 크면 변환등에 시간이 많이 걸린다.)
  * **Transparancy** Check를 해제하고 백색으로 만든다. 그렇지 않은 경우 배경이 검은색으로 나타나기 때문에 다른 문제를 일으킬 수 있다. (추가 테스트 필요) 

4. PNG 파일의 EPS 변환을 지원하는 사이트에서 EPS 파일로 변환한다 대표적인 것은 두가지가 있다.
  * VectorMagic 
      * SIte는 다음과 같다.
          * https://ko.vectormagic.com/
      * PNG 등 그림 파일을 인고지능 기법을 사용하여 선 등을 추출한 다음 이것을 기반으로 새로 Vector Graphics를 그리는 방식이다.
      * 기존 Converter들이 단순하게 그림 자체를 Vector Graphics 로 만드는 데 반하여 훨씬 진보적이다.
      * 가격이 비싸고 해당 파일을 다운로드 하여 보려고 하면 Online 결제를 해야만 한다.
   * ConvertIO
      * Site는 다음과 같다.
         * https://convertio.co/kr/svg-eps/
      * 다른 사이트들 보다 빠르고 확실하게 변환해준다.
      * 단, 24시간에 10분만 무료로 변환이 가능하다. 따라서, 단 시간에 여러 테스트를 하기에는 시간이 모자르다.
      * 그러나 브라우저를 바꿔 접속하면 계속 변경이 가능하다.

## Solution : PDF to EPS through GIMP

해당 문제는 Draw.io 에서도 충분히 인지하고 있었다. 그런데, Draw.io에서 해결책을 내놓았는데 방법은 PDF 로 Export 하고  GIMP 에서 EPS 로 변환하는 것이었다.  이때, Draw.io에서 **Crop**을 선택해 주어야 하나의 페이지가 아닌 선택 영역만 정확하게 PDF가 되어 생성된 EPS도 정확한 해상도를 가지게 된다. (그렇제 않으면 종이 한 페이지의 영역을 차지한다.)

이 방법의 장점은 다음과 같다.

- Draw.IO 에서 LaTeX을 사용하여 수식을 Diagram에 넣어도 정확하게 EPS에 LaTeX 본래의 폰트가  Vector Font로 삽입된다.  오히려 Draw.io 보다 더 깔끔하다.

이 방법의 단점과 해결책은 다음과 같다.

- Draw.io 기본 형식으로 PDF를 만들면 해상도가 떨어진다 (DPI가 떨어져 Vector Graphics 임에도 충분히 좋은 해상도의 그림이 나오지 않는다.)  따라서 이 경우에는 Export->Advanced 옵션을 실행하여 PDF로 내보낸다.
- 이 경우, PDF는 전체 Diagram이 나오므로 Draw.io 에서 Diagram Block을 따로 뗴어내어 신규 Diagram으로 만든 후, GIMP에서 Cropping을 실행 (보통의 경우 사각 영역 선택하여 클립보드로 복사) 하고 새로 만들기에서 클립보드에서 새로 만들기를 선택한 다음 해당 레이어를 EPS로 내보내기를 선택하여 생성하면 된다.
- 최종적으로 LaTex 컴파일을 수행해 본 결과 매우 깔끔한 PDF 결과를 이 방법으로 얻을 수 있었다.




### 결론
현재 발견한 가장 좋은 방법은 PDF to EPS 이다. 이를 GIMP를 통해 수행한다.  현재까지 가장 좋은 솔루션을 제공한다. 


## MathJax 에서 수식에 번호 넣기

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



