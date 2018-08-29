Python 학습노트
==============
Python을 학습하는 과정을 적은 학습 노트이다.
본 학습은 하나의 프로젝트를 완성하기 위하여 Googling을 통하여 Python을 학습하고 이를 확장하기 위함이다.

## Python의 Matrix 연산 

numpy의 dot 연산이 수학에서 사용되는 일반적인 dot product와 동등하다.
numpy에는 inner product도 존재하나, inner product에만 정확하게 표시될 뿐,  matrix Dimension 까지 정확하게 맞춰주기 위해서는 dot 연산을 사용하여야 한다.

~~~python
import numpy as np

basicDim = (np.size(X), 1)
dX      = np.reshape(Xn - X,   basicDim)  # General 2x1 Vector (tuple)
dG      = np.reshape(gr_n - gr,basicDim)  # basicDim = (2,1)
dGdGt   = np.dot(dG, dG.T)                # 2x1, 1x2 => 2x2
test_0  = np.asscalar(np.dot(dX.T, dG))   # 1x2, 2x1 => 1x1 => scalar
~~~

np.dot 연산을 통해 임의 Dimension의 Matrix 연산을 자유자재로 할 수 있다.
단 Scalar의 경우는 추가로 numpy의 **asscalar** 명령을 사용하여야 한다.  이로서 array에서 데이터를 빼내서 일반적인 scalar 값으로 사용할 수 있다.

~~~python
>>> test_0  = np.dot(dx.T, dG)                # 1x2 2x1   
>>> test_0
array([[ 0.00285346]])
>>> test_1 = np.asscalar(test_0)
>>> test_1
0.0028534634171432
~~~

## Lambda 함수를 사용한 C/C++ ? 표현
C/C++ 에서는 ?를 사용하여 if 문을 쉽게 표현하는 방법들이 있다.
~~~cpp
Param = (condition)? True:False;
~~~
Python에서 이를 수행하는 것은 Lambda 함수를 응용하여 가능하다.

~~~python
lambda 인자: 표현식
~~~

이때 인자의 수는 여러개가 들어살 수 있다. 예를 들어 인자 두개를 받아 더하는 경우 

~~~python
>>> (lambda x, y: x+y)(10, 20)
30
~~~

이를 사용하면 다음과 같이 C/C++ 와 동등한 코드를 만들 수 있다.

~~~python
Param = (lambda x: x == True)(condition)
~~~

실제 코드에서는 다음과 같이 사용되었다.
~~~python
>>> param = 1001
>>> Qtest = (lambda x: x == 1000)(param)
>>> Qtest
False
~~~

## VSCode 에서의 Debugging 

MS VSCode 가 Pycharm 보다 시각적으로 더 좋은 것 같아. 이것을 사용하고 있다.
그런데 디버깅의 경우 환경 맞추기가 쉽지 않다.
다행스럽게도 환경을 맞추는 방법을 알아내었다.

다음과 같이 수행한다.

![](https://drive.google.com/uc?id=1_LyZFLDLUvBN6oZ__2PT-NEQNAoOp4rz)

이 그림에서, 먼저 디버그 방식을 **Python:Current File**로 맞추고 다음 옆의 **기어** 표시를 클릭하여  **launch.json** 가 나오도록 한다.

VScode에서의 Debugging 환경 세팅은 **launch.json** 를 적절히 수정하여  Setting 하는 것을 의미한다.

**launch.json**를 open 하면 

~~~
    // IntelliSense를 사용하여 가능한 특성에 대해 알아보세요.
    // 기존 특성에 대한 설명을 보려면 가리킵니다.
    // 자세한 내용을 보려면 https://go.microsoft.com/fwlink/?linkid=830387을(를) 방문하세요.
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}"
        },
        {
            "name": "Python: Attach",
            "type": "python",
            "request": "attach",
            "localRoot": "${workspaceFolder}",
            "remoteRoot": "${workspaceFolder}",
            "port": 3000,
~~~

와 같이 파일이 나온다. 여기에서 첫번째  brace로 둘러쌓인 부분이 **Python: Current File**의 Setting이다. 여기에 다음과 같이 Argument를 추가한다.  예를 들어 Debugging을 위한 파라키터 인자 값이   -a 4 -s 2 -df 1 -q 100 라 하면 

~~~
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}"
            "args": [ "-a", "4",  "-s", "2", "-df", "1", "-q", "100"]
~~~

앞에서는 없었던 **args** 라는 Parameter를 넣고 위의 그림처럼 인자를 넘겨주면 된다. 그러면 해당 인자 값을 받은 Python 에 대한 Debugging이 수행된다. 이는 Vscode의 Terminal을 통해서도 확인할 수 있다.

~~~
d:\Document\Github\python01\nlp_class01.py -a 4 -s 2 -df 1 -q 100 
~~~

## Markdown Web Drive에서 Image 삽입
### Drop box
원래 공유를 하면 다음과 같은 링크가 주어진다
~~~html
    https://www.dropbox.com/s/randomrandom/image.png?dl=0
~~~
여기서 https 를 http로, www를 dl로 변경하고 ?dl=0 을 제거한다
~~~html
    http://dl.dropbox.com/s/randomrandom/image.png
~~~
위 링크를 사용하면 마크다운에서 Dropbox 공유 이미지를 불러올 수 있다

### Google drive

그림에 대한 Shareable Link를 받으면 다음과 같다.
~~~html
https://drive.google.com/open?id=1_LyZFLDLUvBN6oZ__2PT-NEQNAoOp4rz
~~~
여기에서 **open**을 **uc** 로 바꾸어 준다.

~~~html
https://drive.google.com/uc?id=1_LyZFLDLUvBN6oZ__2PT-NEQNAoOp4rz
~~~




