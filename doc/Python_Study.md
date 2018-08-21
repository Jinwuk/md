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

