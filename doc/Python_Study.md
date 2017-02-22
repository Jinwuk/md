Python 학습노트
==============
Python을 학습하는 과정을 적은 학습 노트이다.
본 학습은 하나의 프로젝트를 완성하기 위하여 Googling을 통하여 Python을 학습하고 이를 확장하기 위함이다.

## Python의 Matrix 연산 
Python matrix 연산을 위해서 numpy 모듈을 다운 받고 설치해야 한다. 다음과 같다.

- 일반 CMD에서 다음과 같이 실행한다.
~~~
python -m pip install [numpy Module 경로와 파일 명]
~~~
실제로 다음과 같았다. 중간에 **PIP** 를 업그레이드 하라고 하여 업그레이드 하였다.

~~~
e:\[07]_Python\Study>python -m pip install "..\Module\numpy-1.12.0+mkl-cp35-cp35m-win_amd64.whl"
numpy-1.12.0+mkl-cp35-cp35m-win_amd64.whl is not a supported wheel on this platform.
You are using pip version 7.1.2, however version 9.0.1 is available.
You should consider upgrading via the 'python -m pip install --upgrade pip' command.

e:\[07]_Python\Study>python -m pip install --upgrade pip
Collecting pip
  Downloading pip-9.0.1-py2.py3-none-any.whl (1.3MB)
    100% |################################| 1.3MB 493kB/s
Installing collected packages: pip
  Found existing installation: pip 7.1.2
    Uninstalling pip-7.1.2:
      Successfully uninstalled pip-7.1.2
Successfully installed pip-9.0.1

e:\[07]_Python\Study>
e:\[07]_Python\Study>python -m pip install "..\Module\numpy-1.12.0+mkl-cp35-cp35m-win_amd64.whl"
Processing e:\[07]_python\module\numpy-1.12.0+mkl-cp35-cp35m-win_amd64.whl
Installing collected packages: numpy
Successfully installed numpy-1.12.0+mkl
~~~

