[2018-10-04] Quantization Parameter 분석
======

Test 결과 본인의 예상과 거의 들어 맞는 결과가 도출 되었으나 Difference가 예상과 달리 -1. 0. 1 의 3가지 값을 가진다. 다시말해 어떤 오차에 의해  $\Delta Q_{q_1, q_2} = Q_{q_1} - 2 \cdot Q_{q_2}, \;\;\; q_2 = q_1 + 6$ 에서

$$
-1 < N_{q_1} - 2 N_{q_2} < 2
$$

의 조건이 깨지기 때문이다. 이 원인은 기본적으로 **$|2Q_2|$ 가 $|Q_1|$ 보다 클 수가 없기 때문** 이라는 전제가 깨지기 떄문이다.   이를 분석하기 위해서  HEVC 혹은 H.264/AVC의 양자화 과정을 살펴보자. 

## 기본 양자화 방정식 

비디오 코덱에서는 기본적으로 다음과 같은 방식으로 Quantization을 수행한다. 
$$
Q(\mathcal{X}, q) = \left[ \frac{1}{q^{s}} \mathcal{X} \right] \approx \left[ \frac{1}{q^{s}} \mathcal{X} \times 2^{qbits} \right] \gg qbits

\tag{1}
$$
여기서 $bits$는 14이다. 

그런데, 직접 정수 나눗셈을 수행하는 것이 아니라,  Shift 연산을 통해 정수 나눗셈을 필요한 정확도를 가지면서 수행하는 것이 비디오 코덱의 기본이다. 이를 위해 기본적인 Quantization  범위를 살펴보면 다음과 같다. 

|qp    | 0   | 1   | 2   | 3   | 4 | 5   | 6  |
|---   |---  |---  |---  |---  |---|---  |--- |
|$q^{s}$|0.625|0.703|0.797|0.891|1.0|1.125|1.25|

이를 수식으로 나타내면 다음과 같다고 하면. 

$$
q^s = 0.625 \cdot 2^{\left \lfloor \frac{q}{6} \right \rfloor + k \cdot (q \mod 6)}
\tag{2}
$$

이떄, 비례상수 $k$를 알아보기 위해 위 식을 정리하면 다음과 같은 방정식이 유도된다.  만일 $m = (q \mod 6)$ 이라 하고 $m$ 값일 때의 Quantization Step을 $q_m^s$ 라 하면 

$$
\begin{aligned}
q_m^s &= 0.625 \cdot 2^{\left( \left \lfloor \frac{q}{6} \right \rfloor + k \cdot m \right)} \\

\frac{q_m^s}{0.625} &= 2^{\left( \left \lfloor \frac{q}{6} \right \rfloor + k \cdot m \right)} \\

\log_2 \frac{q_m^s}{0.625} &= \left \lfloor \frac{q}{6} \right \rfloor + k \cdot m \\

k &= \frac{1}{m} \cdot \left( \log_2 \frac{q_m^s}{0.625} - \left \lfloor \frac{q}{6} \right \rfloor \right)

\end{aligned}
\tag{3}
$$

위 표와 같은 결과를 얻기 위해서 $q \in \mathbf{Z}(0, 6)$ 인 경우를 알아보면 $\left \lfloor \frac{q}{6} \right \rfloor = 0$ 이므로,  이것의 Least mean square error를 최소화 하는 $\bar {k} = \frac{1}{N} \sum_{m=1}^N k_m$를 알아본다. 

다음과 같은 Python 프로그램을 통해 결과를 알아본다.

~~~python
npA = np.array([ 0.625,  0.703,  0.797,  0.891,  1.   ,  1.125])
npQA = npA[1:6]
npK = np.log2(npQA/0.625) * 1/np.array(list(range(1, 6)))

>>>sum(npK)/5
0.17093414099804752
~~~

|m     | 1   | 2   | 3   | 4   | 5   |Average|
|---   |---  |---  |---  |---  |---  |---  |
| $k$ |0.1696685|0.17536177|0.17052308|0.16951798|0.16959938|0.17093414|

식 (2)를  식(1)에 대입하여 정리하면
$$
\begin{aligned}
Q(\mathcal{X}, q) &\approx \left[ \frac{1}{0.625 \cdot 2^{\left \lfloor \frac{q}{6} \right \rfloor + k \cdot m}} \mathcal{X} \times 2^{qbits} \right] \gg qbits \\

&= \left[ \frac{8}{5} \cdot 2^{-\left( \left \lfloor \frac{q}{6} \right \rfloor + k \cdot m \right)} \mathcal{X} \times 2^{qbits} \right] \gg qbits \\

&= \left[ \frac{8}{5} \cdot 2^{qbits - k \cdot m} \mathcal{X} \times 2^{- \left \lfloor \frac{q}{6} \right \rfloor} \right] \gg qbits \\

&\approx \left[ \frac{8}{5} \cdot 2^{qbits - \left( \log_2 \frac{q_m^s}{0.625} - \left \lfloor \frac{q}{6} \right \rfloor \right)} \mathcal{X} \right] \gg (qbits + \left \lfloor \frac{q}{6} \right \rfloor) \\

&= \left[ \left( \frac{8}{5} \cdot 2^{qbits - \log_2 \frac{q_m^s}{0.625}} \right) 2^{\left \lfloor \frac{q}{6} \right \rfloor } \mathcal{X} \right] \gg (qbits + \left \lfloor \frac{q}{6} \right \rfloor) \\

&= \left[ \left( \frac{8}{5} \cdot 2^{- \log_2 \frac{q_m^s}{0.625}} \cdot 2^{qbits} \right)  \cdot 2^{\left \lfloor \frac{q}{6} \right \rfloor } \mathcal{X} \right] \gg (qbits + \left \lfloor \frac{q}{6} \right \rfloor) \\

&= \left[ \left( \frac{8}{5} \cdot \frac{0.625}{q_m^s} \cdot 2^{qbits} \right) \cdot 2^{\left \lfloor \frac{q}{6} \right \rfloor } \mathcal{X} \right] \gg (qbits + \left \lfloor \frac{q}{6} \right \rfloor) \\

&= \left[ \left( \frac{1}{q_m^s} \cdot 2^{qbits} \right) \cdot 2^{\left \lfloor \frac{q}{6} \right \rfloor } \mathcal{X} \right] \gg (qbits + \left \lfloor \frac{q}{6} \right \rfloor)
\end{aligned}
\tag{4}
$$

식 (4)에서 $q_m^s$ 의 값은 $m = (q \mod 6)$ 에 따라 표 1 과 같으므로 $\left( \frac{1}{q_m^s} \cdot 2^{qbits} \right)$ 를  $m \in \mathbf{Z}[0,5]$ 의 값으로 다음 Python 코드로 간단히 계산하면 

~~~
>>>
>>> (1/npA[0:5]) * 16384
array([ 26214.4, 23305.83214794,  20557.08908407,  18388.32772166,
        16384. , 14563.55555556 ])
~~~

이를 반올림 하여 정수화 시킨 값과 HEVC의 Quant Scale 값을 비교해 보면 다음과 같다.

|m      | 0   | 1   | 2   | 3   | 4   | 5   |
|---    |---  |---  |---  |---  |---  |---  |
| Eval  |26214|23306|20557|18388|16384|14564|
| HEVC  |26214|23302|20560|18396|16384|14564|

여기에 입력 데이터의 비트심도 (8/10 bit)와 변환 부호화 크기에 따른 변환 부호화의 스케일을 고려하여 5 비트의 추가 Shift가 필요하다.  이를 고려하면 식 (4)는 다음과 같이 쓸 수 있다. 

$$
\begin{aligned}
Q(\mathcal{X}, q) &\approx \left[ \left( \frac{1}{q_m^s} \cdot 2^{qbits} \right) \cdot 2^{\left \lfloor \frac{q}{6} \right \rfloor } \cdot 2^5 \cdot \mathcal{X} \right] \gg (qbits + \left \lfloor \frac{q}{6} \right \rfloor + 5) \\

&= \left[ \left( \frac{1}{q_m^s} \cdot 2^{qbits} \right) \cdot 2^{\left \lfloor \frac{q}{6} \right \rfloor } \cdot \bar{\mathcal{X}} \right] \gg (qbits + \left \lfloor \frac{q}{6} \right \rfloor + 5)
\end{aligned}
$$

따라서 이 값의 반올림을 한다고 가정하면  다음과 같아야 한다.

$$
\begin{aligned}
Q(\mathcal{X}, q) &\approx \left[ \frac{1}{q_m^s} \cdot 2^{qbits + \left \lfloor \frac{q}{6} \right \rfloor} \cdot \bar{\mathcal{X}} \right] \gg (qbits + \left \lfloor \frac{q}{6} \right \rfloor + 5) \\

&= \left \lfloor \frac{1}{q_m^s} \cdot 2^{qbits + \left \lfloor \frac{q}{6} \right \rfloor} \cdot \bar{\mathcal{X}} + f \right \rfloor \gg (qbits + \left \lfloor \frac{q}{6} \right \rfloor + 5)
\end{aligned}
$$

일반적인 반 올림이 되기 위해서는 $f$ 의 값이 다음과 같아야 한다. 
- 외부에서 수행되는 나눗셈 때문에 $2^{qbits + \left \lfloor \frac{q}{6} \right \rfloor + 5}$ 이 곱해져야 한다.

$$
f = \frac{1}{2} q_m^s \cdot 2^{qbits + \left \lfloor \frac{q}{6} \right \rfloor + 5}
$$
따라서,  Scale Factor $2^{qbits + \left \lfloor \frac{q}{6} \right \rfloor + 5}$ 를 생각하지 않는다면 $f$ 는 다음의 범위를 가지는 것이 정상이다. 

$$
0.3125 \leq \frac{1}{2} q_m^s \leq 0.5625
$$

HEVC에서 $f$의 값은 Intra의 경우 $f = \frac{1}{3} \cdot 2^{qbits + \left \lfloor \frac{q}{6} \right \rfloor + 5}$ Inter의 경우 $f = \frac{1}{6} \cdot 2^{qbits + \left \lfloor \frac{q}{6} \right \rfloor + 5}$ 이다. 그러므로 HEVC에서의 반올림 팩터는  Intra에서 $\frac{1}{3} = 0.3333 \cdots$,  Inter에서는 $\frac{1}{6} = 0.16666 \cdots$ 이다. 특히 Inter의 경우에는 지나치게 낮아서 제대로 반올림이 되기에는 낮은 값이다. 이는 상당히 높은 값을 가져야만 반올림되어 값이 수정됨을 의미한다.  이런 경우 다음과 같은 경우가 발생할 수 있다.

- $q_2 = q_1 + 6$ 이고  $f_{q} = f_0  \cdot 2^{qbits + \left \lfloor \frac{q}{6} \right \rfloor + 5}$ 이라 하면 
$$
\begin{aligned}
f_{q_1} &= f_0  \cdot 2^{qbits + \left \lfloor \frac{q_1}{6} \right \rfloor + 5} \\

f_{q_2} &= \cdot f_0  \cdot 2^{qbits + \left \lfloor \frac{q_2}{6} \right \rfloor + 5} = 2 \cdot f_0  \cdot 2^{qbits + \left \lfloor \frac{q_1}{6} \right \rfloor + 5} = 2 f_{q_1} 
\end{aligned}
$$

$$
\begin{aligned}
Q(\mathcal{X}, q_1) &\approx \left \lfloor \frac{1}{q_m^s} \cdot 2^{qbits + \left \lfloor \frac{q_1}{6} \right \rfloor} \cdot \bar{\mathcal{X}} + f_{q_1} \right \rfloor \gg (qbits + \left \lfloor \frac{q_1}{6} \right \rfloor + 5) = \lfloor A \rfloor \gg B\\

Q(\mathcal{X}, q_2) &\approx \left \lfloor \frac{1}{q_m^s} \cdot 2^{qbits + \left \lfloor \frac{q_2}{6} \right \rfloor} \cdot \bar{\mathcal{X}} + f_{q_2} \right \rfloor \gg (qbits + \left \lfloor \frac{q_2}{6} \right \rfloor + 5) \\ 

&=\left \lfloor \frac{1}{q_m^s} \cdot 2^{qbits + \left \lfloor \frac{q_1 + 6}{6} \right \rfloor} \cdot \bar{\mathcal{X}} + f_{q_2} \right \rfloor \gg (qbits + \left \lfloor \frac{q_1 + 6}{6} \right \rfloor + 5) \\

&=\left \lfloor 2 \cdot \frac{1}{q_m^s} \cdot 2^{qbits + \left \lfloor \frac{q_1}{6} \right \rfloor} \cdot \bar{\mathcal{X}} + 2 f_{q_1}  \right \rfloor \gg (qbits + \left \lfloor \frac{q_1}{6} \right \rfloor + 5 + 1 ) = \lfloor 2 A \rfloor \gg (B + 1)
\end{aligned}
$$



## Quantization 과 Gauss Function

식 (1)은 근사 방정식이다. 그런데, 이것이 등식이 되기 위한 조건을 생각해 보자. 

$$
\begin{aligned}
\left[ \frac{1}{q^{s}} \mathcal{X} \right]  &= \left[ \frac{1}{q^{s}} \mathcal{X} \cdot 2^{qbits} \cdot 2^{-qbits}  \right] \\

&= \left[ \frac{8}{5} \cdot 2^{\left( qbits - \left \lfloor \frac{q}{6} \right \rfloor - k \cdot m \right)} \mathcal{X} \cdot  2^{-qbits} \right] \\

&= \left[ \frac{1}{10} \cdot 2^{\left( qbits - \left \lfloor \frac{q}{6} \right \rfloor + 4 - k \cdot m \right)} \mathcal{X} \cdot  2^{-qbits} \right]

\end{aligned}
\tag{5}
$$

정수 $L(q) = qbits - \left \lfloor \frac{q}{6} \right \rfloor+ 4 = 18 - \left \lfloor \frac{q}{6} \right \rfloor \in \mathbf{Z}$  를 정의하자. 

이 경우 방정식 (5)는 다음과 같다.

$$
\begin{aligned}
\left[ \frac{1}{q^{s}} \mathcal{X} \right]  

&= \left[ \frac{2^{- k \cdot m}}{10} \cdot \mathcal{X} \cdot 2^{L(q)}  \cdot  2^{-qbits} \right] \\

&= \left[ \frac{2^{- k \cdot m}}{10} \cdot \mathcal{X} \cdot 2^{10 + 8 - \left \lfloor \frac{q}{6} \right \rfloor}  \cdot  2^{-qbits} \right]

\end{aligned}
\tag{6}
$$

식 (6) 에서 $\left \lfloor \frac{q}{6} \right \rfloor \in \mathbf{Z}[0, 8]$ 이므로 $8 - \left \lfloor \frac{q}{6} \right \rfloor = 2^p$ where $ p \in \mathbf{Z}[0, 8]$ 그리고   $km \in \mathbf{R}[0, 1)$ 이므로 

따라서  다음을 만족 시킬 수 있는 정수 $\lambda \in \mathbf{Z}$가 존재한다고 하면, 
$$
\frac{2^{- k \cdot m}}{10} \cdot \mathcal{X} \cdot 2^{10} = 2^{\lambda}
\tag{7}
$$

그리고, $\lambda + p \geq qbits \Rightarrow \lambda \geq 14 - p \in \mathbf{Z}$ 이면 근사식이 아닌 Shift에 의한 연산이 등식 조건으로 성립한다. 그러나, 식 (7) 에서 
$$
\mathcal{X} = 10 \cdot 2^{\lambda - 10 + k \cdot m} =
\begin{cases}
\in \mathbf{Z}    & m = 0 \\
\notin \mathbf{Z} & m \in \mathbf{Z}[1, 5]
\end{cases}
$$

즉, $q$ 가 6의 배수인 경우에만 Shift 연산과 Gaussian 함수에 의한 양자화와 동일하게 된다. 

그 외의 경우에는,, Shift 연산에 의한 양자화에 오차가 있음을 의미하며, 오차의 크기에 따라,  예상하지 못한 결과가 나타날 수 있음을 의미한다. 



## Quantization 오차 분석 

$q_2 = q_1 + 6$ 이고, $q_1^s = \frac{1}{2} q_2^s$ 일때, 
Let $\mathcal{X} = q_1 \cdot k + m$ where $0 \leq m < q_1$, then 

$$
\left[ \frac{1}{q_1} \mathcal{X} \right] = \left[ k + \frac{1}{q_1} m \right] = \left \lfloor k + \frac{1}{q_1} m + \frac{1}{2} \frac{1}{q_1} q_1 \right \rfloor.
$$

thereby, 
$$
\left[ \frac{1}{q_1} \mathcal{X} \right] = 
\begin{cases}
k   & 0 \leq m < \frac{1}{2} q_1 \\
k+1 & \frac{1}{2} q_1 \leq m < q_1
\end{cases}
$$

In $q_2$, by the same way, we can obtain
$$
\left[ \frac{1}{q_2} \mathcal{X} \right] = \left[ \frac{1}{2 q_1} \mathcal{X} \right] = \left[ \frac{1}{2} k + \frac{1}{2q_1} m \right] = \left \lfloor \frac{1}{2} k + \frac{1}{2 q_1} m + \frac{1}{2} \frac{1}{2 q_1} 2 q_1 \right \rfloor.

\tag{8}
$$

Let $k = 2 \cdot \bar{k} + n$.  In (9), since $m < q_1$, $\frac{m}{2q_1} + \frac{1}{2} < 1$. Therefore,
$$
\left[ \frac{1}{q_2} \mathcal{X} \right] =
\begin{cases}
\left \lfloor \frac{1}{2} k + \frac{1}{2 q_1} m + \frac{1}{2} \right \rfloor = \bar{k}. & n=0 \\
\left \lfloor \frac{1}{2} k + \frac{1}{2} + \frac{1}{2 q_1} m + \frac{1}{2} \right \rfloor = \bar{k} +1 & n=1
\end{cases}
$$

It means that the $k$ is odd number, $\left[ \frac{1}{q_2} \mathcal{X} \right]$ is $\bar{k}+1$ in spite of which value the remainder $m$ has. 

- When $k$ is **odd** value and $m < \frac{1}{2} q_1$
$$
\left[ \frac{1}{q_1} \mathcal{X} \right] = 2 \bar{k} + 1, \;\;\; \left[ \frac{1}{q_1} \mathcal{X} \right] = \bar{k} + 1 
$$

Therefore,
$$
Q_1 - 2 Q_2 = 2 \bar{k} + 1 - 2 (\bar{k} + 1) = -1
$$

- When $k$ is **odd** value and $ \frac{1}{2} q_1 \leq m  < q_1$
$$
\left[ \frac{1}{q_1} \mathcal{X} \right] = (2 \bar{k} + 1) + 1, \;\;\; \left[ \frac{1}{q_1} \mathcal{X} \right] = \bar{k} + 1
$$

Therefore,
$$
Q_1 - 2 Q_2 = 2 \bar{k} + 2 - 2 (\bar{k} + 1) = 0
$$

- When $k$ is **even** value and $m < \frac{1}{2} q_1$
$$
\left[ \frac{1}{q_1} \mathcal{X} \right] = 2 \bar{k}, \;\;\; \left[ \frac{1}{q_1} \mathcal{X} \right] = \bar{k} 
$$

Therefore,
$$
Q_1 - 2 Q_2 = 2 \bar{k} - 2 \bar{k} = 0
$$

- When $k$ is **even** value and $ \frac{1}{2} q_1 \leq m  < q_1$
$$
\left[ \frac{1}{q_1} \mathcal{X} \right] = 2 \bar{k} + 1, \;\;\; \left[ \frac{1}{q_1} \mathcal{X} \right] = \bar{k} 
$$

Therefore,
$$
Q_1 - 2 Q_2 = 2 \bar{k} + 1 - 2 \bar{k} = 1
$$

**Q.E.D.**


그러므로,  양자화 모델은 $|2Q_2|$ 가 $|Q_1|$ 보다 1이 더 클 수 있으므로, 이 경우에는  

$$
\begin{aligned}
\Delta Q_{q_1, q_2} &= Q_1 - 2 \cdot Q_2  \\

&= \frac{1}{q_1^s} \mathcal{X} + N_{q_1} - \frac{2}{q_2^s} \mathcal{X} - 2 N_{q_2} - 1\\

&= \frac{1}{q_1^s} \mathcal{X} + N_{q_1} - \frac{2}{2 q_1^s} \mathcal{X} - 2 N_{q_2} - 1\\

&= N_{q_1} - 2 N_{q_2} - 1
\end{aligned}
$$

에서 
$$
-2 < N_{q_1} - 2 N_{q_2} < 1
$$

이 된다. 



