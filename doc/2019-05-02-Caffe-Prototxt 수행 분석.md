---
title:  "Prototxt 수행 분석"
date:   2019-05-02 13:40:00 +0900
---

Prototxt 분석
===

[TOC]

다음 protoptxt 파일을 분석한다.

~~~python 
train_net_path = 'mnist/custom_auto_train.prototxt'
test_net_path = 'mnist/custom_auto_test.prototxt'
solver_config_path = 'mnist/custom_auto_solver.prototxt'
~~~

해당 내용은 다음 사이트를 통해 자세히 알 수 있다.

https://github.com/ys7yoo/BrainCaffe/wiki/Caffe-Example-:-2.Training-LeNet-on-MNIST-with-Caffe-(Kor)

그럼에도 본 문을 작성하는 이유는,  예제와 조금은 다른 prototxt를 분석함으로서 Caffe에 완전히 익숙해지기 위한 것이다.


## custom_auto_train.prototxt

~~~
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  transform_param {
    scale: 0.003921568859368563
  }
  data_param {
    source: "mnist/mnist_train_lmdb"
    batch_size: 64
    backend: LMDB
  }
}
layer {
  name: "score"
  type: "InnerProduct"
  bottom: "data"
  top: "score"
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "score"
  bottom: "label"
  top: "loss"
}
~~~



### Input Layer 

~~~
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  transform_param {
    scale: 0.003921568859368563
  }
  data_param {
    source: "mnist/mnist_train_lmdb"
    batch_size: 64
    backend: LMDB
  }
}
~~~

- 이름이 data 
- 타입은 Data이다.
- Data와 Label 이 있기 때문에 Top 즉, 출력으로 두개가 나간다.  (두 개의 blobs, 하나는 data blobs, 다른 하나는 label blob를 생성한다.)
  - data 
  - label
-  transform_para 에서 Scale은 1을 256으로 나누기 때문에 $1/256 = 0.0039215...$ 이다.
- data_param 에서 lmdb 파일/포맷 으로 배치 사이즈 64의 크기로 데이터를 읽어 들인다.
- bottom은 바로 밑의 Layer를 의미하지만, Top은 현재 Layer를 의마한다.



###  InnerProduct Layer (Fully Connected Layer)

~~~
layer {
  name: "score"
  type: "InnerProduct"
  bottom: "data"
  top: "score"
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
  }
}
~~~
Inner product의 경우는 Full Connected Layer로서 간주된다.
따라서, Inner_product_param의 데이터가 중요하다.

* Bottom은 data, top은 자기 자신이므로 score이다. 
* inner_product_param 이 중요하며 Output 수는 10개 이다.
* Weight 초기화 방법은 *xavier* 이다. 
  * 초기화 방법은 xavier외에 



## Other Layers

각 Layer별 특성은 다음 사이트를 참조한다.  각 사이트에는 실제 Layer들을 기술하고 있는 CPP 과 Hpp  파일들의 위치가 나타나 있다.
https://caffe.berkeleyvision.org/tutorial/layers.html
https://github.com/ys7yoo/BrainCaffe/wiki/Caffe-Tutorial-:-5.Layer-Catalogue-(Kor)


### Pooling Layers 

~~~
layer {
  name: "pool1"
  type: "Pooling"
  pooling_param {
    kernel_size: 2
    stride: 2
    pool: MAX
  }
  bottom: "conv1"
  top: "pool1"
}
~~~
 pool 커널 사이즈2 그리고 stride of 2로 max pooling 을 의미한다.

 CAFFE_ROOT/examples/mnist/lenet_train_test.prototxt 에서 알 수 있다.


### Convolution Layer

|    |   |
|--- |---|
|Layer type| Convolution |
|CPU implementation| ./src/caffe/layers/convolution_layer.cpp|
|CUDA GPU implementation| ./src/caffe/layers/convolution_layer.cu|
|Parameters | (ConvolutionParameter convolution_param)|
|요구사항.|  |
|num_output (c_o)| the number of filters|
|kernel_size (or kernel_h and kernel_w)| 각 필터의 너비와 높이를 명시한다.|
|강력한 권고사항. | weight_filler [default type: 'constant' value: 0] |

#### 추가 옵션.

- bias_term [default true]: 필터 출력에 추가적인 편향의 집단을 적용할 것인지 학습시킬 것인지 명시한다.
- pad (or pad_h and pad_w) [default 0]: (암묵적으로) 입력의 각각 사이드에 추가할 픽셀의 수를 명시한다.
- stride (or stride_h and stride_w) [default 1]: 출력에 필터를 적용할 간격들 위치을 명시한다
- group (g) [default 1]: 만약 g>1 면, 입력의 부분 집합에 각각 필터의 연결을 제한한다. 특히 입력과 출력 채널이 g 그룹들로 나누어지고, i번째 출력 그룹 채널은 오직 i번째 입력 그룹 채널과 연결될 것이다.

- Input
$$
n \times c_i \times h_i \times w_i
$$
- Output
$$
n \times c_o \times h_o \times w_o
$$
where $h_o = \frac{(h_i + 2 * pad_h - kernel_h)}{stride_h}+ 1$ and $w_o$ likewise. 

- 아래 Sample에서 

  - 첫번재 param은 weights 이고 두번재는 biases 이다. 

  - lr_mult  

    - 특정 layer의 learning rate를 설정할 때 쓰인다. 

    - 예를 들어, solver의 base_lr: 0.001이고 lr_mult: 1이면 해당 layer의 learning rate 는 

$$
\text{learning_rate} = \text{base_lr} \times \text{lr_mult} = 0.001 \times 1
$$
- 따라서, lr_mult: 0 이면 해당 layer의 weight값을 freezing (no update)하는 것이다.

  - decay_mult  
  
    - solver의 weight decay 와 같다.

출처:  https://dongjinlee.tistory.com/entry/퍼옴-Caffe-model-finetuning-하는-방법

#### Sample 
as seen in ./models/bvlc_reference_caffenet/train_val.prototxt

~~~
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  # learning rate and decay multipliers for the filters
  param { lr_mult: 1 decay_mult: 1 }
  # learning rate and decay multipliers for the biases
  param { lr_mult: 2 decay_mult: 0 }
  convolution_param {
    num_output: 96     # learn 96 filters
    kernel_size: 11    # each filter is 11x11
    stride: 4          # step 4 pixels between each filter application
    weight_filler {
      type: "gaussian" # initialize the filters from a Gaussian
      std: 0.01        # distribution with stdev 0.01 (default mean: 0)
    }
    bias_filler {
      type: "constant" # initialize the biases to zero (0)
      value: 0
    }
  }
}
~~~

컨볼루션 계층은 출력이미지에서 각각 하나의 특징 맵을 생산하면서 배울수 있는 필터들의 집단으로 입력 이미지를 감아놓는다.




## LeNet과 ProtoText 관계


### Caffe Example 에서의 Network 

Caffe의 MNIST Example의 Network Spec. 은 다음과 같다.
~~~
state {
  phase: TEST
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  transform_param {
    scale: 0.0039215689
  }
  data_param {
    source: ".\\mnist\\mnist_test_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 20
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  convolution_param {
    num_output: 50
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "fc1"
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "fc1"
  top: "fc1"
}
layer {
  name: "score"
  type: "InnerProduct"
  bottom: "fc1"
  top: "score"
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "score"
  bottom: "label"
  top: "loss"
}
~~~

이를 그래프로 도시하면 다음과 같다.

![](https://drive.google.com/uc?id=1RY0pBFqD6HmsZsn971AS2Pq3KHqT8c-e)

 살펴보면 먼저

- 28x28 Data에 대하여 5x5  Kernel로 Convolution 하게 되면 24x24 의 convolution 결과가 나온다. ($28 - 25 + 1 = 24$) 
- Pool1 에서  2x2 Pooling 이지만 Stride가 2인 관계로 크기는 절반으로 줄어든다. 
- conv2 에서 5x5 Convolution이 수행되므로 $12 - 5 + 1 = 8$ 이 되어 8x8 로 결과가 나온다.
- Pool2 에서  2x2 Pooling 이지만 Stride가 2인 관계로 크기는 절반으로 줄어 $4 \times 4$가 된다. 
- 그 다음 부분은 Fully Connected 이므로 일반적인 Back Propagation이다. 

### LeNet 분석

다음과 같이 LeNet이 주어졌다고 가정하자. 

![](https://drive.google.com/uc?id=17b3GIalF1zQFjuTilMj6CDJwwTmKEMrh)

### 

## Caffe의 Class

 Caffe는 Solver, net, Layer, Blob 의 4개의 class로 이루어져 있다.

### Blob 
- Caffe에서 정의하고 있는 자료구조
- Layer 마다 2개씩 존재
     - Weight Blob
     - Bias Blob
- 1 Blob의 구조
     - 데이터 배열 (Parameter)
     - 변량 배열 (Differential of Parameter)
- 이미지 데이터 일회 처리량에 대한 틀에 적용된 blob 차원은
$$
N (데이터 갯수) \times K(채널) \times H(높이) \times W(밑변)
$$
- 위 공식에 대한 에제
     - 1회 처리 데이터가 256개이면 $N=256$
     - RGB라고 하면 $K=3$
     - 예를 들어 $11 \times 11$ 공간 차원에 대한 96개의 필터로 이루어진 Convolution Layer에 대해 3 채널 입력 Blob는 $96 \times 3 \times 11 \times 11$ 
     - Perceptron or Fully Connected Layer에서 100개의 출력 계층과 1024개의 입력계층 에 대한 Blob 크기는 $100 \times 1024$

### Layer 
Layer는 위에서 설명하였다.
구체적인 동작은 다음 사이트를 참조한다.
https://github.com/ys7yoo/BrainCaffe/wiki/Caffe-Tutorial-:-5.Layer-Catalogue-(Kor)


### Net 
Net은 Layer들의 집합으로 Protext를 통해 명세화 된다.


#### Solver
SOlver는 Layer들에 정의된 Parameter들을 학습하기 위한 명세로서, 별도의 prototext를 통해 정의된다. 


## custom_auto_solver.prototxt

Caffe의 Solver는 다음과 같다.

- Stochastic Gradient Descent ( type : "SGD" )
- AdaDelta ( type : "AdaDelta" )
- Adaptive Gradient (type: "AdaGrad"),
- Adam (type: "Adam"),
- Nesterov’s Accelerated Gradient (type: "Nesterov") and
- RMSprop (type: "RMSProp")

다음을 분석해보자

~~~
train_net: "mnist/custom_auto_train.prototxt"
test_net: "mnist/custom_auto_test.prototxt"
test_iter: 100
test_interval: 500
base_lr: 0.009999999776482582
display: 1000
max_iter: 10000
lr_policy: "inv"
gamma: 9.999999747378752e-05
power: 0.75
momentum: 0.8999999761581421
weight_decay: 0.0005000000237487257
snapshot: 5000
snapshot_prefix: "mnist/custom_net"
solver_mode: GPU
random_seed: 831486
type: "SGD"
~~~

하나씩 살펴본다. Iteration의 경우 

~~~
test_iter: 100
test_interval: 500
~~~

아래 구체적인 알고리즘과 관련된 부분을 제외한 Parameter는 다음과 같다. 

- **test_iter**는 얼마나 정방향과정을 test단계에서 수행해야하는 명시한다.
  MNIST의 경우에서, 전체 실험용 10000개의 이미지에 대하여 실험 일회 처리량을 100으로, 100번의 실험 반복수를 가진다. 즉 **test batch size** 는 100이 되고 이를 100 Iteration 하게 되면 10000개의 이미지에 대하여 1 Epoch 에 대한 학습이 된다.    즉 

$$

\text{test_iter} = \frac{\text{total number of data}}{\text{test batch size}}
$$



- **test_interval**: Test 를 실행하는 간격을 설정. 실험을 매 500번 훈련 반복마다 한번 실행한다. 즉, EPOCH가 500번 임을 표시한다. 
- **base_lr**: [Learning Rate](https://www.quora.com/What-is-the-learning-rate-in-neural-networks) 의 초기값 
- **display**: 학습 도중에 몇번의 Iteration 마다 중간 결과를 보여줄 것인지 설정
- **max_iter**: 학습을 실행하는 최대 Iteration 을 설정
- **snapshot**: 학습 도중에 특정 Iteration 마다 모델의 중간 결과를 저장. 이 설정은 몇번째 Iteration 마다 중간 결과를 저장할 것인지 설정한다.
- **solver_mode**: 모델을 CPU, GPU 중에서 어떤 것으로 학습할지를 설정. 이처럼 Caffe 는 CPU, GPU 사이의 설정 변경이 매우 간단하다.



### Solver 분석

해결사 메소드는 손실 최소화의 일반적 최적화 문제를 다룬다. 데이터세트$D$ 대하여 , 최적화 목적은 데이터 셋을 걸쳐 모든$|D|$데이터 사례에 대한 전체 평균 손실이다.
$$
L(W) = \frac{1}{|D|} \sum_i^{|D|} f_W \left( X^{(i)} \right) + \lambda r(W)
$$
여기서$f_W \left( X^{(i)} \right)$는 데이터 instance $X^{(i)}$에 대한 손실 (혹은 목적함수)이고 $r(W)$는 가중치 $\lambda$를 가진 조직화 항(regularization term)이다. $|D|$는 매우 클 수 있지만, 그래서 실제로는, 우리가 이 목적함수의 stochastic approximation를 사용하는 각 Iteration에 있어, $N<<|D|$ 경우의 최소 일회 처리량(mini-batch)을 의미하여 다음과 같이 근사화 시킬 수 있다.
$$
L(W) \approx \frac{1}{N} \sum_i^N f_W\left(X^{(i)}\right) + \lambda r(W)
\tag{C1}
\label{C1}
$$
### SGD 
실제 코드는 다음에 있다. 본인의 경우 C:\Projects 아래에 caffe를 설치하였다.
~~~
C:\Projects\caffe\src\caffe\solvers\sgd_solver.cpp
C:\Projects\caffe\src\caffe\solvers\sgd_solver.cu
C:\Projects\caffe\include\caffe\sgd_solvers.hpp
~~~

Caffe에서 SGD는 다음과 같이 정의된다.

The **learning rate** $$ \alpha $$ is the weight of the negative gradient.
The **momentum** $$ \mu $$ is the weight of the previous update.

Formally, we have the following formulas to compute the update value $$ V_{t+1} $$ and the updated weights $$ W_{t+1} $$ at iteration $$ t+1 $$, given the previous weight update $$ V_t $$ and current weights $$ W_t $$:
$$
V_{t+1} = \mu V_t - \alpha \nabla L(W_t)
$$

$$
W_{t+1} = W_t + V_{t+1}
$$
#### Step Size in SGD : Learning Parameters in SGD

다음 파라미터를 살펴보자. `momentum` $$ \mu = 0.9 $$. 라 가정하자.

~~~
base_lr: 0.01     # begin training at a learning rate of 0.01 = 1e-2
lr_policy: "step" # learning rate policy: drop the learning rate in "steps"
                  # by a factor of gamma every stepsize iterations
gamma: 0.1        # drop the learning rate by a factor of 10
                  # (i.e., multiply it by a factor of gamma = 0.1)
stepsize: 100000  # drop the learning rate every 100K iterations
max_iter: 350000  # train for 350K iterations total
momentum: 0.9
~~~

base_lr` of $ \alpha = 0.01 = 10^{-2}  $ 이고 이는 stepsize = 100K에서 처음 100k Iteration에서 유지된다.

그 다음  gamma 가 0.01 이므로  $$ \alpha' = \alpha \gamma = (0.01) (0.1) = 0.001 = 10^{-3} $$ for iterations 100K-200K, 가 된다.  그리고  $$ \alpha'' = 10^{-4} $$ for iterations 200K-300K, 마지막으로 350K Iteration에서 $$ \alpha''' = 10^{-5} $$. 가 된다.

Momentum 계수 $\mu = 0.9$ 로 가정되어 있는데 $\mu$가 증가하면 $\alpha$를 비례하여 감소시키고  그 반대로 해도 된다. 

다시말해 Momentum은 과거의 Update를 무한 등비 급수로 더하는 것과 마찬가지이므로 $ \frac{1}{1 - \mu} $ 만큼 곱해지는 것으로 볼 수 있다. (현재 Update에 대하여) 따라서 다시 생각해 보면 이는 지수가중 이동평균필터와 같은 의미이다.

지수 가중 이동평균 필터는 결국 Convexity를 기반으로 하는 간단한 저주파 필터링이 된다. 
$$
\begin{aligned}
W_{t+1} &= W_t + V_{t+1} \\
&= W_t - \alpha \nabla L(W_t) + \mu V_t \\
&= W_t - \alpha \nabla L(W_t) - \mu \alpha \nabla L(W_{t-1}) + \mu^2 V_{t-1} \\
&= W_t - \alpha \nabla L(W_t) - \mu \alpha \nabla L(W_{t-1}) - \mu^2 \alpha \nabla L(W_{t-2}) + \mu^3 V_{t-2}\\
\end{aligned}
$$
Learning rate Policy는 다음과 같다. (caffe-master/src/caffe/proto/caffe.proto 참조)

 The learning rate decay policy. The currently implemented learning rate  policies are as follows:

|      |      |
| ---- | ---- |
|fixed | always return $\text{base_lr}$. |
|step  | return $\text{base_lr} \times \gamma^{\lfloor\frac{iter}{step} \rfloor}$ |
|exp   | return $\text{base_lr} \times \gamma^{iter} $ |
|inv   | return $\text{base_lr} \times (1 + \gamma \times iter)^{- power}$ |
|multistep| similar to step but it allows non uniform steps defined by stepvalue |
|poly  | the effective learning rate follows a polynomial decay, to be zero by the max_iter. return $\text{base_lr} \left( 1 - (\frac{iter}{\text{max_iter}})^ {power} \right)$ |
|sigmoid | the effective learning rate follows a sigmod decay return $\text{base_lr} ( 1/(1 + \exp(-\gamma \times (iter - stepsize))))$ |

 where **base_lr, max_iter, gamma, step, stepvalue** and **power** are defined in the solver parameter protocol buffer, and **iter** is the **current iteration**.


#### Weight Decay

Solver에서 나오는 것 중 Weight Decay Parameter가 있다. 이는  $\eqref{C1}$  에  정의된 Weight Regularization Term 앞에 있는 Largrangian Multiplier를 의미한다. 

간단히 말하면 다음과 같이 Object Function이 주어지고,
$$
\tilde{E}(w) = E(w) + \frac{\lambda}{2} w^2
$$
이에 대한 Gradient Deascent 방정식을  구하면 다음과 같다.
$$
w_{t} \leftarrow w_{t} - \eta \frac{\partial E}{\partial w_{t}} - \eta \lambda w_{t}
$$
이때, Largrangian Multiplier $\lambda$ 가 Weight Decay가 된다. 



### 그 외 알고리즘들

#### AdaGrad

$$
(W_{t+1})_i =
(W_t)_i - \alpha
\frac{\left( \nabla L(W_t) \right)_{i}}{
    \sqrt{\sum_{t'=1}^{t} \left( \nabla L(W_{t'}) \right)_i^2}
}
$$



[1] J. Duchi, E. Hazan, and Y. Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. The Journal of Machine Learning Research, 2011.

#### AdaDelta
The **AdaDelta** (`type: "AdaDelta"`)  method (M. Zeiler [1]) is a “robust learning rate method”. It is a  gradient-based optimization method (like SGD). The update formulas are

$$
\begin{aligned}
(v_t)_i &= \frac{\operatorname{RMS}((v_{t-1})_i)}{\operatorname{RMS}\left( \nabla L(W_t) \right)_{i}} \left( \nabla L(W_{t'}) \right)_i
\\
\operatorname{RMS}\left( \nabla L(W_t) \right)_{i} &= \sqrt{E[g^2] + \varepsilon}
\\
E[g^2]_t &= \delta{E[g^2]_{t-1} } + (1-\delta)g_{t}^2
\end{aligned}
$$

$$
(W_{t+1})_i = (W_t)_i - \alpha (v_t)_i.
$$

기본 아이디어는 다음과 같다.

1. ADAGRAD 의 경우  모든 $\nabla L(W_{t'})$  의 크기를 더하기 때문에  초기치에 크게 영향을  받으므로 이를 Exponential Decaying 되는 값인 $E[g^2]_t$ (where $g_t = \nabla L(W_t)$ 로 제한시켜 가급적 최근의 Gradient를 가장 많이 반영하도록 한다.)
2. 그리고 각 Dimension 별로 서로 다른 값이 Update 되도록 한다. 
   1. Gradient 값은 Dimension 별로  동일하나 $\operatorname{RMS}((v_{t-1})_i)$ 값이 Dimension $i$에 따라 다르게 나타난다. 
3. Newton Rapson 법에 의하여 다음과 같다.
$$
\Delta x_t = -H^{-1}_t g_t
$$

이는 다음과 같은 개념이다.  

$$
\Delta x_t = \frac{\frac{\partial f}{\partial x}}{\frac{\partial^2 f}{\partial x^2}} \Rightarrow \frac{1}{\frac{\partial^2 f}{\partial x^2}} = \frac{\Delta x}{\frac{\partial f}{\partial x}}
$$

이때 Hessian을 다음과 같이 근사화 하는 것이다.
$$
\left| \frac{\partial^2 f}{\partial x^2} \right| \approx \frac{\mathbb{E} \frac{\partial f}{\partial x}}{\mathbb{E} \Delta x } \approx \frac{\operatorname{RMS} [g]_t}{\operatorname{RMS}[\Delta x]_{t-1}}
$$
이를 위 식과 결합하면 
$$
\frac{\Delta x}{g_t} = -\left( \frac{\partial^2 f}{\partial x^2} \right)^{-1} = -\frac{\operatorname{RMS}[\Delta x]_{t-1}}{\operatorname{RMS} [g]_t}
$$
혹은 unit of $x$ 를 
$$
\text{Unit of } x \triangleq \frac{\Delta x }{\operatorname{RMS}[\Delta x]_{t-1}}
$$
그런데, 최적화 알고리즘에서 (아래 논문의 방정식 (10)) unit of $x$ 가 다음으로 정의되므로 
$$
\text{Unit of } x \triangleq -\frac{1}{\operatorname{RMS} [g]_t}
$$
그러므로 같은 식이 유도된다고 볼 수도 있다. 여기에서는 Hessian의 개념이 없다.


[1] M. Zeiler     [ADADELTA: AN ADAPTIVE LEARNING RATE METHOD](http://arxiv.org/pdf/1212.5701.pdf).     *arXiv preprint*, 2012.

#### Adam
Adaptive moment estimation

- $g_t \triangleq g_t \odot g_t$ , the elementswise square
- $\alpha$ step size
- $m_0 = 0, v_0 = 0, W_0 = 0, t \rightarrow 0$

While $W_t$ not converged do 

- $t \leftarrow t+1$ 

- $g_t  \leftarrow \nabla_{W_t} L(W_t)$
- First moment Estimate 
$$
m_t^i \leftarrow \beta_1 m_{t-1}^i + (1-\beta_1)\nabla L(W_t)^i
$$
- Second moment Estimate 
$$
v_t^i = \beta_2 v_{t-1}^i + (1-\beta_2)(\nabla L(W_t)^i)^2
$$
- Compute bias-corrected first moment estimate 
$$
{\hat{m}}_t \leftarrow \frac{m_t}{1 - \beta_1^t}
$$
- Compute bias-corrected second moment estimate
$$
\hat{v}_t \leftarrow \frac{v_t}{1 - \beta_2^t}
$$
- Update Parameter 
$$
W_t \leftarrow W_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

위 update 방정식에서 $\epsilon > 0$ 은 0으로 나누어 떨어지는 것을 방지하기 위한 것이므로 간단히 변경하면 

$$
\begin{aligned}
W_t &= W_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}} \\
&= W_{t-1} - \alpha \cdot \frac{m_t}{1 - \beta_1^t} \cdot \sqrt{\frac{1 - \beta_2^t}{v_t}} \\
&= W_{t-1} - \alpha \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t} \cdot \frac{m_t}{\sqrt{v_t}}
\end{aligned}
$$

그러므로 $\epsilon$을 닷히 부가하고  Dimension index $i$를 추가하면,
$$
W_{t}^i = W_{t-1}^i - \alpha \frac{\sqrt{1-(\beta_2)_i^t}}{1-(\beta_1)_i^t}\frac{(m_t)_i}{\sqrt{(v_t)_i}+\varepsilon}.
$$

Kingma 와 그의 동료들이 제시한 [1]에서는 $\beta_1=0.9,\beta_2=0.999,\epsilon=10^{−8}$ 를 디폴트 값으로 사용하라고 제시했다. Caffe는 각각 $\beta_1,\beta_2, \epsilon\beta_1,\beta_2, \epsilon$에 대하여 모멘텀, 모멘텀2 델타를 사용한다.

[1] D. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. International Conference for Learning Representations, 2015.

