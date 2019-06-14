---
title:  "Prototxt 수행 분석"
date:   2019-05-02 13:40:00 +0900
---

Prototxt 분석
===

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

각 Layer별 특성은 다음 사이트를 참조한다.
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

    - $$
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

다음과 같이 LeNet이 주어졌다고 가정하자. 

![](https://drive.google.com/uc?id=17b3GIalF1zQFjuTilMj6CDJwwTmKEMrh)





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
