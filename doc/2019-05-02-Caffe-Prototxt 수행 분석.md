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





## Other Layers
### Pooling Layers 
