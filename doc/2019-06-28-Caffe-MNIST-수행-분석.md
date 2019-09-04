---
title:  "Caffe MNIST 수행 분석"
date:   2019-06-28 19:40:00 +0900
---

2019-06-28 Caffe MNIST 수행 분석 : Python Code
===

본 문서는 **Caffe MNIST 수행 분석** 의 후속편으로 prototxt  분석 이후에 대한 Python Code 분석이다.

주요 내용은 다음 Site를 참조한다.

http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html

본 내용은 위 사이트에서 언급되지 않은 부분을 포함하여 정리한다. (언급된 내용들을 대체로 넘어간다.)

## Setup 단계

~~~Python
###########################################################################
# Setup
###########################################################################
from pylab import *
#%matplotlib inline

# To execute Linux Script
import subprocess

#caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
caffe_root = 'c:\\Projects\\caffe\\'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

# run scripts from caffe root
import os
os.chdir(caffe_root)
# Download data
#!data\\mnist\\get_mnist.sh
subprocess.call(["C:\\Program Files\\Git\\git-bash.exe", '.\\data\\mnist\\get_mnist.sh'])

# Prepare data
#!examples\\mnist\\create_mnist.sh
subprocess.call(["C:\\Program Files\\Git\\git-bash.exe", '.\\examples\\mnist\\create_mnist.sh'])
# back to examples
os.chdir('examples')
~~~

- Python을 사용한 Setup 단계이다.
- Linux Script를 Python을 통해 Windows에서 수행하는 방법은 git-bash.exe 등을 사용하여 Linux Script를 수행하도록  Subprocess.call 을 호출하는 방식으로 이루어진다.



### Creating the net

~~~python
###########################################################################
# Creating the net
###########################################################################
from caffe import layers as L, params as P

def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    
    return n.to_proto()

with open('mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('.\\mnist\\mnist_train_lmdb', 64)))


with open('mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('.\\mnist\\mnist_test_lmdb', 100)))

#!cat mnist/lenet_auto_train.prototxt
#!cat mnist/lenet_auto_solver.prototxt
subprocess.call(["C:\\Program Files\\Git\\git-bash.exe", 'cat mnist/lenet_auto_train.prototxt'])
subprocess.call(["C:\\Program Files\\Git\\git-bash.exe", 'cat mnist/lenet_auto_solver.prototxt'])
~~~

- Caffe에서  중요한 부분은 다음 부분이다.

~~~python
with open('mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('.\\mnist\\mnist_train_lmdb', 64)))
~~~



### Loading and checking the solver

~~~
caffe.set_device(0)
caffe.set_mode_gpu()
~~~
를 통해 GPU Device를 세팅한다. 이 부분에 대해서는 나중에 ETRI System과 다를 수 있으므로 더 알아 보아야 한다.

~~~
### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver('mnist/lenet_auto_solver.prototxt')
~~~

이로서 Solver가 Loading 된다. 
Solver Parameter 들을 다음과 같은 형태로 출력하여 알아보면

~~~python
### Printout the solver Information
print("----------- Solver Information -----------")
a = []
print(solver.iter)
a.append([x for x in solver.net.blob_loss_weights.items()])
a.append([x for x in solver.net.blobs.items()]) 
a.append([x for x in solver.net.bottom_names.items()]) 
a.append([x for x in solver.net.layer_dict.items()]) 
a.append([x for x in solver.net.params.items()]) 
a.append([x for x in solver.net.top_names.items()]) 

print("---- solver.net.blob_loss_weights ----\n", a[0])
print("---------- solver.net.blobs ----------\n", a[1])
print("------- solver.net.bottom_names ------\n", a[2])
print("-------- solver.net.layer_dict -------\n", a[3])
print("---------- solver.net.params ---------\n", a[4])
print("---------solver.net.top_names --------\n", a[5])
print("------------------------------------------")

~~~


- solver.iter : Int

- solver.net : Caffe Object 로서 Net 형 Neural Network의 데이터와 파라미터들이 정의되어 있다.
	- blob_loss_weights : Ordered Dict
	- blobs : Ordered Dict, 각 Layer별 주요 Parameter( Dimension, output, Weight, Difference 등이 있다) ,  [0] :  Layer 이름,  [1] Caffe Blob 형식의 Layer 정보에 대한 Pointer.  다음의 정보를 가지고 있다.
	  - channels 
	  - count
	  - data : Python Array 
	  - diff :  Python Array 
	  - height 
	  - num : 
	  - shape : Caffe Object, IntVec Type 
	- bottom_names :  Ordered Dict
	- inputs : 
	- layer_dict : 구성된 Layer의 주요 데이터들이 저장된다.  자체적으로 정의된 데이터 형식이다. 각 Layer 데이터들은 blobs로 구성되어 있다.
	- params : Layer들의 Parameter들이 Blob 형태로 저장되어 있음. Python에서 Access는 불가능하며 Caffe Library 내에서만 참조가 가능하다. BlobVector 형 
	
- top_names : Network의 이름들 
	- _blob_loss_weights : Caffe Object : DTypeVec,   Python에서 직접 Access 불가능
	- _blob_names : Caffe Object : StringVec , Python에서 직접 Access 불가능
	- _blobs : Caffe Object : BlobVec, Python에서 직접 Access 불가능
	- _inputs : Caffe Object : IntVec, Python에서 직접 Access 불가능
	- _layer_names : Caffe Object : StringVec , Python에서 직접 Access 불가능
	- _outputs : Caffe Object : IntVec , Python에서 직접 Access 불가능
	
- solver.param : Caffe Object, Solver 자체의 Parameter 3종을 가지고 있으며, display, layer-wise-reduce, max_iter 3개의 파라미터를 가지고 있다.

- solver.test_nets : Caffe Object, NetVec 형식 Net Type이 Vector형으로 존재한다. test_nets가 실제 test 시 training 결과를 받아  test를 수행한다. Neural network을 의미한다. 

  

Solver.net 데이터를 실제로 print 해보면 다음과 같다.

~~~
---- solver.net.blob_loss_weights ----
 [('data', 0.0), ('label', 0.0), ('conv1', 0.0), ('pool1', 0.0), ('conv2', 0.0), ('pool2', 0.0), ('fc1', 0.0), ('score', 0.0), ('loss', 1.0)]
---------- solver.net.blobs ----------
 [('data', <caffe._caffe.Blob object at 0x000002565CB7EBE0>), 
 ('label', <caffe._caffe.Blob object at 0x000002565CB7EFA8>), 
 ('conv1', <caffe._caffe.Blob object at 0x000002565CB7E978>), 
 ('pool1', <caffe._caffe.Blob object at 0x000002565CB7EE48>), 
 ('conv2', <caffe._caffe.Blob object at 0x000002565CB7ECE8>), 
 ('pool2', <caffe._caffe.Blob object at 0x000002565CB7ED98>), 
 ('fc1', <caffe._caffe.Blob object at 0x000002565CB7ED40>), 
 ('score', <caffe._caffe.Blob object at 0x000002565CB7EB30>), 
 ('loss', <caffe._caffe.Blob object at 0x000002565CB7EC90>)]
------- solver.net.bottom_names ------
 [('data', []), ('conv1', ['data']), ('pool1', ['conv1']), ('conv2', ['pool1']), ('pool2', ['conv2']), ('fc1', ['pool2']), ('relu1', ['fc1']), ('score', ['fc1']), ('loss', ['score', 'label'])]
-------- solver.net.layer_dict -------
 [('data', <caffe._caffe.Layer object at 0x000002565CB7E9D0>), 
 ('conv1', <caffe._caffe.Layer object at 0x000002565CB7EAD8>), 
 ('pool1', <caffe._caffe.Layer object at 0x000002565CB7EA80>), 
 ('conv2', <caffe._caffe.Layer object at 0x000002565CB7EDF0>), 
 ('pool2', <caffe._caffe.Layer object at 0x000002565CB7EA28>), 
 ('fc1', <caffe._caffe.Layer object at 0x000002565CB7E920>), 
 ('relu1', <caffe._caffe.Layer object at 0x000002565CB7EEA0>), 
 ('score', <caffe._caffe.Layer object at 0x000002565CB7EC38>), 
 ('loss', <caffe._caffe.Layer object at 0x000002565CB7EB88>)]
---------- solver.net.params ---------
 [('conv1', <caffe._caffe.BlobVec object at 0x000002565C617620>), 
 ('conv2', <caffe._caffe.BlobVec object at 0x000002565C6176C0>), 
 ('fc1', <caffe._caffe.BlobVec object at 0x000002565C6177B0>), 
 ('score', <caffe._caffe.BlobVec object at 0x000002565C617850>)]
---------solver.net.top_names --------
 [('data', ['data', 'label']), ('conv1', ['conv1']), ('pool1', ['pool1']), ('conv2', ['conv2']), ('pool2', ['pool2']), ('fc1', ['fc1']), ('relu1', ['fc1']), ('score', ['score']), ('loss', ['loss'])]
------------------------------------------
~~~



### Setting Training and Test Network

Network Loading 이 끝나면 학습용 네트워크와 Test용 네트워크를 다음과 같이 1회 forward 계산을 수행한다.

~~~
solver.net.forward()  # train net
solver.test_nets[0].forward()  # test net (there can be more than one)
~~~

solver.net 은 Training  을 의미하고 solver.test_nets[0] 는 test_nets중 0번 네트워크에 대한 forward 계산을 의미한다.

#### Data Set 표시

Data Set을 Python의 imshow 함수로 보인다.

~~~python
imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off'); plt.show() 
print('train labels:', solver.net.blobs['label'].data[:8]) 
~~~

Input 데이터는 net.blobs 에  'data' 로 존재하고 있으며 float32 형식의 python array로 이루어져 있다.

Data는 실제 영상 Data와 Label에 접근하고 있으며 **solver.net.blobs** 를 통해 접근하고 있다.

- **solver.net.blobs['data']** : 영상 Data를 의미
  - solver.net.blobs['data'].**data[:8, 0]** : 각  Label별로 0번째 Data를 Index,  Python Array 형식이다.
  - solver.net.blobs['data'].data[:8, 0].**transpose(1, 0, 2)**  : Python Array의 transpose  (0, 1, 2) axis 규칙을 (1, 0 , 2) 로 Transpose 한다. 따라서 data[:8, 0]은 $ 8 \times 28 \times 28$ 인데,  0번째 Index의 Dimesion인 8이 1번째로 가고, 1번째 Index의 Dimension 28이 0번째 로 가서 $28 \times 8 \times 28$ 이 된다. 
  - **reshape(28, 8*28)** : $28 \times 8 \times 28$  Tensor를 $ 28 \times 224 (=8 \times 28)$ 로 바꾼다.
- **solver.net.blobs['label']**: 영상 Data에 대한 Label을 의미 
  - solver.net.blobs['label'].**data[:8]** : Label 을 의미 

### Iterations 
#### 준비단계

학습 결과를 담기 위한 데이터 Setting이 다음과 같이 우선된다.

~~~python
###########################################################################
# Writing a custom training loop
###########################################################################
#%%time

niter = 200
test_interval = 25
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))
~~~

200 Iteration 을 수행하도록 세팅된다.



#### Learning Process

Learning Process의 전체 코드는 다음과 같다.

~~~python
# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    
    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print ('Iteration', it, 'testing...')
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4
~~~

한 부분씩 살펴본다. 

step 명령은 forward 와 다르게 forward - backward - update 를 모두 수행한다.  forwardsms forward만 수행한다. 

~~~
solver.step(1)  # SGD by Caffe
~~~

만일 solver.step(100) 이면 100번을 수행한다. 

즉, 위 코드는 200번의 iteration을 도는데, 8개의 class 에서 8개씩 (그래서 총 64개) 데이터에 대하여 수행한다. Training 단계에서 데이터를 64개만 읽어 왔기 때문이다. 

다음 매 Iteration당 train_loss를 Blob에서 읽어서 Python array에 저장한다.

~~~
train_loss[it] = solver.net.blobs['loss'].data
~~~

다음 학습이 아닌 테스트 결과를 얻기 위해 8개의 데이터에 대하여 output 결과를 가져온다. 하나의 데이터에 대하여 대체로 가장 큰 하나의 값이 존재할 것이고 해당 index가 class index가 된다.

~~~python
    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]
    
~~~

여기까지가 기본적인 구조이고 실제로 학습 자체는 solver.step() 으로 끝난다. 

다음 코드는 25 Iteration 마다 학습 결과를 계산하고  얼마나 정확한지를 계산하는 부분이다. 

~~~        python
        for test_it in range(100):
			solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
            
        test_acc[it // test_interval] = correct / 1e4
~~~

여기서 **solver.test_nets[0].blobs['score'].data** 는 $100 \times 10$ 이다 그래서 1번 axis (=10) 에 대하여 argmax를 구하면 해당 값이 Class 를 지칭하는 Index 이고 이 값이 Label 데이터와 맞은 갯수를 모두 더하면 1 Batch size (=100 개의 data)에 대하여 계산이 되고 batch size가 100이므로 총 10000만개의 경우에 대한 Correct 값이 최종적으로 test_acc에 저장된다.

나머지 코드는 이렇게 구해진  결과를 Ploting 한다.

~~~python
###########################################################################
# plot the train loss and test accuracy
###########################################################################
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.show()
print ("Test Accuracy plot")
~~~

