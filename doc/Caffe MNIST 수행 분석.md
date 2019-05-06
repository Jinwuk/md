---
title:  "Caffe MNIST 수행 분석"
date:   2019-04-17 19:40:00 +0900
---

Caffe MNIST 수행 분석
===

본 문서는 caffe mnist 분석을 위해 작성한 것이다. 기본적인 caffe 코드를 사용하여 Network을 구성하는 등의 방법들을 알아 본다.

## Import Part 

다음 부분이 기본적으로 imort 되어야 할 부분이다.

~~~python
from pylab import *
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

###########################################################################
# Creating the net
###########################################################################
from caffe import layers as L, params as P

~~~

get_mnist.sh 는 별것 없다. wget을 사용하여 mnist 데이터를 받아오는 것이다.
그 다음에 gzip을 사용하여 압축을 푼다. 

~~~sh
#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

echo "Downloading..."

for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    if [ ! -e $fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
    fi
done
~~~

train-images-idx3-ubyte 
train-labels-idx1-ubyte 
t10k-images-idx3-ubyte 
t10k-labels-idx1-ubyte

의 4개 파일을 받아온다.  이 데이터들은 C:\Projects\caffe\data\mnist 경로에 풀려진다.
각 데이터를 살펴보면 말그대로 8bit Unsigned Byte로 구성된 Training Image이다.  분류없이 촘촘히 모두 연결되어 있다.  이 데이터들을 잘 나누고 의미를 부여하기 위해서는 그 다음에 수행되는 create_mnist.sh 파일을 분석해 보아야 한다.

### create_mnist.sh 분석

분석을 위해 Visual Studio 2015에서 다음을 각각의 Debug Parameter로 하여 디버그 해 보자 

C:\Projects\caffe\build\Caffe.sln 
을 프로젝트로 한다. Debug mode의 경우 Debug  mode로 Command Line에서 컴파일 하면 가능하다. 

convert_minist_data.cpp 파일을 분석한다. 
다음 함수를 살펴 본다.

Debug를 위한 parameter 는 다음과 같다.
~~~
c:\Projects\caffe\data\mnist\train-images-idx3-ubyte  c:\Projects\caffe\data\mnist\train-labels-idx1-ubyte  C:\Projects\caffe\examples\mnist\mnist_train_lmdb  --backend=lmdb
~~~
즉, convert_minist_data.exe의 입력 파라미터는 train, label 데이터 파일, 그리고 lmdb 파일, 마지막으로 저장형식 --backend=lmdb 로 정의 된다.

~~~cpp
void convert_dataset(const char* image_filename, const char* label_filename,
        const char* db_path, const string& db_backend) {
  // Open files
  std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
  std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
  CHECK(image_file) << "Unable to open file " << image_filename;
  CHECK(label_file) << "Unable to open file " << label_filename;
  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items;
  uint32_t num_labels;
  uint32_t rows;
  uint32_t cols;

  image_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
  label_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
  image_file.read(reinterpret_cast<char*>(&num_items), 4);
  num_items = swap_endian(num_items);
  label_file.read(reinterpret_cast<char*>(&num_labels), 4);
  num_labels = swap_endian(num_labels);
  CHECK_EQ(num_items, num_labels);
  image_file.read(reinterpret_cast<char*>(&rows), 4);
  rows = swap_endian(rows);
  image_file.read(reinterpret_cast<char*>(&cols), 4);
  cols = swap_endian(cols);


  scoped_ptr<db::DB> db(db::GetDB(db_backend));
  db->Open(db_path, db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  char label;
  char* pixels = new char[rows * cols];
  int count = 0;
  string value;

  Datum datum;
  datum.set_channels(1);
  datum.set_height(rows);
  datum.set_width(cols);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int item_id = 0; item_id < num_items; ++item_id) {
    image_file.read(pixels, rows * cols);
    label_file.read(&label, 1);
    datum.set_data(pixels, rows*cols);
    datum.set_label(label);
    string key_str = caffe::format_int(item_id, 8);
    datum.SerializeToString(&value);

    txn->Put(key_str, value);

    if (++count % 1000 == 0) {
      txn->Commit();
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
      txn->Commit();
  }
  LOG(INFO) << "Processed " << count << " files.";
  delete[] pixels;
  db->Close();
}
~~~

위 코드에서 magic 은 각 File이 정상적인 File인가를 Check하는 부분이다.
그 다음 부터 data를 가져 오는데, 각 parameter의 값은 다음과 같이 된다.

| parameter | Value |
|---|---|
| num_items  | 6000 |
| num_labels | 6000 |
| rows       | 28 |
| cols       | 28 |

즉, 6000개의 데이터와 그에 대응하는 Label이 있고 $28 \times 28$의 image 데이터가 있는 것이다.  

그 다음에는 lmdb 파일을 만드는 부분이다. 
만일 lmdb 파일을 만드는 폴더와 파일이 존재하면 에러가 나서 더 이상 진행이 안된다. 

~~~cpp
  scoped_ptr<db::DB> db(db::GetDB(db_backend));
  db->Open(db_path, db::NEW);		/// 이 부분에서  error가 난다.
  scoped_ptr<db::Transaction> txn(db->NewTransaction());
~~~

그 후에 lmdb header에 정의된 Datum 형식으로 세팅된다.
Datum 정의는 다음과 같다.

#### Datum
Datum is a Google Protobuf Message class used to store data and optionally a label. A Datum can be thought of a as **a matrix with three dimensions: width, height, and channel**.

즉 Tensorflow의 tensor와 같은 의미이다. 
자세한 설명은 다음에 있다.

https://github.com/BVLC/caffe/wiki/The-Datum-Object

Datum의 사용은 다음과 같다. 먼저, channel, width, height를 다음과 같이 세팅한다.

~~~cpp
  Datum datum;
  datum.set_channels(1);
  datum.set_height(rows);
  datum.set_width(cols);
~~~

그 다음 총 6만개 = num_items 의 데이터를  image file과 label file에서 읽어 Datum 형식으로 세팅한다.  

~~~cpp
    image_file.read(pixels, rows * cols);
    label_file.read(&label, 1);
    datum.set_data(pixels, rows*cols);
    datum.set_label(label);
~~~

이 데이터는 lmdb 파일에 Serialize 되어 저장되어야 하므로 하나의 datum 은 일단,  임시 string (:value) 로 저장되고

~~~cpp
datum.SerializeToString(&value);
~~~

caffe format에 맞춘 후 (txn->Put) 1000개 데이터가 모이면 한꺼번에 Serial 화된 데이터를 File에 쓴다 (commit) 

~~~cpp
    string key_str = caffe::format_int(item_id, 8);
    txn->Put(key_str, value);

    if (++count % 1000 == 0) {
      txn->Commit();
    }
~~~

위 과정을 수행하게 되면 다음과 같이 **Google Protobuf** 에 의해 메시지가 출력된다.

~~~
I0426 01:37:40.410586 48028 common.cpp:36] System entropy source not available, using fallback algorithm to generate seed instead.
I0426 01:37:40.419562 48028 db_lmdb.cpp:40] Opened lmdb C:\Projects\caffe\examples\mnist\mnist_train_lmdb
I0426 01:39:00.183307 48028 convert_mnist_data.cpp:93] A total of 60000 items.
I0426 01:39:09.361768 48028 convert_mnist_data.cpp:94] Rows: 28 Cols: 28
I0426 18:03:10.497315 48028 db_lmdb.cpp:112] Doubling LMDB map size to 2MB ...
I0426 18:03:12.082078 48028 db_lmdb.cpp:112] Doubling LMDB map size to 4MB ...
I0426 18:03:21.562731 48028 db_lmdb.cpp:112] Doubling LMDB map size to 8MB ...
I0426 18:03:54.452795 48028 db_lmdb.cpp:112] Doubling LMDB map size to 16MB ...
I0426 18:03:54.963431 48028 db_lmdb.cpp:112] Doubling LMDB map size to 32MB ...
I0426 18:03:56.209100 48028 db_lmdb.cpp:112] Doubling LMDB map size to 64MB ...
I0426 18:07:51.087132 48028 convert_mnist_data.cpp:113] Processed 60000 files.
~~~

#### CHECK_EQ, LOG(INFO)

해당 매크로는 google의 glog 에 정의되어 있다. 다음 링크에 자세히 설명되어 있다.

http://rpg.ifi.uzh.ch/docs/glog.html


### Input Data (Train/Test Data Format)

다음과 같이 저장된다.

~~~
c:\Projects\caffe\data\mnist\train-images-idx3-ubyte
c:\Projects\caffe\data\mnist\train-labels-idx1-ubyte
C:\Projects\caffe\examples\mnist\mnist_train_lmdb
--backend=lmdb


c:\Projects\caffe\data\mnist\t10k-images-idx3-ubyte 
c:\Projects\caffe\data\mnist\t10k-labels-idx1-ubyte 
C:\Projects\caffe\examples\mnist\mnist_test_lmdb 
--backend=lmdb
~~~

해당 파일의 포맷은  http://yann.lecun.com/exdb/mnist/ 에서 알 수 있다, 기본 형식은 다음과 같다. 

~~~
TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  60000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  10000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  10000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel
~~~

이후에는 다음 prototxt를 분석한다.
이것이 실제적인 Caffe 기반 기계학습의 내용이 된다.

~~~python 
train_net_path = 'mnist/custom_auto_train.prototxt'
test_net_path = 'mnist/custom_auto_test.prototxt'
solver_config_path = 'mnist/custom_auto_solver.prototxt'
~~~


