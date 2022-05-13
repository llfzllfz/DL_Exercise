# DL_Exercise
日常的深度学习练习（采用pytorch）

## 文件目录

```python
CNN
---data.py
---Start_CNN.py
---LeNet.py
---AlexNet.py
---VGG.py
---ResNet.py
RNN
---data_rnn.py
---LSTM.py
---Start_RNN.py
Transformer
---bert-fine-tune.py
---transformer.py
---transformer_config.py
---transformer_data.py
---transformer_test.py
touch_exercise.py
GCN.py
GAT.py
```

## 文件说明

### CNN

#### data.py

包含了数据处理模块，目前载入的数据为Cifar-10，需自行下载数据集，同时修改路径

数据shape为（-1，3，32，32）

范围在0-1之间（/255归一化）

#### Start.py

包含了训练模块和测试模块，支持命令行方式运行

```python
命令行运行参数介绍：
# 批处理大小，默认32
--batch-size
# 训练轮次，默认20
--epochs
# 学习率，默认0.01
--lr
# 采用的模型，目前支持LeNet
--model
# 模型存放的路径，默认同文件夹下'./model.ckpt'
--model-save-path
# 是否采用cuda运行，默认否，当且仅当改参数为True，且机器支持GPU时可运行cuda
--cuda
# 将图片resize为相应的大小，默认为32
--resize
```

例如：

```python
python Start.py --cuda --lr=0.001 --epochs=50 --resize=224 --batch-size=256 --model=ResNet
```



#### LeNet.py

基础的LeNet模型，图片大小为32

具体见blog：[https://llfzllfz.github.io/2022/04/22/LeNet/#more](https://llfzllfz.github.io/2022/04/22/LeNet/#more)



#### AlexNet.py

基础的AlexNet，图片大小为224



#### VGG.py

基础的VGG，图片大小为224



#### ResNet.py

基础的ResNet， 图片大小为224



### RNN

#### data_rnn.py

路径需要在文件中重新设置

将数据转换成（-1，32*32，3）并且载入torch中的数据加载中



#### LSTM.py

使用torch.nn.lstm()+torch.nn.Linear()

对Cifar-10数据集进行分类



#### Start_RNN.py

包含了训练模块和测试模块，支持命令行方式运行

通过python Start_RNN.py -h获取对应参数得相关信息



### Transformer

#### bert-fine-tune.py

熟悉bert的fine-tune

数据集采用的是IMDB

采用last-hidden-state做fine-tune

#### transformer.py

transformer模型，参考https://wmathor.com/index.php/archives/1455/

#### transformer_config.py

transformer的部分参数设置

#### transformer_data.py

自己构造transformer相关的数据

#### transformer_test.py

测试自己实现的transformer模型

### touch_Exercise(感知机)

使用的数据集是kaggle上面的house-prices-advanced-regression-techniques数据集，该文件用来熟悉touch的一些基础操作

### GCN.py

使用的是空手道数据集，参考了网上的一些内容（主要是没想到GCN怎么应用到Cifar-10的分类上）

模型采用的是$D^{-\frac{1}{2}}AD^{-\frac{1}{2}}X$

### GAT.py

实现的GAT模型，同样采用空手道数据集做测试

