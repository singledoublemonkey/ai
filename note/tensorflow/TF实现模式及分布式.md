## 训练模式

深度学习训练中有2种训练方式：
- 同步模式
- 异步模式

同步模式和异步模式的主要区别是参数问题。

### 异步模式
异步模式训练中，每个训练单元都会独立获取参数然后对参数进行更新。相当于多线程操作同一个参数对象，每个处理单元都各自进行获取和更新。

### 同步模式
同步模式训练是将训练步骤中的参数进行统一管理。将等待每个训练单元更新参数之后进行等待，最后将所有的训练参数进行平均后更新。相当于多线程屏障。

### 差异性

同步模式和异步模式启动的方式一样，不同的机器上运行就可以。和异步模式不同，在第一个计算服务器启动之后，并不能直接更新参数。同步模式每一次的参数更新，都需要所有的计算服务器的梯度。

同时，在期初100轮迭代中，第一个计算服务器的平均执行时间要比第二个计算服务器慢很多，这是因为在第一迭代轮开始之前，第一个计算服务器需要等待第二个计算服务器的初始化过程，于是导致前100轮的平均速度是最慢的。

同步模式的瓶颈：若某个计算服务器卡住，其他计算服务器都需要等待这个最慢的服务器。

TF提供的解决方式：

tf.train.SyncReplicasOptimizer相关参数和方法
- replicas_to_aggregate参数，即指定同步更新时需要多少个计算服务器就可以进行更新。当次参数小于计算服务器数量时，每次迭代就不需要收集所有的梯度，从而避免被最慢的计算服务器卡住。
- get_init_tokens_op参数来控制对不同计算服务器之间的同步要求。当此参数大于0时，TF支持多次使用由同一个计算服务器得到的梯度，也可以缓解计算服务器性能瓶颈的问题。



## 分布式实现TF
>使用 tf.device("/gpu:0")可以将计算图任务放到指定的设备上进行运行，可以达到多GPU并行化加速处理模型训练的目的。为了进一步提升深度模型的训练速度，需要在多台机器上分布式运行tensorflow。

简单的示例：


```
import tensorflow as tf

hello = tf.constant("Hello, distributed tensorflow.")
# 创建一个本地集群
server = tf.train.Server.create_local_server()
# 在集群上创建会话
with tf.Session(server.target) as sess:
    # 输出上述字符串
    print(sess.run(hello))
```
在输出上述字符串之前，可以看到有类似如下输出：

```
Initialize GrpcChannelCache for job local -> {0 -> localhost:14401}
Started server with target: grpc://localhost:14401
Start master session c7c88ec51572b4dd with config:
```
第一步，初始化本地job。第二步，启动本地服务，端口是14401。第三步，启动master会话。

上述是一个很简单的启动分布式例子。可以看到启动tensorflow分布式时，有三点，分布式是以job为划分（此例中只有一个任务task），其次分布式会启动一个服务供，分布式其实是以master/salve模式启动会话。

实际上，tensorflow通过一系列task来执行计算图中的运算，task汇聚成为一个job，即一个job包含一个或者多个task。一般情况，不同任务跑在不同的机器上，但是对于GPU而言，不同的任务可以跑在同一台机器的不同GPU上。

当集群中有多个任务时，需要tf.train.ClusterSpec来指定每一个task运行的机器。

以下例子给出运行2个任务的运行情况：

First：

```
import tensorflow as tf

hello = tf.constant("Hello, distributed tensorflow.")
# 生成2个任务的集群，一个跑在20000端口，一个跑在20001端口
cluster = tf.train.ClusterSpec({"local": ["localhost:20000", "localhost:20001"]})
# 根据集群信息创建Server，因为此任务是第一个任务，所以task_index为0
server = tf.train.Server(cluster, job_name="local", task_index=0)

with tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(hello))

server.join()
```
Second：

```
import tensorflow as tf

hello = tf.constant("Hello, distributed tensorflow 2.")

cluster = tf.train.ClusterSpec({"local": ["localhost:20000", "localhost:20001"]})
# 第二个任务，所以task_index为1
server = tf.train.Server(cluster, job_name="local", task_index=1)

with tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(hello))

server.join()
```

启动运行第一个任务，首先会有如下输出：

```
Initialize GrpcChannelCache for job local -> {0 -> localhost:20000, 1 -> localhost:20001}
Started server with target: grpc://localhost:20000
CreateSession still waiting for response from worker: /job:local/replica:0/task:1
CreateSession still waiting for response from worker: /job:local/replica:0/task:1
...
```
前两行依旧是启动服务配置，后面的表示等待其他task的加入，任务一处于阻塞状态。若第二个没有启动，就会一直输出此信息。

启动第二个任务，会有如下输出：

```
Initialize GrpcChannelCache for job local -> {0 -> localhost:20000, 1 -> localhost:20001}
Started server with target: grpc://localhost:20001
Start master session efeb3f2380a99bb5 with config:

...

Const: (Const): /job:local/replica:0/task:0/cpu:0
b'Hello, distributed tensorflow 2.'
```
> 服务启动会一直处于运行状态

接着第一个任务，会有如下输出：

```
Const: (Const): /job:local/replica:0/task:0/cpu:0
b'Hello, distributed tensorflow.'
```
可以看到第二个任务的计算也是由第一个任务去执行的。类似GPU指定方式，也可以通过tf.device指定操作运行在哪个任务上。

比如将第一个任务的做如下修改：

```
with tf.device("/job:local/task:1"):
    hello = tf.constant("Hello, distributed tensorflow.")
```
第一个任务将会有如下输出：

```
Const: (Const): /job:local/replica:0/task:1/cpu:0
b'Hello, distributed tensorflow.'
```
>在本机测试时发现，通过将任务一指定device，运行结果输出上会快一些。

在实际情况中，不会只定义一个local工作，一会定义两个工作，一个用于负责存储、获取和更新变量值，这个工作统称为参数服务器（paramater server，ps）。另外一个工作负责运行反向传播算法获取更新梯度，这个工作涉及的任务统称为计算服务器（worker）

### 分布式训练模型方式

一般有2种方式，

- 计算图内分布式（in-graph replication）：此种方式，所以的任务都会使用一个tensorflow计算图中的变量，即模型中的参数，只是讲计算部分发布到不同的计算服务器上。计算图内分布式需要有一个中心节点来生成这个计算图并分配计算任务，所以当数据量太大，这个中心节点容易造成性能瓶颈。
- 计算图之间分布式（between-graph replication）：此种方式在每一个计算服务器上都会创建一个独立的tensorflow计算图，但不同计算图的相同参数需要以一种固定方式放到同一个参数服务器上。tensorflow提供了tf.train.replica_device_setter函数来完成此功能。同时为了解决参数同步困难问题，也提供了tf.train.SyncReplicasOptimizer函数来完成参数同步。因此此种方式应用也更加广泛。


计算图之间分布式完成深度学习模型训练的异步更新和同步更新。

代码在：同级目录下的example/mnist下
