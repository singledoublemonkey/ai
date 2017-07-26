## MNIST数据集异步模式运行（计算图之间分布式）
```
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference as mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZION_RATE = 0.0001
TRAINING_STEPS = 10000

MODE_SAVE_PATH = "./mnist_model/"
DATA_PATH = "../mnist/"

# 指定程序运行中给出参数
FLAGS = tf.app.flags.FLAGS

# 指定job是计算服务器还是参数服务器
tf.app.flags.DEFINE_string('job_name', 'worker', '"ps" or "worker"')

# 指定参数服务器地址，默认是本机的列表
tf.app.flags.DEFINE_string('ps_hosts', 'localhost:20000,localhost:20001',
                           'Comma-separated list of host:port for the parameter server job.'
                           ' eg: localhost:20000,localhost:20001')

# 指定计算服务器地址，默认是本机的列表
tf.app.flags.DEFINE_string('worker_hosts', 'localhost:20003,localhost:20004',
                           'Comma-separated list of host:port for the parameter server job.'
                           ' eg: localhost:20003,localhost:20004')

# 指定当前任务ID
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the train')

# 定义模型
def build_model(x, y_, is_chief):
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZION_RATE)
    # 计算前向传播结果
    y = mnist_inference.infreence(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    # 定义交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 定义损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               60000 / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return global_step, loss, train_op

def main(argv=None):
    # 解析参数
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    # 集群服务
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # 通过cluster和当前任务创建Server
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)

    # 参数服务器只需要接受参数，不需要参与训练，server.join会一直阻塞在此处
    if FLAGS.job_name == 'ps':
        server.join()

    # 定义计算服务器的操作，在所有计算服务器中，必须要有一个主计算服务器，除了计算负责反向传播结果，还需要负责输出日志和保存模型
    is_chief = (FLAGS.task_id == 0)
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

    with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d' % FLAGS.task_id,
                                                  cluster=cluster)):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], 'x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], 'y-input')
        global_step, loss, train_op = build_model(x, y_, is_chief)
        # 定义保存模型
        saver = tf.train.Saver()
        # 定义日志输出操作，旧版本的tf.merge_all_summaries()已经废弃
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        # Supervisor管理训练深度学习模型的通用功能，能统一管理队列操作、模型保存、日志输出及会话生成
        sv = tf.train.Supervisor(
            is_chief=is_chief,          # 定义当前计算服务器是否主计算服务器，只有主计算服务器才会保存日志和模型
            logdir=MODE_SAVE_PATH,      # 指定日志和模型输出路径
            init_op=init_op,            # 指定初始化操作
            summary_op=summary_op,      # 指定日志输出操作
            saver=saver,                # 指定用于保存模型的Saver
            global_step=global_step,    # 指定当前迭代轮数，会用于生成保存模型的名称
            save_model_secs=60,         # 指定模型保存时间间隔
            save_summaries_secs=60      # 指定日志输出时间间隔
        )
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        #生成会话
        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
        step = 0
        start_time = time.time()
        while not sv.should_stop():
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, global_step_value = sess.run([train_op, loss, global_step],
                                       feed_dict={x: xs, y_: ys})
            if global_step_value >= TRAINING_STEPS: break
            if step > 0 and step % 100 == 0:
                duration = time.time() - start_time
                # 不同的计算服务器都会更新全局的训练轮数，所以使用global_step_value可以获得训练中使用的batch总数
                sec_per_batch = duration / global_step_value
                format_str = "After %d training steps (%d global step), loss on training batch is %g. (%.3f sec/batch)"
                print(format_str % (step, global_step_value, loss_value, sec_per_batch))
            step += 1
    sv.stop()

if __name__ == '__main__':
    tf.app.run()
```



启动1：
python dist_tf_mnist_async.py --job_name='ps' --task_id=0 --ps_host='localhost:20000' --worker_hosts='localhost:20002,localhost:20003'

启动2：
python dist_tf_mnist_async.py --job_name='worker' --task_id=0 --ps_host='localhost:20000' --worker_hosts='localhost:20002,localhost:20003'

启动3：python dist_tf_mnist_async.py --job_name='worker' --task_id=1 --ps_host='localhost:20000' --worker_hosts='localhost:20002,localhost:20003'

依次启动上述命令后，观察输出：

参数服务器会一直等待阻塞。

worker_id为0的，输出类似如下：

```
After 100 training steps (100 global step), loss on training batch is 1.06693. (0.022 sec/batch)
After 200 training steps (200 global step), loss on training batch is 0.804804. (0.019 sec/batch)
After 300 training steps (300 global step), loss on training batch is 0.726759. (0.019 sec/batch)
After 400 training steps (400 global step), loss on training batch is 0.879733. (0.018 sec/batch)
...
After 5700 training steps (9765 global step), loss on training batch is 0.360651. (0.016 sec/batch)
After 5800 training steps (9966 global step), loss on training batch is 0.311247. (0.016 sec/batch)
```
worker_id为1的，输出类似如下：

```
After 100 training steps (1836 global step), loss on training batch is 0.357848. (0.002 sec/batch)
After 200 training steps (2034 global step), loss on training batch is 0.380628. (0.003 sec/batch)
After 300 training steps (2230 global step), loss on training batch is 0.379104. (0.004 sec/batch)
...
After 4000 training steps (9631 global step), loss on training batch is 0.302752. (0.013 sec/batch)
After 4100 training steps (9830 global step), loss on training batch is 0.266647. (0.013 sec/batch)
```
运行中会发现，id为0的task会先运行一段时间，然后id为1的才开始运行。即第二个计算服务器在启动之前，第一个计算服务器已经迭代运行了很多次。

异步模式下，即使有计算服务器没有正常工作，参数服务器仍然可以正常工作。而且全局迭代的次数也是所有计算服务器迭代轮数的和。

在Anaconda3环境中，执行tensorboard --logdir=/path 来启动可视化服务。如下：


```
(D:\Anaconda3) E:\>tensorboard --logdir=mnist_model/
Starting TensorBoard b'41' on port 6006
(You can navigate to http://192.168.101.2:6006)
```
可通过本机端口进行查看计算图
