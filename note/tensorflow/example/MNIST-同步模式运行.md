### MNIST-同步模式运行（计算图之间分布式）

```python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import code.my.book_4_mnist_inference as mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZION_RATE = 0.0001
MOVE_AVEAGE_DECAY = 0.99
TRAINING_STEPS = 10000

MODE_SAVE_PATH = "./mnist_model/"
DATA_PATH = "../mnist/"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', 'worker', '"ps" or "worker"')

tf.app.flags.DEFINE_string('ps_hosts', 'localhost:20000,localhost:20001',
                           'Comma-separated list of host:port for the parameter server job.'
                           ' eg: localhost:20000,localhost:20001')

tf.app.flags.DEFINE_string('worker_hosts', 'localhost:20003,localhost:20004',
                           'Comma-separated list of host:port for the parameter server job.'
                           ' eg: localhost:20003,localhost:20004')

tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the train')

def build_model(x, y_, n_workers, is_chief):
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZION_RATE)

    y = mnist_inference.infreence(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    # # 定义滑动平均模型
    # variable_avagers = tf.train.ExponentialMovingAverage(MOVE_AVEAGE_DECAY, global_step)
    # variable_avagers_op = variable_avagers.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               60000 / BATCH_SIZE, LEARNING_RATE_DECAY)
    # ***和异步模式的区别是，需要使用SyncReplicasOptimizer函数来实现同步更新***
    opt = tf.train.SyncReplicasOptimizer(
        tf.train.GradientDescentOptimizer(learning_rate),
        # 每一轮更新需要多少个计算服务器得出梯度
        replicas_to_aggregate=n_workers,
        # 指定总共有多少个计算服务器
        total_num_replicas=n_workers
    )

    train_op = opt.minimize(loss, global_step=global_step)
    return global_step, loss, train_op, opt

def main(argv=None):

    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    n_workers = len(worker_hosts)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_id == 0)
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

    with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d' % FLAGS.task_id,
                                                  cluster=cluster)):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], 'x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], 'y-input')
        global_step, loss, train_op, opt = build_model(x, y_, n_workers, is_chief)

        saver = tf.train.Saver()

        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

        # 同步模式下，主计算服务器需要协调不同计算服务器计算得到的参数梯度并最终更新参数
        # 此需要额外进行一些初始化操作
        if is_chief:
            # 定义协调不同计算服务器的队列，并定义初始化操作
            chief_queue_runner = opt.get_chief_queue_runner()
            init_tokens_op = opt.get_init_tokens_op(0)

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
        # 生成会话
        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        # 开始训练之前，主计算服务器启动需要协调计算服务器的队列，并执行初始化操作
        if is_chief:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)

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
启动方式同异步模式启动
