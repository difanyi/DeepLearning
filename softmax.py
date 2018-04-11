# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#mnist数据集情况 55000个样本，10000个测试集，5000个验证集
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)


#session
sess=tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])#输入数据的地方，第二个参数表示tensor的shape

#初始化weight和bais
W = tf.Variable(tf.zeros([784, 10]))#特征维数784，10代表10类
b = tf.Variable(tf.zeros([10]))

#softmax算法
y = tf.nn.softmax(tf.matmul(x,W)+b)#matmul 矩阵乘法函数

#定义loss，多分类通常使用交叉熵函数
y_ = tf.placeholder(tf.float32,[None,10])#输入真实的label
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
#reduce_mean 对每个结果求均值

#定义优化算法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) 

#初始化参数并run
tf.global_variables_initializer().run()

#随机取100条样本构成一个mini-batch，并feed给placeholder,随机梯度下降
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})
    
#计算分类是否正确
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))


