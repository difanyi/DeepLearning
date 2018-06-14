import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
#mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

def MLE(x,y):
    X_train, X_test,y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)
    print (X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    sess = tf.InteractiveSession()

    in_units = 5128
    h1_units = 300
    label_count = 30
    W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_units]))
    W2 = tf.Variable(tf.zeros([h1_units, label_count]))
    b2 = tf.Variable(tf.zeros([label_count]))

    x = tf.placeholder(tf.float32, [None, in_units])
    keep_prob = tf.placeholder(tf.float32)

    hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, label_count])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

    # Train
    tf.global_variables_initializer().run()
    for i in range(3000):
        input_queue = tf.train.slice_input_producer([X_train, y_test], shuffle=False)
        image_batch, label_batch = tf.train.batch(input_queue, batch_size=100, num_threads=1, capacity=64)
        #print(image_batch.shape,label_batch.shape)
        train_step.run({x: image_batch, y_: label_batch, keep_prob: 0.75})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("概率是")
    print(accuracy.eval({x: X_test, y_: y_test, keep_prob: 1.0}))