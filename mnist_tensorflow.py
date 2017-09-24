import os
import struct
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.model_selection import train_test_split

class EarlyStopping(object):
    def __init__(self, patience=0, verbose=0):
        self.step = 0
        self.loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, _loss):
        if self.loss < _loss:
            self.step += 1

            if self.step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True

        else:
            self.step = 0
            self.loss = _loss



mnist = datasets.fetch_mldata('MNIST original', data_home=".")
n = len(mnist.data) 
N = 1000
indices = np.random.permutation(range(n))[:N]
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]       #出力層 1-0f-K表現で出力
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

early_stop = EarlyStopping(patience=10, verbose=1)

Lambda = 1000
C = 1/Lambda
X_train_pre = X_train/255*C
X_test_pre = X_test/255*C

y_train_s = np.zeros((len(y_train),10), dtype=np.int)
for i in range(0,len(y_train),1):
    y_train_s[i,y_train[i]] = 1

y_test_s = np.zeros((len(y_test),10), dtype=np.int)
for i in range(0,len(y_test),1):
    y_test_s[i,y_test[i]] = 1

tf.set_random_seed(20170816)
num_filters = 32

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

W_conv = tf.Variable(tf.truncated_normal([5,5,1,num_filters], stddev=0.1))

h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1,1,1,1], padding='SAME')
h_pool =tf.nn.max_pool(h_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
h_pool_flat = tf.reshape(h_pool, [-1, 14*14*num_filters])

num_units1 = 14*14*num_filters
num_units2 = 1024


#重みW[i]の初期化 
def weight_variable(in_unit, out_unit):
    in_neuron = in_unit
    out_neuron = out_unit
    #Heの初期値
    stddev = np.sqrt( 2 / (in_neuron * out_neuron))
    return tf.Variable(tf.truncated_normal([in_unit, out_unit], stddev=stddev))

w2 = weight_variable(num_units1, num_units2)
b2 = tf.Variable(tf.zeros([num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool_flat, w2) + b2)
output = tf.nn.dropout(hidden2, keep_prob=0.5)

w0 = weight_variable(num_units2, 10)
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(output, w0) + b0)

t = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

k = 100
X_train_p = X_train_pre[0:k,:]
y_train_p = y_train_s[0:k,:]
X_test_p = X_test_pre[0:k,:]
y_test_p = y_test_s[0:k,:]

acc_test_val = []

for i in range(0,900,1):
    sess.run(train_step, feed_dict={x:X_train_p, t:y_train_p})
    
    if i % k == 0 and i !=0:
        acc_test_val.append(sess.run(accuracy, feed_dict={x:X_test_p, t:y_test_p}))
        loss_val, acc_val = sess.run( [loss, accuracy], feed_dict={x:X_train_p, t:y_train_p})
        print ('Step: %d, Loss: %f, Accuracy(train): %f, Accuracy(test): %f' % (i, loss_val, acc_val, acc_test_val[-1]))
        X_train_p = X_train_pre[i:i+k,:]
        y_train_p = y_train_s[i:i+k,:]
        X_test_p = X_test_pre[i:i+k,:]
        y_test_p = y_test_s[i:i+k,:]

        if early_stop.validate(loss_val):
            break

print('train認証精度: ', acc_val)
print('test認証精度: ', acc_test_val[-1])


