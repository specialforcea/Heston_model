import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import time
tf.disable_v2_behavior()



print(tf.__version__)



training_set = pd.read_csv('training set.csv').iloc[:,1:] 

training_set_neg = training_set[training_set['heston_price']<0]
training_set_neg.to_csv('negs.csv')
training_set = training_set[training_set['heston_price']>=0]
#print (training_set.head())
training_X = training_set.loc[:,training_set.columns!='heston_price']
#print(training_X.head())
training_Y = training_set['heston_price']

training_X, training_Y = training_X.values,training_Y.values
#print(training_X.shape,training_Y.shape)
#############################################################
test_set = pd.read_csv('test set.csv').iloc[:,1:] 
test_set = test_set[test_set['heston_price']>=0]
#print (training_set.head())
test_X = test_set.loc[:,test_set.columns!='heston_price']
#print(training_X.head())
test_Y = test_set['heston_price']

#training_X, training_Y = training_X.values,training_Y.values
test_X, test_Y = test_X.values,test_Y.values

#print(training_X.shape,training_Y.shape)

#normalize
meanX = np.mean(training_X,axis=0)
stdX = np.std(training_X,axis=0)
meanY = np.mean(training_Y,axis=0)
stdY = np.std(training_Y,axis=0)

normX = (training_X - meanX) / stdX
normY = (training_Y - meanY) / stdY
#print(meanX,stdX)
dim = normX.shape[1]


# inference = sequence of feed-forward equations from input to output 
# TensorFlow provides higher level function for all kinds of standard layers
# for vanilla layers, the function is tf.layers.dense() 

# the weights and biases are encapsulated and do not explicitly appear in the code

# the argument kernel_initializer allows to control the initialization of the weights
# (the biases are all initialized to 0)
# tf.variance_scaling_initializer() implements the Xavier-He initialization
# (centred Gaussian with variance 1.0 / num_inputs)
# widely considered an effective default, see e.g. Andrew Ng's DL spec on Coursera

def inference(xs):
    
    # hidden layers, note that the weights and biases are encpasulated in the tf functions
    a1 = tf.layers.dense(xs, 5, activation = tf.nn.elu, kernel_initializer = tf.variance_scaling_initializer)
    a2 = tf.layers.dense(a1, 5, activation = tf.nn.elu, kernel_initializer = tf.variance_scaling_initializer)
    a3 = tf.layers.dense(a2, 3, activation = tf.nn.elu, kernel_initializer = tf.variance_scaling_initializer)
    a4 = tf.layers.dense(a3, 2, activation = tf.nn.elu, kernel_initializer = tf.variance_scaling_initializer)
    #a5 = tf.layers.dense(a4, 2, activation = tf.nn.elu, kernel_initializer = tf.variance_scaling_initializer)
    # output payer
    ys = tf.layers.dense(a4, 1, activation = None, kernel_initializer = tf.variance_scaling_initializer)
    
    return ys


# calculation graph for prediction and loss

# the instructions below don't calculate anything, they initialize a calculation graph in TensorFlow's memory space
# when the graph is complete, we can run it in a TensorFlow session, on CPU or GPU

# since TensorFlow knows the calculation graph, it will not only evaluate the results, but also the gradients, 
# very effectively, with the back-propagation equations

# reserve space for inputs and labels
inputs = tf.compat.v1.placeholder(shape=[None,dim], dtype = tf.float32)
labels = tf.compat.v1.placeholder(shape=[None,1], dtype = tf.float32)

# calculation graphs for predictions given inputs and loss (= mean square error) given labels
predictions = inference(inputs)
loss = tf.losses.mean_squared_error(labels, predictions)


# definition of the optimizer
# the optimizer computes the gradient of loss to all weights and biases,
# and modifies them all by a small step (learning rate) in the direction opposite to the gradient
# in order to progressively decrease the loss and identify the set of weights that minimize it

learning_rate = 0.05
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # optimizer obejct
optimize = optimizer.minimize(loss) #  this op now computes gradient and moves weights
# the op 'optimize' performs one iteration of gradient descent

# we can display predictions before, during and after training
# to do this, we execute the inference result named 'predictions' on the session
# with some arbitrary inputs xs
def predict(xs, sess):
    # first, normalize
    nxs = (xs - meanX) / stdX
    # forward feed through ANN
    nys = sess.run(predictions,feed_dict={inputs:nxs})
    # de-normalize output
    ys = meanY + stdY * nys
    # we get a matrix of shape [size of xs][1], which we reshape as vector [size of xs]
    return np.reshape(ys, [-1])


# training set
feed_dict = {inputs:normX, labels:normY[:,np.newaxis]}

# run the optimizer a few times (called epochs)
epochs = 200000

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# we save results after each epoch for visualization
yAxes = [predict(test_X, sess)] 
losses = [sess.run(loss, feed_dict=feed_dict)]
pltepoche = []
# go
epo = 0
MSE_test =10.
while epo<epochs and MSE_test>0.01:
    for e in range(1000):
        _, l = sess.run([optimize, loss], feed_dict=feed_dict)
    #yAxes.append(predict(test_X, sess))
    epo += 1000
    losses.append(l)
    pltepoche.append(epo)

#print(len(losses),losses[:5])
#plt.scatter(test_Y,yAxes[-1])
    MSE_train = round(np.sqrt(sum((training_Y-predict(training_X,sess))**2)/training_Y.shape[0]),4)
    MAE_train = round(sum(abs(training_Y-predict(training_X,sess)))/training_Y.shape[0],4)

    MSE_test = round(np.sqrt(sum((test_Y-predict(test_X,sess))**2)/test_Y.shape[0]),4)
    MAE_test = round(sum(abs(test_Y-predict(test_X,sess)))/test_Y.shape[0],4)


    print('MSE_train:{},MAE_train:{},MSE_test:{},MAE_test:{}'.format(MSE_train,MAE_train,MSE_test,MAE_test))
#print(epo)
# test_df = test_set
# test_df['predict_Y'] = predict(test_X,sess)
# test_df_1 = test_df[abs(test_df['predict_Y']-test_df['heston_price'])<5]
# test_df_2 = test_df[abs(test_df['predict_Y']-test_df['heston_price'])>=5]
# test_df_2.to_csv('problems.csv')
# print(test_df_1.mean(),test_df_2.mean())
# print('good samples,MSE_train:{},MAE_train:{}'.format(
#     np.mean((test_df_1['predict_Y']-test_df_1['heston_price'])**2), 
#     np.mean(abs(test_df_1['predict_Y']-test_df_1['heston_price']))))
# plt.scatter(test_df_1['heston_price'],test_df_1['predict_Y'],c='r')
# plt.scatter(test_df_2['heston_price'],test_df_2['predict_Y'],c='b')

plt.subplot(1,3,1)
plt.scatter(pltepoche,np.log(losses[1:]))

plt.subplot(1,3,2)
plt.scatter(test_Y,predict(test_X,sess))
plt.plot(test_Y,test_Y,c='r')

plt.subplot(1,3,3)
plt.scatter(training_Y,predict(training_X,sess))
plt.plot(training_Y,training_Y,c='r')

plt.show()
#
#t = time.time()
#predict(test_X, sess)
#print('takes {}'.format(time.time() - t))
#saver = tf.train.Saver()
#saver.save(sess,'/Users/yueyuchen/Documents/Academy/Research/Notes/Algo trading & pricing/heston/DL32222.ckpt')
#saver.restore(sess, "/Users/yueyuchen/Documents/Academy/Research/Notes/Algo trading & pricing/heston/DL32222.ckpt")
#sess.close()