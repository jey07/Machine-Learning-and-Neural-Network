import tensorflow as tf 
import seaborn as sb
import numpy as np
import math

data = tf.keras.datasets.mnist.load_data()


x_train,y_train = data[0]   #Load train data
x_test,y_test = data[1]     #Load test data

#Split train data into train and validation
x_valid = x_train[55000:,:,:]
y_valid = y_train[55000:]
x_train = x_train[:55000,:,:]
y_train = y_train[:55000]


#sb.heatmap(x_train[2,:,:])  To check how image looks like

#flattening the image for neural network
def flattenimage(x):
    num_pixels = x.shape[1]*x.shape[2]
    x=x.reshape(x.shape[0],num_pixels).astype('float32')
    
    return(x)
    
x_train = flattenimage(x_train)
x_test = flattenimage(x_test)
x_valid = flattenimage(x_valid)

#one hot encoding of labels.
y_train = np.eye(10)[y_train]
y_test =  np.eye(10)[y_test]
y_valid = np.eye(10)[y_valid]
    

# Making the neural netowrk architecture
n_input = 784  # input layer (28x28 pixels)
n_hidden1 = 512  # 1st hidden layer
n_hidden2 = 256  # 2nd hidden layer
n_hidden3 = 128  # 3rd hidden layer
n_output = 10  # output layer (0-9 digits)

learning_rate = 1e-3
n_iterations = 1000
batch_size = 128
dropout = 0.5  #To prevent overfitting


x_out = tf.placeholder("float", [None, n_input])
y_out = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}


l1 =  tf.add(tf.matmul(x_out,weights['w1']),biases['b1'])
l2 = tf.add(tf.matmul(l1, weights['w2']), biases['b2'])
l3 = tf.add(tf.matmul(l2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(l3, keep_prob)

op = tf.matmul(l3, weights['out']) + biases['out']

cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(
                                labels=y_out, logits=op))

#Adam optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(op, 1), tf.argmax(y_out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#%%
for i in range(n_iterations):
    st = 0
    for j in range(math.ceil(x_train.shape[0]/batch_size)) :
        batch_x = x_train[st:(batch_size+st)]
        batch_y = y_train[st:(batch_size+st)]
        
        sess.run(train_step, feed_dict={
                x_out: batch_x, y_out: batch_y, keep_prob: dropout})
        st = st + batch_size
    # print loss and accuracy (per minibatch)
     
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy],
        feed_dict={x_out: x_train, y_out:y_train, keep_prob: 1.0})
        
        print("Iteration",str(i),"\t| Loss =",str(minibatch_loss),"\t| Accuracy =",str(minibatch_accuracy))
    
        
test_accuracy = sess.run(accuracy, feed_dict={x_out: x_test, y_out: y_test, keep_prob: 1.0})

print("\nAccuracy on test set:", test_accuracy)