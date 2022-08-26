import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.reset_default_graph()
input_data = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)
output_data = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)

# y = mx+b
# Set up training variables for line equation
slopeM = tf.Variable(0.5, dtype=tf.float32)
yInterceptB = tf.Variable(3, dtype=tf.float32)

# Find output value
modelOperation = slopeM * input_data + yInterceptB

error = modelOperation - output_data
squared_error = tf.square(error)
loss = tf.reduce_mean(squared_error)

init = tf.compat.v1.global_variables_initializer()

# Input and output values
x_values = [0, 1, 2, 3, 4]
y_values = [1, 3, 5, 7, 9]

# Weight that gets changed, learning rate is η
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.005)

train = optimizer.minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for i in range(2000):
        sess.run(train, feed_dict={input_data: x_values, output_data: y_values})
        if i % 100 == 0:
            print(sess.run([slopeM, yInterceptB]))
            plt.plot(x_values, sess.run(modelOperation, feed_dict={input_data: x_values}))

    print(sess.run(loss, feed_dict={input_data: x_values, output_data: y_values}))
    plt.plot(x_values, y_values, 'ro', 'Training Data')
    plt.plot(x_values, sess.run(modelOperation, feed_dict={input_data: x_values}))

    plt.show()

# 50,000 steps combined error: 1.8950459e-10
# 2000 steps combined error: 1.0895243e-05
