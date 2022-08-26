import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.datasets as skd
from tensorflow.keras.datasets import mnist

from Basic_MNIST_ImageRecognition import image_height, image_width

china_image = skd.load_sample_image("china.jpg")
plt.imshow(china_image)
plt.show()

print(china_image.shape)

# (427, 640, 3) --> height, width, # of color channels

input_layer = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 427, 640, 3])
conv_1 = tf.compat.v1.layers.conv2d(input_layer, filters=64, kernel_size=[2, 2], padding="same", activation=tf.nn.relu)
print(conv_1.shape)

# Placeholder shape is [None/?, 427, 640, 3]
# The None/? means that means the placeholder is expecting something

# Filters - changing the color pallet but find patterns instead of just changing  color
# 64 filters for the above convolutional network

init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    output = sess.run(conv_1, feed_dict={input_layer: [china_image]})

print(output[0].shape)

# plt.imshow(output[0]); This won't work because the computer can't print an image with 64 color channels

tf.compat.v1.reset_default_graph()
# image_height = 28
# image_width = 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()


class CNN:

    def __init__(self, image_height, image_width, channels):
        self.input_layer = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, channels])
        conv_layer_1 = tf.compat.v1.layers.conv2d(self.input_layer, filters=32, kernel_size=[2, 2],
                                                  activation=tf.nn.relu)
        # pooling_layer_1 = tf.layers.max_pooling2d(conv_layer_1, pool_size=[2, 2], strides=2)

        # Number of image numbers
        num_classes = 10

        # Second convolutional and pooling layer to speed up the process by clearing more pixels and being efficient
        # conv_layer_2 = tf.layers.conv2d(self.input_layer, filters=32, kernel_size=[2, 2], activation=tf.nn.relu)
        pooling_layer_2 = tf.compat.v1.layers.max_pooling2d(conv_layer_1, pool_size=[2, 2], strides=2)

        flattened_pooling = tf.compat.v1.layers.flatten(pooling_layer_2)
        dense_layer = tf.compat.v1.layers.dense(flattened_pooling, 1024, activation=tf.nn.relu)
        dropout = tf.compat.v1.layers.dropout(dense_layer, rate=0.4, training=True)
        outputs = tf.compat.v1.layers.dense(dropout, num_classes)

        self.choice = tf.argmax(outputs, axis=1)  # because the output is only 1 dimensional there's only 1 axis/vector
        self.probability = tf.nn.softmax(outputs)
        self.labels = tf.compat.v1.placeholder(dtype=tf.float32, name="labels")

        # shorthand for the function returning two things
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.choice)
        one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, dtype=tf.int32), depth=num_classes)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=outputs)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
        self.train_operation = optimizer.minimize(self.loss, global_step=tf.compat.v1.train.get_global_step())


steps = 10000
batch_size = 32

test_img = x_test[1]
plt.imshow(test_img)
test_img = test_img.reshape(-1, 28, 28, 1)

x_train = x_train.reshape(-1, image_height, image_width, 1)

cnn = CNN(28, 28, 1)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    step = 0
    while step < steps:
        none, current_accuracy = sess.run((cnn.train_operation, cnn.accuracy_op),
                                          feed_dict={cnn.input_layer: x_train[step:step + batch_size],
                                                     cnn.labels: y_train[step:step + batch_size]})
        step += batch_size

        if steps % 100 == 0:
            print(current_accuracy)

    print("\nIt is done.")

    # prints the number the computer thinks  is correct
    print("\n" + str(sess.run(cnn.choice, feed_dict={cnn.input_layer: test_img})))

# Multiply steps and batch size, and then if the number is bigger than the total number of images,
# then drop the step value. MNIST has 60000 images.

# Testing and Revising the convolutional neural network
test_img = x_test[1]
plt.imshow(test_img)
plt.show()

test_img = test_img.reshape(-1, 28, 28, 1)

x_train = x_train.reshape(-1, image_height, image_width, 1)
cnn = CNN(28, 28, 1)
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    step = 0
    while step < steps:
        sess.run((cnn.train_operation, cnn.accuracy_op), feed_dict={cnn.input_layer: x_train[step:step + batch_size],
                                                                    cnn.labels: y_train[step:step + batch_size]})
        step += batch_size

    print("The computer thinks that image is " + str(sess.run(cnn.choice, feed_dict={cnn.input_layer: test_img})))
