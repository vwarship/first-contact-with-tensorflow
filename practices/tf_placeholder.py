import tensorflow as tf

a = tf.placeholder("float")
b = tf.placeholder("float")
y = tf.multiply(a, b)

with tf.Session() as sess:
    print(sess.run(y, feed_dict={a: 3, b: 3}))
