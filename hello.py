import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.float32, [1000, 500])
b = tf.placeholder(tf.float32, [300, 1000])
c = tf.matmul(a, b)

with tf.Session() as sess:
	for i in range(10000):
		d = sess.run(c, feed_dict = {a: np.ones((1000, 500)), b: np.ones((300, 1000))})
		print(i, d)
