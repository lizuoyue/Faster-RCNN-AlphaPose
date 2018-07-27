import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.float32, [500, 1000])
b = tf.placeholder(tf.float32, [1000, 300])
c = tf.matmul(a, b)

with tf.Session() as sess:
	for i in range(10000):
		d = sess.run(c, feed_dict = {a: np.ones((500, 1000)), b: np.ones((1000, 300))})
		print(i, d)
