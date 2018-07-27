import tensorflow as tf
import numpy as np
import time

a = tf.placeholder(tf.float32, [500, 1000])
b = tf.placeholder(tf.float32, [1000, 300])
c = tf.matmul(a, b)

with tf.Session() as sess:
	t = time.time()
	for i in range(1000):
		d = sess.run(c, feed_dict = {a: np.ones((500, 1000)), b: np.ones((1000, 300))})
		print(i, d.shape)
	print(time.time() - t)
