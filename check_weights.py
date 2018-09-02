import numpy as np
from tensorflow.python import pywrap_tensorflow

def read_tensorflow_weights(file_name):
	reader = pywrap_tensorflow.NewCheckpointReader(file_name)
	var_list = reader.get_variable_to_shape_map()
	return {var: reader.get_tensor(var) for var in var_list}

d1 = read_tensorflow_weights('human-detection/output/res152/coco_2017_train/default/res152.ckpt')
d2 = read_tensorflow_weights('hm-all-sep/output/res152/coco_2017_train/default/res152_faster_rcnn_iter_900000.ckpt')

for var in d1:
	assert(var in d2)
