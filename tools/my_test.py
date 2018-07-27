from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys, time

import _init_paths
from nets.resnet_v1 import resnetv1

def setAvaiGPUs(num_gpus = 1):
	import subprocess as sp
	ACC_AVAI_MEM = 10240
	COMMAND = 'nvidia-smi --query-gpu=memory.free --format=csv'
	# try:
	_output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
	memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
	memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
	avai_gpus = [i for i, x in enumerate(memory_free_values) if x > ACC_AVAI_MEM]
	if len(avai_gpus) < num_gpus:
		raise ValueError('Found only %d usable GPUs in the system.' % len(avai_gpus))
	print(avai_gpus[:num_gpus])
	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, avai_gpus[:num_gpus]))
	# except Exception as e:
		# print('\"nvidia-smi\" is probably not installed. GPUs are not masked.', e)
	return

if __name__ == '__main__':
	assert(len(sys.argv) == 3)
	num_gpus = int(sys.argv[2])
	if socket.gethostname() == 'ait-server-03':
		setAvaiGPUs(num_gpus)

	tfconfig = tf.ConfigProto(allow_soft_placement = True)
	tfconfig.gpu_options.allow_growth = True
	sess = tf.Session(config=tfconfig)

	net = resnetv1(num_layers = 152)
	net.create_architecture('TEST', 81, tag = 'default', anchor_scales = [2,4,8,16,32])
	saver = tf.train.Saver()
	saver.restore(sess, './output/res152/coco_2014_train+coco_2014_valminusminival/default/res152.ckpt')
