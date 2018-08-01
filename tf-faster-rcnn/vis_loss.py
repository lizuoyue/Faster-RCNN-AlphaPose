import re
import matplotlib.pyplot as plt
import os
import numpy as np

os.popen('scp ait:/disks/data4/zyli/Faster-RCNN-AlphaPose/human-detection/experiments/logs/res152_coco_2017_train__res152.txt.2018* ./')
filename = 'res152_coco_2017_train__res152.txt.2018-07-31_16-33-05'

def mov_avg(li, n = 500):
	assert(len(li) >= n)
	s = sum(li[0: n])
	res = [s / float(n)]
	for i in range(n, len(li)):
		s += (li[i] - li[i - n])
		res.append(s / float(n))
	return res

res = [re.findall('(total loss|rpn_loss_cls|rpn_loss_box|loss_cls|loss_box): ([0-9\.]+)\n', line) for line in open(filename).readlines()]
res = [item[0] for item in res if item]
keys = 'total loss|rpn_loss_cls|rpn_loss_box|loss_cls|loss_box'.split('|')
d = {key: [] for key in keys} 
for item in res:
	d[item[0]].append(float(item[1]))

for item in d:
	d[item] = mov_avg(d[item])
	plt.plot(np.arange(len(d[item])) * 20 + 10000, d[item], label=item)

plt.title('Training Loss')
plt.ylim(ymin = 0, ymax = 0.15)
# plt.xlim(xmin = 84000)
plt.legend(loc='upper right')
plt.show()
