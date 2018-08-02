import json
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import random
import time

colors = [
	(255, 255, 255), (255, 255,   0), (255,   0, 255), (  0, 255, 255), (  0, 127, 255),
	(  0, 255, 127), (255,   0, 127), (255, 127,   0), (127, 255,   0), (127,   0, 255)
]

idxs = [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 8, 9, 9]

choose = 'val2017'
os.popen('mkdir heatmap_%s' % choose)
img_json = json.load(open('../../coco2017data/annotations/person_keypoints_%s.json' % choose))
hm_json = json.load(open('heatmap_%s.json' % choose))
image_ids = set(list([item['image_id'] for item in img_json['annotations']]))
for i, img_info in enumerate(img_json['images']):
	key = img_info['file_name'].replace('.jpg', '')
	h, w = img_info['height'], img_info['width']
	img = io.imread(img_info['coco_url'])
	xx, yy = np.meshgrid(np.arange(w), np.arange(h))
	t = time.time()
	hm = np.ones((18, h, w)) * (-1e9)
	for j, part in enumerate(hm_json[key]):
		for x, y, v in part:
			if v < 600:
				continue
			hm[j] = np.maximum(hm[j], -((xx - x) ** 2 + (yy - y) ** 2) / 100 + np.log(v / 6000))
		hm[j] = np.exp(hm[j])
	hm_c = np.zeros((h, w, 3))
	for j in range(3):
		for k in range(18):
			hm_c[..., j] += hm[k] * colors[idxs[k]][j]
	print(i, time.time() - t)
	hm_c = np.array(np.maximum(np.minimum(hm_c, 255), 0), np.uint8)
	np.save('heatmap_%s/%s.npy' % (choose, key), hm_c)
