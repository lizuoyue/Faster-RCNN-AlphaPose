import json
heatmap = {}
heatmap['train2017'] = json.load(open('/disks/data4/zyli/Faster-RCNN-AlphaPose/heatmap/heatmap_train2017.json'))
heatmap['val2017'] = json.load(open('/disks/data4/zyli/Faster-RCNN-AlphaPose/heatmap/heatmap_val2017.json'))

colors = [
	(255, 255, 255), (255, 255,   0), (255,   0, 255), (  0, 255, 255), (  0, 127, 255),
	(  0, 255, 127), (255,   0, 127), (255, 127,   0), (127, 255,   0), (127,   0, 255)
]

idxs = [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 8, 9, 9]

from pycocotools.coco import COCO
import numpy as np
import skimage

for dataset in ['val2017', 'train2017']:
	coco = COCO('../../coco2017data/annotations/instances_%s.json' % dataset)
	imgIds = coco.getImgIds()
	imgInfos = coco.loadImgs(imgIds)
	for i, imgId, imgInfo in enumerate(zip(imgIds, imgInfos)):
		print(dataset, i, imgId)
		imgIdStr = str(imgId).zfill(12)
		h, w = imgInfo['height'], imgInfo['width']
		xx, yy = np.meshgrid(np.arange(w), np.arange(h))
		hm = np.ones((18, h, w)) * (-1e9)
		for j, part in enumerate(heatmap[dataset][imgIdStr]):
			for x, y, v in part:
				if v < 600:
					continue
				hm[j] = np.maximum(hm[j], -((xx - x) ** 2 + (yy - y) ** 2) / 100 + np.log(v / 6000))
			hm[j] = np.exp(hm[j])
		hm_c = np.zeros((3, h, w))
		for j in range(3):
			for k in range(18):
				hm_c[j] += hm[k] * colors[idxs[k]][j]
		hm_c = np.array(np.maximum(np.minimum(hm_c, 255), 0), np.float32).transpose([1, 2, 0])
		skimage.io.imsave('/disks/data4/zyli/Faster-RCNN-AlphaPose/heatmap/%s/%s.png' % (dataset, imgIdStr), hm_c)
