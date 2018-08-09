import json
heatmap = {}
heatmap['train2017'] = json.load(open('/disks/data4/zyli/Faster-RCNN-AlphaPose/heatmap/heatmap_train2017.json'))
heatmap['val2017'] = json.load(open('/disks/data4/zyli/Faster-RCNN-AlphaPose/heatmap/heatmap_val2017.json'))

from pycocotools.coco import COCO
import numpy as np

for dataset in ['val2017', 'train2017']:
	coco = COCO('../../coco2017data/annotations/instances_%s.json' % dataset)
	imgIds = coco.getImgIds()
	imgInfos = coco.loadImgs(imgIds)
	for i, (imgId, imgInfo) in enumerate(zip(imgIds, imgInfos)):
		print(dataset, i, imgId)
		imgIdStr = str(imgId).zfill(12)
		h, w = imgInfo['height'], imgInfo['width']
		xx, yy = np.meshgrid(np.arange(w), np.arange(h))
		hm = np.ones((18, h, w)) * (-1e9)
		for j, part in enumerate(heatmap[dataset][imgIdStr]):
			for x, y, v in part:
				if v < 600:
					continue
				hm[j] = np.maximum(hm[j], -((xx - x) ** 2 + (yy - y) ** 2) / 100.0 + np.log(v / 6000.0))
			hm[j] = np.exp(hm[j])
		hm = np.maximum(np.minimum(hm, 1), 0)
		hm = np.array(hm * 255.0, np.uint8).transpose([1, 2, 0])
		np.save('/disks/data4/zyli/Faster-RCNN-AlphaPose/heatmap/%s/%s.npy' % (dataset, imgIdStr), hm)
