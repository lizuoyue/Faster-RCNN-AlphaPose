import h5py
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
	d = h5py.File('coco_output/BBOX/test-bbox.h5')
	boxes = [item for item in zip(d['xmin'], d['ymin'], d['xmax'], d['ymax'])]
	box_scores = [[float(item) for item in line.strip().split()] for line in open('../../Comb/box_score.txt').readlines()]
	choose = 'all box'
	if choose == 'final box':
		boxes = [boxes[int(line.strip())] for line in open('finalBoxIdx.txt').readlines()]
		imgIds = [int(item.strip().split('\t')[0].replace('.jpg', '')) for item in open('../POSE/scores.txt').readlines()]
	if choose == 'all box':
		imgIds = [int(item.strip().split('\t')[0].replace('.jpg', '')) for item in open('../BBOX/test-images.txt').readlines()]
	assert(len(imgIds) == len(boxes))

	# res = [{
	# 	'image_id'    : imgId,
	# 	'category_id' : 1,
	# 	'bbox'        : [x1, y1, x2 - x1, y2 - y1],
	# 	'score'       : 1,
	# } for imgId, (x1, y1, x2, y2) in zip(imgIds, boxes)]

	# assert(len(res_anns) == len(box_scores))
	# # res = [res_ann for res_ann, box_score in zip(res_anns, box_scores) if box_score[3] > 0.2]
	# print(len(res))

	# gtCoco = COCO('../../two-step/instances_val2017.json')
	# dtCoco = gtCoco.loadRes(res)
	# cocoEval = COCOeval(gtCoco, dtCoco, 'bbox')
	# cocoEval.evaluate()
	# cocoEval.params.catIds = [1]
	# cocoEval.accumulate()
	# cocoEval.summarize()

	d = {}
	for imgId, box in zip(imgIds, boxes):
		if imgId in d:
			d[imgId].append(box)
		else:
			d[imgId] = [box]

	ious = []

	for imgId in d:
		box = d[imgId]
		for i in range(len(box)):
			ax1, ay1, ax2, ay2 = box[i]
			sa = (ax2 - ax1) * (ay2 - ay1)
			for j in range(i + 1, len(box)):
				bx1, by1, bx2, by2 = box[j]
				sb = (bx2 - bx1) * (by2 - by1)
				iw = min(ax2, bx2) - max(ax1, bx1)
				ih = min(ay2, by2) - max(ay1, by1)
				if iw > 0 and ih > 0:
					si = sa + sb - iw * ih
					iou = iw * ih / float(si)
					ious.append(iou)

	plt.hist(ious, bins = 100)
	plt.show()






