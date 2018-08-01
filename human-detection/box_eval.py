import h5py
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
	d = h5py.File('tools/prediction/BBOX/test-bbox.h5')
	boxes = [item for item in zip(d['xmin'], d['ymin'], d['xmax'], d['ymax'])]
	box_scores = [[float(item) for item in line.strip().split()] for line in open('tools/prediction/BBOX/score-proposals.txt').readlines()]
	choose = 'all box'
	if choose == 'final box':
		boxes = [boxes[int(line.strip())] for line in open('finalBoxIdx.txt').readlines()]
		imgIds = [int(item.strip().split('\t')[0].replace('.jpg', '')) for item in open('../POSE/scores.txt').readlines()]
	if choose == 'all box':
		imgIds = [int(item.strip().split('\t')[0].replace('.jpg', '')) for item in open('tools/prediction/BBOX/test-images.txt').readlines()]
	assert(len(imgIds) == len(boxes))

	res = [{
		'image_id'    : imgId,
		'category_id' : 1,
		'bbox'        : [x1, y1, x2 - x1, y2 - y1],
		'score'       : 1,
	} for imgId, (x1, y1, x2, y2) in zip(imgIds, boxes)]

	assert(len(res) == len(box_scores))
	print(len(res))

	gtCoco = COCO('person_keypoints_val2017.json')
	dtCoco = gtCoco.loadRes(res)
	cocoEval = COCOeval(gtCoco, dtCoco, 'bbox')
	cocoEval.evaluate()
	cocoEval.params.catIds = [1]
	cocoEval.accumulate()
	cocoEval.summarize()
