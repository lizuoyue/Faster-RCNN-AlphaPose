import h5py
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
	d = h5py.File('tools/prediction/BBOX/test-bbox.h5')
	boxes = [item for item in zip(d['xmin'], d['ymin'], d['xmax'], d['ymax'])]
	box_scores = [float(line.strip()) for line in open('tools/prediction/BBOX/score-proposals.txt').readlines()]
	assert(len(boxes) == len(box_scores))
	imgIds = [int(item.strip().split('/')[-1].replace('.jpg', '')) for item in open('tools/prediction/BBOX/test-images.txt').readlines()]
	assert(len(boxes) == len(imgIds))

	res = [{
		'image_id'    : imgId,
		'category_id' : 1,
		'bbox'        : [x1, y1, x2 - x1, y2 - y1],
		'score'       : score
	} for imgId, (x1, y1, x2, y2), score in zip(imgIds, boxes, box_scores)]
	print(len(res))

	gtCoco = COCO('data/coco/annotations/instances_val2017.json') # person_keypoints_val2017.json
	dtCoco = gtCoco.loadRes(res)
	cocoEval = COCOeval(gtCoco, dtCoco, 'bbox')
	cocoEval.evaluate()
	cocoEval.params.catIds = [1]
	cocoEval.accumulate()
	cocoEval.summarize()
