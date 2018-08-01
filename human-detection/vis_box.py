from PIL import Image, ImageDraw
import h5py
import matplotlib.pyplot as plt

if __name__ == '__main__':
	d = h5py.File('tools/prediction/BBOX/test-bbox.h5')
	boxes = [item for item in zip(d['xmin'], d['ymin'], d['xmax'], d['ymax'])]
	box_scores = [float(line.strip()) for line in open('tools/prediction/BBOX/score-proposals.txt').readlines()]
	assert(len(boxes) == len(box_scores))
	imgIds = [int(item.strip().split('/')[-1].replace('.jpg', '')) for item in open('tools/prediction/BBOX/test-images.txt').readlines()]
	assert(len(boxes) == len(imgIds))

	res = {}
	for imgId, (x1, y1, x2, y2), score in zip(imgIds, boxes, box_scores):
		d = {
			'image_id'    : imgId,
			'category_id' : 1,
			'bbox'        : [x1, y1, x2, y2],
			'score'       : score
		}
		if imgId in res:
			res[imgId].append(d)
		else:
			res[imgId] = [d]

	for imgId in set(imgIds):
		img = Image.open('/disks/data4/zyli/Faster-RCNN-AlphaPose/human-detection/data/coco/val2017/%s.jpg' % str(imgId).zfill(12)).convert('RGB')
		draw = ImageDraw.Draw(img)
		for item in res[imgId]:
			draw.rectangle(item['bbox'], outline = (255, 0, 0))
		img.save('%d.jpg' % imgId)
