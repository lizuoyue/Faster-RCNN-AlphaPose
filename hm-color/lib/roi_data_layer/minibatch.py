# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

import json, os
_heatmap = {}
_heatmap['train2017'] = json.load(open('/disks/data4/zyli/Faster-RCNN-AlphaPose/heatmap/heatmap_train2017.json'))
_heatmap['val2017'] = json.load(open('/disks/data4/zyli/Faster-RCNN-AlphaPose/heatmap/heatmap_val2017.json'))

colors = [
  (255, 255, 255), (255, 255,   0), (255,   0, 255), (  0, 255, 255), (  0, 127, 255),
  (  0, 255, 127), (255,   0, 127), (255, 127,   0), (127, 255,   0), (127,   0, 255)
]

idxs = [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 8, 9, 9]

def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  blobs = {'data': im_blob}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
    dtype=np.float32)

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    ##################
    dataset = roidb[i]['image'].split('/')[-2]
    img_id = roidb[i]['image'].split('/')[-1].replace('.jpg', '')
    file_name = '/disks/data4/zyli/Faster-RCNN-AlphaPose/heatmap/%s/%s.png' % (dataset, img_id)
    if os.path.exists(file_name):
      hm_c = cv2.imread(file_name)
    else:
      h, w = im.shape[0], im.shape[1]
      xx, yy = np.meshgrid(np.arange(w), np.arange(h))
      hm = np.ones((18, h, w)) * (-1e9)
      for j, part in enumerate(_heatmap[dataset][img_id]):
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
      cv2.imwrite(file_name, cv2.cvtColor(hm_c, cv2.COLOR_RGB2BGR))
    im = np.concatenate([im, hm_c], axis = 2)
    ##################
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales
