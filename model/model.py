import os
from PIL import Image
import numpy as np
import glob
from data.sosc_tools import SOSC

class DummyModel:
	"""docstring for DummyModel"""
	def __init__(self, arg, anno_file):
		super(DummyModel, self).__init__()
		self.arg = arg
		self.sosc = SOSC(anno_file)
		# self.root = 'E:\\05fa6395e3241f67e33b9e8fe8467612\\'
		self.root = 'C:\\Users\\Son\\Desktop\\demo_gui\\data\\sosc\\05fa6395e3241f67e33b9e8fe8467612\\'
		model_id = '05fa6395e3241f67e33b9e8fe8467612'
		scenes = self.sosc.dataset['scenes']
		self.cat_ids = self.sosc.getCatIds()
		# self.cat_ids = self.cat_ids[1:]
		self.cat2label = {
			cat_id: i + 1
			for i, cat_id in enumerate(self.cat_ids)
		}
		print(self.cat2label)
		self.data_anno = dict()
		for item in scenes:
			img_name = item['img_name'].split('/')
			if img_name[0] == model_id:
				self.data_anno[img_name[-1][:-4]] = item['id']

	def forward(self, image, fname):
		
		#get anno for each image in the folder
		basename = os.path.basename(fname)[:-4]
		scene_id = self.data_anno[basename]
		ann_ids = self.sosc.getAnnIds(imgIds=[scene_id])
		ann_info = self.sosc.loadAnns(ann_ids)
		#get gt information
		gt_bboxes = []
		gt_labels = []
		gt_f_rgb = []
		gt_f_depth = []
		gt_orders = []
		# gt_v_mask = []
		gt_f_mask = []
		for i, ann in enumerate(ann_info):
			f_name = ann['f_img_name'].split('/')[-1]
			file = self.root + 'f_rgb\\' + f_name
			if not os.path.isfile(file):
				continue
			# if ann['category_id'] == 0:
			# 	continue
			if ann.get('ignore', False):
				continue
			x, y, w, h = ann['f_bbox']
			if w < 1 or h < 1:
				continue
			bbox = [x, y, x + w - 1, y + h - 1]
			gt_bboxes.append(bbox)
			gt_labels.append(self.cat2label[ann['category_id']])
			gt_orders.append(ann['layer_order'])
			#get object images
			f_name = ann['f_img_name'].split('/')[-1]
			f_rgb_pil = Image.open(self.root + 'f_rgb\\' + f_name)
			f_mask = (np.array(f_rgb_pil)[:, :, :] > 0).astype(np.uint8)
			gt_f_rgb.append(np.array(f_rgb_pil))
			gt_f_mask.append(f_mask)
			f_rgb_pil.close()
			#get object depth
			f_name = ann['f_depth_name'].split('/')[-1]
			f_depth_pil = Image.open(self.root + 'f_depth\\' + f_name)
			gt_f_depth.append(np.array(f_depth_pil).astype(np.uint16))
			f_depth_pil.close()
		gt_bboxes = np.array(gt_bboxes, dtype=np.int)
		gt_labels = np.array(gt_labels, dtype=np.int64)
		gt_f_rgb = np.array(gt_f_rgb)
		gt_f_mask = np.array(gt_f_mask)
		gt_f_depth = np.array(gt_f_depth).astype(np.uint16)

		depth_file = self.root + 'depth\\' + basename + '.png'
		depth = np.array(Image.open(depth_file)).astype(np.uint16)

		seg_file = self.root + 'segmentation\\' + basename + '.png'
		seg = np.array(Image.open(seg_file))

		self.result = dict(
			bbox=gt_bboxes,
			label=gt_labels,
			f_rgb=gt_f_rgb, 
			f_mask=gt_f_mask, 
			f_depth=gt_f_depth,
			depth=depth,
			seg=seg
			)

		print(self.result['bbox'])

		return self.result