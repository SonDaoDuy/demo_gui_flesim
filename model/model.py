import os
from PIL import Image
import numpy as np
import glob

class DummyModel:
	"""docstring for DummyModel"""
	def __init__(self, arg):
		super(DummyModel, self).__init__()
		self.arg = arg

	def forward(self, image, fname):
		self.root = 'E:\\05fa6395e3241f67e33b9e8fe8467612\\'
		basename = os.path.basename(fname)[:-4]
		f_rgb = glob.glob(self.root + 'f_rgb\\' + basename + '*.png')
		print(f_rgb)
		rgb_list = []
		f_mask_list = []
		for file in f_rgb:
			f_rgb_pil = Image.open(file)
			f_mask = (np.array(f_rgb_pil)[:, :, :] > 0).astype(np.uint8)
			rgb_list.append(np.array(f_rgb_pil))
			f_mask_list.append(f_mask)

		f_depth = glob.glob(self.root + 'f_depth\\' + basename + '*.png')
		f_depth_list = []
		for file in f_depth:
			f_depth_pil = Image.open(file)
			f_depth_list.append(np.array(f_depth_pil))

		depth_file = self.root + 'depth\\' + basename + '.png'
		depth = np.array(Image.open(depth_file))

		seg_file = self.root + 'segmentation\\' + basename + '.png'
		seg = np.array(Image.open(seg_file))

		self.result = dict(
			f_rgb=rgb_list, 
			f_mask=f_mask_list, 
			f_depth=f_depth_list,
			depth=depth,
			seg=seg
			)

		return self.result