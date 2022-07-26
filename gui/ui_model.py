from gui.ui_window import Ui_MainWindow
from PIL import Image, ImageQt
import random, io, os
import numpy as np
from model.model import DummyModel
from PyQt5 import QtWidgets, QtGui

class ui_model(QtWidgets.QMainWindow, Ui_MainWindow):
	def __init__(self, opt):
		super(ui_model, self).__init__()
		# MainWindow = QtWidgets.QMainWindow()
		self.setupUi(self)
		self.opt = opt
		# self.opt.anno_file = 'C:\\Users\\Son\\Desktop\\demo_gui\\data\\sosc\\sosc_train.json'
		print(self.opt)
		self.opt.loadSize = [512, 512]
		self.graphicsView.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
		self.graphicsView_2.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
		self.graphicsView_3.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
		self.graphicsView_4.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
		self.graphicsView_5.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
		self.graphicsView_6.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
		# model and data info
		# self.model = self.load_model(self.opt.anno_file)

		# button connection
		# load image
		self.pushButton.clicked.connect(self.load_image)

		# select object
		self.comboBox.activated.connect(self.combobox_update_view)

		# visibility
		self.pushButton_3.clicked.connect(self.turn_on_off)

		#position
		self.move_step = 10
		self.pushButton_2.clicked.connect(self.move_up)
		self.pushButton_4.clicked.connect(self.move_down)
		self.pushButton_5.clicked.connect(self.move_left)
		self.pushButton_6.clicked.connect(self.move_right)

		#zoom (not actualy zoom)
		self.zoom_step = 10
		self.pushButton_7.clicked.connect(self.zoom_in)
		self.pushButton_8.clicked.connect(self.zoom_out)

	def load_model(self, anno_file):
		return DummyModel(self.opt, anno_file)

	def load_image(self, obj_id=None):
		self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'select the image', self.opt.img_file, 'Image files(*.jpg *.png)')
		self.label_2.setText(str(self.fname))
		self.forward_img(self.fname)

	def forward_img(self, file_name):
		# read img rgb and gt
		self.original_img = Image.open(file_name).convert('RGB')
		# self.extra_gt_things = self.get_extra(file_name)
		#pre-process input
		self.input = self.original_img.copy()
		self.input = self.input.resize(self.opt.loadSize) #some pre-process here
		# get output from model
		self.output = self.model.forward(self.input, file_name)
		# update combo_box
		self.no_of_objects = len(self.output['f_rgb'])
		self.update_combobox(self.no_of_objects)
		self.obj_id = 0
		# update visibility
		self.visibility = np.ones(self.no_of_objects) # all on
		# display
		self.show_result()

	def update_combobox(self, no_of_objects):
		self.comboBox.clear()       # delete all items from comboBox
		list_obj = list(range(no_of_objects))
		combo_list = ['all'] + [str(i) for i in list_obj]
		self.comboBox.addItems(combo_list)

	def combobox_update_view(self):
		self.obj_id = self.comboBox.currentIndex() - 1
		if self.obj_id < 0:
			self.obj_id = 0
		print(self.obj_id)
		self.show_result()

	def show_result(self):
		show_images = []
		# input scence
		self.show_original_img = ImageQt.ImageQt(self.original_img)
		# show_images.append(self.show_original_img)
		self.graphicsView.scene = QtWidgets.QGraphicsScene()
		item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(self.show_original_img))
		self.graphicsView.scene.addItem(item)
		self.graphicsView.setScene(self.graphicsView.scene)
		self.graphicsView.fitInView(item)
		# segmentation
		self.show_seg_img = self.make_seg_img()
		self.show_seg_img = ImageQt.ImageQt(self.show_seg_img)
		self.graphicsView_2.scene = QtWidgets.QGraphicsScene()
		item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(self.show_seg_img))
		self.graphicsView_2.scene.addItem(item)
		self.graphicsView_2.setScene(self.graphicsView_2.scene)
		self.graphicsView_2.fitInView(item)
		# depth
		self.show_depth_img = self.make_depth_img()
		self.show_depth_img = ImageQt.ImageQt(self.show_depth_img)
		self.graphicsView_3.scene = QtWidgets.QGraphicsScene()
		item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(self.show_depth_img))
		self.graphicsView_3.scene.addItem(item)
		self.graphicsView_3.setScene(self.graphicsView_3.scene)
		self.graphicsView_3.fitInView(item)
		# output scene
		self.show_recompose_img = self.make_recompose_img()
		self.show_recompose_img = ImageQt.ImageQt(self.show_recompose_img)
		self.graphicsView_4.scene = QtWidgets.QGraphicsScene()
		item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(self.show_recompose_img))
		self.graphicsView_4.scene.addItem(item)
		self.graphicsView_4.setScene(self.graphicsView_4.scene)
		self.graphicsView_4.fitInView(item)
		# single RGB
		self.show_single_rgb = self.make_single_rgb()
		self.show_single_rgb = ImageQt.ImageQt(self.show_single_rgb)
		self.graphicsView_5.scene = QtWidgets.QGraphicsScene()
		item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(self.show_single_rgb))
		self.graphicsView_5.scene.addItem(item)
		self.graphicsView_5.setScene(self.graphicsView_5.scene)
		self.graphicsView_5.fitInView(item)
		# single depth
		self.show_single_depth = self.make_single_depth()
		self.show_single_depth = ImageQt.ImageQt(self.show_single_depth)
		self.graphicsView_6.scene = QtWidgets.QGraphicsScene()
		item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(self.show_single_depth))
		self.graphicsView_6.scene.addItem(item)
		self.graphicsView_6.setScene(self.graphicsView_6.scene)
		self.graphicsView_6.fitInView(item)

	def make_seg_img(self):
		return Image.fromarray(self.output['seg'])

	def make_depth_img(self):
		depth = self.output['depth']
		depth_img = (depth - depth.min())/(depth.max() - depth.min()) * 255
		depth_pil = Image.fromarray(depth_img.astype(np.uint8))
		return depth_pil

	def make_recompose_img(self):
		depth_list = []
		rgb_list = []
		for i in range(self.no_of_objects):
			if self.visibility[i] == 1:
				depth_list.append(self.output['f_depth'][i])
				rgb_list.append(self.output['f_rgb'][i])
		_, recompose_img = self.min_depth_pooling(depth_list, rgb_list)
		return recompose_img

	def make_single_rgb(self):
		return Image.fromarray(self.output['f_rgb'][self.obj_id])

	def make_single_depth(self):
		depth = self.output['f_depth'][self.obj_id]
		depth_img = (depth - depth.min())/(depth.max() - depth.min()) * 255
		depth_pil = Image.fromarray(depth_img.astype(np.uint8))
		return depth_pil

	def min_depth_pooling(self, depth, rgbs):
		depth = np.array(depth)
		depth = np.where(depth == 0, np.ones_like(depth)*(2**16-1), depth)
		min_depth = np.min(depth, axis=0)
		min_index = np.where(depth == min_depth)
		index = np.zeros_like(depth).astype(np.uint8)
		index[min_index] = 1
		rgb_value = np.zeros_like(rgbs[0])
		index_unique = np.zeros((512, 512, 1)).astype(np.uint8)
		for i, rgb in enumerate(rgbs):
			index_each = index[i,:,:].reshape(512, 512, 1) * (1-index_unique)
			rgb_value += index_each * rgb
			index_unique += index_each
		min_depth = np.where(min_depth == (2**16 -1), np.zeros_like(min_depth), min_depth)
		min_depth = Image.fromarray(min_depth.astype(np.uint16))
		r_rgb = Image.fromarray(rgb_value)
		return min_depth, r_rgb

	def turn_on_off(self):
		#update visibility
		if self.visibility[self.obj_id] == 1:
			self.visibility[self.obj_id] = 0
		else:
			self.visibility[self.obj_id] = 1
		self.show_result()

	def move_up(self):
		#update bbox
		old_bbox = self.output['bbox'][self.obj_id].copy()
		self.output['bbox'][self.obj_id][1] -= self.move_step
		self.output['bbox'][self.obj_id][3] -= self.move_step
		new_bbox = self.output['bbox'][self.obj_id]
		# bbox[1] += self.move_step
		# bbox[3] += self.move_step
		# self.output['bbox'][self.obj_id] = bbox
		print(old_bbox)
		print(self.output['bbox'][self.obj_id])

		#update object depth map
		object_depth = self.output['f_depth'][self.obj_id].copy()
		depth_map = np.zeros(np.shape(object_depth)).astype(np.uint16)
		depth_map[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]] = \
			object_depth[old_bbox[1]:old_bbox[3], old_bbox[0]:old_bbox[2]]
		self.output['f_depth'][self.obj_id] = depth_map

		#update object rgb????
		object_rgb = self.output['f_rgb'][self.obj_id].copy()
		rgb = np.zeros(np.shape(object_rgb))
		rgb[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2], :] = \
			object_rgb[old_bbox[1]:old_bbox[3], old_bbox[0]:old_bbox[2], :]
		self.output['f_rgb'][self.obj_id] = rgb

		#update seg

		#update scene depth map

		#show result
		self.show_result()

	def move_down(self):
		#update bbox
		old_bbox = self.output['bbox'][self.obj_id].copy()
		self.output['bbox'][self.obj_id][1] += self.move_step
		self.output['bbox'][self.obj_id][3] += self.move_step
		new_bbox = self.output['bbox'][self.obj_id]
		# bbox[1] += self.move_step
		# bbox[3] += self.move_step
		# self.output['bbox'][self.obj_id] = bbox
		print(old_bbox)
		print(self.output['bbox'][self.obj_id])

		#update object depth map
		object_depth = self.output['f_depth'][self.obj_id].copy()
		depth_map = np.zeros(np.shape(object_depth)).astype(np.uint16)
		depth_map[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]] = \
			object_depth[old_bbox[1]:old_bbox[3], old_bbox[0]:old_bbox[2]]
		self.output['f_depth'][self.obj_id] = depth_map

		#update object rgb????
		object_rgb = self.output['f_rgb'][self.obj_id].copy()
		rgb = np.zeros(np.shape(object_rgb))
		rgb[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2], :] = \
			object_rgb[old_bbox[1]:old_bbox[3], old_bbox[0]:old_bbox[2], :]
		self.output['f_rgb'][self.obj_id] = rgb

		#update seg

		#update scene depth map

		#show result
		self.show_result()

	def move_left(self):
		#update bbox
		old_bbox = self.output['bbox'][self.obj_id].copy()
		self.output['bbox'][self.obj_id][0] -= self.move_step
		self.output['bbox'][self.obj_id][2] -= self.move_step
		new_bbox = self.output['bbox'][self.obj_id]
		# bbox[1] += self.move_step
		# bbox[3] += self.move_step
		# self.output['bbox'][self.obj_id] = bbox
		print(old_bbox)
		print(self.output['bbox'][self.obj_id])

		#update object depth map
		object_depth = self.output['f_depth'][self.obj_id].copy()
		depth_map = np.zeros(np.shape(object_depth)).astype(np.uint16)
		depth_map[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]] = \
			object_depth[old_bbox[1]:old_bbox[3], old_bbox[0]:old_bbox[2]]
		self.output['f_depth'][self.obj_id] = depth_map

		#update object rgb????
		object_rgb = self.output['f_rgb'][self.obj_id].copy()
		rgb = np.zeros(np.shape(object_rgb))
		rgb[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2], :] = \
			object_rgb[old_bbox[1]:old_bbox[3], old_bbox[0]:old_bbox[2], :]
		self.output['f_rgb'][self.obj_id] = rgb

		#update seg

		#update scene depth map

		#show result
		self.show_result()

	def move_right(self):
		#update bbox
		old_bbox = self.output['bbox'][self.obj_id].copy()
		self.output['bbox'][self.obj_id][0] += self.move_step
		self.output['bbox'][self.obj_id][2] += self.move_step
		new_bbox = self.output['bbox'][self.obj_id]
		# bbox[1] += self.move_step
		# bbox[3] += self.move_step
		# self.output['bbox'][self.obj_id] = bbox
		print(old_bbox)
		print(self.output['bbox'][self.obj_id])

		#update object depth map
		object_depth = self.output['f_depth'][self.obj_id].copy()
		depth_map = np.zeros(np.shape(object_depth)).astype(np.uint16)
		depth_map[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]] = \
			object_depth[old_bbox[1]:old_bbox[3], old_bbox[0]:old_bbox[2]]
		self.output['f_depth'][self.obj_id] = depth_map

		#update object rgb????
		object_rgb = self.output['f_rgb'][self.obj_id].copy()
		rgb = np.zeros(np.shape(object_rgb))
		rgb[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2], :] = \
			object_rgb[old_bbox[1]:old_bbox[3], old_bbox[0]:old_bbox[2], :]
		self.output['f_rgb'][self.obj_id] = rgb

		#update seg

		#update scene depth map

		#show result
		self.show_result()

	def zoom_in(self):
		depth = self.output['f_depth'][self.obj_id].copy()
		self.output['f_depth'][self.obj_id][depth[:,:] > 0] += self.zoom_step
		self.show_result()

	def zoom_out(self):
		depth = self.output['f_depth'][self.obj_id].copy()
		self.output['f_depth'][self.obj_id][depth[:,:] > 0] -= self.zoom_step
		self.show_result()