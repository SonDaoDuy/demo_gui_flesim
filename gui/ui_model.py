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
		print(self.opt)
		self.opt.loadSize = [512, 512]
		self.graphicsView.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
		self.graphicsView_2.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
		self.graphicsView_3.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
		self.graphicsView_4.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
		self.graphicsView_5.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
		self.graphicsView_6.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
		# model and data info
		self.model = self.load_model()

		# button connection
		# load image
		self.pushButton.clicked.connect(self.load_image)

		# select object
		self.comboBox.activated.connect(self.combobox_update_view)

	def load_model(self):
		return DummyModel(self.opt)

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
		no_of_objects = len(self.output['f_rgb'])
		self.update_combobox(no_of_objects)
		# self.obj_id = self.comboBox.currentIndex()
		self.obj_id = 0
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
		return self.original_img

	def make_single_rgb(self):
		return Image.fromarray(self.output['f_rgb'][self.obj_id])

	def make_single_depth(self):
		depth = self.output['f_depth'][self.obj_id]
		depth_img = (depth - depth.min())/(depth.max() - depth.min()) * 255
		depth_pil = Image.fromarray(depth_img.astype(np.uint8))
		return depth_pil

	def setup_view(self, graphicsView, image):
		imageQt = ImageQt.ImageQt(image)
		graphicsView.scene = QtWidgets.QGraphicsScene()
		item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(imageQt)) 
		graphicsView.scene.addItem(item)
		graphicsView.setScene(self.graphicsView.scene)
		