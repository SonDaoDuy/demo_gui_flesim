from gui.ui_main import Ui_MainWindow
from PIL import Image, ImageQt
import random, io, os
import numpy as np
# from model.model import DummyModel
from PyQt5 import QtWidgets, QtGui
from data.base_dataset import BaseDataset, get_transform
from util import util
from models import create_model


def save_images(visuals, image_dir, name, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """

    for label, im_data in visuals.items():
        if label != 'fake_B':
            continue
        im = util.tensor2im(im_data)
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        print('done saving at %s' % (save_path))
        image_pil = Image.fromarray(im)
    return image_pil

class ui_model(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, opt):
        super(ui_model, self).__init__()
        # MainWindow = QtWidgets.QMainWindow()
        self.setupUi(self)
        self.opt = opt
        self.opt.loadSize = [256, 256]
        self.graphicsView.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
        self.graphicsView_2.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
        self.graphicsView_3.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])
        self.graphicsView_4.setMaximumSize(self.opt.loadSize[0], self.opt.loadSize[1])

        # button connection
        # load image
        self.pushButton.clicked.connect(self.load_image_h2z)
        self.pushButton_4.clicked.connect(self.load_image_s2i)

        # load image
        # self.pushButton_2.clicked.connect(self.forward_img)
        # self.pushButton_4.clicked.connect(self.forward_img)

        # image processing
        self.transform_fn = get_transform(opt, grayscale=(opt.input_nc == 1))

        # testing model
        self.model = create_model(opt)
        opt.name = 's2i'
        self.model_s2i = create_model(opt)

        #count 
        self.count_h2z = 0
        self.count_s2i = 0
        

    def load_image_s2i(self, obj_id=None):
        self.opt.name = 'h2z'
        image_folder = os.path.join(self.opt.dataroot, 's2i')
        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'select the image', image_folder, 'Image files(*.jpg *.png *.jpeg)')
        # self.label_2.setText(str(self.fname))
        self.forward_img_s2i(self.fname)

    def load_image_h2z(self, obj_id=None):
        self.opt.name = 's2i'
        image_folder = os.path.join(self.opt.dataroot, 'h2z')
        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'select the image', image_folder, 'Image files(*.jpg *.png *.jpeg)')
        # self.label_2.setText(str(self.fname))
        self.forward_img_h2z(self.fname)

    def forward_img_h2z(self, file_name):
        # data processing
        
        A_path = file_name
        self.original_img = Image.open(A_path).convert('RGB')
        A = self.transform_fn(self.original_img)

        data = {'A': A.unsqueeze(0), 'B': A.unsqueeze(0), 'A_paths': A_path}
        # get output from model
        if self.count_h2z == 0:
            self.model.data_dependent_initialize(data)
            self.model.setup(self.opt)
            self.model.parallelize()
            if self.opt.eval:
                self.model.eval()
            self.count_h2z += 1
        self.model.set_input(data)  # unpack data from data loader
        self.model.test()           # run inference
        visuals = self.model.get_current_visuals()  # get image results
        img_path = self.model.get_image_paths()     # get image paths

        save_dir = os.path.join(self.opt.dataroot, self.opt.name)
        save_name = 'output'
        self.output = save_images(visuals, save_dir, save_name, aspect_ratio=self.opt.aspect_ratio, width=self.opt.display_winsize)
        # self.output = self.original_img
        # display
        self.show_result_h2z()

    def show_result_h2z(self):
        show_images = []
        # input scence
        self.show_original_img = ImageQt.ImageQt(self.original_img)
        # show_images.append(self.show_original_img)
        self.graphicsView.scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(self.show_original_img))
        self.graphicsView.scene.addItem(item)
        self.graphicsView.setScene(self.graphicsView.scene)
        self.graphicsView.fitInView(item)
        # output
        self.show_seg_img = self.output
        self.show_seg_img = ImageQt.ImageQt(self.show_seg_img)
        self.graphicsView_2.scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(self.show_seg_img))
        self.graphicsView_2.scene.addItem(item)
        self.graphicsView_2.setScene(self.graphicsView_2.scene)
        self.graphicsView_2.fitInView(item)

    def forward_img_s2i(self, file_name):
        # data processing
        
        A_path = file_name
        self.original_img = Image.open(A_path).convert('RGB')
        A = self.transform_fn(self.original_img)

        data = {'A': A.unsqueeze(0), 'B': A.unsqueeze(0), 'A_paths': A_path}
        # get output from model
        if self.count_s2i == 0:
            self.model_s2i.data_dependent_initialize(data)
            self.model_s2i.setup(self.opt)
            self.model_s2i.parallelize()
            if self.opt.eval:
                self.model_s2i.eval()
            self.count_s2i += 1
        self.model_s2i.set_input(data)  # unpack data from data loader
        self.model_s2i.test()           # run inference
        visuals = self.model_s2i.get_current_visuals()  # get image results
        img_path = self.model_s2i.get_image_paths()     # get image paths

        save_dir = os.path.join(self.opt.dataroot, 's2i')
        save_name = 'output'
        self.output = save_images(visuals, save_dir, save_name, aspect_ratio=self.opt.aspect_ratio, width=self.opt.display_winsize)
        # self.output = self.original_img
        # display
        self.show_result_s2i()

    def show_result_s2i(self):
        show_images = []
        # input scence
        self.show_original_img = ImageQt.ImageQt(self.original_img)
        # show_images.append(self.show_original_img)
        self.graphicsView_3.scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(self.show_original_img))
        self.graphicsView_3.scene.addItem(item)
        self.graphicsView_3.setScene(self.graphicsView_3.scene)
        self.graphicsView_3.fitInView(item)
        # output
        self.show_seg_img = self.output
        self.show_seg_img = ImageQt.ImageQt(self.show_seg_img)
        self.graphicsView_4.scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(self.show_seg_img))
        self.graphicsView_4.scene.addItem(item)
        self.graphicsView_4.setScene(self.graphicsView_4.scene)
        self.graphicsView_4.fitInView(item)