import sys
# from options.test_options import TestOptions
from gui.ui_model_test import ui_model
from PyQt5 import QtWidgets
from options.test_options import TestOptions
import argparse

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataroot', type=str, default='E:\\05fa6395e3241f67e33b9e8fe8467612\\rgb', help='dummy')
	opt = parser.parse_args()
	return opt

if __name__=="__main__":
	# opt = parse_args()
	opt = TestOptions().parse()  # get test options
	# hard-code some parameters for test
	opt.num_threads = 0   # test code only supports num_threads = 1
	opt.batch_size = 1    # test code only supports batch_size = 1
	opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
	opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
	opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
	app = QtWidgets.QApplication(sys.argv)
	# MainWindow = QtWidgets.QMainWindow()
	# opt = parse_args()
	my_gui = ui_model(opt)
	my_gui.show()
	sys.exit(app.exec_())