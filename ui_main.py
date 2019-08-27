import sys
# from options.test_options import TestOptions
from gui.ui_model import ui_model
from PyQt5 import QtWidgets
import argparse

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--img_file', type=str, default='E:\\05fa6395e3241f67e33b9e8fe8467612\\rgb', help='dummy')
	opt = parser.parse_args()
	return opt

if __name__=="__main__":
	opt = parse_args()
	app = QtWidgets.QApplication(sys.argv)
	# MainWindow = QtWidgets.QMainWindow()
	# opt = parse_args()
	my_gui = ui_model(opt)
	my_gui.show()
	sys.exit(app.exec_())