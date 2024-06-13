import sys
import os
from PyQt5.QtWidgets import QApplication
from se_mainwindow import SE
from utils import source_path
from PyQt5.QtCore import *
from PyQt5 import QtCore

if __name__ == '__main__':
    count_progress = 0
    cd = source_path('')
    os.chdir(cd)

    with open('resources/py_dracula_light.qss') as f:
        qss = f.read()
        app = QApplication(sys.argv)
        app.setStyleSheet(qss)
    window = SE()
    window.show()
    sys.exit(app.exec_())
