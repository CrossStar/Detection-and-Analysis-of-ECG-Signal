#!/usr/bin/env python
# ------------------------------------------------------
#
#       This is a program for PCF8591 Module.
#
#       Warning! The Analog input MUST NOT be over 3.3V!
#
#       In this script, we use a poteniometer for analog
#   input, and a LED on AO for analog output.
#
#       you can import this script to another by:
#   import PCF8591 as ADC
#
#   ADC.Setup(Address)  # Check it by sudo i2cdetect -y -1
#   ADC.read(channal)   # Channal range from 0 to 3
#   ADC.write(Value)    # Value range from 0 to 255
#
# ------------------------------------------------------
import smbus
from itertools import count
import time
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal, QMutex

fs = 360
max_length = 3750
bus = smbus.SMBus(1)

class PCF8591_Worker(QThread):
    collected_signal = pyqtSignal(list)
    progress_signal = pyqtSignal(int)
    
    def __init__(self):
        super(PCF8591_Worker, self).__init__()
        self.ecg_signal = []
        self.Addr = 0x48
        self.address = self.Addr

    def read(self, chn):
        if chn == 0:
            bus.write_byte(self.address, 0x40)
        bus.read_byte(self.address)
        return bus.read_byte(self.address)

    def run(self):
        Vcc = 3.3

        ecg_signal = []

        for index in range(0, max_length):
            result = self.read(0)
            ecg_signal_item = result / 256 * Vcc
            print(result,ecg_signal_item)
            ecg_signal.append(ecg_signal_item)
            time.sleep(1 / fs)
            self.progress_signal.emit(index + 1)
        self.collected_signal.emit(ecg_signal)
        print('the thread has stopped')

