import json
import statistics
import time

from PCF8591_worker import PCF8591_Worker, fs, max_length
import pyqtgraph as pg
from myWfdb.io import rdrecord

from ecg_data_analysis import *
from datetime import datetime
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import pyqtSlot, QTimer
from Ui_ui_design import Ui_MainWindow
from openai_worker import Worker
from openai import OpenAI
import numpy as np

class SE(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(SE, self).__init__()
        self.filtered_signal = None
        self.timer2 = None
        self.HRV = None
        self.QT_time = None
        self.ST_time = None
        self.PR_time = None
        self.sex = None
        self.age = None
        self.QRS_time_PanTompkins_avg = None
        self.QRS_time_PanTompkins_min = None
        self.QRS_time_PanTompkins_max = None
        self.fs = None
        self.client = None
        self.ecg_signal = []
        self.ecg_signal_draw = None
        self.detector_PanTompkins = None
        self.detector_Elgendi_QRS = None
        self.detector_Elgendi_PT = None
        self.heartrate_Elgendi_QRS_avg = None
        self.heartrate_PanTompkins_avg = None
        self.heartrate_Elgendi_QRS_max = None
        self.heartrate_PanTompkins_max = None
        self.heartrate_Elgendi_QRS_min = None
        self.heartrate_PanTompkins_min = None
        self.setupUI()
        self.setupData()
        self.setupConnections()
        self.chunk_str = ''  # 存储从GPT接收到的全部字符
        self.index = 0  # 当前正在显示的字符的索引
        self.index2 = 0
        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(self.update_filtered_signal)
        self.timer2.setInterval(1)
        self.timer3 = QTimer(self)
        self.timer3.timeout.connect(self.ecg_signal_drawing)
        self.timer3.setInterval(1)
        self.index3 = 0
        self.filtered_signal_draw = []
        self.ecg_signal_draw = []
        self.showMaximized()    
    def setupUI(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.graphicsView.setBackground('w')
        self.ui.graphicsView_2.setBackground('w')
        self.ui.textEdit.setReadOnly(True)
        self.ui.heartrate1.setReadOnly(True)
        self.ui.heartrate1_4.setReadOnly(True)
        self.ui.heartrate1_6.setReadOnly(True)
        self.ui.heartrate2.setReadOnly(True)
        self.ui.heartrate2_3.setReadOnly(True)
        self.ui.heartrate2_4.setReadOnly(True)
        self.ui.QRStime.setReadOnly(True)
        self.ui.QRStime_2.setReadOnly(True)
        self.ui.QRStime_3.setReadOnly(True)
        self.ui.textEdit.setReadOnly(True)
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.setRange(0, max_length)

    def setupData(self):
        self.chunk_str = ''
        self.client = OpenAI(
            api_key = OPENAI_API_KEY
        )
        self.detector_Elgendi_PT = None
        self.detector_Elgendi_QRS = None
        self.detector_PanTompkins = None
        if self.ui.comboBox_2.currentText() == '实测模式':
            self.fs = fs
        else:
            self.fs = 360
        self.index = 0
        self.sex = None
        self.age = 0
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.collect.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.filter.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.analysis.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        # self.ui.report.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
    def update_progress(self, value):
        self.ui.progressBar.setValue(value)  # 更新进度条

    def update_filtered_signal(self):
        if self.index2 < len(self.filtered_signal):
            self.filtered_signal_draw.extend(self.filtered_signal[self.index2:self.index2+99])
            self.index2 += 100
            pen = pg.mkPen(color='b', width=3)
            self.ui.graphicsView_2.plot().setData(self.filtered_signal_draw, pen=pen)

    def update_ecg_signal(self, ecg_signal):
        self.ecg_signal = ecg_signal
    
    def ecg_signal_drawing(self):
        if self.index3 < len(self.ecg_signal):
            self.ecg_signal_draw.extend(self.ecg_signal[self.index3:self.index3+199])
            self.index3 += 200
            pen = pg.mkPen(color='b', width=3)
            self.ui.graphicsView.plot().setData(self.ecg_signal_draw, pen=pen)
    
    def setupConnections(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.type_text)
        self.timer.setInterval(20)  # 设置间隔时间，例如20毫秒

    def update_text_edit(self, text):
        self.chunk_str += text  # 添加新接收到的文本到待显示的字符串
        self.timer.start()  # 开始或继续定时器

    def type_text(self):
        if self.index < len(self.chunk_str):
            self.ui.textEdit.insertPlainText(self.chunk_str[self.index])
            self.ui.textEdit.ensureCursorVisible()  # 确保光标可见
            self.index += 1
        else:
            self.timer.stop()  # 如果所有字符都已显示，停止定时器

    def heart_report_generator(self, **kwargs):
        ecg_data = json.dumps(kwargs)
        if self.ui.sex_man.isChecked():
            self.sex = '男'
        else:
            self.sex = '女'
        # self.age = self.ui.spinBox.value()
        # 获取当前日期
        currentDate = datetime.now().date()
        # print(currentDate)
        # 获取用户输入的日期
        # birthDate = self.ui.dateEdit.getDate()
        birthDate = self.ui.dateEdit.date()
        # print(birthDate)
        # 计算年龄
        self.age = currentDate.year - birthDate.year()
        # 如果当前日期还没过生日那么就减一岁
        if currentDate < birthDate.addYears(self.age):
            self.age -= 1
        print(self.sex,self.age)
        myPrompt = f'''
        我的年龄是{self.age}，我的性别是{self.sex}
        我的基于ECG信号的各种参数为: {ecg_data}，所有时间的单位都是秒，其中的HRV通过所有正常RR间期的标准差来反映
        请根据上面的参数，和正常情况下的参数进行比较，从中判断我的心脏问题，并给我具有可行性的生活方面的建议。
        注意：我的心脏问题和建议都需要分条，有条理地回答，
        '''
        messages = [
            {'role': 'system', 'content': '你是我的心脏问题专家'},
            {'role': 'user', 'content': myPrompt}
        ]
        # 创建并启动线程
        model = self.ui.comboBox.currentText()
        # print(model)
        self.worker = Worker(self.client, messages,model)
        self.worker.update_signal.connect(self.update_text_edit)
        self.worker.start()

    # 其他方法和类的定义...
    @pyqtSlot()
    def on_confirm_button_clicked(self):
        if self.ui.record_number.text().isdigit():
            record_number = int(self.ui.record_number.text())
            if 100 <= record_number <= 234:
                recordPath = 'resources/database/'
                record = rdrecord(recordPath + str(record_number))
                ecg_length = 3750
                ecg_data = record.p_signal
                ecg_data_0 = ecg_data[:, 0]
                ecg_data_0 = ecg_data_0[:ecg_length]
                self.ecg_signal = ecg_data_0
                self.fs = 360
    
    @pyqtSlot()
    def on_collect_clicked(self):
        self.ui.graphicsView.clear()
        self.ui.graphicsView.setXRange(0,3750)
        # self.ui.graphicsView.setYRange(-0.4,+2)
        if hasattr(self, 'PCF8591_worker'):
            self.PCF8591_worker.terminate()
        if self.ui.comboBox_2.currentText() == '实测模式':
            self.PCF8591_worker = PCF8591_Worker()
            self.PCF8591_worker.collected_signal.connect(self.update_ecg_signal)
            self.PCF8591_worker.progress_signal.connect(self.update_progress)  # 连接进度信号
            self.PCF8591_worker.start()
            self.timer3.start()
        else:
            self.timer3.start()


    @pyqtSlot()
    def on_filter_clicked(self):
        self.ui.graphicsView_2.clear()
        self.ui.graphicsView_2.setXRange(0,3750)
        self.ui.graphicsView_2.setYRange(-0.4,+2)

        self.detector_PanTompkins, self.detector_Elgendi_QRS, self.detector_Elgendi_PT = ecg_data_analysis(
            self.ecg_signal, self.fs)
        self.detector_PanTompkins.detect_QRS_peaks()
        self.detector_Elgendi_QRS.detect_QRS_peaks()
        self.detector_Elgendi_PT.detect_PT_peaks()
        # self.filtered_signal = self.detector_Elgendi_QRS.filtered_signal
        filtered_signal = self.detector_Elgendi_QRS.filtered_signal
        self.filtered_signal = (filtered_signal - np.min(filtered_signal)) / np.ptp(filtered_signal) * 2
        # pen = pg.mkPen(color='b', width=3)
        # self.ui.graphicsView_2.plot().setData(self.filtered_signal, pen=pen)
        self.timer2.start()


            # self.worker2 = Worker2(self.filtered_signal)
            # self.worker2.update_signal.connect(self.update_drawing)
            # self.worker2.start()

    @pyqtSlot()
    def on_peaks_clicked(self):
        self.ui.graphicsView_2.clear()
        self.detector_PanTompkins, self.detector_Elgendi_QRS, self.detector_Elgendi_PT = ecg_data_analysis(
            self.ecg_signal, self.fs)
        self.detector_PanTompkins.detect_QRS_peaks()
        self.detector_Elgendi_QRS.detect_QRS_peaks()
        self.detector_Elgendi_PT.detect_PT_peaks()
        filtered_signal = self.detector_Elgendi_QRS.filtered_signal
        pen = pg.mkPen(color='b', width=3)
        self.ui.graphicsView_2.plot().setData(x=range(0, len(filtered_signal)), y=filtered_signal, pen=pen)
        signal_T_Peaks = self.detector_Elgendi_PT.T_peaks
        # print(type(signal_T_Peaks))
        pen2 = pg.mkPen(color='#7180AC')
        scatter = pg.ScatterPlotItem(size=12, pen=pen2,brush='#7180AC')
        for T_Peak in signal_T_Peaks:
            scatter.addPoints(x=[T_Peak], y=[filtered_signal[T_Peak]])
        self.ui.graphicsView_2.addItem(scatter,name='T-peaks')

        signal_R_Peaks = self.detector_Elgendi_QRS.R_peaks
        # print(signal_R_Peaks)
        pen3 = pg.mkPen(color='#E49273')
        scatter2 = pg.ScatterPlotItem(size=12, pen=pen3,brush='#E49273')
        for R_Peak in signal_R_Peaks:
            scatter2.addPoints(x=[R_Peak], y=[filtered_signal[R_Peak]])

        self.ui.graphicsView_2.addItem(scatter2)

        signal_P_Peaks = self.detector_Elgendi_PT.P_peaks
        # print(signal_P_Peaks)
        pen4 = pg.mkPen(color='#2CA6A4')
        scatter3 = pg.ScatterPlotItem(size=12, pen=pen4,brush='#2CA6A4')
        for P_Peak in signal_P_Peaks:
            scatter3.addPoints(x=[P_Peak], y=[filtered_signal[P_Peak]])
        self.ui.graphicsView_2.addItem(scatter3)

        signal_Q_Peaks = self.detector_Elgendi_QRS.Q_peaks
        # print(signal_Q_Peaks)
        pen5 = pg.mkPen(color='#D90368')
        scatter4 = pg.ScatterPlotItem(size=12, pen=pen5,brush='#D90368')
        for Q_Peak in signal_Q_Peaks:
            scatter4.addPoints(x=[Q_Peak], y=[filtered_signal[Q_Peak]])
        self.ui.graphicsView_2.addItem(scatter4)

        signal_S_Peaks = self.detector_Elgendi_QRS.S_peaks
        # print(signal_S_Peaks)
        pen6 = pg.mkPen(color='#493B2A')
        scatter5 = pg.ScatterPlotItem(size=12, pen=pen6,brush='#493B2A')
        for S_Peak in signal_S_Peaks:
            scatter5.addPoints(x=[S_Peak], y=[filtered_signal[S_Peak]])
        self.ui.graphicsView_2.addItem(scatter5)

    @pyqtSlot()
    def on_analysis_clicked(self):

        R_peaks_Pantompkins = self.detector_PanTompkins.R_peaks
        R_peaks_Elgendi = self.detector_Elgendi_QRS.R_peaks
        Q_peaks = self.detector_Elgendi_QRS.Q_peaks
        S_peaks = self.detector_Elgendi_QRS.S_peaks
        P_peaks = self.detector_Elgendi_PT.P_peaks
        T_peaks = self.detector_Elgendi_PT.T_peaks

        heart_rate_Pantompkins, heart_rate_min_Pantompkins, heart_rate_max_Pantompkins, heart_rate_Elgendi, heart_rate_min_Elgendi, heart_rate_max_Elgendi, PR_interval_mean, ST_interval_mean, QT_interval_mean, QRS_interval_mean, QRS_interval_min, QRS_interval_max, HRV = heart_parameters(
            R_peaks_Pantompkins, R_peaks_Elgendi, P_peaks, T_peaks, Q_peaks, S_peaks, self.fs)

        if self.ui.comboBox_2.currentText() == '实测模式':
            heart_rate_Pantompkins = heart_rate_Pantompkins * 0.5
            heart_rate_min_Pantompkins = heart_rate_min_Pantompkins * 0.5
            heart_rate_max_Pantompkins = heart_rate_max_Pantompkins * 0.5
            heart_rate_Elgendi = heart_rate_Elgendi * 0.5
            heart_rate_min_Elgendi = heart_rate_min_Elgendi * 0.5
            heart_rate_max_Elgendi = heart_rate_max_Elgendi * 0.5

        self.heart_rate_PanTompkins_mean = heart_rate_Pantompkins
        self.heart_rate_PanTompkins_min = heart_rate_min_Pantompkins
        self.heart_rate_PanTompkins_max = heart_rate_max_Pantompkins
        self.QRS_time_PanTompkins_mean = QRS_interval_mean
        self.PR_time = PR_interval_mean
        self.ST_time = ST_interval_mean
        self.QT_time = QT_interval_mean
        self.HRV = HRV

        # print(self.QRS_time_PanTompkins)
        self.ui.QRStime.setText(str(round(QRS_interval_min, 3)))
        self.ui.QRStime_2.setText(str(round(QRS_interval_mean, 3)))
        self.ui.QRStime_3.setText(str(round(QRS_interval_max, 3)))

        # 两种心率显示
        self.ui.heartrate1.setText(str(round(heart_rate_min_Pantompkins, 3)))
        self.ui.heartrate1_4.setText(str(round(heart_rate_Pantompkins, 3)))
        self.ui.heartrate1_6.setText(str(round(heart_rate_max_Pantompkins, 3)))
        self.ui.heartrate2.setText(str(round(heart_rate_min_Elgendi, 3)))
        self.ui.heartrate2_3.setText(str(round(heart_rate_Elgendi, 3)))
        self.ui.heartrate2_4.setText(str(round(heart_rate_max_Elgendi, 3)))

        self.ui.lineEdit.setText(str(round(PR_interval_mean, 3)))
        self.ui.lineEdit.setReadOnly(True)
        self.ui.lineEdit_3.setText(str(round(QT_interval_mean, 3)))
        self.ui.lineEdit_3.setReadOnly(True)

        self.ui.lineEdit_4.setText(str(round(ST_interval_mean, 3)))
        self.ui.lineEdit_4.setReadOnly(True)

        self.ui.lineEdit_6.setText(str(round(HRV, 3)))
        self.ui.lineEdit_6.setReadOnly(True)

    @pyqtSlot()
    def on_report_2_clicked(self):
        self.heart_report_generator(平均心率=self.heart_rate_PanTompkins_mean, 最大心率=self.heart_rate_PanTompkins_max,
                                    最小心率=self.heart_rate_PanTompkins_min, QRS波宽度=self.QRS_time_PanTompkins_mean,
                                    PR间期=self.PR_time, ST间期=self.ST_time, QT间期=self.QT_time, HRV=self.HRV)

    @pyqtSlot()
    def on_clear_clicked(self):
        self.ui.textEdit.clear()