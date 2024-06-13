from scipy import signal
import numpy as np
import pandas as pd
from vmdpy import VMD
import pywt

def ecg_data_analysis(ecg_signal, fs):
    # 使用陷波滤波器
    trap_filtered_signal = trap_filter(ecg_signal, fs)
    # 使用小波变换降噪
    denoised_signal = denoise_with_coeffs(trap_filtered_signal)
    # 去趋势化
    detrended_signal = detrend_wave(denoised_signal)

    processed_signal = 4 * detrended_signal ** 2 * np.sign(detrended_signal)

    # normalized_signal = (processed_signal - np.min(processed_signal)) / np.ptp(processed_signal)
    # trap_filtered_signal = detrended_signal
    # 使用 VMD 降噪
    # denoised_signal = denoise_with_VMD(ecg_signal)
    # 使用原始信号
    # denoised_signal = trap_filtered_signal

    detector_PanTompkins = PanTompkinsDetector(processed_signal, fs)
    detector_Elgendi_QRS = ElgendiDetector_QRS(processed_signal, fs)
    detector_Elgendi_PT = ElgendiDetector_PT(processed_signal, fs, detector_Elgendi_QRS.R_peaks)

    return detector_PanTompkins, detector_Elgendi_QRS, detector_Elgendi_PT

def heart_parameters(R_peaks_Pantompkins, R_peaks_Elgendi, P_peaks, T_peaks, Q_peaks, S_peaks, fs):
    R_peaks_Pantompkins = pd.Series(R_peaks_Pantompkins)
    R_peaks_Elgendi = pd.Series(R_peaks_Elgendi)

    R_peaks_Pantompkins_low = R_peaks_Pantompkins.quantile(0.25)
    R_peaks_Pantompkins_high = R_peaks_Pantompkins.quantile(0.75)
    R_peaks_Elgendi_low = R_peaks_Elgendi.quantile(0.25)
    R_peaks_Elgendi_high = R_peaks_Elgendi.quantile(0.75)

    R_peaks_Pantompkins = R_peaks_Pantompkins[(R_peaks_Pantompkins > R_peaks_Pantompkins_low) & (R_peaks_Pantompkins < R_peaks_Pantompkins_high)]
    R_peaks_Elgendi = R_peaks_Elgendi[(R_peaks_Elgendi > R_peaks_Elgendi_low) & (R_peaks_Elgendi < R_peaks_Elgendi_high)]

    RR_intervals_Pantompkins = np.diff(R_peaks_Pantompkins) / fs
    RR_intervals_Pantompkins_mean = np.mean(RR_intervals_Pantompkins)
    RR_intervals_Pantompkins_min = np.min(RR_intervals_Pantompkins)
    RR_intervals_Pantompkins_max = np.max(RR_intervals_Pantompkins)

    heart_rate_Pantompkins = 60 / RR_intervals_Pantompkins_mean
    heart_rate_min_Pantompkins = 60 / RR_intervals_Pantompkins_max
    heart_rate_max_Pantompkins = 60 / RR_intervals_Pantompkins_min

    RR_intervals_Elgendi = np.diff(R_peaks_Elgendi) / fs
    RR_intervals_Elgendi_mean = np.mean(RR_intervals_Elgendi)
    RR_intervals_Elgendi_min = np.min(RR_intervals_Elgendi)
    RR_intervals_Elgendi_max = np.max(RR_intervals_Elgendi)

    heart_rate_Elgendi = 60 / RR_intervals_Elgendi_mean
    heart_rate_min_Elgendi = 60 / RR_intervals_Elgendi_max
    heart_rate_max_Elgendi = 60 / RR_intervals_Elgendi_min

    PR_interval, ST_interval, QT_interval, QRS_interval = [], [], [], []

    wave_group = []

    for R_peak_Elgendi in R_peaks_Elgendi:
        P_peak = P_peaks[np.argmin(np.abs(np.subtract(P_peaks, R_peak_Elgendi)))]
        T_peak = T_peaks[np.argmin(np.abs(np.subtract(T_peaks, R_peak_Elgendi)))]
        Q_peak = Q_peaks[np.argmin(np.abs(np.subtract(Q_peaks, R_peak_Elgendi)))]
        S_peak = S_peaks[np.argmin(np.abs(np.subtract(S_peaks,  R_peak_Elgendi)))]

        PR_interval_temp = np.abs(P_peak - R_peak_Elgendi) / fs
        QR_interval_temp = np.abs(Q_peak - R_peak_Elgendi) / fs
        SR_interval_temp = np.abs(R_peak_Elgendi - S_peak) / fs
        TR_interval_temp = np.abs(R_peak_Elgendi - T_peak) / fs

        if [PR_interval_temp, QR_interval_temp, SR_interval_temp, TR_interval_temp] < [0.4, 0.4, 0.4, 0.4]:
            wave_group.append([P_peak, Q_peak, R_peak_Elgendi, S_peak, T_peak])

    print(wave_group)

    for wave_group_item in wave_group:
        PR_interval.append(np.abs(wave_group_item[0] - wave_group_item[2]) / fs)
        ST_interval.append(np.abs(wave_group_item[3] - wave_group_item[4]) / fs)
        QT_interval.append(np.abs(wave_group_item[1] - wave_group_item[4]) / fs)
        QRS_interval.append(np.abs(wave_group_item[1] - wave_group_item[3]) / fs)

    PR_interval_mean = np.mean(PR_interval)
    ST_interval_mean = np.mean(ST_interval)
    QT_interval_mean = np.mean(QT_interval)

    QRS_interval_mean = np.mean(QRS_interval)
    QRS_interval_min = np.min(QRS_interval)
    QRS_interval_max = np.max(QRS_interval)

    HRV = np.std(RR_intervals_Elgendi)

    return heart_rate_Pantompkins, heart_rate_min_Pantompkins, heart_rate_max_Pantompkins, heart_rate_Elgendi, heart_rate_min_Elgendi, heart_rate_max_Elgendi, PR_interval_mean, ST_interval_mean, QT_interval_mean, QRS_interval_mean, QRS_interval_min, QRS_interval_max, HRV


def denoise_with_coeffs(ecg_signal):
    # 执行小波变换
    coeffs = pywt.wavedec(ecg_signal, wavelet='db4', level=4)

    # 应用阈值降噪
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(ecg_signal)))
    coeffs = [pywt.threshold(c, threshold) for c in coeffs]
    denoised_ecg_signal = pywt.waverec(coeffs, wavelet='db4')
    return denoised_ecg_signal

def trap_filter(wave, fs):
    b, a = signal.iirnotch(w0=50, Q=30, fs=fs)
    filtered_wave = signal.filtfilt(b, a, wave)

    return filtered_wave

def detrend_wave(wave):
    return signal.detrend(wave)



class PanTompkinsDetector:
    def __init__(self, signal, fs):
        self.signal = signal
        self.fs = fs
        self.filtered_signal = None
        self.derivative_signal = None
        self.squared_signal = None
        self.preprocessed_signal = None
        self.study_signal = None
        self.preprocessed_signal = None
        self.QRS_widths = []
        self.Q_peaks , self.P_peaks = None, None
        self.peaks = []
        self.thresholds = {'THRESHOLD1': 0, 'THRESHOLD2': 0}
        self.RR_average1, self.RR_average2, self.SPK, self.NPK = 0, 0, 0, 0
        self.RR_low_limit, self.RR_high_limit, self.RR_missed_limit = 0, 0, 0
        self.R_peaks = []

    def bandpass_filter(self):
        # 低通滤波器
        low_coef_numerator = [1] + [0]*5 + [-2] + [0]*5 + [1]
        low_coef_denominator = [1, -2, 1]
        low_filtered_ecg = signal.lfilter(low_coef_numerator, low_coef_denominator, self.signal)

        # 高通滤波器
        high_coef_numerator = [-1/32] + [0]*15 + [1] + [-1] + [0]*14 + [1/32]
        high_coef_denominator = [1, -1]
        high_filtered_ecg = signal.lfilter(high_coef_numerator, high_coef_denominator, low_filtered_ecg)

        self.filtered_signal = (high_filtered_ecg - np.min(high_filtered_ecg)) / (np.max(high_filtered_ecg) - np.min(high_filtered_ecg))
        # b, a = signal.butter(3, [5, 15], btype='bandpass', fs=self.fs)
        # self.filtered_signal = signal.filtfilt(b, a, self.signal)

    def derivative_filter(self):
        b, a = [1, 2, 0, -2, -1], [1]
        self.derivative_signal = signal.lfilter(b, a, self.filtered_signal)

    def moving_average_window(self):
        window_size = 25
        window = np.ones(window_size) / window_size
        self.preprocessed_signal = np.convolve(self.squared_signal, window, 'same')
        self.preprocessed_signal = np.convolve(self.preprocessed_signal, window, 'same')

        # # 步骤2: 突出显著部分
        # # 这里使用平方来增加对比度
        # self.preprocessed_signal = signal_smooth * np.abs(signal_smooth)

    def further_operaton(self):
        self.derivative_filter()
        self.squared_signal = self.derivative_signal ** 2

        self.moving_average_window()

    def search_back(self, peak, last_R_peak):
        search_back_list = self.peaks[last_R_peak:peak]

        # 找到最大值所在处的索引
        max_signal_index = np.argmax(self.preprocessed_signal[search_back_list])
        max_signal_index = search_back_list[max_signal_index]

        max_signal_value = self.preprocessed_signal[max_signal_index]

        if max_signal_value > self.thresholds['THRESHOLD2']:
            # 如果这个峰值大于阈值2，那么就认为它是一个 R 波
            self.R_peaks.append(max_signal_index)
            last_R_peak = max_signal_index
            self.SPK = 0.25 * max_signal_value + 0.75 * self.SPK
            self.update_RR_average()

    def detect_T_peaks(self, peak, last_R_peak):
        if last_R_peak != None and peak - last_R_peak < 0.36 * self.fs and self.preprocessed_signal[peak] < 0.5 * self.preprocessed_signal[last_R_peak]:
            self.NPK = 0.125 * self.preprocessed_signal[peak] + 0.875 * self.NPK
        else:
            self.R_peaks.append(peak)
            last_R_peak = peak
            self.SPK = 0.125 * self.preprocessed_signal[peak] + 0.875 * self.SPK
            self.update_RR_average()

    def update_RR_average(self):
        RR_intervals = np.diff(self.R_peaks)

        self.RR_low_limit = 0.92 * self.RR_average2
        self.RR_high_limit = 1.16 * self.RR_average2
        self.RR_missed_limit = 1.66 * self.RR_average2

        conditions = np.logical_and(RR_intervals <self.RR_low_limit, RR_intervals > self.RR_high_limit)
        RR_prime_intervals = RR_intervals[conditions]

        if len(self.R_peaks) < 8:
            self.RR_average1 = np.mean(RR_intervals)
            if len(RR_prime_intervals) < 8:
                self.RR_average2 = np.mean(RR_prime_intervals)
            else:
                self.RR_average2 = np.mean(RR_prime_intervals[:-8])
        else:
            self.RR_average1 = np.mean(RR_intervals[:-8])

        if self.RR_low_limit < RR_intervals[:-8].all() < self.RR_high_limit:
            self.RR_average2 = self.RR_average1

    def detect_QRS_peaks(self):
        self.bandpass_filter()
        self.further_operaton()

        # 首先开始 2 秒的学习阶段
        signal_percentage = 0.6
        noise_percentage = 0.3
        study_time = 2

        self.SPK = signal_percentage * np.max(self.preprocessed_signal[:int(study_time * self.fs)])
        self.NPK = noise_percentage * np.mean(self.preprocessed_signal[:int(study_time * self.fs)])
        self.thresholds["THRESHOLD1"] = self.NPK + 0.25 * (self.SPK - self.NPK)
        self.thresholds["THRESHOLD2"] = 0.5 * self.thresholds["THRESHOLD1"]

        # 然后开始正式检测
        self.peaks, _ = signal.find_peaks(self.preprocessed_signal[int(study_time * self.fs):],
                                          distance=0.2*self.fs)
        self.peaks = np.array(self.peaks) + int(study_time * self.fs)

        # 从2s处开始检测，找到所有可能的峰值，并确保相互之间距离不少于200ms
        last_R_peak = None
        self.RR_average1 = self.RR_average2 = np.mean(np.diff(self.peaks))
        self.update_RR_average()


        for peak in self.peaks:
            if self.preprocessed_signal[peak] > self.thresholds["THRESHOLD1"]:
                self.detect_T_peaks(peak, last_R_peak) # 如果这个峰值大于阈值1，进行最后是否为 T 波的判断再下结论
            else:   # 如果这个峰值小于阈值1，那么直接把它暂时判断为噪声
                self.NPK = 0.125 * self.preprocessed_signal[peak] + 0.875 * self.NPK
            if last_R_peak != None and peak - last_R_peak > self.RR_missed_limit:
                self.search_back(peak, last_R_peak)

            self.thresholds["THRESHOLD1"] = self.NPK + 0.25 * (self.SPK - self.NPK)
            self.thresholds["THRESHOLD2"] = 0.5 * self.thresholds["THRESHOLD1"]

        # 进行QRS宽度的检测
        self.QRS_widths = signal.peak_widths(self.preprocessed_signal, self.peaks, rel_height=0.1)[0]

# 使用 Elgendi 方法检测 QRS 波
class ElgendiDetector_QRS():
    def __init__(self, ecg_data, fs):
        self.signal = ecg_data
        self.fs = fs
        self.R_peaks = []
        self.P_peaks, self.T_peaks = [], []
        self.filtered_signal = None
        self.squared_signal = None
        self.W1 = int(0.060 * self.fs)
        self.W2 = int(0.611 * self.fs)
        self.MA_QRS = None
        self.MA_beat = None
        self.QRS_widths = []
        self.Q_peaks, self.S_peaks = [], []
        self.THR1 = self.THR2 = 0
        self.beta, self.alpha = 0.01, 0
        self.block_of_interest, self.Blocks = None, []
    def bandpass_filter(self):
        # 使用 3 阶巴特沃思带通滤波器滤波
        b, a = signal.butter(3, [0.5, 50], btype='bandpass', fs=self.fs)
        # b, a = signal.butter(3, [8, 20], btype='bandpass', fs=self.fs)
        self.filtered_signal = signal.filtfilt(b, a, self.signal)
    def square_filtered_signal(self):
        # 对滤波后的信号进行平方操作
        self.squared_signal = self.filtered_signal ** 2
    def integrate_operation(self):
        QRS_window_size = self.W1
        self.MA_QRS = np.convolve(self.squared_signal, np.ones(QRS_window_size) / QRS_window_size, mode='same')

        beat_window_size = self.W2
        self.MA_beat = np.convolve(self.squared_signal, np.ones(beat_window_size) / beat_window_size, mode='same')

        z = np.mean(self.squared_signal)
        self.alpha = self.beta * z + self.MA_beat

        self.THR1 = self.MA_beat + self.alpha
        self.THR2 = self.W1
    def generate_interest_blocks(self):
        for n in range(len(self.MA_QRS)):
            if self.MA_QRS[n] > self.THR1[n]:
                self.block_of_interest[n] = 1
            else:
                self.block_of_interest[n] = 0
        # Blocks的第一列存储block_of_interest非零项开始的位置，第二列存储非零项持续的样本长度
        for n in range(len(self.block_of_interest) - 1):
            if self.block_of_interest[n] == 0 and self.block_of_interest[n + 1] == 1:
                self.Blocks.append([n, 0])
            if self.block_of_interest[n] == 1 and len(self.Blocks) != 0:
                self.Blocks[-1][1] += 1
    def detect_QRS_peaks(self):
        self.bandpass_filter()
        self.square_filtered_signal()
        self.integrate_operation()

        self.block_of_interest = np.zeros(len(self.MA_QRS))
        self.generate_interest_blocks()
        for block in self.Blocks:
            if block[1] >= self.THR2:
                max_value_location = np.argmax(self.filtered_signal[block[0]:block[0] + block[1]])
                self.R_peaks.append(block[0] + max_value_location)

        find_interval = 100
    # 找到每个R波左右两侧交叉点位置
        for R_peak in self.R_peaks:
            self.Q_peaks.append(np.argmin(self.filtered_signal[max(R_peak-find_interval, 0) : R_peak]) + max(R_peak-find_interval, 0)) # Q点位置
            self.S_peaks.append(np.argmin(self.filtered_signal[R_peak : min(R_peak+find_interval, len(self.filtered_signal))]) + R_peak)

        self.QRS_widths = np.array(self.S_peaks) - np.array(self.Q_peaks)

class ElgendiDetector_PT():
    def __init__(self, ecg_data, fs, R_peaks):
        self.signal = ecg_data
        self.fs = fs
        self.R_peaks = R_peaks
        self.P_peaks, self.T_peaks = [], []
        self.filtered_signal = None
        self.removed_R_signal = None
        self.MA_Peak = None
        self.MA_P_wave = None
        self.block_of_interest = None
        self.Blocks = []
    def detect_PT_peaks(self):
        self.bandpass_filter()
        self.R_remove()
        self.integrate_operation()

        self.block_of_interest = np.zeros(len(self.MA_Peak))
        self.generate_interest_blocks()
        self.detect_P_peaks()
        self.detect_T_peaks()

    def bandpass_filter(self):
        # 使用 3 阶巴特沃思带通滤波器滤波
        b, a = signal.butter(3, [0.5, 50], btype='bandpass', fs=self.fs)
        # b, a = signal.butter(3, [8, 20], btype='bandpass', fs=self.fs)
        self.filtered_signal = signal.filtfilt(b, a, self.signal)

    def R_remove(self):
        self.removed_R_signal = self.filtered_signal.copy()
        # R波的前0.030s和后0.050s被设置为零
        remove_prev = int(0.030 * self.fs)
        remove_post = int(0.050 * self.fs)
        for R_peak in self.R_peaks:
            self.removed_R_signal[R_peak- remove_prev: R_peak + remove_post] = 0

    def integrate_operation(self):
        P_wave_window_size = int(0.11 * self.fs)
        self.MA_P_wave = np.convolve(self.removed_R_signal, np.ones(P_wave_window_size) / P_wave_window_size, mode='same')

        peak_window_size = int(0.055 * self.fs)
        self.MA_Peak = np.convolve(self.removed_R_signal, np.ones(peak_window_size) / peak_window_size, mode='same')

    def generate_interest_blocks(self):
        for n in range(len(self.MA_Peak)):
            if self.MA_Peak[n] <= self.MA_P_wave[n]:
                self.block_of_interest[n] = 0
            else:
                self.block_of_interest[n] = 1
        # Blocks的第一列存储block_of_interest非零项开始的位置，第二列存储非零项持续的样本长度
        for n in range(len(self.block_of_interest) - 1):
            if self.block_of_interest[n] == 0 and self.block_of_interest[n + 1] == 1:
                self.Blocks.append([n, 0])
            if self.block_of_interest[n] == 1 and len(self.Blocks) != 0:
                self.Blocks[-1][1] += 1
        # 移除包含 R_peaks 的 Block
        for R_peak in self.R_peaks:
            for block in self.Blocks:
                if block[0] <= R_peak <= block[0] + block[1]:
                    self.Blocks.remove(block)

    def detect_P_peaks(self):
        R_intervals = np.diff(self.R_peaks)
        P_min = R_intervals * 0.15
        P_max = R_intervals * 0.32
        Block_duration = 0

        for n in range(1, len(self.Blocks)):
            Block_duration = range(self.Blocks[n-1][0], self.Blocks[n-1][0] + self.Blocks[n-1][1])

            max_value_location = np.argmax(self.removed_R_signal[Block_duration]) + self.Blocks[n-1][0]  # 计算块的最大值位置
            # 找到这个最大值的位置在哪两个R波之间
            index = np.searchsorted(self.R_peaks, max_value_location)
            if 0 < index < len(self.R_peaks):
                R_prev = self.R_peaks[index-1]
                R_next = self.R_peaks[index]

                if P_min[index-1] <=  R_next - max_value_location <= P_max[index-1]:
                    self.P_peaks.append(max_value_location)
                if len(self.P_peaks) >= 2 and self.P_peaks[-1] - self.P_peaks[-2] < 0.15 * self.fs:
                    self.P_peaks.pop(-2)

    def detect_T_peaks(self):
        R_intervals = np.diff(self.R_peaks) # 计算 R 波之间的间隔，注意第 n-1 个元素代表第 n-1 个 R 波和第 n 个 R 波之间的间隔
        T_min = R_intervals * 0.30
        T_max = R_intervals * 0.50

        for n in range(1, len(self.Blocks)):
            Block_duration = range(self.Blocks[n-1][0], self.Blocks[n-1][0] + self.Blocks[n-1][1])
            max_value_location = np.argmax(self.removed_R_signal[Block_duration]) + self.Blocks[n-1][0]

            index = np.searchsorted(self.R_peaks, max_value_location)
            if 0 < index < len(self.R_peaks):
                R_prev = self.R_peaks[index-1]
                R_next = self.R_peaks[index]
                if T_min[index-1] <= max_value_location - R_prev <= T_max[index-1]:
                    self.T_peaks.append(max_value_location)
                if len(self.T_peaks) >= 2 and self.T_peaks[-1] - self.T_peaks[-2] < 0.15 * self.fs:
                    self.T_peaks.pop(-1)


