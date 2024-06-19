# 信号处理项目

## 项目简介

本项目旨在设计和实现一个基于树莓派的实时采集和分析心电数据的系统。系统包括数据采集、信号处理、图形用户界面以及报告生成等功能。

## 目录结构

```plaintext
- .git/                   # Git版本控制目录
- __pycache__/            # Python字节码缓存目录
- ecg_data_analysis.py    # 心电数据分析脚本
- main.py                 # 项目的主要入口文件
- myWfdb/                 # 自定义心电数据处理库
- openai_worker.py        # 与OpenAI API交互的脚本
- PCF8591_worker.py       # 处理PCF8591传感器数据的脚本
- README.md               # 项目说明文件
- requirements.txt        # 项目依赖的Python库列表
- resources/              # 资源文件夹（心电数据）
- se_mainwindow.py        # PyQt生成的主窗口脚本
- ui_design.ui            # Qt Designer生成的UI文件
- Ui_ui_design.py         # 由UI设计文件生成的Python代码
- utils.py                # 辅助函数脚本
```

## 安装与运行
### 安装依赖
本项目需要用户在树莓派 4B 上运行，确保已安装Python 3.7或更高版本。然后，使用以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```
### API 配置
在 `se_mainwindow.py` 文件中，需要配置OpenAI API的密钥。请在 [OpenAI](https://platform.openai.com/playground) 注册账号并获取API密钥。

### 运行项目

运行以下命令启动项目：

```bash
python main.py
```

## 文件解析

### `ecg_data_analysis.py`

该文件实现了对心电图信号的分析和处理。主要功能包括使用陷波滤波器、小波变换降噪、去趋势化、PanTompkins心律检测器、Elgendi QRS波检测器和Elgendi PT波检测器。

主要函数：
- `ecg_data_analysis(ecg_signal, fs)`: 对心电信号进行处理并使用PanTompkins和Elgendi算法检测心律和心律参数。
- `heart_parameters(R_peaks_Pantompkins, R_peaks_Elgendi, P_peaks, T_peaks, Q_peaks, S_peaks, fs)`: 计算心电信号的各种心率参数。
- `denoise_with_coeffs(ecg_signal)`: 使用小波变换对心电信号进行降噪。
- `trap_filter(wave, fs)`: 使用陷波滤波器对信号进行滤波。
- `detrend_wave(wave)`: 对信号进行去趋势化处理。
- `PanTompkinsDetector`: PanTompkins心律检测器类，用于检测QRS波。
- `ElgendiDetector_QRS`: Elgendi QRS波检测器类，用于使用Elgendi方法检测QRS波。
- `ElgendiDetector_PT`: Elgendi PT波检测器类，用于检测PT波。

### `main.py`

这是一个基于PyQt5的应用程序，主要功能是加载心电图信号数据并进行分析。程序通过引入一个名为 "SE" 的自定义窗口类用于显示心电图数据和分析结果。

主要功能：
- 加载并应用样式文件。
- 创建QApplication实例，初始化并显示SE窗口。
- 运行应用程序。

### `openai_worker.py`

这个文件包含一个名为Worker的类，继承自QThread类，用于与客户端进行聊天并处理返回的数据流。

主要功能：
- 实现一个处理客户端聊天逻辑的线程类。

### `PCF8591_worker.py`

用于PCF8591模块的Python程序，通过模拟输入读取来自模拟传感器的电压值，并将结果发送到一个列表中。

主要功能：
- 读取模拟传感器数据并发送到列表。

### `se_mainwindow.py`

基于PyQt5和pyqtgraph的心电图信号检测和分析程序。

主要功能：
- 实时采集心电信号。
- 进行滤波处理。
- ECG波峰检测。
- 心率参数分析。
- 生成心电报告。

### `Ui_ui_design.py`

基于PyQt5的用户界面设计文件，用于心电图数据处理和分析。

主要功能：
- 提供四个按钮（采集心电图、过滤心电图、峰值显示、心电图数据分析）。
- 提供输入框和图形显示部分。
- 设计用于心电图数据处理和分析的用户界面。

### `utils.py`

提供用于处理路径问题的工具函数。

主要功能：
- 获取相对路径的绝对路径。

### `myWfdb`目录

包含多个子模块和文件，用于处理心电图信号的检测和分析。

#### 主要文件及功能：

| 文件名 | 功能描述 |
| --- | --- |
| version.py | 定义当前项目的版本号为4.1.2 |
| __init__.py | 提供对心电图信号进行检测和分析的功能 |
| io/annotation.py | 处理注释文件的功能模块 |
| io/datasource.py | 提供与数据源进行交互的类和函数 |
| io/download.py | 从PhysioNet数据库下载和处理心电图信号的功能 |
| io/header.py | 解析心电图信号文件的头部信息 |
| io/record.py | 实现Record对象和MultiRecord对象的WFDB文件解析器 |
| io/util.py | 通用工具函数模块 |
| io/_coreio.py | 提供打开数据文件作为随机访问文件对象的功能 |
| io/_header.py | 处理WFDB信号文件的头部信息 |
| io/_signal.py | 读取心电图信号数据 |
| io/_url.py | 处理远程文件的功能 |
| io/convert/csv.py | 将CSV文件转换为WFDB记录对象或注释对象 |
| io/convert/edf.py | 读取和转换EDF格式文件的功能 |
| io/convert/matlab.py | 将WFDB格式的生理信号转换为Matlab中读取的.mat文件和文本文件 |
| io/convert/tff.py | 用于读取ME6000 .tff格式文件 |
| io/convert/wav.py | 在WFDB记录和.wav文件之间进行转换 |
| plot/plot.py | 绘制心电图信号和注释 |
| plot/__init__.py | 绘制信号和注释的工具 |
| processing/basic.py | 处理心电图信号的基本处理功能 |
| processing/evaluate.py | 比较两组心电图信号注释的工具 |
| processing/filter.py | 对心电图信号进行平均处理 |
| processing/hr.py | 计算瞬时心率和R-R间隔的函数 |
| processing/peaks.py | 检测和处理ECG信号中的峰值 |
| processing/qrs.py | 实现QRS波的检测算法 |
| processing/__init__.py | 包含多个模块的导入，用于信号处理、评估心率、峰值处理、QRS波处理等功能 |

## 贡献指南

欢迎对本项目进行贡献。请遵循以下步骤：

1. Fork本仓库。
2. 创建一个新的分支 (`git checkout -b feature-branch`)。
3. 提交你的修改 (`git commit -am 'Add new feature'`)。
4. 推送到分支 (`git push origin feature-branch`)。
5. 创建一个新的Pull Request。

## 许可证

本项目采用 [MIT 许可证](./LICENSE)。
