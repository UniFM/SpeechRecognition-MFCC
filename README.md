# SpeechRecognition MFCC 实验说明

本项目实现了**手写 MFCC 声学特征提取**流程，并支持：

- 单文件调试模式；
- `\wav` 全目录批量模式；
- 中间结果图、特征文件、批量 CSV 汇总；
- 与 `librosa.feature.mfcc` 的静态 MFCC 对照误差分析；
- `pytest` 自动化单元测试。

## 1. 目录结构

```text
D:\SpeechRecognition\
├── data\
├── wav\
├── src\
├── tests\
├── results\
│   ├── figures\
│   ├── features\
│   ├── comparison\
│   └── batch_summary.csv
├── outputs\
├── report\
│   └── report.md
├── config.py
├── plotting.py
├── main.py
├── requirements.txt
└── README.md
```

## 2. 环境准备（PowerShell）

### 2.1 创建独立 Conda 环境（推荐）

```powershell
conda create -n mfcc python=3.11 -y
conda activate mfcc
python -m pip install -r requirements.txt
```

> 如果环境已存在，请直接 `conda activate mfcc`，不要重复创建。

### 2.2 验证依赖

```powershell
python --version
python -m pip show numpy scipy matplotlib soundfile librosa pytest
```

## 3. 运行方式

### 3.1 批量处理（默认读取 `\wav`）

此处仓库地址以`D:\SpeechRecognition`为例

```powershell
conda activate mfcc
cd D:\SpeechRecognition
python main.py
```

### 3.2 指定单条音频

```powershell
python main.py --audio "D:\SpeechRecognition\wav\1089-134686-0000.wav"
```

### 3.3 指定音频目录

```powershell
python main.py --audio-dir "D:\SpeechRecognition\wav"
```

### 3.4 指定时间片段

```powershell
python main.py --audio "D:\SpeechRecognition\wav\1089-134686-0000.wav" --start-time 0.5 --end-time 3.0
```

### 3.5 跳过参考比较（不调用 librosa）

```powershell
python main.py --no-reference
```

### 3.6 进行参考比较（调用 librosa）

```powershell
python main.py
```

### 3.7 为每个文件都生成完整图表

默认只为排序后的第一个文件（批量模式）或 `--audio` 指定的文件生成 01~17 号图，避免 38 套图互相覆盖且耗时过长。需要全部生成时：

```powershell
python main.py --plot-all
```

## 4. 自动化测试

```powershell
python -m pytest tests -q
```

## 5. 输出说明

- `results/batch_summary.csv`：批量汇总（采样率、声道数、时长、帧数、形状、状态、误差指标）
- `results/single_<文件名>_summary.csv`：`--audio` 单文件模式的汇总（不覆盖批量汇总）
- `results/features/<文件名>_arrays.npz`：全部中间数组
- `results/features/<文件名>_feature39_cmvn.csv`：最终 39 维特征
- `results/comparison/<文件名>_comparison.npz`：自定义/librosa MFCC、误差矩阵、两种 Log-Mel
- `results/comparison/<文件名>_metrics.json`：误差指标与完整 librosa 调用参数
- `results/figures/<文件名>/01~17_*.png`：按文件分目录保存的图表
- `outputs/*.png`：代表文件图表的副本（课程要求的固定文件名）

### 5.1 图表清单（与 `report/report.md` 一致）

每个音频目录 `results/figures/<文件名>/` 下包含：

1. `01_waveform.png`
2. `02_preemphasis.png`
3. `03_framing_window.png`
4. `04_hamming_window.png`
5. `05_stft_magnitude.png`
6. `06_power_spectrum.png`
7. `07_mel_filterbank.png`
8. `08_mel_energy.png`
9. `09_log_mel.png`
10. `10_static_mfcc.png`
11. `11_log_energy.png`
12. `12_delta.png`
13. `13_delta_delta.png`
14. `14_features_before_cmvn.png`
15. `15_features_after_cmvn.png`
16. `16_mfcc_comparison.png`
17. `17_mfcc_error.png`

附加图：

- `11b_energy_dynamic.png`（静态能量 / Delta 能量 / Delta-Delta 能量三曲线）

### 5.2 坐标轴约定（与报告一致）

- 时频图（`05`/`06`/`08`/`09`）：横轴 `Time (s)`，纵轴 `Frequency (Hz)`。
- 能量曲线（`11` 与 `11b`）：横轴 `Time (s)`。
- MFCC/Delta/39维热力图：横轴 `Time (s)`，纵轴为系数索引或特征维度（非频率量）。

## 6. 与 librosa 对照的对齐方式

`src/reference_mfcc.py` 是唯一调用 `librosa.feature.mfcc` 的模块，对齐措施：

- 输入使用同一条预加重信号；`window` 传入自定义 Hamming 数组；`htk=True`、`mel_norm=None`、`dct_type=2`、`norm=None`、`lifter=0`；
- librosa 在 `center=False` 时按 `n_fft=512` 计帧且不补尾，并把 400 点窗居中放在 512 点内（偏移 56 点）。因此输入左侧补 56 个零、尾部补零到自定义填充长度、右侧再补 56 个零，使两者帧数相同且每帧覆盖相同采样点，不做任何截断；
- 无法对齐的部分：`librosa.feature.mfcc` 内部固定使用 `power_to_db`（10·log10，`top_db=80` 截断），而自定义实现用自然对数，因此报告乘以 ln(10)/10 后的诊断误差。

## 7. 实现边界说明

- 如果 `wav` 目录不存在或无 `.wav` 文件，程序会写入失败状态到 `batch_summary.csv`，不伪造实验结果。
- 单个文件处理或对照失败时，错误写入汇总 CSV 的 `error` / `mfcc_compare_error` 列，其余文件继续处理。
