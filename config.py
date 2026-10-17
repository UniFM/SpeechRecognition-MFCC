from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# @dataclass 装饰器: 自动为类添加 __init__ / __repr__ 等方法
# slots: 限制实例动态添加属性
@dataclass(slots=True)
class MFCCConfig:
    """统一管理 MFCC 实验参数与项目路径。"""

    # 绝对/相对路径 -> 规范化解析 -> 获取文件目录
    project_root: Path = Path(__file__).resolve().parent
    wav_dir: Path = Path(__file__).resolve().parent / "wav"
    data_dir: Path = Path(__file__).resolve().parent / "data"
    results_dir: Path = Path(__file__).resolve().parent / "results"
    outputs_dir: Path = Path(__file__).resolve().parent / "outputs"
    figures_dir: Path = Path(__file__).resolve().parent / "results" / "figures"
    features_dir: Path = Path(__file__).resolve().parent / "results" / "features"
    comparison_dir: Path = Path(__file__).resolve().parent / "results" / "comparison"
    report_dir: Path = Path(__file__).resolve().parent / "report"

    # 采样率
    sample_rate: int = 16000
    # 预加重系数 y[n] = x[n] - pre_emphasis_alpha x[n-1]
    pre_emphasis_alpha: float = 0.97

    # 帧长
    frame_length_ms: float = 25.0
    # 帧移
    frame_step_ms: float = 10.0
    # FFT 点数
    n_fft: int = 512
    # 梅尔滤波器数量
    n_mels: int = 26
    # 保留的 MFCC 系数数量，即静态 MFCC维度
    n_mfcc: int = 12
    
    # 梅尔频率范围
    mel_fmin: float = 300.0
    # 8000 Hz 正好对应 16kHz 采样率下的奈奎斯特频率（16000 / 2）
    mel_fmax: float = 8000.0
    # 差分窗口大小，计算动态特征时，使用前后各 2 帧（共跨度 5 帧）的回归公式来估计特征随时间变化的斜率
    delta_window: int = 2
    # 对数平滑常数，避免对数运算时出现负无穷大
    log_epsilon: float = 1e-10
    # 归一化常数
    cmvn_epsilon: float = 1e-8

    start_time: float | None = None
    end_time: float | None = None

    @property
    def frame_length(self) -> int:
        return int(round(self.frame_length_ms * self.sample_rate / 1000))

    @property
    def frame_step(self) -> int:
        return int(round(self.frame_step_ms * self.sample_rate / 1000))

    @property
    def batch_summary_csv(self) -> Path:
        return self.results_dir / "batch_summary.csv"


def ensure_project_dirs(cfg: MFCCConfig) -> None:
    """创建实验输出目录。"""
    for path in (
        cfg.data_dir,
        cfg.results_dir,
        cfg.outputs_dir,
        cfg.figures_dir,
        cfg.features_dir,
        cfg.comparison_dir,
        cfg.report_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
