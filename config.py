from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MFCCConfig:
    """统一管理 MFCC 实验参数与项目路径。"""

    project_root: Path = Path(__file__).resolve().parent
    wav_dir: Path = Path(__file__).resolve().parent / "wav"
    data_dir: Path = Path(__file__).resolve().parent / "data"
    results_dir: Path = Path(__file__).resolve().parent / "results"
    outputs_dir: Path = Path(__file__).resolve().parent / "outputs"
    figures_dir: Path = Path(__file__).resolve().parent / "results" / "figures"
    features_dir: Path = Path(__file__).resolve().parent / "results" / "features"
    comparison_dir: Path = Path(__file__).resolve().parent / "results" / "comparison"
    report_dir: Path = Path(__file__).resolve().parent / "report"

    sample_rate: int = 16000
    pre_emphasis_alpha: float = 0.97

    frame_length_ms: float = 25.0
    frame_step_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 26
    n_mfcc: int = 12

    mel_fmin: float = 300.0
    mel_fmax: float = 8000.0
    delta_window: int = 2
    log_epsilon: float = 1e-10
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
