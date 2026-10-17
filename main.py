"""
主函数：执行完整手写 MFCC 39 维特征提取实验

核心执行流程说明：
1. 参数解析与环境初始化：调用 `parse_args()` 解析命令行参数（如单文件路径、音频目录、时间截取范围、是否跳过对照、是否全量出图），初始化全局配置 `MFCCConfig` 并创建所需的项目输出目录。
2. 运行模式分发：
   - 单文件调试模式：若通过 `--audio` 指定了单个文件，则仅处理该音频，生成全套图表并输出单文件汇总。
   - 批量处理模式：若未指定单文件，则自动发现目标目录（默认 `wav/`）下的所有 WAV 文件，开启批量循环处理。
3. 核心特征流水线执行：对每个音频调用 `run_mfcc_pipeline`，依次完成音频加载、预加重、分帧加窗、STFT、功率谱、梅尔滤波器组、Log-Mel、DCT、静态 MFCC、对数能量、一/二阶差分（Delta/Delta-Delta）、39维特征拼接与 CMVN 归一化。
4. 实验产物持久化存储：将每条音频的中间计算数组保存为 `.npz` 压缩文件、最终 39 维归一化特征保存为 `.csv`、各项运行指标保存为 `.json`。
5. 可视化与图表导出：根据策略生成全套全过程图表（01~17号图与11b），并将代表性文件的图表镜像复制到 `outputs/` 目录以满足固定文件名要求。
6. librosa 参考对照（可选）：在未开启 `--no-reference` 的前提下，通过对齐补零将手写结果与工业级 `librosa.feature.mfcc` 进行严谨对照，计算 MAE/RMSE 误差指标，并生成对比曲线与误差热力图。
7. 结果汇总与结案输出：将所有音频的处理状态、特征维度与误差指标汇总写入全局的 `batch_summary.csv` 统计表，并在控制台打印最终成功率与耗时提示。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from config import MFCCConfig, ensure_project_dirs
from plotting import (
    plot_energy_curve,
    plot_error_heatmap,
    plot_first_frame_compare,
    plot_hamming,
    plot_mel_filterbank,
    plot_mfcc_comparison,
    plot_spectrogram,
    plot_three_energy_curves,
    plot_two_waveforms,
    plot_waveform,
)
from src.mfcc_pipeline import PipelineResult, run_mfcc_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手写 MFCC 39 维特征提取实验")
    parser.add_argument("--audio", type=Path, default=None, help="单文件调试模式 WAV 路径")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="批量模式音频目录，默认使用 D:\\SpeechRecognition\\wav",
    )
    parser.add_argument("--start-time", type=float, default=None, help="起始时间(秒)")
    parser.add_argument("--end-time", type=float, default=None, help="结束时间(秒)")
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="跳过 librosa 参考 MFCC 比较",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="批量模式下为每个文件都生成完整图表（默认只为排序后的第一个文件生成）",
    )
    return parser.parse_args()


def discover_wavs(audio_dir: Path) -> list[Path]:
    return sorted(audio_dir.glob("*.wav"))


def save_arrays(result: PipelineResult, cfg: MFCCConfig) -> Path:
    out = cfg.features_dir / f"{result.audio_path.stem}_arrays.npz"
    np.savez_compressed(out, **result.arrays)
    return out


def save_final_csv(result: PipelineResult, cfg: MFCCConfig) -> Path:
    out = cfg.features_dir / f"{result.audio_path.stem}_feature39_cmvn.csv"
    np.savetxt(out, result.arrays["feature_39_after_cmvn"], delimiter=",")
    return out


def save_json(data: dict, path: Path) -> Path:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def export_figures(result: PipelineResult, cfg: MFCCConfig, fig: Path) -> None:
    """生成 01~15 号中间结果图到 fig 目录。"""
    arrays = result.arrays
    fig.mkdir(parents=True, exist_ok=True)

    plot_waveform(arrays["raw_signal"], cfg.sample_rate, fig / "01_waveform.png", "Raw Waveform")
    plot_two_waveforms(
        arrays["raw_signal"],
        arrays["pre_emphasis_signal"],
        cfg.sample_rate,
        fig / "02_preemphasis.png",
        "Pre-emphasis Comparison",
        "Raw",
        "Pre-emphasis",
    )
    fft_freq_hz = np.linspace(0.0, cfg.sample_rate / 2.0, cfg.n_fft // 2 + 1)
    mel_center_hz = arrays["mel_hz_points"][1:-1]

    plot_first_frame_compare(arrays["frames"][0], arrays["windowed_frames"][0], cfg.sample_rate, fig / "03_framing_window.png")
    plot_hamming(arrays["hamming_window"], fig / "04_hamming_window.png")
    plot_spectrogram(
        arrays["magnitude_spectrum"],
        fig / "05_stft_magnitude.png",
        "STFT Magnitude",
        "Frequency (Hz)",
        frame_step=cfg.frame_step,
        sample_rate=cfg.sample_rate,
        freq_hz=fft_freq_hz,
    )
    plot_spectrogram(
        arrays["power_spectrum"],
        fig / "06_power_spectrum.png",
        "Power Spectrum",
        "Frequency (Hz)",
        frame_step=cfg.frame_step,
        sample_rate=cfg.sample_rate,
        freq_hz=fft_freq_hz,
    )
    plot_mel_filterbank(arrays["mel_filterbank"], fft_freq_hz, fig / "07_mel_filterbank.png")
    plot_spectrogram(
        arrays["mel_energy"],
        fig / "08_mel_energy.png",
        "Mel Energy",
        "Frequency (Hz)",
        frame_step=cfg.frame_step,
        sample_rate=cfg.sample_rate,
        freq_hz=mel_center_hz,
    )
    plot_spectrogram(
        arrays["log_mel"],
        fig / "09_log_mel.png",
        "Log Mel",
        "Frequency (Hz)",
        frame_step=cfg.frame_step,
        sample_rate=cfg.sample_rate,
        freq_hz=mel_center_hz,
    )
    plot_spectrogram(
        arrays["static_mfcc"],
        fig / "10_static_mfcc.png",
        "Static MFCC(12D)",
        "Coefficient Index",
        frame_step=cfg.frame_step,
        sample_rate=cfg.sample_rate,
    )
    plot_energy_curve(arrays["log_energy"], cfg.frame_step, cfg.sample_rate, fig / "11_log_energy.png")
    plot_spectrogram(
        arrays["delta_feature"][:, :12],
        fig / "12_delta.png",
        "Delta MFCC",
        "Coefficient Index",
        frame_step=cfg.frame_step,
        sample_rate=cfg.sample_rate,
    )
    plot_spectrogram(
        arrays["delta_delta_feature"][:, :12],
        fig / "13_delta_delta.png",
        "Delta-Delta MFCC",
        "Coefficient Index",
        frame_step=cfg.frame_step,
        sample_rate=cfg.sample_rate,
    )
    plot_spectrogram(
        arrays["feature_39_before_cmvn"],
        fig / "14_features_before_cmvn.png",
        "39D Features Before CMVN",
        "Feature Dimension",
        frame_step=cfg.frame_step,
        sample_rate=cfg.sample_rate,
    )
    plot_spectrogram(
        arrays["feature_39_after_cmvn"],
        fig / "15_features_after_cmvn.png",
        "39D Features After CMVN",
        "Feature Dimension",
        frame_step=cfg.frame_step,
        sample_rate=cfg.sample_rate,
    )
    plot_three_energy_curves(
        arrays["static_feature"][:, 12:13],
        arrays["delta_feature"][:, 12:13],
        arrays["delta_delta_feature"][:, 12:13],
        cfg.frame_step,
        cfg.sample_rate,
        fig / "11b_energy_dynamic.png",
    )


def run_reference_comparison(
    result: PipelineResult, cfg: MFCCConfig, fig: Path | None
) -> dict[str, dict[str, float]]:
    """与 librosa 对照，保存数组和指标；fig 不为 None 时生成 16/17 号图。"""
    from src.reference_mfcc import compute_reference_mfcc, error_metrics, reference_config

    arrays = result.arrays
    custom_mfcc = arrays["static_mfcc"]
    ref = compute_reference_mfcc(
        signal=arrays["pre_emphasis_signal"],
        num_frames=custom_mfcc.shape[0],
        sample_rate=cfg.sample_rate,
        frame_length=cfg.frame_length,
        frame_step=cfg.frame_step,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        n_mfcc=cfg.n_mfcc,
        fmin=cfg.mel_fmin,
        fmax=cfg.mel_fmax,
    )
    ref_mfcc = ref["mfcc_librosa"]

    metrics = {
        # 主指标：自定义 MFCC vs librosa.feature.mfcc 原始输出
        "mfcc_vs_librosa_mfcc": error_metrics(custom_mfcc, ref_mfcc),
        # 诊断：librosa 输出乘 ln(10)/10，把 10*log10 标度换回自然对数标度
        "mfcc_vs_librosa_mfcc_scaled_ln": error_metrics(
            custom_mfcc, ref_mfcc * np.log(10.0) / 10.0
        ),
        # 诊断：librosa Mel 功率谱取 ln 后与自定义 Log-Mel 比较，隔离滤波器组差异
        "log_mel_vs_librosa_log_mel": error_metrics(arrays["log_mel"], ref["log_mel_librosa"]),
        "mfcc_vs_librosa_ln_dct": error_metrics(custom_mfcc, ref["mfcc_librosa_ln"]),
    }

    stem = result.audio_path.stem
    np.savez_compressed(
        cfg.comparison_dir / f"{stem}_comparison.npz",
        custom_mfcc=custom_mfcc,
        librosa_mfcc=ref_mfcc,
        mfcc_error=custom_mfcc - ref_mfcc,
        custom_log_mel=arrays["log_mel"],
        librosa_log_mel=ref["log_mel_librosa"],
        librosa_mfcc_ln=ref["mfcc_librosa_ln"],
    )
    ref_cfg = reference_config(
        cfg.sample_rate, cfg.frame_length, cfg.frame_step, cfg.n_fft,
        cfg.n_mels, cfg.n_mfcc, cfg.mel_fmin, cfg.mel_fmax,
    )
    ref_cfg["window"] = "自定义 Hamming 数组 (0.54-0.46cos(2πn/(L-1)), L=400)"
    ref_cfg["input"] = "预加重信号，左补 56 零 + 尾部补零到自定义填充长度 + 右补 56 零"
    save_json(
        {"metrics": metrics, "librosa_config": ref_cfg},
        cfg.comparison_dir / f"{stem}_metrics.json",
    )

    if fig is not None:
        plot_mfcc_comparison(
            custom_mfcc,
            ref_mfcc,
            cfg.frame_step,
            cfg.sample_rate,
            fig / "16_mfcc_comparison.png",
        )
        plot_error_heatmap(
            custom_mfcc,
            ref_mfcc,
            cfg.frame_step,
            cfg.sample_rate,
            fig / "17_mfcc_error.png",
        )
    return metrics


def write_summary(rows: list[dict[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def process_single(
    audio_path: Path, cfg: MFCCConfig, do_reference: bool, make_figures: bool
) -> dict[str, str]:
    result = run_mfcc_pipeline(audio_path, cfg)
    row = {
        "audio_file": audio_path.name,
        "status": result.status,
        "error": result.error,
        "sample_rate": str(result.metrics.get("sample_rate", "")),
        "channels": str(result.metrics.get("channels", "")),
        "duration_sec": str(result.metrics.get("duration_sec", "")),
        "num_samples": str(result.metrics.get("num_samples", "")),
        "num_frames": str(result.metrics.get("num_frames", "")),
        "mfcc_shape": str(result.arrays.get("static_mfcc", np.empty((0, 0))).shape),
        "final_feature_shape": str(result.arrays.get("feature_39_after_cmvn", np.empty((0, 0))).shape),
    }
    if result.status != "ok":
        print(f"[FAILED] {audio_path.name}: {result.error}")
        return row

    save_arrays(result, cfg)
    save_final_csv(result, cfg)
    save_json(result.metrics, cfg.features_dir / f"{audio_path.stem}_metrics.json")

    fig = cfg.figures_dir / audio_path.stem if make_figures else None
    if fig is not None:
        export_figures(result, cfg, fig)

    if do_reference:
        try:
            metrics = run_reference_comparison(result, cfg, fig)
            main_m = metrics["mfcc_vs_librosa_mfcc"]
            scaled_m = metrics["mfcc_vs_librosa_mfcc_scaled_ln"]
            row["mfcc_mae"] = f"{main_m['mae']:.6f}"
            row["mfcc_rmse"] = f"{main_m['rmse']:.6f}"
            row["mfcc_max_abs_error"] = f"{main_m['max_abs_error']:.6f}"
            row["mfcc_scaled_mae"] = f"{scaled_m['mae']:.6f}"
            row["log_mel_mae"] = f"{metrics['log_mel_vs_librosa_log_mel']['mae']:.6f}"
        except Exception as exc:  # noqa: BLE001
            row["mfcc_compare_error"] = str(exc)
            print(f"[WARN] {audio_path.name} 参考比较失败: {exc}")
    print(f"[OK] {audio_path.name} frames={row['num_frames']} mae={row.get('mfcc_mae', '-')}")
    return row


def mirror_to_outputs(fig: Path, cfg: MFCCConfig) -> None:
    """把代表文件的完整图表复制到 outputs/，满足课程要求的固定文件名。"""
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    for png in fig.glob("*.png"):
        shutil.copyfile(png, cfg.outputs_dir / png.name)


def main() -> None:
    args = parse_args()
    cfg = MFCCConfig()
    cfg.start_time = args.start_time
    cfg.end_time = args.end_time
    if args.audio_dir is not None:
        cfg.wav_dir = args.audio_dir
    ensure_project_dirs(cfg)
    do_reference = not args.no_reference

    rows: list[dict[str, str]] = []
    if args.audio is not None:
        rows.append(process_single(args.audio, cfg, do_reference, make_figures=True))
        mirror_to_outputs(cfg.figures_dir / args.audio.stem, cfg)
        summary_path = cfg.results_dir / f"single_{args.audio.stem}_summary.csv"
    else:
        wavs = discover_wavs(cfg.wav_dir)
        print(f"[INFO] 在 {cfg.wav_dir} 发现 {len(wavs)} 个 WAV 文件")
        if not wavs:
            rows.append({"audio_file": "", "status": "failed", "error": f"未发现 WAV 文件: {cfg.wav_dir}"})
        for i, wav_path in enumerate(wavs):
            make_figures = args.plot_all or i == 0
            rows.append(process_single(wav_path, cfg, do_reference, make_figures))
        if wavs:
            mirror_to_outputs(cfg.figures_dir / wavs[0].stem, cfg)
        summary_path = cfg.batch_summary_csv
    write_summary(rows, summary_path)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"[DONE] 成功 {ok}/{len(rows)}，汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
