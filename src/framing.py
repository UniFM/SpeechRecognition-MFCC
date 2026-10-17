"""兼容模块：分帧与加窗函数入口。"""

from src.preprocessing import apply_window, build_hamming_window, frame_signal

__all__ = ["frame_signal", "build_hamming_window", "apply_window"]
