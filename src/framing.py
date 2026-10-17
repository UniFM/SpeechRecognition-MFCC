"""兼容模块：分帧与加窗函数入口。"""
"""把实现与调用解耦，上层调用分帧加窗实际从 src.framing 导入函数"""
"""当前分帧加窗实际实现在 src.preprocessing 模块中"""

from src.preprocessing import apply_window, build_hamming_window, frame_signal

# __all__ : 当别人使用 from src.framing import * 导入这个模块时，
# 只允许导入 __all__ 列表里明确指定的这三个核心函数
__all__ = ["frame_signal", "build_hamming_window", "apply_window"]
