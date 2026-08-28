"""可选：带 torch 时的神经 CNN 编码。无 torch 时自动跳过，不影响主流程。"""
from ..core import SoloConfig


class LingjingCNN:
    """4 层卷积 + 全局池化的轻量编码器（参考 StochasticGoose 思路）。

    评测期若无 torch，encoder.py 的 fallback 特征会自动顶上，
    架构层无需任何改动 —— 这是「可插拔编码」的设计意图。
    """
    def __init__(self, cfg: SoloConfig):
        self.cfg = cfg
        self._model = None

    def build(self):
        try:
            import torch.nn as nn
            self._model = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, self.cfg.cnn_feature_dim),
            )
        except ImportError:
            self._model = None

    def forward(self, grid):
        if self._model is None:
            return None
        import torch
        x = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            return self._model(x).squeeze(0).numpy()
