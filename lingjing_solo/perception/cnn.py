"""可选：带 torch 时的神经 CNN 编码。无 torch 时自动走 NumpyCNN，不影响主流程。"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core import SoloConfig


class LingjingCNN:
    """感知 CNN：优先 torch；否则纯 numpy NumpyCNN（同构嵌入）。"""

    def __init__(self, cfg: SoloConfig):
        self.cfg = cfg
        self._model = None
        self._numpy_cnn = None
        self.backend = "none"

    def build(self):
        dim = int(getattr(self.cfg, "cnn_feature_dim", 128) or 128)
        colors = int(getattr(self.cfg, "num_colors", 16) or 16)
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
                nn.Linear(64, dim),
            )
            self.backend = "torch"
        except ImportError:
            self._model = None
            from ..neural.conv import NumpyCNN

            # NumpyCNN native embed is 64; PerceptionEncoder pads to cnn_feature_dim
            self._numpy_cnn = NumpyCNN(num_colors=colors, embed_dim=min(64, dim))
            self.backend = "numpy"

    def forward(self, grid) -> Optional[np.ndarray]:
        if self._model is None and self._numpy_cnn is None:
            self.build()
        if self._model is not None:
            import torch

            x = torch.from_numpy(np.asarray(grid)).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                return self._model(x).squeeze(0).numpy()
        if self._numpy_cnn is not None:
            dim = int(getattr(self.cfg, "cnn_feature_dim", 128) or 128)
            return self._numpy_cnn.encode_padded(grid, dim)
        return None
