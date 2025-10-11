import numpy as np
import torch
import torch.nn as nn

from ss_baselines.common.utils import Flatten
from sen_baselines.enmus.models.seldnet import MultiHeadAttentionLayer
from sen_baselines.enmus.models.visual_cnn import conv_output_dim, layer_init

class AudioCRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._number_input_audio = 4
        cnn_dimensions = np.array(
            (65,126), dtype=np.int32
        )

        if cnn_dimensions[0] < 30 or cnn_dimensions[1] < 30:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dimensions = conv_output_dim(
                dimension=cnn_dimensions,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )
            
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._number_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            Flatten(),
            nn.Linear(64 * cnn_dimensions[0] * cnn_dimensions[1], 128),
            nn.ReLU(True),
        )

        self.attn = MultiHeadAttentionLayer(
            hidden_size=128,
            n_heads=16,
            dropout=0.05,
        )
        self.fnn = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
        layer_init(self.cnn)
        layer_init(self.fnn)

    def forward(self, audio):
        audio = audio.permute(0, 3, 1, 2)

        x = self.cnn(audio)
        x = self.attn.forward(x, x, x)

        x = x.squeeze(1)
        x = self.fnn(x)
        x = x*180
        return x