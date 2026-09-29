"""
Lightweight CNN for CIFAR-10, split into a client-side and a server-side
part at a fixed "cut layer". This mimics a split-inference deployment:
the client-side part runs identically on every replica, the server-side
part runs once on the (reconstructed) activation received from the
replicas.
"""
import torch
import torch.nn as nn


class SplitCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ---- Client-side network (identical on every replica) ----
        self.client_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        # >>> CUT LAYER: activation `a` has shape [B, 64, 8, 8] <<<

        # ---- Server-side network (runs on the aggregated activation) ----
        self.server_layers = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward_client(self, x: torch.Tensor) -> torch.Tensor:
        """Runs on each client replica. Returns the cut-layer activation a."""
        return self.client_layers(x)

    def forward_server(self, a: torch.Tensor) -> torch.Tensor:
        """Runs once on the server, on the (reconstructed) activation."""
        h = self.server_layers(a)
        h = h.flatten(1)
        return self.classifier(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_server(self.forward_client(x))
