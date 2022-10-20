from torch import nn


class EmotionsNetwork2LinLayers(nn.Module):
    def __init__(self):
        super(EmotionsNetwork2LinLayers, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(369 * 496 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Linear(100, 6)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
