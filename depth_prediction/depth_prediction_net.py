import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthPred(nn.Module):
    def __init__(self, D=8, W=256, input_ch=6, output_ch=1):
        super(DepthPred, self).__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch

        self.linear_layers = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [nn.Linear(W, W) for _ in range(D - 1)]
        )

        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        h = x

        for layer in self.linear_layers:
            h = layer(h)
            h = F.relu(h)

        output = self.output_linear(h)

        return output


def create_depth_pred(lr=0.0005):
    model = DepthPred()
    grad_vars = list(model.parameters())

    optimiser = torch.optim.Adam(params=grad_vars, lr=lr, betas=(0.9, 0.999))

    return model, optimiser
