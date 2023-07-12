import torch
import torch.nn as nn
import torchsummary


class PBSModel(nn.Module):
    def __init__(self):
        super(PBSModel, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.inc = self.input_conv()
        self.dense_block_1 = self.backbone.features.denseblock1
        self.transition1 = self.backbone.features.transition1
        self.dense_block_2 = self.backbone.features.denseblock2
        self.transition2 = self.backbone.features.transition2

        self.conv_1x1 = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=1, padding=0)
        self.linear = nn.Linear(14*14, 1)

    def input_conv(self):
        return nn.Sequential(
            self.backbone.features.conv0,
            self.backbone.features.norm0,
            self.backbone.features.relu0,
            self.backbone.features.pool0,
        )

    def forward(self, x):
        x_inc = self.inc(x)
        x_dense_1 = self.dense_block_1(x_inc)
        x_transition_1 = self.transition1(x_dense_1)
        x_dense_2 = self.dense_block_2(x_transition_1)
        x_transition_2 = self.transition2(x_dense_2)

        x_out = self.conv_1x1(x_transition_2)
        out_map = nn.functional.sigmoid(x_out)
        out = self.linear(out_map.view(-1, 14*14))
        out = nn.functional.sigmoid(out)
        out = torch.flatten(out)

        return out_map, out

