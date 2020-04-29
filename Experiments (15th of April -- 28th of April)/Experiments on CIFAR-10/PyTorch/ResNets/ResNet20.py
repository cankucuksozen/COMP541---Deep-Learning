from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from layers.batch_norm import FrozenBatchNorm2d
from layers.self_attention_2d_strided import selfAttn2d

# ResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet20 model
# ----------------------------------------------------------------------------

# ResNet-20 up to stage 3 (excludes stage 5)
ResNet20StagesTo3 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 2, False), (2, 2, False), (3, 2, True))
)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        stage_specs = ResNet20StagesTo3 # "R-50-C4"
        transformation_module = BasicBlock
        # Construct the stem module
        stem_out_channels = 16
        self.stem = Stem(stem_out_channels)

        # Constuct the specified ResNet stages
        num_groups = 1
        width_per_group = 32
        in_channels = stem_out_channels
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = 32
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            planes = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            module = _make_stage(
                    transformation_module,
                    in_channels,
                    planes,
                    1,
                    stage_spec.block_count,
                    first_stride=int(stage_spec.index > 0) + 1,
                )

            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features
        
        # Optionally freeze (requires_grad=False) parts of the backbone
        #self._freeze_backbone(2)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs



def _make_stage(
    transformation_module,
    in_channels,
    planes,
    expansion,
    block_count,
    first_stride,
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                planes,
                expansion,
                stride
            )
        )
        stride = 1
        in_channels = planes*expansion
    return nn.Sequential(*blocks)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        planes,
        expansion,
        stride
    ):
        super(BasicBlock, self).__init__()
                
        assert (stride == 1 or stride == 2), "Stride must be either of the following 1 , 2"
        
        self.out_channels = planes * expansion
        self.stride = stride
        self.padding = 1
        
        #print(self.stride)
        
        norm_func =  FrozenBatchNorm2d
        
        self.downsample = None
        if in_channels != self.out_channels:
            down_stride = stride
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, self.out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(self.out_channels),
            )
            for modules in [self.downsample]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)


        self.conv1 = nn.Conv2d(
            in_channels,
            self.out_channels,
            kernel_size = 3,
            stride = self.stride,
            padding = self.padding,
            bias=False,
        )
        self.bn1 = norm_func(self.out_channels)
    
        self.conv2 = nn.Conv2d(
                 self.out_channels,
                 self.out_channels,
                kernel_size = 3,
                stride = 1,
                padding= 1,
                bias=False
            )
        self.bn2 = norm_func(self.out_channels)


        for l in [self.conv1, self.conv2,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out

class Stem(nn.Module):
    def __init__(self, out_channels):
        super(Stem, self).__init__()

        self.conv1 = nn.Conv2d(
            3, out_channels, kernel_size=3, stride=1, padding=1, bias=False #imagenet stride = 2 
        )
        self.bn1 = FrozenBatchNorm2d(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        return x

