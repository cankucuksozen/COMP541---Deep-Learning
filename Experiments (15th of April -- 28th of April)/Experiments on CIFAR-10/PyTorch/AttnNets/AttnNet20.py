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
        "block_version",
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard AttnNet20 model
# ----------------------------------------------------------------------------

# AttnNet20 up to stage 3 (excludes stage 5)
AttnNet20Stages = tuple(
    StageSpec(index=i, block_version=v, block_count=c, return_features=r)
    for (i, v, c, r) in ((1, "conv", 2, False), (2, "conv", 2, False), (3, "attn", 3, True))
)


class AttnNet20(nn.Module):
    def __init__(self):
        super(AttnNet20, self).__init__()
        
        stage_specs = AttnNet20Stages #
        transformation_module = BasicBlock
        # Construct the stem module
        self.stem_out_channels = 16
        self.stem = Stem(self.stem_out_channels)

        # Constuct the specified ResNet stages
        self.in_channels = self.stem_out_channels
        stage1_out_channels = 32
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage1_relative_factor = 2 ** (stage_spec.index - 1)
            planes = stage1_out_channels * stage1_relative_factor
            module = _make_stage(
                    transformation_module,
                    stage_spec.block_version,
                    self.in_channels,
                    planes,
                    1,
                    stage_spec.block_count,
                    first_stride=int(stage_spec.index > 0) + 1,
                )

            self.in_channels = planes
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
    version,
    in_planes,
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
                version,
                in_planes,
                planes,
                expansion,
                stride
            )
        )
        stride = 1
        in_planes = planes*expansion
    return nn.Sequential(*blocks)


class BasicBlock(nn.Module):
    
    def __init__(
        self,
        version,
        in_planes,
        planes,
        expansion,
        stride = 1
    ):
        super(BasicBlock, self).__init__()
        
        self.version = version
        self.in_planes = in_planes
        self.out_planes = planes*expansion
        self.stride = stride
        self.padding = 1
        
        norm_func =  nn.BatchNorm2d ## Try Dropout 
        
        self.downsample = None
        if self.in_planes != self.out_planes or stride != 1:
            down_stride = stride
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, self.out_planes,
                    kernel_size=1, stride=down_stride, 
                    bias=False
                ),
                norm_func(self.out_planes),
            )
            for modules in [self.downsample]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)
        
        if self.version == "attn":
            self.layer1 = selfAttn2d(self.in_planes, kernel_size = 3, stride = self.stride,
                                     padding = self.padding, Nh = 4, dk = self.out_planes, 
                                     dv = self.out_planes)
            
        elif self.version == "conv":
            self.layer1 = nn.Conv2d(self.in_planes,
                                    self.out_planes,
                                    kernel_size = 3,
                                    stride = self.stride,
                                    padding = self.padding,
                                    bias=False)
            nn.init.kaiming_uniform_(self.layer1.weight, a=1)
    
        self.bn1 = norm_func(self.out_planes)
    
        if self.version == "attn":
            self.layer2 = selfAttn2d(self.out_planes, kernel_size = 3, stride = 1, 
                                     padding = self.padding, Nh = 4, dk = self.out_planes, 
                                     dv = self.out_planes)
            
        elif self.version == "conv":
            self.layer2 = nn.Conv2d(self.out_planes,
                                    self.out_planes,
                                    kernel_size = 3,
                                    stride = 1,
                                    padding= 1,
                                    bias=False)
            nn.init.kaiming_uniform_(self.layer2.weight, a=1)
                
        self.bn2 = norm_func(self.out_planes)


    def forward(self, x):
        identity = x  
        
        out = self.layer1(x)
        out = self.bn1(out)
        
        #if self.version == "conv":
        out = F.relu(out)

        out = self.layer2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        
        #if self.version == "conv":
        out = F.relu(out)

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
        x = F.relu(x)
        return x

