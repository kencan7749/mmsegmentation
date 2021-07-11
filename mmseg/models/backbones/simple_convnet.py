import torch.nn as nn
from .unet import BasicConvBlock

from ..builder import BACKBONES

@BACKBONES.register_module
class SimpleConvNet(nn.Module):
    def __init__(self,in_channels,
                 out_channels,
                 base_channels= 64,
                 num_convs=2,
                 strides=(1,1),
                 dilations=(1,1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None):
        super(SimpleConvNet, self).__init__()
        self.encoder = nn.ModuleList()
        if num_convs == 1:
            enc_conv_block = []
            enc_conv_block.append(
                    BasicConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        num_convs=1,
                        stride=strides[0],
                        dilation=dilations[0],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        dcn=None,
                        plugins=None))
            self.encoder.append((nn.Sequential(*enc_conv_block)))
            
        else:
            enc_conv_block = []
            for i in range(num_convs-1):
                enc_conv_block.append(
                        BasicConvBlock(
                            in_channels=in_channels,
                            out_channels=base_channels * 2**i,
                            num_convs=1,
                            stride=strides[i],
                            dilation=dilations[i],
                            with_cp=with_cp,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            dcn=None,
                            plugins=None))
                in_channels = base_channels * 2**i
            i+=1
            enc_conv_block.append(
                        BasicConvBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            num_convs=1, 
                            stride=strides[i],
                            dilation=dilations[i],
                            with_cp=with_cp,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            dcn=None,
                            plugins=None))
            
        self.encoder = nn.Sequential(*enc_conv_block)
            
    def forward(self, x):
        return self.encoder(x),

    
        