import torch.nn as nn
from model.residual_blocks import ResModule, ResBasicBlock, ResBottleneckBlock

class UNet_Encoder(nn.Module):
    def __init__(self,
                 input_channel,
                 channel_layers,
                 nblock_layers,
                 bottleneck_channel,
                 bottleneck_nblock,
                 scale_factor_layers,
                 ksize,
                 block_func=ResBasicBlock,
                 act_func=nn.LeakyReLU,
                 norm_func=nn.BatchNorm2d):
        super().__init__()
        # 设置多层
        self.encoder = nn.ModuleList()
        self.down_sampler = nn.ModuleList()
        self.layer_channels = []
        in_c = input_channel
        # 附带下采样的编码器部分
        for idx in range(len(channel_layers)):
            self.encoder.append(
                ResModule(in_channel=in_c, out_channel=channel_layers[idx], ksize=ksize, num_block=nblock_layers[idx],
                          block_func=block_func, stride=1, act_func=act_func, bn_func=norm_func))
            # 采用池化下采样
            self.down_sampler.append(
                nn.AvgPool2d(kernel_size=scale_factor_layers[idx], stride=scale_factor_layers[idx]))
            in_c = channel_layers[idx] * block_func.expansion
            self.layer_channels.append(in_c)

        # bottleneck部分
        self.bottleneck = ResModule(in_c, out_channel=bottleneck_channel, ksize=ksize, num_block=bottleneck_nblock,
                                    block_func=block_func, stride=1, act_func=act_func, bn_func=norm_func)

        in_c = bottleneck_channel * block_func.expansion
        self.layer_channels.append(in_c)

    def forward(self, x):
        encoder_feats = []
        for idx in range(len(self.encoder)):
            x = self.encoder[idx](x)
            encoder_feats.append(x)
            x = self.down_sampler[idx](x)

        x = self.bottleneck(x)

        return x, encoder_feats
