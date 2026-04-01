from model.UNet import UNet_Encoder
from model.residual_blocks import ResBasicBlock, ResBottleneckBlock, ResModule
from model.transformer_blocks import TransformerEncoder_BlockO, TransformerEncoder_BlockV1, TFEncoder
import torch.nn as nn
import torch
import torch.nn.functional as F


# 格网化的采样
def s2dmap_features_extraction(s2dmap_features, coords, mask, scale_factor):
    _channel = s2dmap_features.shape[1]
    output_feat = torch.ones([coords.shape[0], coords.shape[1], _channel]) * -111
    device = s2dmap_features.device
    output_feat = output_feat.to(device)
    for b in range(s2dmap_features.shape[0]):
        pos = coords[b, :, :]
        valid_pos = pos[mask[b, :] == 1, :]
        _x = valid_pos[:, 0] // scale_factor
        _y = valid_pos[:, 1] // scale_factor
        output_feat[b, mask[b, :] == 1, :] = s2dmap_features[b,:, _x, _y].permute(1, 0)
    # r = output_feat.clone().detach().cpu().numpy()
    # c = coords.clone().detach().cpu().numpy()
    return output_feat


def map_coordinates_bi(feat_map, coords, x_range, y_range):
    a = feat_map.clone().detach().cpu().numpy()
    # 四个顶点坐标
    x_l = torch.floor(coords[:, 0]).to(torch.int32)
    x_h = torch.ceil(coords[:, 0]).to(torch.int32)
    # 判断是否有相同的，如果x_h == x_l，那就意味着其为整数
    _integer = x_h == x_l
    x_h[_integer] = x_h[_integer] + 1
    y_l = torch.floor(coords[:, 1]).to(torch.int32)
    y_h = torch.ceil(coords[:, 1]).to(torch.int32)
    # 判断是否有相同
    _integer = y_h == y_l
    y_h[_integer] = y_h[_integer] + 1
    # clip function
    x_l = x_l.clip(0, x_range - 2)
    x_h = x_h.clip(1, x_range - 1)
    y_l = y_l.clip(0, y_range - 2)
    y_h = y_h.clip(1, y_range - 1)
    #
    f_xlyl = feat_map[:, x_l, y_l].to(torch.float64)
    f_xlyh = feat_map[:, x_l, y_h].to(torch.float64)
    f_xhyl = feat_map[:, x_h, y_l].to(torch.float64)
    f_xhyh = feat_map[:, x_h, y_h].to(torch.float64)
    delta_hx = x_h.to(torch.float64) - coords[:, 0].to(torch.float64)
    delta_xl = coords[:, 0].to(torch.float64) - x_l.to(torch.float64)
    delta_hy = y_h.to(torch.float64) - coords[:, 1].to(torch.float64)
    delta_yl = coords[:, 1].to(torch.float64) - y_l.to(torch.float64)
    output = delta_yl * (delta_hx * f_xlyh + delta_xl * f_xhyh) + delta_hy * (delta_hx * f_xlyl + delta_xl * f_xhyl)
    output = output.to(torch.float32)
    b = feat_map.clone().detach().cpu().numpy()

    return output


def s2dmap_features_extraction_bib(s2dmap_features, coords, mask, scale_factor):
    b = s2dmap_features.shape[0]
    _channel = s2dmap_features.shape[1]
    output_feat = torch.zeros([b, coords.shape[1], _channel])
    device = s2dmap_features.device
    output_feat = output_feat.to(device)
    height = s2dmap_features.shape[2]
    width = s2dmap_features.shape[3]
    for b_idx in range(b):
        # 变为[1, 1, num_samples, 2]
        pos = coords[b_idx, mask[b_idx, :] == 1, :] / scale_factor
        _v = map_coordinates_bi(s2dmap_features[b_idx, :, :, :], pos, height, width)
        output_feat[b_idx, mask[b_idx, :] == 1, :] = _v.squeeze().permute(1, 0)

    return output_feat


class fuse_model_v1(nn.Module):
    def __init__(self,
                 s2d_input_channel=1,
                 channel_layers=None,
                 nblock_layers=None,
                 d_channel_layers=None,
                 d_nblock_layers=None,
                 bottleneck_channel=128,
                 bottleneck_nblock=6,
                 scale_factor_layers=None,
                 ksize=3,
                 block_func=ResBasicBlock,
                 s2dmap_act_func=nn.LeakyReLU,
                 s2dmap_norm_func=nn.BatchNorm2d,
                 ptc_block_nums=None,
                 ptc_block_sa=TransformerEncoder_BlockO,
                 ptc_num_head=4,
                 ptc_input_channel=3,  # x y and c
                 ptc_ffn_hidden_channel=16,
                 ptc_attn_dropout_rate=0.1,
                 ptc_proj_dropout_rate=0.1,
                 ptc_ffn_dropout_rate=0.1,
                 num_class=4
                 ):
        super().__init__()
        if ptc_block_nums is None:
            ptc_block_nums = [3, 3, 4]
        if d_nblock_layers is None:
            d_nblock_layers = [4, 4, 5]
        if d_channel_layers is None:
            d_channel_layers = [32, 64, 64]
        if scale_factor_layers is None:
            scale_factor_layers = [2, 3, 2]
        if nblock_layers is None:
            nblock_layers = [4, 4, 5]
        if channel_layers is None:
            channel_layers = [32, 64, 64]
        # 编码器部分
        self.s2dmap_encoder = UNet_Encoder(input_channel=s2d_input_channel, channel_layers=channel_layers,
                                           nblock_layers=nblock_layers, bottleneck_nblock=bottleneck_nblock,
                                           bottleneck_channel=bottleneck_channel, ksize=ksize,
                                           scale_factor_layers=scale_factor_layers, block_func=block_func,
                                           act_func=s2dmap_act_func, norm_func=s2dmap_norm_func)

        encoder_channels = self.s2dmap_encoder.layer_channels
        # 构建与图像分支编码器匹配的transformer主干
        self.ptc_transformer = nn.ModuleList()
        ptc_trans_channel = 32
        self.branches_channel_aligner = nn.ModuleList()
        self.s2dmap_decoder = nn.ModuleList()
        self.up_sampler = nn.ModuleList()
        en_c = encoder_channels[-1]
        assert len(channel_layers) == len(self.s2dmap_encoder.layer_channels) - 1, ('The number of encoder levels and '
                                                                                    'decoder levels should match.')
        self.rescaled_factors = []
        self.pos_dim = 2
        self.pos_embed = nn.Sequential(
            nn.Linear(self.pos_dim, self.pos_dim),
            nn.LayerNorm(self.pos_dim),
            nn.GELU(),
            nn.Linear(self.pos_dim, ptc_trans_channel)
        )
        ptc_n_count = 0
        for idx in range(len(channel_layers) - 1, -1, -1):
            higher_level_c = encoder_channels[idx]
            self.up_sampler.append(
                nn.Upsample(scale_factor=scale_factor_layers[idx], align_corners=True, mode='bilinear'))

            self.s2dmap_decoder.append(
                ResModule(in_channel=en_c + higher_level_c, out_channel=d_channel_layers[idx], ksize=ksize,
                          num_block=d_nblock_layers[idx], block_func=block_func, stride=1, act_func=s2dmap_act_func,
                          bn_func=s2dmap_norm_func))
            en_c = d_channel_layers[idx]

            v = 1
            for s_idx in range(idx):
                v = v * scale_factor_layers[s_idx]
            self.rescaled_factors.append(v)

            self.branches_channel_aligner.append(
                nn.Sequential(
                    nn.Linear(ptc_input_channel, ptc_trans_channel, bias=False),
                    nn.LayerNorm(ptc_trans_channel),
                    nn.GELU()))

            self.ptc_transformer.append(
                TFEncoder(sa_block_num=ptc_block_nums[ptc_n_count],
                 input_dim=ptc_trans_channel + d_channel_layers[idx],
                 num_head=ptc_num_head,
                 ffn_hidden_dim=ptc_ffn_hidden_channel,
                 sa_block=ptc_block_sa,
                 attn_drops=ptc_attn_dropout_rate,
                 proj_drops=ptc_proj_dropout_rate,
                 ffn_drops=ptc_ffn_dropout_rate)
            )
            ptc_n_count += 1
            ptc_input_channel = ptc_trans_channel + d_channel_layers[idx]
        #
        self.seg_head = nn.Sequential(
            nn.Linear(ptc_input_channel, ptc_input_channel // 2),
            nn.LayerNorm(ptc_input_channel // 2),
            nn.GELU(),
            nn.Linear(ptc_input_channel // 2, num_class),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, dt_elev, density_map, mask, coords):
        # 先对二维堆叠密度图像进行处理，获取不同尺度的特征
        _feats, h_feats = self.s2dmap_encoder(density_map)
        # 对应每个上采样模块，先后获取对应的特征
        # pos = x[:, :, :self.pos_dim] # 获取前两位
        pos_embedding_v = self.pos_embed(dt_elev)
        for branch_idx in range(len(self.s2dmap_decoder)):
            _feats = self.up_sampler[branch_idx](_feats)
            hb_idx = len(h_feats) - 1 - branch_idx
            _feats = torch.cat([h_feats[hb_idx], _feats], dim=1)  # 沿channel拼接起来
            _feats = self.s2dmap_decoder[branch_idx](_feats)
            # extract s2d map features by x and y
            extract_s2dmap_f = s2dmap_features_extraction_bib(_feats, coords, mask, scale_factor=self.rescaled_factors[branch_idx])
            x = self.branches_channel_aligner[branch_idx](x) + pos_embedding_v
            x = torch.cat([x, extract_s2dmap_f], dim=-1)
            x = self.ptc_transformer[branch_idx](x, mask=mask)

        pred = self.seg_head(x)
        return pred


if __name__ == '__main__':
    a = torch.arange(0, 2 * 40 * 30)
    a = a.view(2, 40, 30).to(torch.float32)
    coords = torch.Tensor([[2, 2], [2.2, 2.3], [10.22, 15.3]])
    out = map_coordinates_bi(a, coords, 40, 30)
