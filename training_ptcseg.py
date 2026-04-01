import os.path
import shutil
import numpy as np
from utils.metric import get_confusion_matrix, accuracy, miou, accuracy_of_file, get_confusion_matrix_wo_mask, print_acc
import torch.optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from utils.loss_func import masked_nll_loss
from model.model_v1 import fuse_model_v1
from dataloader.Dataloader_RTS_V2 import RTS_Dataloader_V2
import torch.nn as nn
from model.transformer_blocks import TransformerEncoder_BlockO, TransformerEncoder_BlockV1
from utils.logger import logger
import argparse
from model.residual_blocks import ResBasicBlock, ResBottleneckBlock


def dataset_update_and_build(dataloader, batch_size, is_training):
    dataloader.update_data()
    return torch.utils.data.DataLoader(dataloader, batch_size=batch_size, shuffle=is_training, num_workers=4,
                                       pin_memory=True)

def raw_p_gt(out, f_names, raw, mask, pred, gt):
    #
    for b in range(len(f_names)):
        f = f_names[b]
        m = mask[b, :]
        m = m == 1
        _raw = raw[b, m, :]
        _p = pred[b, m]
        _gt = gt[b, m]
        if _p.ndim == 1:
            _p = _p[:, None]
        if _gt.ndim == 1:
            _gt = _gt[:, None]

        # cat
        out_segment = np.concatenate([_raw, _p, _gt], axis=-1)

        if f not in out:
            out[f] = []
            out[f].append(out_segment)
        else:
            out[f].append(out_segment)

    return out


def args():
    paras = argparse.ArgumentParser('Fuse_Model_V1')
    # model architecture
    paras.add_argument('--s2d_input_channel', type=int, default=1)
    paras.add_argument('--channel_layers', type=list, default=[32, 64, 128], help='The channel of UNet encoder part.')
    paras.add_argument('--nblock_layers', type=list, default=[3, 3, 3], help='The number of blocks contained in the encoder.')
    paras.add_argument('--d_channel_layers', type=list, default= [32, 64, 128], help='The channel of UNet decoder part.')
    paras.add_argument('--d_nblock_layers', type=list, default=[3, 3, 3], help='The number of blocks contained in the decoder.')
    paras.add_argument('--bottleneck_channel', type=int, default=128, help='The channel of bottleneck part of the UNet.')
    paras.add_argument('--bottleneck_nblock', type=int, default=4, help='The number of blocks contained in the bottleneck of UNet.')
    paras.add_argument('--scale_factor_layers', type=int, default=[2, 3, 2], help='The scale factors for UNet stages.')
    paras.add_argument('--ksize', type=int, default=3, help='The kernel size of residual block in the UNet.')
    paras.add_argument('--block_func', type=str, default=ResBasicBlock, help='The type of block function.')
    paras.add_argument('--s2dmap_act_func', type=nn.Module, default=nn.LeakyReLU, help='The activate function used in the UNet.')
    paras.add_argument('--s2dmap_norm_func', type=nn.Module, default=nn.BatchNorm2d, help='The norm func used in the UNet.')
    paras.add_argument('--ptc_block_nums', type=list, default=[3, 3, 3], help='The number of transformer blocks in ptc branch.')
    paras.add_argument('--ptc_block_sa', type=nn.Module, default=TransformerEncoder_BlockV1, help='The type pf transformer block in ptc branch.')
    paras.add_argument('--ptc_num_head', type=int, default=4, help='The num heads used by transformer block in ptc branch.')
    paras.add_argument('--ptc_input_channel', type=int, default=4, help='The channel of input of ptc branch.')  # 3 + 1
    paras.add_argument('--ptc_ffn_hidden_channel', type=int, default=64, help='The hidden channel of ptc branch.')
    paras.add_argument('--ptc_attn_dropout_rate', type=float, default=0.05, help='The dropout rate for attention module.')
    paras.add_argument('--ptc_proj_dropout_rate', type=float, default=0.05, help='The dropout rate for projection module.')

    # s2d function and data preparation params
    paras.add_argument('--num_class', type=int, default=3)
    paras.add_argument('--land_reassign', type=str, default='to_surface')
    # dataloader part
    paras.add_argument('--at_dist_dim', type=int, default=4, help='The indices of along track distance in raw point cloud.')
    paras.add_argument('--h_dim', type=int, default=-2, help='The indices of height in raw point cloud.')
    paras.add_argument('--gt_dim', type=int, default=-1, help='The indices of gt label in raw point cloud.')
    paras.add_argument('--f_dim', type=list, default=[5], help='The indices of feature in raw point cloud.')
    paras.add_argument('--s2d_h_res', type=float, default=2.5, help='The resolution in horizontal direction (along track distance direction).')
    paras.add_argument('--s2d_v_res', type=float, default=0.25, help='The resolution in vertical direction (height direction).')
    paras.add_argument('--s2d_pad_mode', type=str, default='symmetric', help='The padding mode used in extent horizontal direction.')
    paras.add_argument('--seg_len', type=float, default=100, help='The length of segment in raw point cloud along the along-track distance direction.')
    paras.add_argument('--seg_sample_num_max', type=int, default=1024, help='The maximum sample contained in each segment.')
    paras.add_argument('--seg_overlap', type=float, default=0.5, help='The overlap ratio in sampling point cloud.')
    paras.add_argument('--s2d_expand_ratio', type=int, default=9, help='The expansion ratio used in s2d mapping.')
    paras.add_argument('--random_shift_at', type=int, default=64, help='The random shift at along track distance direction.')
    paras.add_argument('--batch_size', type=int, default=24, help='Batch size.')
    paras.add_argument('--utm_x_dim', type=int, default=2)
    paras.add_argument('--utm_y_dim', type=int, default=3)

    # training setting
    paras.add_argument('--optim', type=str, default='AdamW')
    paras.add_argument('--loss_weight', type=list, default=[1., 1., 1.])
    paras.add_argument('--learning_rate', type=float, default=1e-3)

    # 学习率调整器参数
    paras.add_argument('--lr_active_epochs', type=int, default=13)
    paras.add_argument('--cycle_mul', type=float, default=1.0)
    paras.add_argument('--lr_min', type=float, default=1e-6)
    paras.add_argument('--warmup_lr', type=float, default=1e-3)
    paras.add_argument('--warmup_epoch', type=int, default=0)
    paras.add_argument('--cycle_limit', type=int, default=1)
    paras.add_argument('--cycle_decay', type=float, default=1.0)
    paras.add_argument('--epoch', type=int, default=15)

    return paras.parse_args()


def train(args, log):
    # build the model
    model = fuse_model_v1(s2d_input_channel=args.s2d_input_channel,
                          channel_layers=args.channel_layers,
                          nblock_layers=args.nblock_layers,
                          d_channel_layers=args.d_channel_layers,
                          d_nblock_layers=args.d_nblock_layers,
                          bottleneck_nblock=args.bottleneck_nblock,
                          bottleneck_channel=args.bottleneck_channel,
                          scale_factor_layers=args.scale_factor_layers,
                          ksize=args.ksize,
                          block_func=args.block_func,
                          s2dmap_act_func=args.s2dmap_act_func,
                          s2dmap_norm_func=args.s2dmap_norm_func,
                          ptc_block_nums=args.ptc_block_nums,
                          ptc_block_sa=args.ptc_block_sa,
                          ptc_num_head=args.ptc_num_head,
                          ptc_input_channel=args.ptc_input_channel,
                          ptc_ffn_hidden_channel=args.ptc_ffn_hidden_channel,
                          ptc_attn_dropout_rate=args.ptc_attn_dropout_rate,
                          ptc_proj_dropout_rate=args.ptc_proj_dropout_rate,
                          num_class=args.num_class)

    # build the datasets
    log.log_string('Loading Training Datasets......')
    training_dataloader = RTS_Dataloader_V2(data_dict=r'/rd5m/QXM/PTC_SEG/datasets_v1/training',
                                            at_dist_dim=args.at_dist_dim,
                                            h_dim=args.h_dim,
                                            label_dim=args.gt_dim, 
                                            utm_dims=[2, 3, args.h_dim],
                                            feature_dims=args.f_dim,
                                            s2d_h_res=args.s2d_h_res,
                                            s2d_v_res=args.s2d_v_res,
                                            s2d_pad_mode=args.s2d_pad_mode,
                                            seg_len=args.seg_len,
                                            seg_sample_num_max=args.seg_sample_num_max,
                                            mini_valid_ratio=0.1,
                                            s2d_expand_ratios=args.s2d_expand_ratio,
                                            random_shift_range=args.random_shift_at,
                                            num_class=args.num_class,
                                            land_reassign=args.land_reassign,
                                            is_training=True)

    log.log_string('Loading Testing Datasets......')
    testing_dataloader = RTS_Dataloader_V2(data_dict=r'/rd5m/QXM/PTC_SEG/datasets_v1/testing',
                                           at_dist_dim=args.at_dist_dim,
                                           h_dim=args.h_dim,
                                           label_dim=args.gt_dim,
                                           utm_dims=[2, 3, args.h_dim],
                                           feature_dims=args.f_dim,
                                           s2d_h_res=args.s2d_h_res,
                                           s2d_v_res=args.s2d_v_res,
                                           s2d_pad_mode=args.s2d_pad_mode,
                                           seg_len=args.seg_len,
                                           seg_sample_num_max=args.seg_sample_num_max,
                                           mini_valid_ratio=0.1,
                                           s2d_expand_ratios=args.s2d_expand_ratio,
                                           random_shift_range=0,
                                           num_class=args.num_class,
                                           land_reassign=args.land_reassign,
                                           is_training=False)
    # training_dataset = dataset_update_and_build(training_dataloader, batch_size=args.batch_size, is_training=True)
    testing_dataset = dataset_update_and_build(testing_dataloader, batch_size=args.batch_size, is_training=False)
    model = model.cuda()
    loss_func = masked_nll_loss(weight=torch.Tensor(args.loss_weight).cuda(), num_class=args.num_class)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,  weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999))
    schedular = CosineLRScheduler(optim, t_initial=args.lr_active_epochs, cycle_mul=args.cycle_mul,
                                  lr_min=args.lr_min, warmup_lr_init=args.warmup_lr, warmup_t=args.warmup_epoch,
                                  cycle_limit=args.cycle_limit, cycle_decay=args.cycle_decay, t_in_epochs=True)
    for idx in range(args.epoch):
        # create a empty n_class x n_class matrix
        cf_matrix = np.zeros([args.num_class, args.num_class])
        acc_stat = np.zeros([args.num_class, 2])
        loss_v = 0
        # 更新训练集
        training_dataset = dataset_update_and_build(training_dataloader, batch_size=args.batch_size, is_training=True)
        for iters, (norm_xy, utm_xyz, feat, mask, gt, s2dmap, dm_coord, f_name, raw) in enumerate(training_dataset):
            norm_xy = norm_xy.cuda()
            utm_xyz = utm_xyz.cuda()
            f = feat.cuda()
            mask = mask.cuda()
            gt = gt.cuda()
            s2dmap = s2dmap.cuda()
            dm_coord = dm_coord.cuda()
            _input = torch.cat([utm_xyz, f], dim=-1)
            # _input = utm_xyz
            optim.zero_grad()
            pred = model(_input, norm_xy, s2dmap, mask, dm_coord)
            loss = loss_func(pred, gt, mask)
            loss.backward()
            optim.step()

            # get the output and calculate the metrics
            p = pred.clone().detach().cpu().numpy()
            g = gt.clone().detach().cpu().numpy()
            m = mask.clone().detach().cpu().numpy()
            _p = p.argmax(axis=-1)
            conf_matrix = get_confusion_matrix(_p, g, m, n_class=args.num_class)
            cf_matrix += conf_matrix
            mean_iou = miou(conf_matrix)
            _acc = accuracy(_p, g, m, n_class=args.num_class)
            print_str = 'TRAINING - EPOCH->[{:d} / {:d}] ITER->[{:d} / {:d}]: Loss is {:.6f}; MIOU is {:.6f}; '.format(
                idx + 1, args.epoch, iters + 1, len(training_dataset), float(loss.item()), mean_iou)
            acc_str = print_acc(_acc, args.num_class)
            acc_stat += _acc
            loss_v += float(loss.item())
            #
            log_output = print_str + acc_str
            log.log_string(log_output)

        schedular.step(idx + 1)
        # statistic analyze on this training epoch.
        F_MIOU = miou(cf_matrix)
        log.log_string('######################## STATISTIC ANALYZE ON EPOCH %d (%d) #########################' % (idx + 1, args.epoch))
        print_str = '----> Mean Loss is {:.6f}; MIOU on this epoch is {:.6f}; '.format(loss_v / len(training_dataset), F_MIOU)
        print_str += print_acc(acc_stat, args.num_class)
        log.log_string(print_str)

    # performance on testing dataset
    with torch.no_grad():
        model.eval()
        out = {}
        cf_matrix = np.zeros([args.num_class, args.num_class])
        acc_stat = np.zeros([args.num_class, 2])
        for t_iters, (norm_xy, utm_xyz, feat, mask, gt, s2dmap, dm_coord, f_name, raw) in enumerate(testing_dataset):
            norm_xy = norm_xy.cuda()
            utm_xyz = utm_xyz.cuda()
            f = feat.cuda()
            mask = mask.cuda()
            gt = gt.cuda()
            s2dmap = s2dmap.cuda()
            dm_coord = dm_coord.cuda()
            _input = torch.cat([utm_xyz, f], dim=-1)
            # _input = utm_xyz
            pred = model(_input, norm_xy, s2dmap, mask, dm_coord)
            # calculating the metrics
            p = pred.clone().detach().cpu().numpy()
            g = gt.clone().detach().cpu().numpy()
            m = mask.clone().detach().cpu().numpy()
            r = raw.numpy()
            _p = p.argmax(axis=-1)
            out = raw_p_gt(out, f_name, r, m, _p, g)
            conf_matrix = get_confusion_matrix(_p, g, m, n_class=args.num_class)
            cf_matrix += conf_matrix
            mean_iou = miou(conf_matrix)
            _acc = accuracy(_p, g, m, n_class=args.num_class)
            acc_stat += _acc
            print_str = 'TESTING - ITER->[{:d} / {:d}]: MIOU is {:.6f}; '.format(t_iters + 1, len(testing_dataset), mean_iou)
            print_str += print_acc(_acc, args.num_class)
            log.log_string(print_str)

        # get the whole accuracy on testing dataset
        F_MIOU = miou(cf_matrix)
        log.log_string('######################## TESTING STATISTIC ANALYZE ON EPOCH %d #########################' % (idx + 1))
        print_str = '---> [TESTING] MIOU on epoch is {:.6f}; '.format(F_MIOU)
        print_str += print_acc(acc_stat, args.num_class)
        log.log_string(print_str)
    # output the labeled raw point clouds
    for key in out:
        k = key + '.txt'
        filepath = os.path.join(str(log.tif_dict), k)
        rpg = np.concatenate(out[key], axis=0)
        rpg = rpg[:, [0, 1, 2, 3, 4, 5, 6, 7, -2, -1]]
        pred = rpg[:, -2]
        gt = rpg[:, -1]
        pred = pred.astype(np.int64)
        gt = gt.astype(np.int64)
        acc_stat = accuracy_of_file(pred, gt, n_class=args.num_class)
        np.savetxt(filepath, rpg, delimiter=' ')
        coffe_matrix = get_confusion_matrix_wo_mask(pred, gt, n_class=args.num_class)
        mean_iou = miou(coffe_matrix)
        print_str = '[FINAL ANALYSIS] --> '+ key + '\n --> MIOU is {:.6f}; '.format(mean_iou)
        print_str += print_acc(acc_stat, args.num_class)
        log.log_string(print_str)

    # 保存模型权重
    torch.save(model.state_dict(), os.path.join(str(log.checkpoint_dict), 'ckpt.pt'))
    # 保存训练文件
    shutil.copy('training_rtsv2.py', str(log.log_dir))

    return

if __name__ == '__main__':
    args = args()
    log = logger('DECOMPOSE_3dutm+1df_FinalVersion')
    train(args, log)
