from copy import deepcopy

import numpy as np
import math
from numba import jit
import copy
ndv = -111
seg_ratio_threshold = 0.1


# 将点云数据在纵向上进行切割，范围为[bin_center - 50 * bin_width, bin_center + 50 * bin_width]内的点
def clip_point_clouds(xy, bin_width=1, num_bins=60):
    height_min = np.floor(np.min(xy[:, 1]))
    height_max = np.ceil(np.max(xy[:, 1]))
    indices = np.arange(height_min, height_max + bin_width, bin_width)
    center = []
    photo_num_list = []
    for idx in range(indices.shape[0] - 1):
        upper = indices[idx + 1]
        lower = indices[idx]
        selected = (xy[:, 1] < upper) * (xy[:, 1] >= lower)
        photo_num_list.append(np.sum(selected))
        # 计算了在纵向上每个bin的中心点高度
        center.append((upper + lower) / 2)

    # find the largest bin
    photo_num_list = np.array(photo_num_list)
    center = np.array(center)
    # 确定密度最高的bin的索引
    max_density_bin = np.argmax(photo_num_list)
    # determine the range and update the data set，确定了该bin的中心点高度
    max_height_center = center[max_density_bin]
    # 以该中心点为基准，涵盖上下num_bins + 0.5个bin_width所包含的光子点
    valid_range = [max_height_center - num_bins * bin_width, max_height_center + num_bins * bin_width]
    # 过滤得到基础裁剪样本
    valid_indices = (xy[:, 1] >= valid_range[0]) * (xy[:, 1] <= valid_range[1])

    return valid_indices, valid_range


def map_generator(density_map, xy, res_h, res_v):
    nums_ptc = np.int32(xy.shape[0])
    for idx in range(nums_ptc):
        x = xy[idx, 0]
        y = xy[idx, 1]
        # 取整则以为这[0, res_h)内的点归属到（r_idx, c_idx）像素点的密度上
        r_idx = np.int32(x // res_h)
        c_idx = np.int32(y // res_v)
        v = density_map[r_idx, c_idx]
        density_map[r_idx, c_idx] = v + 1.0
    return density_map


# 其横向和纵向长度应该是常数的
def stack_to_2d(xy, res_h, res_v, h_range):
    # 首先计算了裁剪后的
    x_max, x_min = np.max(xy[:, 0]), np.min(xy[:, 0])
    y_max, y_min = h_range[1], h_range[0]
    _length = np.ceil((x_max - x_min) / res_h)
    _length = np.int32(_length)
    _height = np.ceil((y_max - y_min) / res_v)
    _height = np.int32(_height)
    xy[:, 1] = xy[:, 1] - h_range[0]
    density_map = np.zeros((_length, _height), dtype=np.float32)

    m = map_generator(density_map, xy, res_h, res_v)
    # 完成遍历，像素值每个box内样本点的数量（左合右开，下合上开）

    return m


# 根据x_idx和basic_scales在填充过的密度图上进行裁剪，并按照expand_ratio最大采样
def dens_map_sampler(padded_map, x_idx, basic_width, pad_value):
    left = x_idx - pad_value
    right = x_idx + basic_width + pad_value
    cropped_map = padded_map[left: right, :]

    return cropped_map


def read_data(path, atd_dim, h_dim, s2d_res_h=2.5, s2d_res_v=0.5, s2d_pad_mode='symmetric', at_seg_dist=100,
              seg_max_num=1024, seg_overlap=0.5, s2d_expand_ratios=9, random_shift_at=10):
    path = path.replace('\\', '/')
    data = np.loadtxt(path)  # loading data
    f_name = path.split('/')[-1]
    f_name = f_name.split('.')[0]
    # 为保险，先沿at_dist方向排序
    indices = np.argsort(data[:, atd_dim])
    data = data[indices, :]
    # 再做一遍纵向上的大致裁剪，中心点为密度最高的bin的中心
    clip_mask, h_range = clip_point_clouds(data[:, [atd_dim, h_dim]])
    # 计算确定高度数据的范围
    data = data[clip_mask, :]
    # 将at dist转为从0开始
    data[:, atd_dim] = data[:, atd_dim] - np.min(data[:, atd_dim])
    # 由于高度被转换为[h_range[0], h_range[1]]范围内，所以此处其起始点应该是h_range[0]，此处用作偏移
    data[:, h_dim] = data[:, h_dim] - h_range[0]
    h_scale = h_range[1] - h_range[0]
    # 提取出沿航迹（以最小值为0点）和垂直方向（以最小值为0点）的距离值
    xy = copy.deepcopy(data[:, [atd_dim, h_dim]])
    # 确定叠加的2d密度图的左右补充尺寸，此时横向上像素点的尺寸为s2d_res_h，也就是说每个片段（其长度由at_seg_dist控制）内包含
    # 了（at_seg_dist // s2d_res_h）个像素，也就是基础尺度；此处补全的宽度等于基础尺度 x (最大的扩张系数 // 2)。
    print('Starting process point clouds to get the density map.')
    # 获得密度图density_map，即dm，其x轴一个元素的长度s2d_res_h，y轴上一个元素的长度为s2d_res_v
    # 此时xy中的x已经调整至[0, ~], y则将起点调整至h_range为0点
    dm = stack_to_2d(xy=xy, res_h=s2d_res_h, res_v=s2d_res_v, h_range=h_range)
    print('The density map and padded density map are generated.')
    # 根据s2d_pad_width参数来沿横向扩充
    pad_v = int((at_seg_dist // s2d_res_h) * ((s2d_expand_ratios - 1) // 2))
    # 其后补充一个at_seg_dist // s2d_res_h是为了防止采样时最后一个片段的范围可能很小，例如仅20米，此时按照正常的at_seg_dist // s2d_res_h采样则极有可能超出边界
    padded_dm = np.pad(dm, pad_width=((pad_v, int(pad_v + (at_seg_dist // s2d_res_h))), (0, 0)), mode=s2d_pad_mode)

    # 点云样本生成
    ptc_groups = []
    ptc_mask = []
    ptc_stack2d_groups = []
    ptc_aux_info = []
    ptc_dm_coords = []
    # 双重限制，如果密度过高，则按密度裁剪；如果密度过低，则按距离裁剪
    pts_num = data.shape[0]
    _flag = True
    start_pt_idx = 0
    while _flag:
        # 起始点在更新后值的基础上加一些随机变化（当random_shift_at>0时生效）
        start_pt_idx += np.random.randint(-1 * random_shift_at, random_shift_at) if random_shift_at > 0 else 0
        # 确定start_pt_idx在[0, pts_num-1]范围内
        start_pt_idx = np.clip(start_pt_idx, 0, data.shape[0] - 1)
        # 根据预设的样本点数量计算终止点索引
        _end_pt_idx = start_pt_idx + seg_max_num
        # 如果终止点超出了边界，则将其缩回边界
        if _end_pt_idx > pts_num:
            start_pt_idx = pts_num - seg_max_num
            _end_pt_idx = pts_num
            _flag = False
        # 提取出按seg_max_num取出的点云簇，这个地方暴露了一个问题，按索引提取的片段与原始的数据共享内存地址，如果更改则可能修改原始数据，
        # 因此要注意使用deepcopy
        _select = copy.deepcopy(data[start_pt_idx: _end_pt_idx, :])
        # 提取沿航迹距离
        _at_dists = _select[:, atd_dim]
        # 以最小值为原点，计算其横跨长度
        assert _at_dists[0] == np.min(_at_dists), '排序出现了问题！！！'
        _along_track = np.max(_at_dists) - np.min(_at_dists)
        # 生成一个0、1向量，用于应对样本不足时的扩充点的标记
        valid_mask = np.ones([_select.shape[0], ])
        # 如果该样本点的沿航迹距离超过at_seg_dist，则裁剪至at_seg_dist范围内
        if _along_track > at_seg_dist:
            _dists_from_start = _at_dists - np.min(_at_dists)
            # 将超出at_seg_dist制作成bool掩码
            ndv_mask = _dists_from_start > at_seg_dist
            # 这些样本被标记为空值，相应的在_select中也需要将其更新标记为ndv
            _select[ndv_mask, :] = ndv
            # 在mask向量中将这部分赋值为0，即False
            valid_mask[ndv_mask] = 0
            # 更新终止点坐标，其移动量相当于ndv_mask中有效样本的数量
            corr_end_pt_idx = start_pt_idx + np.sum(valid_mask) - 1  # 注意此处ndv_mask是包含了起始点的，所以要减1
        else:
            # 如果没超出，那说明_select内全是有效样本点，则mask为全为1的向量
            corr_end_pt_idx = start_pt_idx + seg_max_num

        # 如果该片段内有效样本点数量大于最大样本点数量的10%，则其为有效样本，否则为无效
        if corr_end_pt_idx != start_pt_idx:
            if np.sum(valid_mask) / seg_max_num >= seg_ratio_threshold:
                # 根据此时的start_pt_idx计算其在stacked 2d density map上的坐标位置，并采样，注意，为对齐其basic scale都是一样的，这意味着在
                # 较为密集的区域会存在多余的density map部分，这是一个折中，为了避免实时带来的损耗
                # 根据沿航向数据确定其在dm上的位置
                start_pt_atd = _select[0, atd_dim]
                # 将其转换为s2d_res_h的横向分辨率
                dmap_h_idx = np.int32(start_pt_atd // s2d_res_h)
                # 考虑到pad，此处需要加一个偏移
                dmap_h_idx += pad_v
                # 提取出裁剪后的dm片段，此处对应的实际上是最大的，后续可以进一步裁剪
                cropped_d_map = dens_map_sampler(padded_map=padded_dm, x_idx=dmap_h_idx,
                                                 basic_width=int(at_seg_dist // s2d_res_h), pad_value=pad_v)

                # 转换点云坐标至固定空间范围内，即沿航迹向上坐标起点变为每个分块的起始点，即变成[0, 100]
                along_track_dist = deepcopy(_select[:, atd_dim])
                # 下标0对应的一定是起始点，因此其at_dist肯定是最小（之前data部分已经排过序了）
                along_track_dist[valid_mask == 1] = (along_track_dist[valid_mask == 1] - _select[0, atd_dim])
                density_map_coords = np.concatenate([along_track_dist[:, None], _select[:, h_dim][:, None]], axis=-1)
                density_map_coords[valid_mask == 1, 0] = density_map_coords[valid_mask == 1, 0] // s2d_res_h + pad_v # 要考虑加上pad宽度
                density_map_coords[valid_mask == 1, 1] = density_map_coords[valid_mask == 1, 1] // s2d_res_v
                # 将所属采样的片段数据收入list中
                ptc_groups.append(_select)
                ptc_mask.append(valid_mask)
                ptc_stack2d_groups.append(cropped_d_map)
                ptc_aux_info.append(f_name)
                ptc_dm_coords.append(density_map_coords)
                if cropped_d_map.shape[0] != 360:
                    dd = 0
                # np.savetxt('./crop_dm.txt', cropped_d_map, delimiter=' ')
                # 此处的seg_overlap是为了保证相邻的点云片段之间有交集，亦或是一种数据增强的方法用来尽可能多生成数据
                start_pt_idx = start_pt_idx + math.ceil((corr_end_pt_idx - start_pt_idx) * seg_overlap)
            else:
                start_pt_idx += 1
        else:  # 起始点和结束点为同一个点，说明该区域内没有其它点，则起始点索引+1
            start_pt_idx += 1

    return ptc_groups, ptc_mask, ptc_stack2d_groups, ptc_dm_coords, ptc_aux_info, h_scale


if __name__ == '__main__':
    file_path = r"D:\ATL03_DATA_V1\testing\AGDA_20200117095437_03310607_gt1r.txt"
    pt, ptl, pts2d, ptc_dm_c, p_info, h_scale = read_data(file_path, 4, -2, random_shift_at=0)
    dd = 0