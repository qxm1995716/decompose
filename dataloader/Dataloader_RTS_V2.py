import numpy as np
import copy
import torch
from glob import glob
from dataloader.data_reader import clip_point_clouds, map_generator

# rts means real-time sampling


# 将陆地标签转换为表面标签
def convert_land_to_surface(gt):
    _valid = gt == 3
    gt[_valid] = 2
    return gt

def convert_land_to_noise(gt):
    _valid = gt == 3
    gt[_valid] = 0
    return gt

def convert_land_to_seafloor(gt):
    _valid = gt == 3
    gt[_valid] = 1
    return gt

# 其横向和纵向长度应该是常数的
def stack_to_2d_norm(xy, res_h, res_v, h_range):
    # 首先计算了裁剪后的
    x_max, x_min = np.max(xy[:, 0]), np.min(xy[:, 0])
    y_max, y_min = h_range[1], h_range[0]
    xy[:, 1] = xy[:, 1] - y_min
    _length = np.ceil((x_max - x_min) / res_h)
    _length = np.int32(_length)
    _height = np.ceil((y_max - y_min) / res_v)
    _height = np.int32(_height)
    density_map = np.zeros((_length, _height), dtype=np.float32)

    m = map_generator(density_map, xy, res_h, res_v)
    # 完成遍历，像素值每个box内样本点的数量（左合右开，下合上开）
    # 进行归一化操作
    m = m / np.sum(m)

    return m


def data_preprocess(path, atd_dim, h_dim, s2d_res_h=2.5, s2d_res_v=0.5, s2d_pad_mode='symmetric', at_seg_dist=100,
                    seg_max_num=1024, s2d_expand_ratios=9, minimum_valid_ratio=0.1, is_training=True):
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
    # 重新将data沿at_dist来进行sort
    n_indices = np.argsort(data[:, atd_dim])
    data = data[n_indices]
    # 将at dist转为从0开始
    data[:, atd_dim] = data[:, atd_dim] - np.min(data[:, atd_dim])
    # 提取出沿航迹（以最小值为0点）和垂直方向（以最小值为0点）的距离值
    xy = copy.deepcopy(data[:, [atd_dim, h_dim]])
    # 确定叠加的2d密度图的左右补充尺寸，此时横向上像素点的尺寸为s2d_res_h，也就是说每个片段（其长度由at_seg_dist控制）内包含
    # 了（at_seg_dist // s2d_res_h）个像素，也就是基础尺度；此处补全的宽度等于基础尺度 x (最大的扩张系数 // 2)。
    print('Starting process point clouds to get the density map.')
    # 获得密度图density_map，即dm，其x轴一个元素的长度s2d_res_h，y轴上一个元素的长度为s2d_res_v
    # 此时xy中的x已经调整至[0, ~], y则将起点调整至h_range为0点；进一步地，对密度图进行归一化计算
    dm = stack_to_2d_norm(xy=xy, res_h=s2d_res_h, res_v=s2d_res_v, h_range=h_range)
    print('The density map and padded density map are generated.')
    # 根据s2d_pad_width参数来沿横向扩充
    pad_v = int((at_seg_dist // s2d_res_h) * ((s2d_expand_ratios - 1) // 2))
    # 其后补充一个at_seg_dist // s2d_res_h是为了防止采样时最后一个片段的范围可能很小，例如仅20米，此时按照正常的at_seg_dist // s2d_res_h采样则极有可能超出边界
    padded_dm = np.pad(dm, pad_width=((pad_v, int(pad_v + (at_seg_dist // s2d_res_h))), (0, 0)), mode=s2d_pad_mode)

    # 确定一个初步范围，后面随机扰动是在这个预先设置好的索引上进行扰动
    _flag = True
    candidate_indices = []
    start_idx = 0
    if is_training:
        while _flag:
            valid_nums_in_segment = np.sum((data[start_idx:, atd_dim]) < (data[start_idx, atd_dim] + at_seg_dist))
            if valid_nums_in_segment >= int(seg_max_num * minimum_valid_ratio):
                candidate_indices.append(start_idx)
                start_idx += int(seg_max_num * minimum_valid_ratio)
            else:
                start_idx += 1

            if start_idx > data.shape[0] - 1:
                _flag = False # 跳出
    else:  # for testing
        while _flag:
            valid_nums_in_segment = np.sum((data[start_idx:, atd_dim]) < (data[start_idx, atd_dim] + at_seg_dist))
            if valid_nums_in_segment >= int(seg_max_num * minimum_valid_ratio):
                candidate_indices.append(start_idx)
                start_idx += valid_nums_in_segment
            else:
                start_idx += 1

            if start_idx > data.shape[0] - 1:
                _flag = False

    return padded_dm, data, f_name, h_range, candidate_indices


class RTS_Dataloader_V2(torch.utils.data.Dataset):
    def __init__(self,
                 data_dict,
                 at_dist_dim,
                 h_dim,
                 label_dim,
                 utm_dims, 
                 feature_dims=None,
                 s2d_h_res=10.,
                 s2d_v_res=0.5,
                 s2d_pad_mode='symmetric',
                 seg_len=100,
                 seg_sample_num_max=1024,
                 mini_valid_ratio=0.1,
                 s2d_expand_ratios=9,
                 random_shift_range=100,
                 num_class=4,
                 land_reassign='to_surface',
                 is_training=True):
        super().__init__()
        files = glob(data_dict + '/*.txt')
        data = {}
        # 形成一个字典，其元素也为字典，包括f_name项和
        self.pad_width = int((seg_len // s2d_h_res) * (s2d_expand_ratios // 2))
        self.ndim = None
        ind = 0
        candidate_segments = {}#  构建一个字典，保存该数据所有的索引坐标以及其所属测线名称
        for f_idx in range(len(files)):
            f = files[f_idx]
            pdm, ptc, f_name, h_range, candi_inds = \
                data_preprocess(f, at_dist_dim, h_dim, s2d_res_h=s2d_h_res, s2d_res_v=s2d_v_res,  s2d_pad_mode=s2d_pad_mode,
                                at_seg_dist=seg_len, seg_max_num=seg_sample_num_max, s2d_expand_ratios=s2d_expand_ratios,
                                minimum_valid_ratio=mini_valid_ratio, is_training=is_training)
            _dict = {'pdm': pdm, 'ptc': ptc, 'h_range': h_range}
            data[f_name] = _dict
            self.ndim = ptc.shape[-1]
            for idx in range(len(candi_inds)):
                candidate_segments[ind] = {'f_name': f_name, 'c_indices': candi_inds[idx]}
                ind += 1

        self.data = data
        self.candidates = candidate_segments
        self.seg_len = seg_len
        self.seg_sample_num_max = seg_sample_num_max
        self.num_samples = ind
        self.h_res = s2d_h_res
        self.v_res = s2d_v_res
        self.at_dim = at_dist_dim
        self.h_dim = h_dim
        self.gt_dim = label_dim
        self.utm_dims = utm_dims
        self.f_dim = feature_dims
        self.random_shift_scale = random_shift_range
        self.dm_core_len = int(self.seg_len // self.h_res)
        self.mini_valid_r = mini_valid_ratio
        if num_class != 4:
            if land_reassign == 'to_noise':
                self.land_reassign_func = convert_land_to_noise
            elif land_reassign == 'to_seafloor':
                self.land_reassign_func = convert_land_to_seafloor
            elif land_reassign == 'to_surface':
                self.land_reassign_func = convert_land_to_surface
        else:
            self.land_reassign = None

        #
        self.ptc_xy = []
        self.ptc_utms = []
        self.ptc_f = []
        self.mask = []
        self.gt = []
        self.dmap = []
        self.dm_coord = []
        self.filename = []
        self.ptc_raw = []

    def update_data(self):
        # 清除所有样本，更新样本
        if len(self.ptc_xy) != 0:
            self.ptc_xy = []
            self.ptc_f = []
            self.mask = []
            self.gt = []
            self.dmap = []
            self.dm_coord = []
            self.filename = []
            self.ptc_raw = []

        # 开始采样
        for idx in range(len(self.candidates)):
            sample = self.candidates[idx]
            f_name = sample['f_name']
            ind = sample['c_indices']
            raw_ind = ind
            if self.random_shift_scale > 0:
                ind = ind + np.random.randint(-1 * self.random_shift_scale, self.random_shift_scale, [1])
                ind = int(ind)

            # get the data
            pdm = self.data[f_name]['pdm']
            ptc = self.data[f_name]['ptc']
            h_range = self.data[f_name]['h_range']
            h_scale = h_range[1] - h_range[0]
            # get the segment
            at_dist_start = ptc[ind, self.at_dim]
            scaled_at_ind = int(at_dist_start // self.h_res + self.pad_width)
            segment_mask = ptc[ind:, self.at_dim] <= at_dist_start + self.seg_len
            included_nums = np.sum(segment_mask)
            if included_nums / self.seg_sample_num_max < self.mini_valid_r:
                ind = raw_ind
                # get the segment again based on the original index, ind = raw_ind
                at_dist_start = ptc[ind, self.at_dim]
                scaled_at_ind = int(at_dist_start // self.h_res) + self.pad_width
                segment_mask = ptc[ind:, self.at_dim] <= at_dist_start + self.seg_len
                included_nums = np.sum(segment_mask)

            data = np.ones([self.seg_sample_num_max, self.ndim]) * -1
            mask = np.zeros([self.seg_sample_num_max, ])
            # sampling based on the ind
            if included_nums > self.seg_sample_num_max:
                data[: self.seg_sample_num_max, :] = ptc[ind: ind + self.seg_sample_num_max, :]
                included_nums = self.seg_sample_num_max
            else:
                data[: included_nums, :] = ptc[ind: ind + included_nums, :]
            #
            mask[:included_nums] = 1
            dm_segment = pdm[scaled_at_ind - self.pad_width: scaled_at_ind + self.dm_core_len + self.pad_width, :]
            dm_coords = np.ones([self.seg_sample_num_max, 2]) * -1
            # 坐标实际上包含了该区域的数据
            dm_coords[:included_nums, 0] = data[:included_nums,
                                           self.at_dim] / self.h_res - at_dist_start // self.h_res + self.pad_width
            dm_coords[:included_nums, 1] = (data[:included_nums, self.h_dim] - h_range[0]) / self.v_res

            # 进一步的，将其转变为tensor，并考虑数据增强
            data[:, -1] = self.land_reassign_func(data[:, -1]) if self.land_reassign_func is not None else data[:, -1]
            ptc_xy = copy.deepcopy(data[:, [self.at_dim, self.h_dim]])
            utm_xyz = copy.deepcopy(data[:, self.utm_dims])
            min_x = np.min(ptc_xy[:included_nums, 0])
            ptc_xy[:included_nums, 0] = (ptc_xy[:included_nums, 0] - min_x) / self.seg_len
            ptc_xy[:included_nums, 1] = 2 * (ptc_xy[:included_nums, 1] - h_range[0]) / h_scale - 1  # 高度转变到[-1, 1]
            # utm xyz同样需要归一化
            utm_x_min = np.min(utm_xyz[: included_nums, 0])
            utm_y_min = np.min(utm_xyz[: included_nums, 1]) 
            utm_xyz[: included_nums, 0] = (utm_xyz[: included_nums, 0] - utm_x_min) / self.seg_len
            utm_xyz[: included_nums, 1] = (utm_xyz[: included_nums, 1] - utm_y_min) / self.seg_len
            utm_xyz[: included_nums, 2] = 2 * (utm_xyz[:included_nums, 2] - h_range[0]) / h_scale - 1  # 高度转变到[-1, 1]
            # n_ptc_xy = ptc_xy.clone().detach().cpu().numpy()
            ptc_f = data[:, self.f_dim] # for cross-entropy loss, it should be long tensor
            dm = dm_segment / np.max(dm_segment)
            dm = dm[None, :, :]
            _mask = torch.Tensor(mask)

            self.ptc_xy.append(copy.deepcopy(ptc_xy))
            self.ptc_utms.append(copy.deepcopy(utm_xyz))
            self.ptc_f.append(copy.deepcopy(ptc_f))
            self.mask.append(mask)
            self.gt.append(copy.deepcopy(data[:, -1]))
            self.dmap.append(copy.deepcopy(dm))
            self.dm_coord.append(dm_coords)
            self.ptc_raw.append(data[:, :-1])
            self.filename.append(f_name)

        #
        print('The dataset has been updated......')
        return

    def __getitem__(self, item):
        ptc_xy = torch.Tensor(self.ptc_xy[item])
        utm = torch.Tensor(self.ptc_utms[item])
        ptc_f = torch.Tensor(self.ptc_f[item])
        ptc_r = torch.Tensor(self.ptc_raw[item])
        gt = torch.LongTensor(self.gt[item])  # for cross-entropy loss, it should be long tensor
        dm = torch.Tensor(self.dmap[item])
        dmc = torch.Tensor(self.dm_coord[item])
        _mask = torch.Tensor(self.mask[item])
        f_name = self.filename[item]

        # norm_xy, feat, mask, gt, s2dmap, dm_coord, f_name, raw
        return ptc_xy, utm, ptc_f, _mask, gt, dm, dmc, f_name, ptc_r

    def __len__(self):
        return len(self.candidates)


def dataloader(batch_size,
               data_dict,
               at_dist_dim=4,
               h_dim=-2,
               label_dim=-1,
               f_dims=None,
               s2d_h_res=10,
               s2d_v_res=0.5,
               s2d_pad_mode='symmetric',
               seg_len=100,
               seg_sample_num_max=1024,
               mini_valid_ratio=0.1,
               s2d_expand_ratios=9,
               random_shift_range=100,
               num_class=4,
               land_reassign='to_surface',
               is_training=True):
    # get the data
    rts_dataloader = RTS_Dataloader_V2(data_dict=data_dict, at_dist_dim=at_dist_dim, h_dim=h_dim, label_dim=label_dim,
                                       feature_dims=f_dims, s2d_h_res=s2d_h_res, s2d_v_res=s2d_v_res,
                                       s2d_pad_mode=s2d_pad_mode, seg_len=seg_len, seg_sample_num_max=seg_sample_num_max,
                                       mini_valid_ratio=mini_valid_ratio, s2d_expand_ratios=s2d_expand_ratios,
                                       random_shift_range=random_shift_range, num_class=num_class,
                                       land_reassign=land_reassign, is_training=is_training)

    datasets = torch.utils.data.DataLoader(rts_dataloader, batch_size=batch_size, shuffle=is_training, num_workers=1,
                                           pin_memory=True)

    return datasets


if __name__ == '__main__':
    data_dict = r'.\training'
    at_dist_dim = 4
    h_dim = -2
    label_dim = -1
    f_dims = [5]
    s2d_h_res = 2.5
    s2d_v_res = 0.5
    s2d_pad_mode = 'symmetric'
    seg_len = 100
    seg_sample_num_max = 1024
    mini_valid_ratio = 0.1
    s2d_expand_ratios = 9
    random_shift_range = 10
    num_class = 3
    land_reassign = 'to_surface'
    rts_dataloader = RTS_Dataloader_V2(data_dict=data_dict, at_dist_dim=at_dist_dim, h_dim=h_dim, label_dim=label_dim,
                                       feature_dims=f_dims, s2d_h_res=s2d_h_res, s2d_v_res=s2d_v_res,
                                       s2d_pad_mode=s2d_pad_mode, seg_len=seg_len, seg_sample_num_max=seg_sample_num_max,
                                       mini_valid_ratio=mini_valid_ratio, s2d_expand_ratios=s2d_expand_ratios,
                                       random_shift_range=random_shift_range, num_class=num_class,
                                       land_reassign=land_reassign, is_training=True)

    rts_dataloader.update_data()

    rts_dataloader.update_data()

    rts_dataloader.update_data()
