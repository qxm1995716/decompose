import numpy as np
from sympy.logic.inference import valid


def get_confusion_matrix(pred, gt, mask, n_class):
    pred = pred[mask == 1]
    gt = gt[mask == 1]
    cf_matrix = np.bincount(pred * n_class + gt, minlength=n_class ** 2)
    cf_matrix = cf_matrix.reshape(n_class, n_class)
    return cf_matrix


def miou(cf_matrix):
    n_class = cf_matrix.shape[0]
    ious = []
    for idx in range(n_class):
        iou = cf_matrix[idx, idx] / (np.sum(cf_matrix[:, idx]) + np.sum(cf_matrix[idx, :]) - cf_matrix[idx, idx])
        if not np.isnan(iou):
            ious.append(iou)
    ious = np.array(ious)
    miou = np.mean(ious)
    return miou


def accuracy(pred, gt, mask, n_class):
    pred = pred[mask == 1]
    gt = gt[mask == 1]
    acc = np.zeros([n_class, 2])
    for idx in range(n_class):
        label_mask = gt == idx
        n_pred = np.sum(pred[label_mask] == idx)
        acc[idx, 0] = n_pred
        acc[idx, 1] = np.sum(label_mask)

    return acc


def accuracy_of_file(pred, gt, n_class):
    acc = np.zeros([n_class, 2])
    for idx in range(n_class):
        valid_mask = gt == idx
        pred_masked = pred[valid_mask]
        correct_num = np.sum(pred_masked == idx)
        total_num = np.sum(valid_mask)
        acc[idx, 0] = correct_num
        acc[idx, 1] = total_num

    return acc


def get_confusion_matrix_wo_mask(pred, gt, n_class):
    cf_matrix = np.bincount(pred * n_class + gt, minlength=n_class ** 2)
    cf_matrix = cf_matrix.reshape(n_class, n_class)
    return cf_matrix


def print_acc(acc_mat, n_class):
    category_ = ['Noise', 'Seafloor', 'Surface', 'Land']
    p =''
    for idx in range(n_class):
        p += category_[idx] + ' Acc is {:.6f} ({:.2f})'.format(acc_mat[idx, 0] / acc_mat[idx, 1], 100* acc_mat[idx, 1] / np.sum(acc_mat[:, 1])) + '; '
    overall_acc = np.sum(acc_mat[:, 0]) / np.sum(acc_mat[:, 1])
    p += 'Overall Acc is {:.6f}'.format(overall_acc) + '.'
    return p
