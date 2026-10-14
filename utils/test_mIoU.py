from PIL import Image
import numpy as np
import torch
from collections import OrderedDict

def imread(path):
    img = Image.open(path)
    array = np.array(img)
    if array.ndim >= 3 and array.shape[2] >= 3:  # color image
        array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    return array

def intersect_and_union(pred_label_path,
                        label_path,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    # 强制加载到 CPU
    pred_label = torch.load(pred_label_path, map_location='cpu')
    
    # 如果保存的是 numpy 数组，转换为 tensor
    if isinstance(pred_label, np.ndarray):
        pred_label = torch.from_numpy(pred_label)
        
    label = torch.from_numpy(imread(label_path))

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
            
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    # 统一强制使用 CPU
    device = torch.device('cpu')

    mask = (label != ignore_index)
    pred_label = pred_label[mask].to(device)
    label = label[mask].to(device)

    # 计算交集
    intersect = pred_label[pred_label == label]
    
    # 全部在 CPU 上进行直方图统计
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    
    return area_intersect, area_union, area_pred_label, area_label

def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):

    device = torch.device('cpu')
    total_area_intersect = torch.zeros((num_classes,), dtype=torch.float64).to(device)
    total_area_union = torch.zeros((num_classes,), dtype=torch.float64).to(device)
    total_area_pred_label = torch.zeros((num_classes,), dtype=torch.float64).to(device)
    total_area_label = torch.zeros((num_classes,), dtype=torch.float64).to(device)
    
    img_id = 0
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        

        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
        
        img_id += 1
        if img_id % 50 == 0: # 减少打印频率
            print(f'Evaluated {img_id} images...')
            
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label

def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
    if isinstance(metrics, str):
        metrics = [metrics]
    
    # 全局准确率aAcc
    total_area_label_sum = total_area_label.sum()
    all_acc = total_area_intersect.sum() / (total_area_label_sum + 1e-6)
    ret_metrics = OrderedDict({'aAcc': all_acc})
    
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / (total_area_union + 1e-6)
            acc = total_area_intersect / (total_area_label + 1e-6)
            precision = total_area_intersect / (total_area_pred_label + 1e-6)
            f1_score = 2 * precision * acc / (precision + acc + 1e-6)
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
            ret_metrics['Prec'] = precision
            ret_metrics['F1'] = f1_score
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label + 1e-6)
            acc = total_area_intersect / (total_area_label + 1e-6)
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc

    # 转换为 numpy
    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics

def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
        results, gt_seg_maps, num_classes, ignore_index, label_map,
        reduce_zero_label)
    
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)
    return ret_metrics

def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result
