import copy
from PIL import Image
import cv2 as cv
import os

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def calculate_metric_percase(pred, gt):
    # 确保输入为 NumPy 数组
    pred = np.array(pred).astype(int)
    gt = np.array(gt).astype(int)

    # 检查形状一致性
    if pred.shape != gt.shape:
        raise ValueError("预测和标签形状不一致")

    # 计算交集和并集
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    
    sum_pred = np.sum(pred)
    sum_gt = np.sum(gt)
    sum_intersection = np.sum(intersection)
    sum_union = np.sum(union)

    # 处理边界情况
    if sum_pred + sum_gt == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0  # Dice, Precision, Recall, mIoU, Accuracy
    elif sum_pred == 0 or sum_gt == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # 计算各项指标
    dice = (2.0 * sum_intersection) / (sum_pred + sum_gt)
    
    # Precision = TP / (TP + FP)
    precision = sum_intersection / sum_pred if sum_pred != 0 else 0.0
    
    # Recall = TP / (TP + FN)
    recall = sum_intersection / sum_gt if sum_gt != 0 else 0.0
    
    # mIoU = intersection / union
    miou = sum_intersection / sum_union if sum_union != 0 else 0.0
    
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = np.sum(pred == gt) / pred.size
    
    # HD95
    hd95 = metric.binary.hd95(pred, gt)

    return dice, precision, recall, miou, accuracy

def test_single_volume(image, label, text, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # 处理原始三通道图像
    original_img = image.transpose(1, 2, 0)  # 转换为HxWxC格式

    _,x, y = image.shape
    if x != patch_size[0] or y != patch_size[1]:
        #缩放图像符合网络输入
        image = zoom(image, (1,patch_size[0] / x, patch_size[1] / y), order=3)
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input, text), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        if x != patch_size[0] or y != patch_size[1]:
            #缩放图像至原始大小
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        # 修改后的颜色映射（BGR格式）
        color_map = {
            1: (0, 255, 0, 0.6),       # 绿色 leaf
            2: (169, 169, 169, 0.7),   # 深灰色  Tea White Scab
            3: (255, 0, 0, 0.7),       # 蓝色  Worm Holes
            4: (153, 51, 255, 0.7),    # 粉色  Tea Sooty Mold
            5: (0, 0, 255, 0.7),       # 红色  Red Leaf Spot
            6: (170, 0, 127, 0.7),     # 紫色  Soft Rot
            7: (255, 255, 0, 0.7),     # 青色  Anthracnose
            8: (0, 255, 255, 0.7)      # 黄色  Algae Leaf Spot
        }

        # 归一化并转换为uint8（保持原始色彩）
        original_img = cv.normalize(original_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        # 直接使用原始图像作为背景（假设已经是BGR格式）
        background = original_img.copy()
        
        # 创建叠加层
        overlay = background.copy()
        
        # 为每个类别创建掩码并叠加颜色
        for value in color_map:
            mask = (prediction == value)
            if np.any(mask):
                bgr_color = color_map[value][:3]  # 取BGR值
                alpha = color_map[value][3]
                # 应用颜色到叠加层
                overlay[mask] = bgr_color
                
        # 混合原始图像和叠加层
        blended = cv.addWeighted(overlay, alpha, background, 1 - alpha, 0)

        # 保存结果
        cv.imwrite(os.path.join(test_save_path, f'{case}.png'), blended)

    return metric_list
