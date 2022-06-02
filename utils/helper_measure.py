import numpy as np
import os

import torch


def cal_precision_recall_mae(prediction, gt):

    assert prediction.shape == gt.shape
    mae1 = torch.mean(torch.abs(prediction - gt))
    print(mae1)
    loss = torch.nn.L1Loss()
    mae = loss(prediction,gt)
    eps = 1e-4
    # rescale 0-1 to 0-255
    prediction = prediction * 255.
    t = torch.sum(gt)

    precision, recall = [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = torch.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1

        tp = torch.sum(hard_prediction * gt)
        p = torch.sum(hard_prediction)

        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall, mae


def unit_test():
    prediction = torch.randn((2,2))
    gt  = torch.zeros((2,2))
    print(prediction)
    precision, recall, mae = cal_precision_recall_mae(prediction,gt)
    print('precision:{}, recall:{}, mae:{}'.format(precision, recall, mae))


if __name__ == "__main__":
    unit_test()
