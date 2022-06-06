import numpy as np
import os
import copy
import torch

#
# input: prediction(0,255)
# input2: label(0,1)
#
class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# measure torch
def cal_precision_recall_mae(pre, gt):
    prediction = copy.deepcopy(pre)
    prediction = prediction / 255.
    assert prediction.shape == gt.shape
    mae1 = torch.mean(torch.abs(prediction - gt))
    eps = 1e-4
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

    return precision, recall, mae1


def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])
    return max_fmeasure

def cal_Jaccard(pre, gt):
    # KNOWN AS IoU
    # gt in eange(0,1)
    # pre in range (0,255)

    assert pre.shape == gt.shape
    prediction = copy.deepcopy(pre)
    prediction = prediction / 255.
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0
    # print(prediction)
    Jaccard =torch.sum(prediction * gt) / (torch.sum(torch.logical_or(gt,prediction)))


    return Jaccard

def cal_BER(pre, label, thr = 127.5):
    # print('==========ber========')
    prediction = torch.zeros(pre.shape)
    label_tmp = label
    # print(prediction)
    # print(pre > thr)
    prediction[pre > thr] = 1
    eps = 1e-4
    # print(prediction)
    # print(label_tmp)
    TP = torch.sum(prediction * label_tmp)
    TN = torch.sum((1 - prediction) * (1 - label_tmp))
    # Shadow pixels, non-shaodow pixel
    Np = torch.sum(label_tmp)
    Nn = torch.sum((1-label_tmp))
    # print('tp,tn,np,nf',TP,TN,Np,Nn)
    shadow_BER = (1 - TP / (Np+eps)) * 100
    non_shadow_BER = (1 - TN / (Nn+eps)) * 100
    BER = 1 / 2 * (shadow_BER + non_shadow_BER)

    return BER, shadow_BER, non_shadow_BER


def unit_test():
    prediction = torch.randn((2,1,3,3)).uniform_(0, 255)
    gt  = torch.ones((2,1,3,3))
    print(gt)
    print(prediction)
    precision_record, recall_record, mae = cal_precision_recall_mae(prediction,gt)
    BER, shadow_BER, non_shadow_BER = cal_BER(prediction, gt)
    Jaccard = cal_Jaccard(prediction, gt)
    fmeasure = cal_fmeasure([precord for precord in precision_record],
                            [rrecord for rrecord in recall_record])
    log = 'MAE:{}, F-beta:{}, Jaccard:{}, BER:{}, SBER:{}, non-SBER:{}'.format(mae, fmeasure,
                                                                               Jaccard, BER,
                                                                               shadow_BER,
                                                                               non_shadow_BER)
    print(log)



if __name__ == "__main__":
    unit_test()
